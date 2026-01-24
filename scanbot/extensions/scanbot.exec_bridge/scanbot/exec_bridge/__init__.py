"""HTTP bridge for executing code in the running Isaac Sim session."""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import omni.ext
import omni.kit.app

import carb

from scanbot.scripts import scanbot_context

import numpy as np  # type: ignore
import torch  # type: ignore
import isaaclab.sim as sim_utils  # type: ignore
from scanbot.common import pos_util  # type: ignore


@dataclass
class _ExecJob:
    code: str
    mode: str
    result_expr: str | None
    timeout_sec: float
    created_time: float
    done_event: threading.Event
    response: dict[str, Any] | None = None


class _ExecHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, bridge) -> None:
        super().__init__(server_address, handler_cls)
        self.bridge = bridge


class _ExecHandler(BaseHTTPRequestHandler):
    server: _ExecHTTPServer  # type: ignore[assignment]

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _parse_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        max_len = self.server.bridge.max_body_bytes
        if length <= 0 or length > max_len:
            self._send_json(413, {"ok": False, "error": "invalid content length"})
            return None
        try:
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": f"invalid json: {exc}"})
            return None

    def _check_auth(self) -> bool:
        token = self.server.bridge.token
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            return header.split(" ", 1)[1].strip() == token
        alt = self.headers.get("X-Scanbot-Token", "")
        return alt == token

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_auth():
            self._send_json(401, {"ok": False, "error": "unauthorized"})
            return
        path = self.path.split("?", 1)[0]
        if path != "/exec":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        payload = self._parse_json()
        if payload is None:
            return
        resp, status = self.server.bridge.handle_exec_request(payload)
        self._send_json(status, resp)

    def log_message(self, _format: str, *_args) -> None:  # noqa: D401, N802
        """Silence default HTTP server logging."""


class Extension(omni.ext.IExt):
    def __init__(self) -> None:
        super().__init__()
        self._ext_id = ""
        self._server: _ExecHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._exec_globals: dict[str, Any] = {"__name__": "__scanbot_exec__"}

        self._enabled = False
        self._host = "127.0.0.1"
        self._port = 7311
        self._token = ""
        self._max_body_bytes = 1_000_000
        self._default_timeout = 5.0

    @property
    def token(self) -> str:
        return self._token

    @property
    def max_body_bytes(self) -> int:
        return self._max_body_bytes

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._enabled = str(os.getenv("SCANBOT_ENABLE_CODE_EXEC", "0")).lower() in {"1", "true", "yes"}
        if not self._enabled:
            carb.log_info("[scanbot.exec_bridge] Disabled (set SCANBOT_ENABLE_CODE_EXEC=1 to enable).")
            return

        self._host = os.getenv("SCANBOT_EXEC_BRIDGE_HOST", "127.0.0.1")
        self._port = int(os.getenv("SCANBOT_EXEC_BRIDGE_PORT", "7311"))
        self._token = os.getenv("SCANBOT_EXEC_BRIDGE_TOKEN", "")
        self._max_body_bytes = int(os.getenv("SCANBOT_EXEC_MAX_BYTES", "1000000"))
        self._default_timeout = float(os.getenv("SCANBOT_EXEC_TIMEOUT_SEC", "5.0"))

        try:
            self._server = _ExecHTTPServer((self._host, self._port), _ExecHandler, bridge=self)
        except Exception as exc:
            carb.log_error(f"[scanbot.exec_bridge] Failed to bind {self._host}:{self._port}: {exc}")
            self._server = None
            return

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        carb.log_info(f"[scanbot.exec_bridge] Listening on http://{self._host}:{self._port}")

    def on_shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def handle_exec_request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        code = payload.get("code")
        if not isinstance(code, str) or not code.strip():
            return {"ok": False, "error": "code is required"}, 400

        mode = str(payload.get("mode", "exec")).lower()
        if mode not in {"exec", "eval"}:
            return {"ok": False, "error": "mode must be exec or eval"}, 400

        result_expr = payload.get("result_expr")

        timeout = float(payload.get("timeout_sec", self._default_timeout))
        timeout = max(0.1, min(timeout, 120.0))

        job = _ExecJob(
            code=code,
            mode=mode,
            result_expr=result_expr,
            timeout_sec=timeout,
            created_time=time.monotonic(),
            done_event=threading.Event(),
        )
        scanbot_context.enqueue_hook(lambda: self._run_job(job))

        if not job.done_event.wait(timeout=timeout):
            return {"ok": False, "error": "timeout waiting for execution"}, 504

        return job.response or {"ok": False, "error": "no response"}, 200

    def _run_job(self, job: _ExecJob) -> None:
        start = time.monotonic()
        stdout = io.StringIO()
        stderr = io.StringIO()
        result = None
        error = None
        tb = None

        exec_globals = self._build_exec_globals()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                if job.mode == "eval":
                    result = eval(job.code, exec_globals, exec_globals)
                else:
                    exec(job.code, exec_globals, exec_globals)
                    if job.result_expr:
                        result = eval(job.result_expr, exec_globals, exec_globals)
        except Exception as exc:  # pragma: no cover
            error = str(exc)
            tb = traceback.format_exc()

        duration = time.monotonic() - start
        job.response = {
            "ok": error is None,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "result": self._safe_json(result),
            "error": error,
            "traceback": tb,
            "duration_sec": duration,
        }
        job.done_event.set()

    def _build_exec_globals(self) -> dict[str, Any]:
        g = self._exec_globals
        g["env"] = scanbot_context.get_env()
        g["app"] = omni.kit.app.get_app()
        g["scanbot_context"] = scanbot_context
        g["np"] = np
        g["torch"] = torch
        g["sim_utils"] = sim_utils
        g["pos_util"] = pos_util
        return g

    @staticmethod
    def _safe_json(value: Any) -> Any:
        if value is None:
            return None
        try:
            json.dumps(value)
            return value
        except Exception:
            return repr(value)
