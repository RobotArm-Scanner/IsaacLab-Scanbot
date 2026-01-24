#!/usr/bin/env python3
"""Example client for Scanbot exec_bridge.

Usage:
  export SCANBOT_EXEC_HOST=127.0.0.1
  export SCANBOT_EXEC_PORT=7311
  export SCANBOT_EXEC_TOKEN=optional_token
  python scanbot/snippets/example.py

Server side:
  SCANBOT_ENABLE_CODE_EXEC=1 /workspace/isaaclab/scanbot/bin/scanbot.sh
"""

from __future__ import annotations

import json
import os
import urllib.request


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = os.getenv("SCANBOT_EXEC_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    host = os.getenv("SCANBOT_EXEC_HOST", "127.0.0.1")
    port = int(os.getenv("SCANBOT_EXEC_PORT", "7311"))
    url = f"http://{host}:{port}/exec"

    payload = {
        "mode": "exec",
        "code": "print('hello from Isaac Sim')\nresult = 2 + 2",
        "result_expr": "result",
        "timeout_sec": 5.0,
    }
    resp = _post_json(url, payload)
    print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()
