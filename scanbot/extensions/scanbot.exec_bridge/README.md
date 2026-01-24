# scanbot.exec_bridge

Lightweight HTTP bridge for executing Python code inside a running Isaac Sim session.

## Enable

The extension is disabled by default. Enable it via environment variable when launching Isaac Sim:

```
SCANBOT_ENABLE_CODE_EXEC=1 /workspace/isaaclab/scanbot/bin/scanbot.sh
```

## Endpoints

- `GET /health` -> `{ "ok": true }`
- `POST /exec` -> execute Python code

Request body:

```
{
  "code": "print('hello')",
  "mode": "exec",               // exec or eval (default: exec)
  "result_expr": "answer",      // optional, evaluated after exec
  "timeout_sec": 5.0            // optional
}
```

Response body (success or error):

```
{
  "ok": true/false,
  "stdout": "...",
  "stderr": "...",
  "result": ...,
  "error": "string or null",
  "traceback": "string or null",
  "duration_sec": 0.123
}
```

## Config via env

- `SCANBOT_EXEC_BRIDGE_HOST` (default `127.0.0.1`)
- `SCANBOT_EXEC_BRIDGE_PORT` (default `7311`)
- `SCANBOT_EXEC_BRIDGE_TOKEN` (optional; if set, require auth)
- `SCANBOT_EXEC_MAX_BYTES` (default `1000000`)
- `SCANBOT_EXEC_TIMEOUT_SEC` (default `5.0`)

If `SCANBOT_EXEC_BRIDGE_TOKEN` is set, send it as:

- `Authorization: Bearer <token>` or
- `X-Scanbot-Token: <token>`

## Usage (bin helper)

Use the helper to submit a snippet file:

```
docker exec isaac-lab-scanbot /workspace/isaaclab/scanbot/bin/exec_submit.sh \
  /workspace/isaaclab/scanbot/snippets/example.py --result answer
```

Trigger a failing snippet:

```
docker exec isaac-lab-scanbot /workspace/isaaclab/scanbot/bin/exec_submit.sh \
  /workspace/isaaclab/scanbot/snippets/exec_error.py
```

## Notes and safety

- Requests run on Isaac Sim's main thread via `scanbot_context.enqueue_hook`.
- Globals persist between requests (`__scanbot_exec__`).
- `stdout`/`stderr` and exceptions are returned in the HTTP response only.
- Do not expose this beyond localhost unless you fully trust the network.
