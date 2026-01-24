#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <snippet.py> [--mode exec|eval] [--result EXPR] [--timeout SEC]" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

snippet=$1
shift

mode="exec"
result_expr=""
timeout="5.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="$2"
      shift 2
      ;;
    --result)
      result_expr="$2"
      shift 2
      ;;
    --timeout)
      timeout="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "$snippet" ]]; then
  echo "Snippet not found: $snippet" >&2
  exit 1
fi

HOST=${SCANBOT_EXEC_BRIDGE_HOST:-127.0.0.1}
PORT=${SCANBOT_EXEC_BRIDGE_PORT:-7311}
TOKEN=${SCANBOT_EXEC_BRIDGE_TOKEN:-}

python3 - "$snippet" "$mode" "$result_expr" "$timeout" "$HOST" "$PORT" "$TOKEN" <<'PY'
import json
import sys
import urllib.error
import urllib.request

snippet, mode, result_expr, timeout, host, port, token = sys.argv[1:]
with open(snippet, "r", encoding="utf-8") as f:
    code = f.read()

payload = {
    "code": code,
    "mode": mode,
    "timeout_sec": float(timeout),
}
if result_expr:
    payload["result_expr"] = result_expr

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    f"http://{host}:{port}/exec",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
if token:
    req.add_header("Authorization", f"Bearer {token}")

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        print(f"HTTP {resp.status}")
        print(body)
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8")
    print(f"HTTP {exc.code}")
    print(body)
PY
