#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SANDEVISTAN_PROJECT_ROOT="$PROJECT_ROOT"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"
export UV_CACHE_DIR="$PROJECT_ROOT/runtime/cache/uv"
export XDG_CACHE_HOME="$PROJECT_ROOT/runtime/cache"
export TMPDIR="$PROJECT_ROOT/runtime/tmp"
mkdir -p "$PROJECT_ROOT/runtime/run" "$PROJECT_ROOT/runtime/logs" "$PROJECT_ROOT/runtime/tmp"
if [[ -f "$PROJECT_ROOT/runtime/run/server.pid" ]] && kill -0 "$(cat "$PROJECT_ROOT/runtime/run/server.pid")" 2>/dev/null; then echo "Sandevistan-Read is already running"; exit 0; fi
nohup "$PROJECT_ROOT/.venv/bin/sandevistan-read" >>"$PROJECT_ROOT/runtime/logs/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$PROJECT_ROOT/runtime/run/server.pid"
read -r BIND_HOST BIND_PORT < <("$PROJECT_ROOT/.venv/bin/python" -c "from sandevistan_read.config import CONFIG; print(CONFIG.server.host, CONFIG.server.port)")
for _ in {1..30}; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    rm -f "$PROJECT_ROOT/runtime/run/server.pid"
    echo "Sandevistan-Read failed to start; inspect runtime/logs/server.log" >&2
    exit 1
  fi
  if "$PROJECT_ROOT/.venv/bin/python" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$BIND_PORT/', timeout=1)" >/dev/null 2>&1; then
    sleep 0.5
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Sandevistan-Read started: http://$BIND_HOST:$BIND_PORT (local access: http://127.0.0.1:$BIND_PORT)"
      exit 0
    fi
  fi
  sleep 0.2
done
kill "$SERVER_PID" 2>/dev/null || true
rm -f "$PROJECT_ROOT/runtime/run/server.pid"
echo "Sandevistan-Read did not become healthy; inspect runtime/logs/server.log" >&2
exit 1
