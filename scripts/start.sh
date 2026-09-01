#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SANDEVISTAN_PROJECT_ROOT="$PROJECT_ROOT"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"
export UV_CACHE_DIR="$PROJECT_ROOT/runtime/cache/uv"
export XDG_CACHE_HOME="$PROJECT_ROOT/runtime/cache"
export TMPDIR="$PROJECT_ROOT/runtime/tmp"

RUN_DIR="$PROJECT_ROOT/runtime/run"
LOG_DIR="$PROJECT_ROOT/runtime/logs"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$LOG_DIR/server.log"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
EXECUTABLE="$PROJECT_ROOT/.venv/bin/sandevistan-read"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$PROJECT_ROOT/runtime/tmp"

if [[ ! -x "$PYTHON" || ! -x "$EXECUTABLE" ]]; then
  echo "Application environment is missing. Run ./scripts/bootstrap.sh first." >&2
  exit 1
fi

process_command() {
  ps -p "$1" -o command= 2>/dev/null || true
}

is_server_process() {
  local pid="$1"
  local command
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command="$(process_command "$pid")"
  [[ "$command" == *"$EXECUTABLE"* ]]
}

remove_pid_file_if_matches() {
  local pid="$1"
  local recorded=""
  [[ -f "$PID_FILE" ]] || return 0
  recorded="$(<"$PID_FILE")"
  if [[ "$recorded" == "$pid" ]]; then
    rm -f "$PID_FILE"
  fi
}

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(<"$PID_FILE")"
  if is_server_process "$EXISTING_PID"; then
    echo "Sandevistan-Read is already running (PID $EXISTING_PID)"
    exit 0
  fi
  if [[ "$EXISTING_PID" =~ ^[1-9][0-9]*$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Ignoring stale PID file: PID $EXISTING_PID belongs to another process." >&2
  fi
  rm -f "$PID_FILE"
fi

read -r BIND_HOST BIND_PORT < <(
  "$PYTHON" -c "from sandevistan_read.config import CONFIG; print(CONFIG.server.host, CONFIG.server.port)"
)
HEALTH_URL="http://127.0.0.1:$BIND_PORT/auth/status"

SERVER_PID="$(
  "$PYTHON" - "$EXECUTABLE" "$LOG_FILE" "$PROJECT_ROOT" <<'PY'
import subprocess
import sys

executable, log_path, project_root = sys.argv[1:]
with open(log_path, "ab", buffering=0) as log:
    process = subprocess.Popen(
        [executable],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
print(process.pid, flush=True)
PY
)"

PID_TEMP="$PID_FILE.$$"
printf '%s\n' "$SERVER_PID" >"$PID_TEMP"
mv -f "$PID_TEMP" "$PID_FILE"

STARTUP_COMPLETE=0
cleanup_failed_start() {
  if [[ "$STARTUP_COMPLETE" != "1" ]]; then
    if is_server_process "$SERVER_PID"; then
      kill "$SERVER_PID" 2>/dev/null || true
    fi
    remove_pid_file_if_matches "$SERVER_PID"
  fi
}
trap cleanup_failed_start EXIT INT TERM

STARTUP_DEADLINE=$((SECONDS + 60))
while (( SECONDS < STARTUP_DEADLINE )); do
  if ! is_server_process "$SERVER_PID"; then
    echo "Sandevistan-Read failed to start; inspect runtime/logs/server.log" >&2
    exit 1
  fi
  if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('$HEALTH_URL', timeout=1)" >/dev/null 2>&1; then
    STARTUP_COMPLETE=1
    trap - EXIT INT TERM
    echo "Sandevistan-Read started: http://$BIND_HOST:$BIND_PORT (local access: http://127.0.0.1:$BIND_PORT, PID $SERVER_PID)"
    exit 0
  fi
  sleep 0.25
done

echo "Sandevistan-Read did not become healthy within 60 seconds; inspect runtime/logs/server.log" >&2
exit 1
