#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/runtime/run/server.pid"
EXECUTABLE="$PROJECT_ROOT/.venv/bin/sandevistan-read"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Sandevistan-Read is not running (no PID file)."
  exit 0
fi

PID="$(<"$PID_FILE")"
if [[ ! "$PID" =~ ^[1-9][0-9]*$ ]]; then
  rm -f "$PID_FILE"
  echo "Removed an invalid PID file; no process was stopped." >&2
  exit 1
fi

if ! kill -0 "$PID" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "Removed a stale PID file; Sandevistan-Read was not running."
  exit 0
fi

COMMAND="$(ps -p "$PID" -o command= 2>/dev/null || true)"
if [[ "$COMMAND" != *"$EXECUTABLE"* ]]; then
  rm -f "$PID_FILE"
  echo "PID $PID belongs to another process; it was not stopped." >&2
  exit 1
fi

if ! kill "$PID" 2>/dev/null; then
  echo "Could not signal Sandevistan-Read process $PID; the PID file was retained." >&2
  exit 1
fi

for _ in {1..30}; do
  if ! kill -0 "$PID" 2>/dev/null; then
    if [[ -f "$PID_FILE" && "$(<"$PID_FILE")" == "$PID" ]]; then
      rm -f "$PID_FILE"
    fi
    echo "Sandevistan-Read stopped."
    exit 0
  fi
  sleep 0.2
done

echo "Process $PID did not stop; the PID file was retained." >&2
exit 1
