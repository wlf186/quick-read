#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/runtime/run/server.pid"
if [[ ! -f "$PID_FILE" ]]; then echo "Sandevistan-Read is not running"; exit 0; fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then kill "$PID"; for _ in {1..30}; do kill -0 "$PID" 2>/dev/null || break; sleep .2; done; fi
rm -f "$PID_FILE"
echo "Sandevistan-Read stopped"
