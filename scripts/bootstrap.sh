#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SANDEVISTAN_PROJECT_ROOT="$PROJECT_ROOT"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"
export UV_CACHE_DIR="$PROJECT_ROOT/runtime/cache/uv"
export COREPACK_HOME="$PROJECT_ROOT/runtime/cache/corepack"
export PNPM_HOME="$PROJECT_ROOT/.tools/pnpm"
export XDG_CACHE_HOME="$PROJECT_ROOT/runtime/cache"
export TMPDIR="$PROJECT_ROOT/runtime/tmp"
mkdir -p "$PROJECT_ROOT/.tools/bin" "$PROJECT_ROOT/runtime/cache" "$PROJECT_ROOT/runtime/tmp"
UV_BIN="$PROJECT_ROOT/.tools/bin/uv"
if [[ ! -x "$UV_BIN" ]]; then
  echo "[bootstrap] downloading uv into .tools/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PROJECT_ROOT/.tools/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
"$UV_BIN" sync --extra ai --extra dev --frozen 2>/dev/null || "$UV_BIN" sync --extra ai --extra dev
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/install-models.py"
if [[ "${SANDEVISTAN_SKIP_TOOLS:-0}" != "1" ]]; then
  "$PROJECT_ROOT/scripts/install-tools.sh"
fi
cd "$PROJECT_ROOT/frontend"
corepack pnpm install --store-dir "$PROJECT_ROOT/runtime/cache/pnpm-store" --frozen-lockfile 2>/dev/null || corepack pnpm install --store-dir "$PROJECT_ROOT/runtime/cache/pnpm-store"
corepack pnpm build
echo "[bootstrap] ready — run scripts/start.sh"
