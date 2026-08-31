#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="$PROJECT_ROOT/scripts/tools.lock.json"
DOWNLOAD_DIR="$PROJECT_ROOT/runtime/cache/downloads"
TEMP_ROOT="$PROJECT_ROOT/runtime/tmp"
TOOLS_ROOT="$PROJECT_ROOT/.tools"
mkdir -p "$DOWNLOAD_DIR" "$TEMP_ROOT" "$TOOLS_ROOT"

case "$(uname -m)" in
  x86_64|amd64) PLATFORM="linux-x86_64" ;;
  aarch64|arm64) PLATFORM="linux-arm64" ;;
  *) echo "Unsupported Linux architecture: $(uname -m)" >&2; exit 1 ;;
esac

download_verified() {
  "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/fetch-tool.py" \
    --lock "$LOCK_FILE" --tool "$1" --platform "$PLATFORM" --download-dir "$DOWNLOAD_DIR"
}

if [[ ! -x "$TOOLS_ROOT/ffmpeg/bin/ffmpeg" ]]; then
  FFMPEG_ARCHIVE="$(download_verified ffmpeg)"
  STAGE_DIR="$(mktemp -d "$TEMP_ROOT/ffmpeg-install.XXXXXX")"
  trap 'rm -rf "$STAGE_DIR"' EXIT
  tar -xJf "$FFMPEG_ARCHIVE" -C "$STAGE_DIR"
  FFMPEG_SOURCE="$(find "$STAGE_DIR" -type f -path '*/bin/ffmpeg' -print -quit)"
  FFPROBE_SOURCE="$(find "$STAGE_DIR" -type f -path '*/bin/ffprobe' -print -quit)"
  [[ -n "$FFMPEG_SOURCE" && -n "$FFPROBE_SOURCE" ]] || { echo "FFmpeg archive layout is invalid" >&2; exit 1; }
  mkdir -p "$TOOLS_ROOT/ffmpeg/bin"
  install -m 0755 "$FFMPEG_SOURCE" "$TOOLS_ROOT/ffmpeg/bin/ffmpeg"
  install -m 0755 "$FFPROBE_SOURCE" "$TOOLS_ROOT/ffmpeg/bin/ffprobe"
  rm -rf "$STAGE_DIR"; trap - EXIT
fi

if [[ ! -x "$TOOLS_ROOT/libreoffice/program/soffice" ]]; then
  command -v dpkg-deb >/dev/null || { echo "dpkg-deb is required to unpack LibreOffice" >&2; exit 1; }
  LIBREOFFICE_ARCHIVE="$(download_verified libreoffice)"
  STAGE_DIR="$(mktemp -d "$TEMP_ROOT/libreoffice-install.XXXXXX")"
  trap 'rm -rf "$STAGE_DIR"' EXIT
  mkdir -p "$STAGE_DIR/archive" "$STAGE_DIR/root"
  tar -xzf "$LIBREOFFICE_ARCHIVE" -C "$STAGE_DIR/archive"
  while IFS= read -r -d '' package; do
    case "$package" in */desktop-integration/*) continue ;; esac
    dpkg-deb -x "$package" "$STAGE_DIR/root"
  done < <(find "$STAGE_DIR/archive" -type f -name '*.deb' -print0)
  SOFFICE_SOURCE="$(find "$STAGE_DIR/root/opt" -type f -path '*/program/soffice' -print -quit)"
  [[ -n "$SOFFICE_SOURCE" ]] || { echo "LibreOffice archive layout is invalid" >&2; exit 1; }
  LIBREOFFICE_SOURCE="$(dirname "$(dirname "$SOFFICE_SOURCE")")"
  mv "$LIBREOFFICE_SOURCE" "$TOOLS_ROOT/libreoffice"
  rm -rf "$STAGE_DIR"; trap - EXIT
fi

"$TOOLS_ROOT/ffmpeg/bin/ffmpeg" -version | head -1
SAL_USE_VCLPLUGIN=svp "$TOOLS_ROOT/libreoffice/program/soffice" --headless --version
echo "Project-local media tools are ready."
