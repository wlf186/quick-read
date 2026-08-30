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

lock_value() {
  "$PROJECT_ROOT/.venv/bin/python" - "$LOCK_FILE" "$1" "$PLATFORM" "$2" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
print(data[sys.argv[2]][sys.argv[3]][sys.argv[4]])
PY
}

download_verified() {
  local tool="$1" url sha destination partial actual
  url="$(lock_value "$tool" url)"
  sha="$(lock_value "$tool" sha256)"
  destination="$DOWNLOAD_DIR/${url##*/}"
  partial="$destination.part"
  if [[ -f "$destination" ]]; then
    actual="$(sha256sum "$destination" | awk '{print $1}')"
    if [[ "$actual" != "$sha" ]]; then
      echo "[$tool] cached archive checksum mismatch; downloading again" >&2
      unlink "$destination"
    fi
  fi
  if [[ ! -f "$destination" ]]; then
    echo "[$tool] downloading ${url##*/}" >&2
    curl --fail --location --retry 6 --retry-all-errors --connect-timeout 20 --speed-time 30 --speed-limit 1024 --continue-at - --output "$partial" "$url"
    actual="$(sha256sum "$partial" | awk '{print $1}')"
    [[ "$actual" == "$sha" ]] || { echo "[$tool] SHA-256 verification failed" >&2; exit 1; }
    mv "$partial" "$destination"
  fi
  actual="$(sha256sum "$destination" | awk '{print $1}')"
  [[ "$actual" == "$sha" ]] || { echo "[$tool] SHA-256 verification failed" >&2; exit 1; }
  printf '%s\n' "$destination"
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
