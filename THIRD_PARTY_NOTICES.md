# Third-party tools

Sandevistan-Read downloads these unmodified executables into the project working directory during bootstrap. They are not part of the Sandevistan-Read source code license.

## FFmpeg

- Version line: FFmpeg 9.0 release series
- Pinned build: `autobuild-2026-09-04-14-01`, assets with version `n9.0.1-11-ge47273f4d9`
- Build provider: BtbN/FFmpeg-Builds, linked from the FFmpeg download page
- Variant: LGPL, static
- Project: https://ffmpeg.org/
- Build source: https://github.com/BtbN/FFmpeg-Builds
- License information: https://ffmpeg.org/legal.html

## LibreOffice

- Version: 26.2.5
- Provider: The Document Foundation
- Package source: https://download.documentfoundation.org/libreoffice/stable/26.2.5/
- License information: https://www.libreoffice.org/about-us/licenses/

[`scripts/tools.lock.json`](scripts/tools.lock.json) records the FFmpeg Release API URL with a fixed tag and exact asset names for each platform. At download time, `scripts/fetch-tool.py` reads the selected asset's download API URL and SHA-256 digest from that Release's metadata; the FFmpeg digests are not embedded in the lock file. LibreOffice uses fixed download URLs and SHA-256 digests recorded directly in the lock file. Both paths verify the archive digest before installation.

Bootstrap reuses an existing FFmpeg executable. Changing the lock file and rerunning bootstrap does not replace an already installed build; stop the application and move the old `.tools/ffmpeg` directory aside before rerunning bootstrap when a replacement is intended.
