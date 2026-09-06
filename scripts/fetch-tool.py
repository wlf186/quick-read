from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Mapping
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
USER_AGENT = "sandevistan-read-tool-installer/1"


@dataclass(frozen=True)
class ResolvedArchive:
    name: str
    url: str
    sha256: str
    headers: dict[str, str]


OpenUrl = Callable[[Request, float], BinaryIO]


class ToolRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: BinaryIO | None, code: int, msg: str,
        headers: Mapping[str, str], newurl: str,
    ) -> Request | None:
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlparse(req.full_url)[:2] != urlparse(newurl)[:2]:
            redirected.remove_header("Authorization")
        return redirected


def _open(request: Request, timeout: float) -> BinaryIO:
    return build_opener(ToolRedirectHandler()).open(request, timeout=timeout)


def _github_headers(url: str, accept: str) -> dict[str, str]:
    headers = {"Accept": accept, "User-Agent": USER_AGENT, "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and urlparse(url)[:2] == ("https", "api.github.com"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str, opener: OpenUrl) -> dict:
    request = Request(
        url,
        headers=_github_headers(url, "application/vnd.github+json"),
    )
    with opener(request, 30) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_archive(lock: dict, tool: str, platform: str, opener: OpenUrl = _open) -> ResolvedArchive:
    try:
        tool_config = lock[tool]
        entry = tool_config[platform]
    except KeyError as exc:
        raise ValueError(f"Unsupported tool or platform: {tool}/{platform}") from exc

    source = tool_config.get("source", {"kind": "fixed"})
    kind = source.get("kind", "fixed")
    if kind == "fixed":
        url = str(entry.get("url", ""))
        digest = str(entry.get("sha256", "")).lower()
        name = Path(urlparse(url).path).name
        if not url.startswith("https://") or not name or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"Invalid fixed archive metadata for {tool}/{platform}")
        return ResolvedArchive(name=name, url=url, sha256=digest, headers={"User-Agent": USER_AGENT})

    if kind != "github_release":
        raise ValueError(f"Unsupported source kind for {tool}: {kind}")
    api_url = str(source.get("api_url", ""))
    asset_name = str(entry.get("asset", ""))
    if not api_url.startswith("https://api.github.com/") or not asset_name:
        raise ValueError(f"Invalid GitHub release metadata for {tool}/{platform}")
    release = _request_json(api_url, opener)
    matches = [asset for asset in release.get("assets", []) if asset.get("name") == asset_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one GitHub release asset named {asset_name}, found {len(matches)}")
    asset = matches[0]
    digest_value = str(asset.get("digest", ""))
    digest = digest_value.removeprefix("sha256:").lower()
    asset_url = str(asset.get("url", ""))
    if not asset_url.startswith("https://api.github.com/") or not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"GitHub release asset {asset_name} has no valid SHA-256 digest")
    return ResolvedArchive(
        name=asset_name,
        url=asset_url,
        sha256=digest,
        headers=_github_headers(asset_url, "application/octet-stream"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_archive(
    lock: dict,
    tool: str,
    platform: str,
    download_dir: Path,
    *,
    opener: OpenUrl = _open,
    attempts: int = 3,
) -> Path:
    resolved = resolve_archive(lock, tool, platform, opener)
    download_dir.mkdir(parents=True, exist_ok=True)
    destination = download_dir / resolved.name
    partial = destination.with_name(destination.name + ".part")
    if destination.exists() and sha256_file(destination) == resolved.sha256:
        return destination
    if destination.exists():
        print(f"[{tool}] cached archive checksum mismatch; downloading again", file=sys.stderr)
        destination.unlink()

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        partial.unlink(missing_ok=True)
        print(f"[{tool}] downloading {resolved.name} ({attempt}/{attempts})", file=sys.stderr)
        try:
            request = Request(resolved.url, headers=resolved.headers)
            with opener(request, 120) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            actual = sha256_file(partial)
            if actual != resolved.sha256:
                raise ValueError(f"SHA-256 verification failed: expected {resolved.sha256}, got {actual}")
            partial.replace(destination)
            return destination
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(attempt, 2))
    raise RuntimeError(f"[{tool}] download failed after {attempts} attempts: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify a project-local tool archive")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    print(fetch_archive(lock, args.tool, args.platform, args.download_dir))


if __name__ == "__main__":
    main()
