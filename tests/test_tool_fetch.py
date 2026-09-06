from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from urllib.request import Request

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "fetch-tool.py"
SPEC = importlib.util.spec_from_file_location("fetch_tool", SCRIPT)
assert SPEC and SPEC.loader
fetch_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fetch_tool
SPEC.loader.exec_module(fetch_tool)


class Response(io.BytesIO):
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def fixed_lock(payload: bytes) -> dict:
    return {
        "demo": {
            "windows-x86_64": {
                "url": "https://downloads.example.test/demo.zip",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        }
    }


def test_resolve_fixed_archive() -> None:
    resolved = fetch_tool.resolve_archive(fixed_lock(b"archive"), "demo", "windows-x86_64")

    assert resolved.name == "demo.zip"
    assert resolved.url == "https://downloads.example.test/demo.zip"
    assert resolved.headers["User-Agent"]


def test_resolve_github_release_asset_uses_api_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "fixture-token")
    digest = "a" * 64
    lock = {
        "demo": {
            "source": {
                "kind": "github_release",
                "api_url": "https://api.github.com/repos/example/demo/releases/latest",
            },
            "windows-x86_64": {"asset": "demo.zip"},
        }
    }
    seen: list[Request] = []

    def opener(request: Request, _timeout: float) -> Response:
        seen.append(request)
        return Response(
            json.dumps(
                {
                    "assets": [
                        {
                            "name": "demo.zip",
                            "url": "https://api.github.com/repos/example/demo/releases/assets/123",
                            "digest": f"sha256:{digest}",
                        }
                    ]
                }
            ).encode()
        )

    resolved = fetch_tool.resolve_archive(lock, "demo", "windows-x86_64", opener)

    assert resolved.sha256 == digest
    assert resolved.headers["Accept"] == "application/octet-stream"
    assert seen[0].get_header("Accept") == "application/vnd.github+json"
    assert seen[0].get_header("Authorization") == "Bearer fixture-token"
    assert resolved.headers["Authorization"] == "Bearer fixture-token"


@pytest.mark.parametrize("url", ["https://downloads.example.test/file", "http://api.github.com/file", "https://api.github.com.example.test/file"])
def test_tool_token_is_only_attached_to_github_api(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "fixture-token")
    assert "Authorization" not in fetch_tool._github_headers(url, "application/json")
    assert "Authorization" not in fetch_tool.resolve_archive(fixed_lock(b"archive"), "demo", "windows-x86_64").headers


@pytest.mark.parametrize("target,authorized", [
    ("https://api.github.com/renamed", True),
    ("https://release-assets.githubusercontent.com/file", False),
    ("http://api.github.com/file", False),
])
def test_tool_redirect_strips_credentials_on_origin_change(target: str, authorized: bool) -> None:
    request = Request("https://api.github.com/assets/1", headers={"Authorization": "Bearer fixture-token"})
    redirected = fetch_tool.ToolRedirectHandler().redirect_request(request, None, 302, "Found", {}, target)
    assert (redirected.get_header("Authorization") is not None) is authorized


@pytest.mark.parametrize(
    "assets",
    [
        [],
        [{"name": "demo.zip", "url": "https://api.github.com/assets/1", "digest": None}],
        [
            {"name": "demo.zip", "url": "https://api.github.com/assets/1", "digest": "sha256:" + "a" * 64},
            {"name": "demo.zip", "url": "https://api.github.com/assets/2", "digest": "sha256:" + "b" * 64},
        ],
    ],
)
def test_rejects_missing_duplicate_or_unverified_github_asset(assets: list[dict]) -> None:
    lock = {
        "demo": {
            "source": {
                "kind": "github_release",
                "api_url": "https://api.github.com/repos/example/demo/releases/latest",
            },
            "windows-x86_64": {"asset": "demo.zip"},
        }
    }

    with pytest.raises(ValueError):
        fetch_tool.resolve_archive(lock, "demo", "windows-x86_64", lambda *_args: Response(json.dumps({"assets": assets}).encode()))


def test_fetch_archive_retries_then_reuses_verified_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"verified archive"
    calls = 0

    def opener(_request: Request, _timeout: float) -> Response:
        nonlocal calls
        calls += 1
        return Response(b"corrupt" if calls == 1 else payload)

    monkeypatch.setattr(fetch_tool.time, "sleep", lambda _seconds: None)
    path = fetch_tool.fetch_archive(fixed_lock(payload), "demo", "windows-x86_64", tmp_path, opener=opener)

    assert path.read_bytes() == payload
    assert calls == 2

    def should_not_open(_request: Request, _timeout: float) -> Response:
        raise AssertionError("verified cache should not access the network")

    assert fetch_tool.fetch_archive(fixed_lock(payload), "demo", "windows-x86_64", tmp_path, opener=should_not_open) == path
