from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sandevistan_read import config as config_module


def load_text_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str):
    config_path = tmp_path / "runtime" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(config_module, "PATHS", SimpleNamespace(root=tmp_path, config=config_path))
    return config_module.load_config()


def test_audio_url_replaces_legacy_tts_url_with_compatible_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    current = load_text_config(monkeypatch, tmp_path, '[development]\naudio_url = "http://audio-current:20810"\ntts_url = "http://audio-legacy:20810"\n')
    assert current.development.audio_url == "http://audio-current:20810"

    legacy = load_text_config(monkeypatch, tmp_path, '[development]\ntts_url = "http://audio-legacy:20810"\n')
    assert legacy.development.audio_url == "http://audio-legacy:20810"


def test_deprecated_job_poll_seconds_is_ignored_but_unknown_keys_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    loaded = load_text_config(monkeypatch, tmp_path, "[runtime]\njob_poll_seconds = 0.5\nmax_upload_mib = 128\n")
    assert loaded.runtime.max_upload_mib == 128
    assert not hasattr(loaded.runtime, "job_poll_seconds")

    with pytest.raises(TypeError):
        load_text_config(monkeypatch, tmp_path, "[runtime]\nunknown_setting = true\n")


def test_example_config_uses_current_audio_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    example = (root / "config.example.toml").read_text(encoding="utf-8")
    loaded = load_text_config(monkeypatch, tmp_path, example)
    assert loaded.development.audio_url == "http://127.0.0.1:20810"
    assert "tts_url" not in example
    assert "job_poll_seconds" not in example
