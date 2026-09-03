from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_podcast.py"
SPEC = importlib.util.spec_from_file_location("evaluate_podcast", SCRIPT)
assert SPEC and SPEC.loader
evaluate_podcast = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_podcast)
sys.modules.setdefault("evaluate_podcast", evaluate_podcast)

SUITE_SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_podcast_suite.py"
SUITE_SPEC = importlib.util.spec_from_file_location("evaluate_podcast_suite", SUITE_SCRIPT)
assert SUITE_SPEC and SUITE_SPEC.loader
evaluate_suite = importlib.util.module_from_spec(SUITE_SPEC)
SUITE_SPEC.loader.exec_module(evaluate_suite)


def test_evaluator_accepts_reference_matched_durations() -> None:
    for minutes in (14, 22, 25):
        args = evaluate_podcast.build_parser().parse_args(["--notebook-id", "n1", "--minutes", str(minutes)])
        assert args.minutes == minutes
    rendered = evaluate_podcast.build_parser().parse_args(["--notebook-id", "n1", "--minutes", "25", "--render-candidate"])
    assert rendered.render_candidate is True


@pytest.mark.asyncio
async def test_candidate_json_skips_main_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps({
        "version": 4,
        "language": "zh-CN",
        "duration": {"target_minutes": 5},
        "turns": [{"speaker": "HOST_A", "text": "已通过门禁的候选。", "dialogue_act": "intro", "citation_ids": []}],
        "quality": {"passed": True},
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evaluate_podcast, "PATHS", SimpleNamespace(runtime=tmp_path))

    async def forbidden_generation(*args, **kwargs):
        raise AssertionError("candidate-json must not call MAIN generation")

    monkeypatch.setattr(evaluate_podcast, "build_podcast_script", forbidden_generation)
    args = evaluate_podcast.build_parser().parse_args(["--candidate-json", str(candidate_path)])
    output, passed = await evaluate_podcast.run(args)
    assert passed is True
    assert (output / "candidate.json").is_file()


def test_anonymous_transcript_normalizes_candidate_and_asr_speakers() -> None:
    candidate = evaluate_podcast.anonymous_transcript({"turns": [
        {"speaker": "HOST_A", "text": "First idea", "citation_ids": ["S1"]},
        {"speaker": "HOST_B", "text": "Second idea"},
    ]})
    reference = evaluate_podcast.anonymous_transcript({"segments": [
        {"speaker": "person-id-1", "speaker_label": "SPEAKER_00", "text": "First"},
        {"speaker": "person-id-1", "speaker_label": "SPEAKER_00", "text": "continued"},
        {"speaker": "person-id-2", "speaker_label": "SPEAKER_01", "text": "Reply"},
    ]}, asr=True)
    assert candidate == "Speaker 1: First idea\nSpeaker 2: Second idea"
    assert reference == "Speaker 1: First continued\nSpeaker 2: Reply"
    assert "HOST" not in candidate and "SPEAKER_" not in reference and "S1" not in candidate


def test_suite_manifest_is_generic_and_requires_reference_audio(tmp_path: Path) -> None:
    reference = tmp_path / "reference.m4a"
    reference.write_bytes(b"audio")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [{
        "id": "sample-one",
        "notebook_id": "n1",
        "source_ids": ["s1"],
        "minutes": 14,
        "language": "zh-CN",
        "reference_audio": str(reference),
    }]}), encoding="utf-8")
    loaded = evaluate_suite.load_manifest(manifest)
    assert loaded["samples"][0]["id"] == "sample-one"
    assert loaded["samples"][0]["reference_language"] == "Chinese"


@pytest.mark.asyncio
async def test_development_suite_reuses_candidate_without_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = tmp_path / "reference.m4a"
    reference.write_bytes(b"audio")
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps({
        "language": "zh-CN",
        "turns": [{"speaker": "HOST_A", "text": "受支持的候选内容。", "dialogue_act": "explain", "claim_ids": ["C1"]}],
        "quality": {"passed": True},
        "context_usage": {"requests": 4, "accounted_total_tokens": 12000},
    }, ensure_ascii=False), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [{
        "id": "sample-one", "minutes": 14, "language": "zh-CN",
        "reference_audio": str(reference), "candidate_json": str(candidate_path),
    }]}), encoding="utf-8")
    monkeypatch.setattr(evaluate_suite, "PATHS", SimpleNamespace(root=Path(__file__).parents[1], runtime=tmp_path))
    monkeypatch.setattr(evaluate_suite, "_implementation_snapshot", lambda: {"podcast.py": "hash"})
    monkeypatch.setattr(evaluate_suite, "_provider_fingerprint", lambda: {"main": {"model": "test"}})

    async def forbidden_main(*args, **kwargs):
        raise AssertionError("development candidate reuse must not call MAIN")

    async def fake_render(candidate, output, stamp, **kwargs):
        (output / "candidate-asr.json").write_text(json.dumps({
            "segments": [{"speaker_label": "S1", "text": "受支持的候选内容。"}],
        }, ensure_ascii=False), encoding="utf-8")
        return {"passed": True, "duration": {"passed": True}, "error_rate": 0.01}

    async def fake_reference(*args, **kwargs):
        return {"segments": [{"speaker_label": "S1", "text": "参考内容。"}]}, tmp_path / "cache.json", True

    monkeypatch.setattr(evaluate_suite, "build_podcast_script", forbidden_main)
    monkeypatch.setattr(evaluate_suite, "render_candidate", fake_render)
    monkeypatch.setattr(evaluate_suite, "_reference_asr", fake_reference)
    output, prepared = await evaluate_suite.prepare_suite(manifest, mode="development", output=tmp_path / "run")
    assert prepared is True
    assert json.loads((output / "suite.json").read_text())["status"] == "awaiting_scores"
    assert (output / "scores-template.json").is_file()


def test_suite_finalizer_requires_publishable_candidate_not_worse_than_reference(tmp_path: Path) -> None:
    suite = {
        "status": "awaiting_scores", "mode": "frozen", "manifest_hash": "m", "implementation": {"p": "h"},
        "provider": {}, "samples": {"sample-one": {
            "script_gate": True, "audio_gate": True, "main_tokens": 39000, "audio_quality": {"passed": True},
        }},
    }
    (tmp_path / "suite.json").write_text(json.dumps(suite), encoding="utf-8")
    (tmp_path / "private-mapping.json").write_text(json.dumps({"sample-one": {"A": "reference", "B": "candidate"}}), encoding="utf-8")
    scores = {"samples": {"sample-one": {
        "A": {name: 4 for name in evaluate_suite.DIMENSIONS},
        "B": {name: 4 for name in evaluate_suite.DIMENSIONS},
    }}}
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    result, passed = evaluate_suite.finalize_suite(tmp_path, scores_path)
    assert passed is True and result["status"] == "passed"
    assert result["samples"]["sample-one"]["candidate_total"] == 24


def _write_wav(path: Path, seconds: float = 0.1) -> None:
    rate = 24000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))


def _fake_tts_provider() -> dict:
    return {
        "name": "saved-tts", "kind": "sandevistan_tts", "model": "saved-tts-1.7b",
        "config": {"compute_device": "gpu", "host_a": "Vivian", "host_b": "Dylan"},
        "capabilities": {"models": [
            {"id": "saved-tts-1.7b", "controls": {"instruction_voice_modes": ["preset"]}},
            {"id": "qwen3-tts-0.6b", "controls": {}},
        ]},
    }


def _render_candidate(monkeypatch: pytest.MonkeyPatch, captured: list[dict]) -> dict:
    candidate = {
        "language": "zh-CN",
        "duration": {"target_minutes": 5},
        "turns": [{"speaker": "HOST_A", "text": "测试轮次。", "dialogue_act": "explain", "claim_ids": ["C1"]}],
        "quality": {"passed": True},
    }
    monkeypatch.setattr(evaluate_podcast, "active_provider", lambda role: _fake_tts_provider())
    monkeypatch.setattr(evaluate_podcast, "CONFIG", SimpleNamespace(tools=SimpleNamespace(ffmpeg_path="ffmpeg")))

    async def fake_synthesize(text, voice, output, **kwargs):
        captured.append(kwargs)
        _write_wav(Path(output))
        return Path(output)

    async def fake_ffmpeg(command, timeout, cancel):
        _write_wav(Path(command[-1]))
        return 0, ""

    async def fake_transcribe(path, **kwargs):
        return {"segments": [{"speaker_label": "S1", "text": "测试轮次。"}]}

    monkeypatch.setattr(evaluate_podcast, "synthesize", fake_synthesize)
    monkeypatch.setattr(evaluate_podcast, "_run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(evaluate_podcast, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(evaluate_podcast, "assess_transcription", lambda *args, **kwargs: {"passed": True, "turn_errors": []})
    monkeypatch.setattr(evaluate_podcast, "_actual_duration_check", lambda *args: {"passed": True, "actual_minutes": 5.0})
    return candidate


def test_parser_accepts_tts_overrides() -> None:
    args = evaluate_podcast.build_parser().parse_args(["--notebook-id", "n1"])
    assert args.tts_model is None and args.tts_device is None
    args = evaluate_podcast.build_parser().parse_args(
        ["--candidate-json", "c.json", "--render-candidate", "--tts-model", "qwen3-tts-0.6b", "--tts-device", "gpu"]
    )
    assert args.tts_model == "qwen3-tts-0.6b" and args.tts_device == "gpu"


@pytest.mark.asyncio
async def test_render_candidate_forwards_tts_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict] = []
    candidate = _render_candidate(monkeypatch, captured)
    result = await evaluate_podcast.render_candidate(
        candidate, tmp_path, "stamp0001", tts_model="qwen3-tts-0.6b", tts_device="cpu"
    )
    assert result["passed"] is True
    assert captured and captured[0]["model"] == "qwen3-tts-0.6b"
    assert captured[0]["compute_device"] == "cpu"
    # 覆盖模型在 capabilities 中没有 preset 能力时不应携带 instruct
    assert captured[0]["instruct"] is None
    assert result["execution"]["model"] == "qwen3-tts-0.6b"


@pytest.mark.asyncio
async def test_render_candidate_defaults_to_saved_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict] = []
    candidate = _render_candidate(monkeypatch, captured)
    result = await evaluate_podcast.render_candidate(candidate, tmp_path, "stamp0001")
    assert result["passed"] is True
    assert captured[0]["model"] == "saved-tts-1.7b"
    assert captured[0]["compute_device"] == "gpu"
    assert captured[0]["instruct"]


@pytest.mark.asyncio
async def test_render_candidate_cache_key_isolates_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first: list[dict] = []
    candidate = _render_candidate(monkeypatch, first)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    await evaluate_podcast.render_candidate(candidate, tmp_path / "a", "stamp0001", tts_model="qwen3-tts-0.6b")
    second: list[dict] = []
    _render_candidate(monkeypatch, second)
    await evaluate_podcast.render_candidate(candidate, tmp_path / "b", "stamp0001", tts_model="qwen3-tts-1.7b")
    key_a = first[0]["idempotency_key"].split("-")
    key_b = second[0]["idempotency_key"].split("-")
    assert key_a[-1] != key_b[-1]


def _two_sample_manifest(tmp_path: Path) -> Path:
    reference = tmp_path / "reference.m4a"
    reference.write_bytes(b"audio")
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps({
        "language": "zh-CN",
        "turns": [{"speaker": "HOST_A", "text": "受支持的候选内容。", "dialogue_act": "explain", "claim_ids": ["C1"]}],
        "quality": {"passed": True},
        "context_usage": {"requests": 4, "accounted_total_tokens": 12000},
    }, ensure_ascii=False), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [
        {"id": "sample-one", "minutes": 14, "language": "zh-CN",
         "reference_audio": str(reference), "candidate_json": str(candidate_path)},
        {"id": "sample-two", "minutes": 14, "language": "zh-CN",
         "reference_audio": str(reference), "candidate_json": str(candidate_path)},
    ]}), encoding="utf-8")
    return manifest


def _stub_suite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, captured: list[dict] | None = None) -> None:
    monkeypatch.setattr(evaluate_suite, "PATHS", SimpleNamespace(root=Path(__file__).parents[1], runtime=tmp_path))
    monkeypatch.setattr(evaluate_suite, "_implementation_snapshot", lambda: {"podcast.py": "hash"})
    monkeypatch.setattr(evaluate_suite, "_provider_fingerprint", lambda: {"main": {"model": "test"}})

    async def forbidden_main(*args, **kwargs):
        raise AssertionError("candidate reuse must not call MAIN")

    async def fake_render(candidate, output, stamp, **kwargs):
        if captured is not None:
            captured.append(kwargs)
        (output / "candidate-asr.json").write_text(json.dumps({
            "segments": [{"speaker_label": "S1", "text": "受支持的候选内容。"}],
        }, ensure_ascii=False), encoding="utf-8")
        return {"passed": True, "duration": {"passed": True}, "error_rate": 0.01}

    async def fake_reference(*args, **kwargs):
        return {"segments": [{"speaker_label": "S1", "text": "参考内容。"}]}, tmp_path / "cache.json", True

    monkeypatch.setattr(evaluate_suite, "build_podcast_script", forbidden_main)
    monkeypatch.setattr(evaluate_suite, "render_candidate", fake_render)
    monkeypatch.setattr(evaluate_suite, "_reference_asr", fake_reference)


@pytest.mark.asyncio
async def test_prepare_sample_filter_selects_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _two_sample_manifest(tmp_path)
    _stub_suite(monkeypatch, tmp_path)
    output, prepared = await evaluate_suite.prepare_suite(
        manifest, mode="development", output=tmp_path / "run", samples=["sample-one"]
    )
    assert prepared is True
    suite = json.loads((output / "suite.json").read_text())
    assert set(suite["samples"]) == {"sample-one"}
    assert suite["requested_samples"] == ["sample-one"]
    mapping = json.loads((output / "private-mapping.json").read_text())
    assert set(mapping) == {"sample-one"}


@pytest.mark.asyncio
async def test_prepare_sample_filter_rejects_unknown_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _two_sample_manifest(tmp_path)
    _stub_suite(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="sample-three"):
        await evaluate_suite.prepare_suite(manifest, mode="development", output=tmp_path / "run", samples=["sample-three"])


@pytest.mark.asyncio
async def test_sample_filter_is_part_of_suite_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _two_sample_manifest(tmp_path)
    _stub_suite(monkeypatch, tmp_path)
    output, prepared = await evaluate_suite.prepare_suite(
        manifest, mode="development", output=tmp_path / "run", samples=["sample-one"]
    )
    assert prepared is True
    with pytest.raises(ValueError, match="不一致"):
        await evaluate_suite.prepare_suite(manifest, mode="development", output=output, resume=True)


@pytest.mark.asyncio
async def test_prepare_forwards_tts_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _two_sample_manifest(tmp_path)
    captured: list[dict] = []
    _stub_suite(monkeypatch, tmp_path, captured)
    output, prepared = await evaluate_suite.prepare_suite(
        manifest, mode="development", output=tmp_path / "run",
        samples=["sample-one"], tts_model="qwen3-tts-0.6b", tts_device="cpu",
    )
    assert prepared is True
    assert captured and captured[0] == {"tts_model": "qwen3-tts-0.6b", "tts_device": "cpu"}
    suite = json.loads((output / "suite.json").read_text())
    assert suite["tts_override"] == {"model": "qwen3-tts-0.6b", "device": "cpu"}


def _interrupted_suite(tmp_path: Path, manifest: Path, run_dir: Path) -> None:
    loaded = evaluate_suite.load_manifest(manifest)
    run_dir.mkdir(parents=True)
    (run_dir / "suite.json").write_text(json.dumps({
        "status": "failed", "mode": "development", "created_at": "2026-09-02T00:00:00+00:00",
        "manifest_hash": evaluate_suite._json_hash(loaded), "implementation": {"podcast.py": "hash"},
        "provider": {}, "samples": {
            "sample-one": {"status": "awaiting_scores"},
            "sample-two": {"status": "failed", "stage": "pipeline"},
        },
    }), encoding="utf-8")


@pytest.mark.asyncio
async def test_resume_preserves_existing_mappings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _two_sample_manifest(tmp_path)
    run_dir = tmp_path / "run"
    _interrupted_suite(tmp_path, manifest, run_dir)
    saved_mapping = {"sample-one": {"A": "reference", "B": "candidate"}, "removed-sample": {"A": "candidate", "B": "reference"}}
    (run_dir / "private-mapping.json").write_text(json.dumps(saved_mapping, ensure_ascii=False), encoding="utf-8")
    (run_dir / "scores-template.json").write_text(json.dumps({"samples": {"sample-one": {
        label: {name: None for name in evaluate_suite.DIMENSIONS} for label in ("A", "B")
    }}}), encoding="utf-8")
    _stub_suite(monkeypatch, tmp_path)
    output, prepared = await evaluate_suite.prepare_suite(manifest, mode="development", output=run_dir, resume=True)
    assert prepared is True
    mapping = json.loads((output / "private-mapping.json").read_text())
    # 旧 mapping 逐字节保留、新样本补齐、已移出 manifest 的键被剔除
    assert mapping["sample-one"] == saved_mapping["sample-one"]
    assert set(mapping["sample-two"]) == {"A", "B"}
    assert "removed-sample" not in mapping
    template = json.loads((output / "scores-template.json").read_text())
    assert set(template["samples"]) == {"sample-one", "sample-two"}


@pytest.mark.asyncio
async def test_resume_rebuilds_mapping_when_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _two_sample_manifest(tmp_path)
    run_dir = tmp_path / "run"
    _interrupted_suite(tmp_path, manifest, run_dir)
    sample_dir = run_dir / "sample-one"
    sample_dir.mkdir()
    candidate_asr = {"segments": [{"speaker_label": "S1", "text": "候选音频转写。"}]}
    reference_asr = {"segments": [{"speaker_label": "S1", "text": "参考音频转写。"}]}
    (sample_dir / "candidate-asr.json").write_text(json.dumps(candidate_asr, ensure_ascii=False), encoding="utf-8")
    (sample_dir / "reference-asr.json").write_text(json.dumps(reference_asr, ensure_ascii=False), encoding="utf-8")
    _stub_suite(monkeypatch, tmp_path)
    output, prepared = await evaluate_suite.prepare_suite(manifest, mode="development", output=run_dir, resume=True)
    assert prepared is True
    mapping = json.loads((output / "private-mapping.json").read_text())
    seed = evaluate_suite.hashlib.sha256(
        f"2026-09-02T00:00:00+00:00:{evaluate_suite._json_hash(evaluate_suite.load_manifest(manifest))}".encode()
    ).hexdigest()
    _, expected = evaluate_suite._blind_packet("sample-one", candidate_asr, reference_asr, seed)
    assert mapping["sample-one"] == expected
    suite = json.loads((output / "suite.json").read_text())
    assert suite["rebuilt_mappings"] == ["sample-one"]
