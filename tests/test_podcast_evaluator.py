from __future__ import annotations

import importlib.util
import json
import sys
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

    async def fake_render(candidate, output, stamp):
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
