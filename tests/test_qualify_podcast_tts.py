from __future__ import annotations

import importlib.util
import json
import math
import wave
import sys
from pathlib import Path

import pytest


EVALUATOR_SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_podcast.py"
EVALUATOR_SPEC = importlib.util.spec_from_file_location("evaluate_podcast", EVALUATOR_SCRIPT)
assert EVALUATOR_SPEC and EVALUATOR_SPEC.loader
evaluate_podcast = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(evaluate_podcast)
sys.modules.setdefault("evaluate_podcast", evaluate_podcast)

SCRIPT = Path(__file__).parents[1] / "scripts" / "qualify_podcast_tts.py"
SPEC = importlib.util.spec_from_file_location("qualify_podcast_tts", SCRIPT)
assert SPEC and SPEC.loader
qualify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualify)


def _quality(*, error_rate: float = 0.02, alignment: float = 1.0) -> dict:
    return {
        "passed": True,
        "error_rate": error_rate,
        "speaker_alignment": alignment,
        "silence_outliers": 0,
        "duration": {"passed": True},
    }


def _metrics(rate: float = 4.0) -> dict:
    return {
        "HOST_A": {
            "rate": {"median": rate, "cv": 0.10},
            "pitch": {"median": 200.0, "cv": 0.10},
            "rms_db": {"median": -20.0, "cv": 0.05},
        }
    }


def test_automatic_gate_accepts_stable_candidate_and_rejects_regression() -> None:
    assert qualify.automatic_gate(_quality(), _metrics(), _quality(), _metrics())["passed"] is True

    result = qualify.automatic_gate(
        _quality(error_rate=0.04), _metrics(rate=5.0),
        _quality(error_rate=0.02), _metrics(),
    )
    assert result["passed"] is False
    assert result["checks"]["error_rate"] is False
    assert result["checks"]["HOST_A_rate_median"] is False


def test_load_manifest_requires_exact_same_candidate(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"turns": []}', encoding="utf-8")
    samples = []
    for index in range(2):
        baseline_06 = tmp_path / f"sample-{index}-06"
        baseline_17 = tmp_path / f"sample-{index}-17"
        for baseline in (baseline_06, baseline_17):
            baseline.mkdir()
            (baseline / "candidate.json").write_bytes(candidate.read_bytes())
            (baseline / "candidate.m4a").write_bytes(b"audio")
        samples.append({
            "id": f"sample-{index}", "candidate_json": str(candidate),
            "baseline_06_dir": str(baseline_06), "baseline_17_dir": str(baseline_17),
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    assert len(qualify.load_manifest(manifest)) == 2

    (tmp_path / "sample-1-17" / "candidate.json").write_text('{"turns": [1]}', encoding="utf-8")
    with pytest.raises(ValueError, match="exact same candidate"):
        qualify.load_manifest(manifest)


def test_finalize_requires_publishable_sequence_not_worse_than_baselines(tmp_path: Path) -> None:
    sample_ids = ("sample-one", "sample-two")
    mapping = {
        sample_id: {
            "A": "qwen-1.7b-cpu-single",
            "B": "qwen-0.6b-gpu-sequence",
            "C": "qwen-0.6b-gpu-single",
        }
        for sample_id in sample_ids
    }
    (tmp_path / "qualification.json").write_text(json.dumps({
        "objective_passed": True,
        "samples": {sample_id: {"passed": True} for sample_id in sample_ids},
    }), encoding="utf-8")
    (tmp_path / "private-mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    scores = {"samples": {
        sample_id: {
            "A": {dimension: 4 for dimension in qualify.DIMENSIONS},
            "B": {dimension: 4 for dimension in qualify.DIMENSIONS},
            "C": {dimension: 4 for dimension in qualify.DIMENSIONS},
        }
        for sample_id in sample_ids
    }}
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    result, passed = qualify.finalize(tmp_path, scores_path)
    assert passed is True and result["status"] == "passed"

    scores["samples"]["sample-two"]["B"]["naturalness"] = 3
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    result, passed = qualify.finalize(tmp_path, scores_path)
    assert passed is False
    assert result["samples"]["sample-two"]["blind_passed"] is False


def _write_tone(path: Path, frequency: float = 180.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 24000
    frames = bytearray()
    frame_count = rate // 2
    fade_frames = rate // 10
    for index in range(frame_count):
        envelope = min(1.0, index / fade_frames, (frame_count - index - 1) / fade_frames)
        value = round(math.sin(2 * math.pi * frequency * index / rate) * 6000 * envelope)
        frames.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(frames)


def test_automated_finalize_closes_dual_baseline_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    samples = []
    report_samples = {}
    for sample_id in ("sample-one", "sample-two"):
        baseline_06 = tmp_path / sample_id / "baseline-06"
        baseline_17 = tmp_path / sample_id / "baseline-17"
        sequence = run_dir / sample_id / "sequence"
        for directory in (baseline_06, baseline_17, sequence):
            _write_tone(directory / "candidate-parts" / "0000.wav")
        samples.append({
            "id": sample_id, "baseline_06_dir": baseline_06, "baseline_17_dir": baseline_17,
        })
        report_samples[sample_id] = {
            "passed": True,
            "quality": {
                "passed": True, "error_rate": 0.01, "speaker_alignment": 1.0,
                "repaired_turns": [],
            },
            "against_06_single": {"passed": True},
            "against_17_single": {"passed": True},
        }
    (run_dir / "qualification.json").write_text(json.dumps({
        "status": "awaiting_scores", "objective_passed": True, "samples": report_samples,
    }), encoding="utf-8")
    monkeypatch.setattr(qualify, "load_manifest", lambda path: samples)

    result, passed = qualify.automated_finalize(run_dir, tmp_path / "manifest.json")

    assert passed is True
    assert result["method"] == "automated_dual_baseline_v1"
    assert all(sample["passed"] for sample in result["samples"].values())
    assert (run_dir / "automated-result.json").is_file()
    assert json.loads((run_dir / "qualification.json").read_text())["status"] == "passed"


def test_automated_signal_gate_rejects_spectral_regression() -> None:
    metrics = {
        "silence_ratio": {"median": 0.01},
        "zero_crossing_rate": {"median": 0.10},
        "high_frequency_ratio": {"median": 0.05},
        "spectral_centroid": {"median": 900.0},
        "crest_db": {"median": 12.0},
    }
    baseline = {"metrics": metrics}
    candidate = {"metrics": {
        **metrics,
        "high_frequency_ratio": {"median": 0.10},
        "spectral_centroid": {"median": 1400.0},
    }}

    result = qualify.automated_signal_gate(candidate, baseline)

    assert result["passed"] is False
    assert result["checks"]["high_frequency_ratio"] is False
    assert result["checks"]["spectral_centroid"] is False
