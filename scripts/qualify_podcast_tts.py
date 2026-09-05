#!/usr/bin/env python3
"""Prepare and finalize repeatable Podcast TTS performance/quality qualification."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import statistics
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_podcast import render_candidate
from sandevistan_read.paths import PATHS


DIMENSIONS = ("naturalness", "host_consistency", "prosody_stability", "intelligibility", "listening_fatigue")
SOURCES = ("qwen-1.7b-cpu-single", "qwen-0.6b-gpu-single", "qwen-0.6b-gpu-sequence")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_manifest(path: Path) -> list[dict[str, Any]]:
    source = load_json(path).get("samples")
    if not isinstance(source, list) or len(source) < 2:
        raise ValueError("Qualification manifest must contain at least two samples")
    result = []
    seen: set[str] = set()
    for raw in source:
        if not isinstance(raw, dict):
            raise ValueError("Every qualification sample must be an object")
        sample_id = str(raw.get("id") or "").strip()
        if not sample_id or sample_id in seen:
            raise ValueError("Qualification sample IDs must be non-empty and unique")
        seen.add(sample_id)
        item = {"id": sample_id}
        for key in ("candidate_json", "baseline_06_dir", "baseline_17_dir"):
            target = Path(str(raw.get(key) or "")).expanduser().resolve()
            if not target.exists():
                raise ValueError(f"Missing {key} for {sample_id}: {target}")
            item[key] = target
        candidate_hash = file_hash(item["candidate_json"])
        for key in ("baseline_06_dir", "baseline_17_dir"):
            baseline_candidate = item[key] / "candidate.json"
            if not baseline_candidate.is_file() or file_hash(baseline_candidate) != candidate_hash:
                raise ValueError(f"{sample_id} does not use the exact same candidate in {key}")
            if not (item[key] / "candidate.m4a").is_file():
                raise ValueError(f"{sample_id} baseline audio is missing in {key}")
        result.append(item)
    return result


def coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else None


def estimated_pitch(samples: np.ndarray, rate: int) -> float | None:
    frame_size = max(1, round(rate * 0.04))
    hop = max(1, round(rate * 0.02))
    low_lag = max(1, rate // 350)
    high_lag = min(frame_size - 1, rate // 70)
    if high_lag <= low_lag:
        return None
    pitches: list[float] = []
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    energy_floor = max(0.004, peak * 0.04)
    analysis_hop = max(hop, len(samples) // 24)
    fft_size = 1 << (frame_size * 2 - 1).bit_length()
    for start in range(0, max(0, len(samples) - frame_size + 1), analysis_hop):
        frame = samples[start:start + frame_size].astype(np.float64, copy=False)
        frame -= frame.mean()
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < energy_floor:
            continue
        spectrum = np.fft.rfft(frame, n=fft_size)
        correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:frame_size]
        if correlation[0] <= 0:
            continue
        window = correlation[low_lag:high_lag + 1]
        lag = low_lag + int(np.argmax(window))
        if float(correlation[lag] / correlation[0]) >= 0.30:
            pitches.append(rate / lag)
    return statistics.median(pitches) if pitches else None


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        rate = audio.getframerate()
        frames = audio.readframes(audio.getnframes())
    if width != 2:
        raise ValueError(f"Expected PCM16 WAV: {path}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def acoustic_metrics(candidate: dict[str, Any], parts_dir: Path) -> dict[str, Any]:
    by_host: dict[str, dict[str, list[float]]] = {}
    turns = candidate.get("turns") or []
    for index, turn in enumerate(turns):
        path = parts_dir / f"{index:04d}.wav"
        if not path.is_file():
            raise ValueError(f"Missing per-turn WAV: {path}")
        samples, rate = read_wav(path)
        duration = len(samples) / max(rate, 1)
        rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        host = by_host.setdefault(str(turn["speaker"]), {"rate": [], "pitch": [], "rms_db": []})
        host["rate"].append(len(str(turn["text"])) / max(duration, 0.001))
        pitch = estimated_pitch(samples, rate)
        if pitch is not None:
            host["pitch"].append(pitch)
        host["rms_db"].append(20 * math.log10(max(rms, 1e-8)))
    result: dict[str, Any] = {}
    for host, metrics in by_host.items():
        result[host] = {
            key: {
                "median": round(statistics.median(values), 4) if values else None,
                "cv": round(coefficient_of_variation(values), 4) if coefficient_of_variation(values) is not None else None,
                "stdev": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            }
            for key, values in metrics.items()
        }
    return result


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def signal_metrics(parts_dir: Path) -> dict[str, Any]:
    measurements: dict[str, list[float]] = {
        "clipping_ratio": [], "dc_offset": [], "silence_ratio": [],
        "zero_crossing_rate": [], "high_frequency_ratio": [],
        "spectral_centroid": [], "crest_db": [], "edge_peak": [],
    }
    paths = sorted(parts_dir.glob("*.wav"))
    if not paths:
        raise ValueError(f"No per-turn WAV files found: {parts_dir}")
    for path in paths:
        samples, rate = read_wav(path)
        if not len(samples):
            raise ValueError(f"Empty per-turn WAV: {path}")
        absolute = np.abs(samples)
        rms = float(np.sqrt(np.mean(samples * samples)))
        peak = float(np.max(absolute))
        edge_size = max(1, round(rate * 0.02))
        measurements["clipping_ratio"].append(float(np.mean(absolute >= 0.99)))
        measurements["dc_offset"].append(abs(float(np.mean(samples))))
        measurements["crest_db"].append(20 * math.log10(max(peak, 1e-8) / max(rms, 1e-8)))
        measurements["edge_peak"].append(float(max(np.max(absolute[:edge_size]), np.max(absolute[-edge_size:]))))
        frame_size = max(1, round(rate * 0.04))
        analysis_hop = max(frame_size, len(samples) // 24)
        silence: list[float] = []
        zero_crossings: list[float] = []
        high_frequency: list[float] = []
        centroids: list[float] = []
        window = np.hanning(frame_size)
        frequencies = np.fft.rfftfreq(frame_size, 1 / rate)
        for start in range(0, max(1, len(samples) - frame_size + 1), analysis_hop):
            frame = samples[start:start + frame_size]
            if len(frame) != frame_size:
                continue
            frame_rms = float(np.sqrt(np.mean(frame * frame)))
            silence.append(float(frame_rms < 0.005))
            if frame_rms < 0.004:
                continue
            zero_crossings.append(float(np.mean(np.signbit(frame[1:]) != np.signbit(frame[:-1]))))
            power = np.abs(np.fft.rfft(frame * window)) ** 2
            total_power = float(np.sum(power)) or 1.0
            high_frequency.append(float(np.sum(power[frequencies >= 4000]) / total_power))
            centroids.append(float(np.sum(power * frequencies) / total_power))
        measurements["silence_ratio"].append(statistics.fmean(silence) if silence else 0.0)
        measurements["zero_crossing_rate"].append(statistics.fmean(zero_crossings) if zero_crossings else 0.0)
        measurements["high_frequency_ratio"].append(statistics.fmean(high_frequency) if high_frequency else 0.0)
        measurements["spectral_centroid"].append(statistics.fmean(centroids) if centroids else 0.0)
    return {
        "turn_count": len(paths),
        "metrics": {
            key: {
                "median": round(statistics.median(values), 8),
                "p95": round(_percentile(values, 95) or 0.0, 8),
                "max": round(max(values), 8),
            }
            for key, values in measurements.items()
        },
    }


def _ratio_in_range(candidate: float, baseline: float, lower: float, upper: float) -> bool:
    if baseline <= 1e-9:
        return candidate <= 1e-9
    return lower <= candidate / baseline <= upper


def automated_signal_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current = candidate["metrics"]
    previous = baseline["metrics"]
    checks = {
        "silence_distribution": abs(current["silence_ratio"]["median"] - previous["silence_ratio"]["median"]) <= 0.10,
        "zero_crossing_rate": _ratio_in_range(
            current["zero_crossing_rate"]["median"], previous["zero_crossing_rate"]["median"], 0.75, 1.30,
        ),
        "high_frequency_ratio": _ratio_in_range(
            current["high_frequency_ratio"]["median"], previous["high_frequency_ratio"]["median"], 0.70, 1.30,
        ),
        "spectral_centroid": _ratio_in_range(
            current["spectral_centroid"]["median"], previous["spectral_centroid"]["median"], 0.80, 1.25,
        ),
        "crest_factor": abs(current["crest_db"]["median"] - previous["crest_db"]["median"]) <= 3.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def automated_finalize(run_dir: Path, manifest_path: Path) -> tuple[dict[str, Any], bool]:
    report = load_json(run_dir / "qualification.json")
    samples = {sample["id"]: sample for sample in load_manifest(manifest_path)}
    if set(samples) != set(report.get("samples") or {}):
        raise ValueError("Qualification run and manifest samples do not match")
    results: dict[str, Any] = {}
    passed = bool(report.get("objective_passed"))
    for sample_id, sample in samples.items():
        objective = report["samples"][sample_id]
        signals = {
            "qwen-0.6b-gpu-sequence": signal_metrics(run_dir / sample_id / "sequence" / "candidate-parts"),
            "qwen-0.6b-gpu-single": signal_metrics(sample["baseline_06_dir"] / "candidate-parts"),
            "qwen-1.7b-cpu-single": signal_metrics(sample["baseline_17_dir"] / "candidate-parts"),
        }
        current = signals["qwen-0.6b-gpu-sequence"]
        metrics = current["metrics"]
        signal_integrity = {
            "clipping": metrics["clipping_ratio"]["p95"] <= 0.001,
            "dc_offset": metrics["dc_offset"]["p95"] <= 0.02,
            "edge_clicks": metrics["edge_peak"]["p95"] <= 0.05,
        }
        comparisons = {
            source: automated_signal_gate(current, signals[source])
            for source in ("qwen-0.6b-gpu-single", "qwen-1.7b-cpu-single")
        }
        quality = objective.get("quality") or {}
        repaired = list(quality.get("repaired_turns") or [])
        repair_ratio = len(repaired) / max(1, current["turn_count"])
        objective_comparisons = [objective.get("against_06_single") or {}, objective.get("against_17_single") or {}]
        dimensions = {
            "naturalness_proxy": all(signal_integrity.values()) and all(item["passed"] for item in comparisons.values()),
            "host_consistency": float(quality.get("speaker_alignment") or 0) >= 0.95
            and all(item.get("passed") for item in objective_comparisons),
            "prosody_stability": all(item["passed"] for item in comparisons.values())
            and all(item.get("passed") for item in objective_comparisons),
            "intelligibility": bool(quality.get("passed"))
            and float(quality.get("error_rate") if quality.get("error_rate") is not None else 1) <= 0.08,
            "listening_fatigue_proxy": signal_integrity["clipping"]
            and all(
                comparison["checks"]["high_frequency_ratio"]
                and comparison["checks"]["spectral_centroid"]
                and comparison["checks"]["crest_factor"]
                for comparison in comparisons.values()
            ),
            "repair_rate": repair_ratio <= 0.05,
        }
        sample_passed = bool(objective.get("passed") and all(dimensions.values()))
        passed = passed and sample_passed
        results[sample_id] = {
            "passed": sample_passed,
            "dimensions": dimensions,
            "repair_ratio": round(repair_ratio, 4),
            "signal_integrity": signal_integrity,
            "comparisons": comparisons,
            "signal_metrics": signals,
        }
    result = {
        "status": "passed" if passed else "failed",
        "method": "automated_dual_baseline_v1",
        "objective_passed": bool(report.get("objective_passed")),
        "samples": results,
    }
    (run_dir / "automated-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    report["automated_qualification"] = {
        "status": result["status"], "method": result["method"],
    }
    report["status"] = result["status"]
    (run_dir / "qualification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, passed


def automatic_gate(
    sequence_quality: dict[str, Any], sequence_metrics: dict[str, Any],
    baseline_quality: dict[str, Any], baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    current_error = float(sequence_quality.get("error_rate", 1.0))
    baseline_error = float(baseline_quality.get("error_rate", 0.0))
    checks: dict[str, bool] = {
        "audio_gate": bool(sequence_quality.get("passed")),
        "error_rate": current_error <= 0.08 and current_error <= baseline_error + 0.01,
        "speaker_alignment": float(sequence_quality.get("speaker_alignment") or 0) >= 0.95
        and float(sequence_quality.get("speaker_alignment") or 0) >= float(baseline_quality.get("speaker_alignment") or 0) - 0.03,
        "silence": int(sequence_quality.get("silence_outliers") or 0) == 0,
        "duration": bool((sequence_quality.get("duration") or {}).get("passed")),
    }
    details: dict[str, Any] = {}
    for host, current in sequence_metrics.items():
        previous = baseline_metrics.get(host) or {}
        for key in ("rate", "pitch"):
            current_median = (current.get(key) or {}).get("median")
            previous_median = (previous.get(key) or {}).get("median")
            current_cv = (current.get(key) or {}).get("cv")
            previous_cv = (previous.get(key) or {}).get("cv")
            median_ok = bool(
                current_median and previous_median
                and 0.85 <= current_median / previous_median <= 1.15
            )
            cv_ok = bool(
                current_cv is not None and previous_cv is not None
                and current_cv <= previous_cv * 1.20 + 0.05
            )
            checks[f"{host}_{key}_median"] = median_ok
            checks[f"{host}_{key}_variation"] = cv_ok
            details[f"{host}_{key}"] = {
                "candidate": current.get(key), "baseline": previous.get(key),
            }
        current_energy = (current.get("rms_db") or {}).get("median")
        previous_energy = (previous.get("rms_db") or {}).get("median")
        energy_ok = bool(
            current_energy is not None and previous_energy is not None
            and abs(current_energy - previous_energy) <= 3.0
        )
        checks[f"{host}_energy"] = energy_ok
        details[f"{host}_energy"] = {"candidate": current_energy, "baseline": previous_energy}
    return {"passed": all(checks.values()), "checks": checks, "details": details}


def link_blind_audio(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


async def prepare(manifest_path: Path, output: Path | None) -> tuple[Path, bool]:
    samples = load_manifest(manifest_path)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = output or PATHS.runtime / "evals" / f"podcast-tts-qualification-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    seed = hashlib.sha256((file_hash(manifest_path) + stamp).encode()).hexdigest()
    report: dict[str, Any] = {
        "status": "running", "created_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": file_hash(manifest_path), "samples": {},
    }
    mappings: dict[str, Any] = {}
    scores: dict[str, Any] = {"samples": {}}
    objective_passed = True
    for sample in samples:
        sample_id = sample["id"]
        candidate = load_json(sample["candidate_json"])
        sample_dir = output / sample_id
        sequence_dir = sample_dir / "sequence"
        sequence_dir.mkdir(parents=True)
        (sample_dir / "candidate.json").write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        quality = await render_candidate(
            candidate, sequence_dir, f"{stamp}-{sample_id}",
            tts_model="qwen3-tts-0.6b", tts_device="gpu", tts_mode="sequence",
        )
        (sequence_dir / "candidate-audio-quality.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        baseline_06_quality = load_json(sample["baseline_06_dir"] / "candidate-audio-quality.json")
        baseline_17_quality = load_json(sample["baseline_17_dir"] / "candidate-audio-quality.json")
        metrics = {
            "qwen-0.6b-gpu-sequence": acoustic_metrics(candidate, sequence_dir / "candidate-parts"),
            "qwen-0.6b-gpu-single": acoustic_metrics(candidate, sample["baseline_06_dir"] / "candidate-parts"),
            "qwen-1.7b-cpu-single": acoustic_metrics(candidate, sample["baseline_17_dir"] / "candidate-parts"),
        }
        gate_06 = automatic_gate(
            quality, metrics["qwen-0.6b-gpu-sequence"],
            baseline_06_quality, metrics["qwen-0.6b-gpu-single"],
        )
        gate_17 = automatic_gate(
            quality, metrics["qwen-0.6b-gpu-sequence"],
            baseline_17_quality, metrics["qwen-1.7b-cpu-single"],
        )
        sample_passed = bool(gate_06["passed"] and gate_17["passed"])
        objective_passed = objective_passed and sample_passed
        variants = {
            "qwen-1.7b-cpu-single": sample["baseline_17_dir"] / "candidate.m4a",
            "qwen-0.6b-gpu-single": sample["baseline_06_dir"] / "candidate.m4a",
            "qwen-0.6b-gpu-sequence": sequence_dir / "candidate.m4a",
        }
        order = sorted(SOURCES, key=lambda source: hashlib.sha256(f"{seed}:{sample_id}:{source}".encode()).hexdigest())
        mapping = {chr(ord("A") + index): source for index, source in enumerate(order)}
        for label, source in mapping.items():
            link_blind_audio(variants[source], sample_dir / "blind" / f"{label}.m4a")
        mappings[sample_id] = mapping
        scores["samples"][sample_id] = {
            label: {dimension: None for dimension in DIMENSIONS} for label in mapping
        }
        report["samples"][sample_id] = {
            "passed": sample_passed,
            "candidate_sha256": file_hash(sample["candidate_json"]),
            "quality": quality,
            "baseline_06_quality": baseline_06_quality,
            "baseline_17_quality": baseline_17_quality,
            "acoustic_metrics": metrics,
            "against_06_single": gate_06,
            "against_17_single": gate_17,
        }
        (sample_dir / "objective-result.json").write_text(
            json.dumps(report["samples"][sample_id], ensure_ascii=False, indent=2), encoding="utf-8",
        )
    report["status"] = "awaiting_scores" if objective_passed else "objective_failed"
    report["objective_passed"] = objective_passed
    (output / "qualification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "private-mapping.json").write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "scores-template.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    listing = "\n".join(
        f"- {sample['id']}: `{sample['id']}/blind/A.m4a`, `B.m4a`, `C.m4a`"
        for sample in samples
    )
    rubric = (
        "# Podcast TTS anonymous listening review\n\n"
        "Do not open private-mapping.json before scoring. For every A/B/C audio file, score each dimension "
        "from 1 to 5: naturalness, host_consistency, prosody_stability, intelligibility, listening_fatigue. "
        "A score of 4 means publishable and stable; for listening_fatigue, 5 means least fatiguing.\n\n"
        + listing + "\n"
    )
    (output / "blind-review.md").write_text(rubric, encoding="utf-8")
    return output, objective_passed


def finalize(run_dir: Path, scores_path: Path) -> tuple[dict[str, Any], bool]:
    report = load_json(run_dir / "qualification.json")
    mapping = load_json(run_dir / "private-mapping.json")
    scores = load_json(scores_path).get("samples")
    if not isinstance(scores, dict) or set(scores) != set(mapping):
        raise ValueError("Scores must cover every qualification sample")
    results: dict[str, Any] = {}
    passed = bool(report.get("objective_passed"))
    for sample_id, labels in mapping.items():
        submitted = scores.get(sample_id)
        if not isinstance(submitted, dict) or set(submitted) != set(labels):
            raise ValueError(f"Scores for {sample_id} must cover A/B/C")
        by_source: dict[str, dict[str, int]] = {}
        for label, source in labels.items():
            values = submitted[label]
            if not isinstance(values, dict) or any(
                isinstance(values.get(dimension), bool)
                or not isinstance(values.get(dimension), int)
                or not 1 <= values[dimension] <= 5
                for dimension in DIMENSIONS
            ):
                raise ValueError(f"{sample_id}/{label} requires five integer scores from 1 to 5")
            by_source[source] = values
        candidate = by_source["qwen-0.6b-gpu-sequence"]
        candidate_total = sum(candidate.values())
        blind_passed = bool(
            min(candidate.values()) >= 4
            and candidate_total >= sum(by_source["qwen-0.6b-gpu-single"].values())
            and candidate_total >= sum(by_source["qwen-1.7b-cpu-single"].values())
        )
        sample_passed = bool(report["samples"][sample_id]["passed"] and blind_passed)
        passed = passed and sample_passed
        results[sample_id] = {
            "passed": sample_passed, "blind_passed": blind_passed,
            "scores_by_source": by_source,
        }
    result = {"status": "passed" if passed else "failed", "samples": results}
    (run_dir / "blind-scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "final-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report["status"] = result["status"]
    (run_dir / "qualification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualify Podcast TTS batching and the 0.6B/GPU candidate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--output")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run-dir", required=True)
    finalize_parser.add_argument("--scores", required=True)
    automated_parser = subparsers.add_parser("auto-finalize")
    automated_parser.add_argument("--run-dir", required=True)
    automated_parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        output, passed = asyncio.run(prepare(
            Path(args.manifest).expanduser().resolve(),
            Path(args.output).expanduser().resolve() if args.output else None,
        ))
        print(json.dumps({"status": "awaiting_scores" if passed else "objective_failed", "output": str(output)}, ensure_ascii=False))
        if not passed:
            raise SystemExit(2)
    elif args.command == "finalize":
        result, passed = finalize(
            Path(args.run_dir).expanduser().resolve(), Path(args.scores).expanduser().resolve(),
        )
        print(json.dumps({"status": result["status"], "output": str(Path(args.run_dir).resolve())}, ensure_ascii=False))
        if not passed:
            raise SystemExit(2)
    else:
        result, passed = automated_finalize(
            Path(args.run_dir).expanduser().resolve(), Path(args.manifest).expanduser().resolve(),
        )
        print(json.dumps({
            "status": result["status"], "method": result["method"],
            "output": str(Path(args.run_dir).resolve()),
        }, ensure_ascii=False))
        if not passed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
