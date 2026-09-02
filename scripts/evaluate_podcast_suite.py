#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluate_podcast import anonymous_transcript, render_candidate
from sandevistan_read.paths import PATHS
from sandevistan_read.podcast import PodcastQualityError, build_podcast_script
from sandevistan_read.providers import active_provider, transcribe_audio


DIMENSIONS = ("coherence", "naturalness", "depth", "grounding", "roles", "repetition")


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError("suite manifest 必须包含非空 samples 数组")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw in samples:
        if not isinstance(raw, dict):
            raise ValueError("每个 sample 必须是对象")
        sample_id = str(raw.get("id") or "").strip()
        if not sample_id or sample_id in seen:
            raise ValueError("sample id 必须非空且唯一")
        seen.add(sample_id)
        reference = Path(str(raw.get("reference_audio") or "")).expanduser().resolve()
        if not reference.is_file():
            raise ValueError(f"找不到参考音频：{reference}")
        minutes = int(raw.get("minutes") or 0)
        if minutes not in {5, 10, 12, 14, 20, 22, 25, 30}:
            raise ValueError(f"{sample_id} 的 minutes 不受支持")
        language = str(raw.get("language") or "zh-CN")
        if language not in {"zh-CN", "en"}:
            raise ValueError(f"{sample_id} 的 language 无效")
        candidate_json = str(raw.get("candidate_json") or "").strip() or None
        normalized.append(
            {
                "id": sample_id,
                "notebook_id": str(raw.get("notebook_id") or "").strip(),
                "source_ids": [str(value) for value in raw.get("source_ids") or [] if str(value)],
                "minutes": minutes,
                "language": language,
                "focus": str(raw.get("focus") or ""),
                "reference_audio": str(reference),
                "reference_language": str(raw.get("reference_language") or ("English" if language == "en" else "Chinese")),
                "candidate_json": candidate_json,
            }
        )
    return {"version": int(payload.get("version") or 1), "samples": normalized}


def _provider_fingerprint() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("main", "tts"):
        provider = active_provider(role) or {}
        config = provider.get("config") or {}
        result[role] = {
            "name": provider.get("name"),
            "kind": provider.get("kind"),
            "base_url": provider.get("base_url"),
            "model": provider.get("model"),
            "capabilities": provider.get("capabilities") or {},
            "config": {
                key: config.get(key)
                for key in (
                    "temperature", "context_window_tokens", "max_output_tokens", "compute_device",
                    "host_a", "host_b", "host_a_en", "host_b_en", "allow_device_fallback",
                )
                if key in config
            },
        }
    return result


def _implementation_snapshot() -> dict[str, str]:
    paths = (
        PATHS.root / "src/sandevistan_read/podcast.py",
        PATHS.root / "scripts/evaluate_podcast.py",
        PATHS.root / "scripts/evaluate_podcast_suite.py",
    )
    return {str(path.relative_to(PATHS.root)): _file_hash(path) for path in paths}


def _load_candidate(path: Path) -> dict[str, Any]:
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(candidate, dict):
        raise ValueError("development candidate_json 必须是 Podcast 对象")
    quality = candidate.get("quality") or candidate.get("quality_report") or {}
    if not candidate.get("turns") or quality.get("passed") is not True:
        raise ValueError("development candidate_json 必须包含已通过门禁的 Podcast 候选")
    return candidate


async def _reference_asr(sample: dict[str, Any], cache_dir: Path) -> tuple[dict[str, Any], Path, bool]:
    reference = Path(sample["reference_audio"])
    tts = active_provider("tts") or {}
    asr = (tts.get("capabilities") or {}).get("asr") or {}
    cache_identity = {
        "audio_sha256": _file_hash(reference),
        "language": sample["reference_language"],
        "asr_model": asr.get("default_model"),
    }
    cache_path = cache_dir / f"{_json_hash(cache_identity)}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8")), cache_path, True
    result = await transcribe_audio(
        reference,
        language=sample["reference_language"],
        idempotency_key=f"sread-suite-ref-{cache_identity['audio_sha256'][:32]}",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, cache_path, False


def _blind_packet(sample_id: str, candidate_asr: dict[str, Any], reference_asr: dict[str, Any], seed: str) -> tuple[str, dict[str, str]]:
    values = {
        "candidate": anonymous_transcript(candidate_asr, asr=True),
        "reference": anonymous_transcript(reference_asr, asr=True),
    }
    order = sorted(values, key=lambda key: hashlib.sha256(f"{seed}:{sample_id}:{key}".encode()).hexdigest())
    mapping = {chr(ord("A") + index): key for index, key in enumerate(order)}
    rubric = """请在不读取 mapping 的情况下为 A/B 分别评分，每项为 1–5 的整数：
coherence（连贯性）、naturalness（自然度）、depth（论证深度）、grounding（资料忠实度）、roles（角色稳定）、repetition（重复控制）。
4=可发布，5=优秀。只依据同一份资料背景比较，不因篇幅更长自动加分。
"""
    body = "\n\n".join(f"## Transcript {label}\n\n{values[source]}" for label, source in mapping.items())
    return f"# {sample_id} anonymous A/B\n\n{rubric}\n{body}\n", mapping


async def prepare_suite(
    manifest_path: Path,
    *,
    mode: str,
    output: Path | None = None,
    resume: bool = False,
) -> tuple[Path, bool]:
    manifest = load_manifest(manifest_path)
    if mode == "frozen" and any(sample.get("candidate_json") for sample in manifest["samples"]):
        raise ValueError("frozen 模式禁止 candidate_json 或 resume")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = output or PATHS.runtime / "evals" / f"podcast-suite-{stamp}"
    manifest_hash = _json_hash(manifest)
    implementation = _implementation_snapshot()
    if resume:
        if not output.is_dir() or not (output / "suite.json").is_file():
            raise ValueError("--resume 需要已存在的 suite 目录")
        snapshot = json.loads((output / "suite.json").read_text(encoding="utf-8"))
        if snapshot.get("mode") != mode or snapshot.get("manifest_hash") != manifest_hash:
            raise ValueError("恢复的 mode 或 manifest 与原 suite 不一致")
        if mode == "frozen" and snapshot.get("implementation") != implementation:
            raise ValueError("frozen suite 的实现快照已改变，必须开始新批次")
        snapshot.update({
            "status": "running",
            "implementation": implementation,
            "provider": _provider_fingerprint(),
            "resumed_at": datetime.now(UTC).isoformat(),
        })
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot = {
            "status": "running",
            "mode": mode,
            "created_at": datetime.now(UTC).isoformat(),
            "manifest_hash": manifest_hash,
            "implementation": implementation,
            "provider": _provider_fingerprint(),
            "samples": {},
        }
    seed = hashlib.sha256(f"{snapshot['created_at']}:{manifest_hash}".encode()).hexdigest()
    (output / "suite.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    mappings: dict[str, dict[str, str]] = {}
    scores_template: dict[str, Any] = {"samples": {}}
    cache_dir = PATHS.runtime / "evals" / "reference-cache"
    for sample in manifest["samples"]:
        sample_id = sample["id"]
        sample_dir = output / sample_id
        previous = (snapshot.get("samples") or {}).get(sample_id) or {}
        if resume and previous.get("status") == "awaiting_scores":
            continue
        sample_dir.mkdir(exist_ok=resume)
        try:
            if resume and (sample_dir / "candidate.json").is_file():
                candidate = _load_candidate(sample_dir / "candidate.json")
                candidate_source = "resumed"
            elif sample.get("candidate_json"):
                candidate = _load_candidate(Path(sample["candidate_json"]).expanduser().resolve())
                candidate_source = "reused"
            else:
                if not sample["notebook_id"]:
                    raise ValueError(f"{sample_id} 缺少 notebook_id")
                candidate = await build_podcast_script(
                    sample["notebook_id"],
                    {
                        "source_ids": sample["source_ids"] or None,
                        "duration_mode": "fixed",
                        "minutes": sample["minutes"],
                        "language": sample["language"],
                        "focus": sample["focus"],
                    },
                )
                candidate_source = "fresh"
            (sample_dir / "candidate.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
            audio_quality = await render_candidate(candidate, sample_dir, f"{stamp}-{sample_id}")
            (sample_dir / "candidate-rendered.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
            (sample_dir / "candidate-audio-quality.json").write_text(
                json.dumps(audio_quality, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if not audio_quality.get("passed"):
                raise RuntimeError(f"候选音频门禁失败：{audio_quality.get('stage') or 'audio'}")
            reference_asr, cache_path, cache_hit = await _reference_asr(sample, cache_dir)
            (sample_dir / "reference-asr.json").write_text(json.dumps(reference_asr, ensure_ascii=False, indent=2), encoding="utf-8")
            candidate_asr = json.loads((sample_dir / "candidate-asr.json").read_text(encoding="utf-8"))
            packet, mapping = _blind_packet(sample_id, candidate_asr, reference_asr, seed)
            (sample_dir / "blind-review.md").write_text(packet, encoding="utf-8")
            mappings[sample_id] = mapping
            scores_template["samples"][sample_id] = {
                label: {dimension: None for dimension in DIMENSIONS} for label in ("A", "B")
            }
            usage = candidate.get("context_usage") or {}
            sample_result = {
                "status": "awaiting_scores",
                "candidate_source": candidate_source,
                "script_gate": bool((candidate.get("quality") or {}).get("passed")),
                "audio_gate": True,
                "audio_quality": audio_quality,
                "main_requests": usage.get("requests"),
                "main_tokens": usage.get("accounted_total_tokens") or usage.get("actual_total_tokens"),
                "reference_cache": str(cache_path),
                "reference_cache_hit": cache_hit,
            }
            snapshot["samples"][sample_id] = sample_result
            (sample_dir / "result.json").write_text(json.dumps(sample_result, ensure_ascii=False, indent=2), encoding="utf-8")
            failure_path = sample_dir / "failure.json"
            if failure_path.is_file():
                previous_failure = sample_dir / "failure.previous.json"
                if previous_failure.is_file():
                    previous_failure.unlink()
                failure_path.replace(previous_failure)
        except PodcastQualityError as exc:
            failure = {"status": "failed", "stage": "script", "message": str(exc), "quality_report": exc.report}
            (sample_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            snapshot["samples"][sample_id] = failure
            snapshot["status"] = "failed"
            (output / "suite.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            return output, False
        except Exception as exc:
            failure = {"status": "failed", "stage": "pipeline", "message": f"{type(exc).__name__}: {exc}"}
            (sample_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            snapshot["samples"][sample_id] = failure
            snapshot["status"] = "failed"
            (output / "suite.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            return output, False
        (output / "suite.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot["status"] = "awaiting_scores"
    (output / "private-mapping.json").write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "scores-template.json").write_text(json.dumps(scores_template, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "suite.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, True


def _validated_scores(payload: dict[str, Any], sample_ids: set[str]) -> dict[str, Any]:
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, dict) or set(samples) != sample_ids:
        raise ValueError("scores 必须恰好覆盖 suite 的全部 sample")
    for sample_id, labels in samples.items():
        if not isinstance(labels, dict) or set(labels) != {"A", "B"}:
            raise ValueError(f"{sample_id} 必须包含 A/B 两份评分")
        for label, scores in labels.items():
            if not isinstance(scores, dict) or any(
                isinstance(scores.get(name), bool) or not isinstance(scores.get(name), int) or not 1 <= scores[name] <= 5
                for name in DIMENSIONS
            ):
                raise ValueError(f"{sample_id}/{label} 必须包含完整的 1–5 六维整数评分")
    return samples


def finalize_suite(run_dir: Path, scores_path: Path) -> tuple[dict[str, Any], bool]:
    suite_path = run_dir / "suite.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if suite.get("status") != "awaiting_scores":
        raise ValueError("suite 尚未准备好评分，或已经结束")
    mapping = json.loads((run_dir / "private-mapping.json").read_text(encoding="utf-8"))
    scores = _validated_scores(json.loads(scores_path.read_text(encoding="utf-8")), set(mapping))
    results: dict[str, Any] = {}
    passed = True
    for sample_id, labels in mapping.items():
        candidate_label = next(label for label, source in labels.items() if source == "candidate")
        reference_label = next(label for label, source in labels.items() if source == "reference")
        candidate_scores = scores[sample_id][candidate_label]
        reference_scores = scores[sample_id][reference_label]
        candidate_total = sum(candidate_scores[name] for name in DIMENSIONS)
        reference_total = sum(reference_scores[name] for name in DIMENSIONS)
        pipeline = suite["samples"][sample_id]
        token_ok = int(pipeline.get("main_tokens") or 0) <= 45_000
        sample_passed = bool(
            pipeline.get("script_gate")
            and pipeline.get("audio_gate")
            and token_ok
            and min(candidate_scores[name] for name in DIMENSIONS) >= 4
            and candidate_total >= reference_total
        )
        results[sample_id] = {
            "passed": sample_passed,
            "candidate_scores": candidate_scores,
            "reference_scores": reference_scores,
            "candidate_total": candidate_total,
            "reference_total": reference_total,
            "main_tokens": pipeline.get("main_tokens"),
            "main_token_warning": int(pipeline.get("main_tokens") or 0) > 40_000,
            "audio_quality": pipeline.get("audio_quality"),
        }
        passed = passed and sample_passed
    result = {
        "status": "passed" if passed else "failed",
        "mode": suite.get("mode"),
        "manifest_hash": suite.get("manifest_hash"),
        "implementation": suite.get("implementation"),
        "provider": suite.get("provider"),
        "samples": results,
    }
    (run_dir / "blind-scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "final-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Podcast suite final report", "", f"Result: **{result['status']}**", ""]
    for sample_id, sample in results.items():
        lines.append(
            f"- {sample_id}: {'PASS' if sample['passed'] else 'FAIL'}; "
            f"candidate {sample['candidate_total']} vs reference {sample['reference_total']}; "
            f"MAIN {sample['main_tokens']} tokens."
        )
    (run_dir / "final-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    suite["status"] = result["status"]
    suite_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行可冻结、可解盲的 Podcast 多样本端到端质量评测。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--mode", choices=("development", "frozen"), default="development")
    prepare.add_argument("--output")
    prepare.add_argument("--resume", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-dir", required=True)
    finalize.add_argument("--scores", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        destination, prepared = asyncio.run(
            prepare_suite(
                Path(args.manifest).expanduser().resolve(),
                mode=args.mode,
                output=Path(args.output).expanduser().resolve() if args.output else None,
                resume=args.resume,
            )
        )
        print(json.dumps({"status": "awaiting_scores" if prepared else "failed", "output": str(destination)}, ensure_ascii=False))
        if not prepared:
            raise SystemExit(2)
    else:
        result, passed = finalize_suite(Path(args.run_dir).expanduser().resolve(), Path(args.scores).expanduser().resolve())
        print(json.dumps({"status": result["status"], "output": str(Path(args.run_dir).resolve())}, ensure_ascii=False))
        if not passed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
