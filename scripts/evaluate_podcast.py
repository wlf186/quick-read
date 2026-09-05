#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sandevistan_read.database import DB, json_load
from sandevistan_read.audio_quality import assess_transcription, repair_turn_indexes
from sandevistan_read.config import CONFIG
from sandevistan_read.jobs import _actual_duration_check, _run_ffmpeg
from sandevistan_read.paths import PATHS
from sandevistan_read.podcast import PodcastQualityError, _content_minutes, _repeated_stem_ratio, build_podcast_script
from sandevistan_read.providers import active_provider, host_voice_instruction, host_voice_selection, synthesize, synthesize_sequence, transcribe_audio


def transcript(payload: dict[str, Any]) -> str:
    turns = payload.get("turns") or []
    if turns:
        return "\n".join(f"{turn.get('speaker', '?')}: {turn.get('text', '')}" for turn in turns)
    return str(payload.get("script") or "")


def anonymous_transcript(payload: dict[str, Any], *, asr: bool = False) -> str:
    values = payload.get("segments") if asr else payload.get("turns")
    values = [item for item in values or [] if isinstance(item, dict) and str(item.get("text") or "").strip()]
    if not values:
        text = str(payload.get("text") or payload.get("script") or "").strip()
        return f"Speaker 1: {text}" if text else ""
    speaker_order: dict[str, str] = {}
    normalized: list[tuple[str, str]] = []
    for item in values:
        raw_speaker = str(
            item.get("speaker_label") if asr and item.get("speaker_label") else item.get("speaker") or "speaker"
        )
        speaker = speaker_order.setdefault(raw_speaker, f"Speaker {len(speaker_order) + 1}")
        text = " ".join(str(item.get("text") or "").split())
        if asr and normalized and normalized[-1][0] == speaker:
            normalized[-1] = (speaker, normalized[-1][1] + " " + text)
        else:
            normalized.append((speaker, text))
    return "\n".join(f"{speaker}: {text}" for speaker, text in normalized)


def metrics(payload: dict[str, Any]) -> dict[str, Any]:
    turns = payload.get("turns") or []
    return {
        "version": payload.get("version", 1),
        "turn_count": len(turns),
        "estimated_minutes": round(_content_minutes(turns), 2) if turns else None,
        "repeated_stem_ratio": round(_repeated_stem_ratio(turns), 3) if turns and all("dialogue_act" in turn for turn in turns) else None,
        "cited_turn_ratio": round(sum(bool(turn.get("citation_ids")) for turn in turns) / max(1, len(turns)), 3),
        "quality": payload.get("quality") or {},
    }


def load_baseline(artifact_id: str | None) -> dict[str, Any] | None:
    if not artifact_id:
        return None
    row = DB.fetchone("SELECT payload_json FROM artifacts WHERE id=? AND type='podcast'", (artifact_id,))
    if not row:
        raise SystemExit(f"找不到 Podcast 产物：{artifact_id}")
    return json_load(row["payload_json"], {})


def review_sheet(items: list[tuple[str, str]]) -> str:
    rubric = """# Podcast blind review

请不要查看 `mapping.json`，先分别为每份文字稿打分。每项 1–5 分：

- 连贯性：每轮是否回应上一轮，跨章是否自然。
- 自然度：是否像真实双人交流，而不是模板采访或事实清单。
- 论证深度：是否展开前提、机制、限制和含义，而非停留在概念复述。
- 资料忠实度：是否存在无依据的数字、实体、案例或因果。
- 角色稳定：两位主持人的职责是否清晰且不僵硬。
- 重复控制：是否反复重启话题、复述相同事实或使用相同句式。

记录格式：`A: 连贯/自然/深度/忠实/角色/重复 = _/_/_/_/_/_`。
"""
    sections = "\n\n".join(f"## Transcript {label}\n\n{text}" for label, text in items)
    return rubric + "\n\n" + sections + "\n"


async def render_candidate(
    candidate: dict[str, Any],
    output: Path,
    stamp: str,
    *,
    tts_model: str | None = None,
    tts_device: str | None = None,
    tts_mode: str = "single",
) -> dict[str, Any]:
    provider = active_provider("audio")
    if not provider:
        raise RuntimeError("请先启用 AUDIO Provider")
    ffmpeg = CONFIG.tools.ffmpeg_path
    if not ffmpeg:
        raise RuntimeError("完整候选评测需要项目内 FFmpeg")
    config = provider.get("config") or {}
    language = str(candidate.get("language") or "zh-CN")
    voices = {
        "HOST_A": host_voice_selection(config, "host_a", language),
        "HOST_B": host_voice_selection(config, "host_b", language),
    }
    # tts_model/tts_device 只覆盖本次渲染；显式参数经 synthesize() 透传，不修改已保存的 provider
    model = str(tts_model or provider.get("model") or "")
    device = str(tts_device or config.get("compute_device") or "gpu")
    model_caps = next((item for item in (provider.get("capabilities") or {}).get("models", []) if item.get("id") == model), {})
    supports_instruction = "preset" in ((model_caps.get("controls") or {}).get("instruction_voice_modes") or [])
    instructions = {
        "HOST_A": host_voice_instruction(config, "host_a", language, supported=supports_instruction),
        "HOST_B": host_voice_instruction(config, "host_b", language, supported=supports_instruction),
    }
    parts_dir = output / "candidate-parts"
    normalized_dir = output / "candidate-normalized"
    parts_dir.mkdir(exist_ok=True)
    normalized_dir.mkdir(exist_ok=True)
    turns = candidate.get("turns") or []
    if tts_mode not in {"single", "sequence"}:
        raise ValueError("tts_mode must be single or sequence")
    sequence_capability = (provider.get("capabilities") or {}).get("sequence_jobs") or {}
    if tts_mode == "sequence" and not sequence_capability.get("supported"):
        raise RuntimeError("当前 AUDIO Provider 未声明批量 TTS 能力")
    execution: dict[str, Any] = {
        "requested_device": device, "compute_device": device, "fallback_used": False,
        "model": model, "tts_mode": tts_mode, "single_jobs": 0, "sequence_jobs": 0,
        "sequence_item_counts": [], "generation_batch_sizes": [], "oom_fallbacks": [],
    }

    async def synthesize_turn(index: int, retry: bool = False) -> None:
        turn = turns[index]
        part = parts_dir / f"{index:04d}.wav"
        selection = voices[turn["speaker"]]
        instruction = instructions[turn["speaker"]]
        digest = hashlib.sha256(json.dumps({
            "model": model, "device": device, "voice": selection,
            "instruction": instruction, "text": turn["text"],
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
        turn_execution: dict[str, Any] = {}
        await synthesize(
            turn["text"], selection.get("speaker"), part,
            language="English" if language == "en" else "Chinese",
            model=model, compute_device=device, voice_mode=selection["mode"],
            voiceprint_sample_id=selection.get("sample_id"), instruct=instruction,
            idempotency_key=f"sread-eval-{stamp[:20]}-{index:04d}-{digest}{'-r1' if retry else ''}",
            execution=turn_execution,
        )
        if turn_execution.get("fallback_used"):
            execution.update(turn_execution)
        execution["single_jobs"] += 1

    async def synthesize_indexes(indexes: list[int], retry: bool = False) -> None:
        if tts_mode == "single":
            for index in indexes:
                await synthesize_turn(index, retry=retry)
            return
        max_items = max(1, min(100, int(sequence_capability.get("max_items") or 100)))
        max_chars = max(1, int(sequence_capability.get("max_total_chars") or 100000))
        for voice_mode in ("preset", "voiceprint"):
            pending = [index for index in indexes if voices[turns[index]["speaker"]]["mode"] == voice_mode]
            while pending:
                batch: list[int] = []
                chars = 0
                while pending and len(batch) < max_items:
                    length = len(str(turns[pending[0]]["text"]))
                    if batch and chars + length > max_chars:
                        break
                    batch.append(pending.pop(0))
                    chars += length
                items: list[dict[str, Any]] = []
                outputs: dict[str, Path] = {}
                for index in batch:
                    turn = turns[index]
                    selection = voices[turn["speaker"]]
                    item_id = f"turn-{index:04d}"
                    items.append({
                        "id": item_id, "text": turn["text"], "speaker": selection.get("speaker"),
                        "voiceprint_sample_id": selection.get("sample_id"),
                        "instruct": instructions[turn["speaker"]],
                    })
                    outputs[item_id] = parts_dir / f"{index:04d}.wav"
                batch_execution: dict[str, Any] = {}
                digest = hashlib.sha256(
                    "".join(str(turns[index]["text"]) for index in batch).encode()
                ).hexdigest()[:24]
                await synthesize_sequence(
                    items, outputs,
                    language="English" if language == "en" else "Chinese",
                    model=model, compute_device=device, voice_mode=voice_mode,
                    idempotency_key=f"sread-eval-seq-{stamp[:18]}-{digest}{'-r1' if retry else ''}",
                    execution=batch_execution, provider=provider,
                )
                execution["sequence_jobs"] += 1
                execution["sequence_item_counts"].append(len(batch))
                execution["generation_batch_sizes"].append(int(batch_execution.get("generation_batch_size") or 1))
                execution["oom_fallbacks"].extend(batch_execution.get("oom_fallbacks") or [])
                if batch_execution.get("fallback_used"):
                    execution.update(batch_execution)

    async def render(force_indexes: set[int] | None = None) -> tuple[Path, float]:
        force_indexes = force_indexes or set()
        chapter_ends = {int(chapter["turn_end"]) for chapter in (candidate.get("chapters") or [])[:-1]}
        normalized: list[Path] = []
        for index in range(len(turns)):
            source, target = parts_dir / f"{index:04d}.wav", normalized_dir / f"{index:03d}.wav"
            if index in force_indexes or not target.exists():
                pause = 0.60 if index in chapter_ends else 0.22
                code, stderr = await _run_ffmpeg(
                    [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-af", f"apad=pad_dur={pause}", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(target)],
                    180,
                    lambda: False,
                )
                if code:
                    raise RuntimeError(f"FFmpeg 音频标准化失败: {stderr[-500:]}")
            normalized.append(target)
        cursor = 0.0
        for turn, part in zip(turns, normalized):
            with wave.open(str(part), "rb") as audio:
                duration = audio.getnframes() / max(1, audio.getframerate())
            turn["start_seconds"] = round(cursor, 3)
            cursor += duration
            turn["end_seconds"] = round(cursor, 3)
        concat_file = output / "candidate-concat.txt"
        concat_file.write_text("".join(f"file '{part.as_posix()}'\n" for part in normalized), encoding="utf-8")
        destination = output / "candidate.m4a"
        code, stderr = await _run_ffmpeg(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "24000", "-ac", "1", "-c:a", "aac", "-b:a", "128k", str(destination)],
            600,
            lambda: False,
        )
        if code:
            raise RuntimeError(f"FFmpeg 音频合并失败: {stderr[-500:]}")
        return destination, cursor

    synthesis_started = time.perf_counter()
    missing = [index for index in range(len(turns)) if not (parts_dir / f"{index:04d}.wav").is_file()]
    await synthesize_indexes(missing)
    execution["tts_seconds"] = round(time.perf_counter() - synthesis_started, 3)
    destination, seconds = await render()
    duration = _actual_duration_check(float(candidate["duration"]["target_minutes"]), seconds)
    if not duration["passed"]:
        return {"passed": False, "stage": "duration", "duration": duration, "execution": execution}
    asr = await transcribe_audio(
        destination,
        language="English" if language == "en" else "Chinese",
        idempotency_key=f"sread-eval-candidate-{stamp}",
    )
    quality = assess_transcription(turns, asr, language)
    retry_indexes = repair_turn_indexes(quality)
    if retry_indexes and len(retry_indexes) <= 6:
        repair_started = time.perf_counter()
        await synthesize_indexes(retry_indexes, retry=True)
        execution["tts_seconds"] = round(execution["tts_seconds"] + time.perf_counter() - repair_started, 3)
        destination, seconds = await render(set(retry_indexes))
        duration = _actual_duration_check(float(candidate["duration"]["target_minutes"]), seconds)
        if not duration["passed"]:
            return {"passed": False, "stage": "duration_after_audio_repair", "duration": duration, "execution": execution}
        asr = await transcribe_audio(
            destination,
            language="English" if language == "en" else "Chinese",
            idempotency_key=f"sread-eval-candidate-{stamp}-verify",
        )
        quality = assess_transcription(turns, asr, language)
        quality["repaired_turns"] = retry_indexes
    (output / "candidate-asr.json").write_text(json.dumps(asr, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {**quality, "duration": duration, "execution": execution}
    result["passed"] = bool(quality.get("passed") and duration["passed"])
    return result


async def run(args: argparse.Namespace) -> tuple[Path, bool]:
    baseline = load_baseline(args.baseline_artifact)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = PATHS.runtime / "evals" / f"podcast-v4-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    if args.candidate_json:
        candidate_path = Path(args.candidate_json).expanduser().resolve()
        if not candidate_path.is_file():
            raise SystemExit(f"找不到候选 JSON：{candidate_path}")
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        if not isinstance(candidate, dict):
            raise SystemExit("--candidate-json 必须是 Podcast 候选对象")
        candidate_quality = candidate.get("quality") or candidate.get("quality_report") or {}
        if not candidate.get("turns") or candidate_quality.get("passed") is not True:
            raise SystemExit("--candidate-json 必须是已通过质量门禁且包含 turns 的 Podcast 候选")
    else:
        if not args.notebook_id:
            raise SystemExit("生成新候选时必须提供 --notebook-id")
        payload = {
            "source_ids": args.source_id or None,
            "duration_mode": "fixed",
            "minutes": args.minutes,
            "language": args.language,
            "focus": args.focus,
        }
        try:
            candidate = await build_podcast_script(args.notebook_id, payload)
        except PodcastQualityError as exc:
            failure = {"status": "failed", "message": str(exc), "quality_report": exc.report}
            (output / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            return output, False
    (output / "candidate.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = {"candidate": metrics(candidate), "baseline": metrics(baseline) if baseline else None}
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.render_candidate:
        try:
            audio_quality = await render_candidate(
                candidate, output, stamp, tts_model=args.tts_model, tts_device=args.tts_device,
                tts_mode=args.tts_mode,
            )
        except Exception as exc:
            audio_quality = {"passed": False, "stage": "render", "error": f"{type(exc).__name__}: {exc}"}
        (output / "candidate-audio-quality.json").write_text(
            json.dumps(audio_quality, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not audio_quality.get("passed"):
            return output, False
    review_items = [("candidate", anonymous_transcript(candidate))]
    if baseline:
        review_items.append(("baseline", transcript(baseline)))
    for index, value in enumerate(args.reference_audio or [], start=1):
        reference = Path(value).expanduser().resolve()
        if not reference.is_file():
            raise SystemExit(f"找不到参考音频：{reference}")
        reference_key = hashlib.sha256(
            f"{reference}:{reference.stat().st_size}:{reference.stat().st_mtime_ns}".encode()
        ).hexdigest()[:32]
        result = await transcribe_audio(
            reference,
            language=args.reference_language,
            idempotency_key=f"sread-eval-{reference_key}",
        )
        (output / f"reference-{index}-asr.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        review_items.append((f"reference-{index}", anonymous_transcript(result, asr=True)))
    if len(review_items) > 1:
        review_items.sort(key=lambda item: hashlib.sha256((stamp + item[0] + item[1]).encode()).hexdigest())
        labels = [chr(ord("A") + index) for index in range(len(review_items))]
        ordered = [(label, item[1]) for label, item in zip(labels, review_items)]
        mapping = {label: item[0] for label, item in zip(labels, review_items)}
        (output / "blind-review.md").write_text(review_sheet(ordered), encoding="utf-8")
        (output / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 Podcast V4 脚本评测；可选用当前本地 TTS/ASR 渲染并验收同一候选。")
    parser.add_argument("--notebook-id")
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--minutes", type=int, choices=(5, 10, 14, 20, 22, 25, 30), default=5)
    parser.add_argument("--language", choices=("auto", "zh-CN", "en"), default="zh-CN")
    parser.add_argument("--focus", default="")
    parser.add_argument("--baseline-artifact")
    parser.add_argument("--reference-audio", action="append", default=[], help="加入盲评的参考音频，可重复指定")
    parser.add_argument("--reference-language", choices=("Chinese", "English"), default="Chinese")
    parser.add_argument("--render-candidate", action="store_true", help="渲染已生成候选并执行实际时长与 ASR 门禁；不会再次调用 MAIN")
    parser.add_argument("--candidate-json", help="复用已通过门禁的候选 JSON，跳过 MAIN 生成")
    parser.add_argument("--tts-model", help="仅本次渲染覆盖 TTS 模型（如 qwen3-tts-0.6b），不修改已保存的 Provider")
    parser.add_argument("--tts-device", help="仅本次渲染覆盖 TTS 设备（gpu/cpu），不修改已保存的 Provider")
    parser.add_argument("--tts-mode", choices=("single", "sequence"), default="single", help="逐轮或批量渲染")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    destination, passed = asyncio.run(run(args))
    print(json.dumps({"status": "passed" if passed else "failed", "output": str(destination)}, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
