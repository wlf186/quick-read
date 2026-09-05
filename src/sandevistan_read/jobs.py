from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
import wave
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from .database import DB, json_dump, json_load, new_id, utc_now
from .config import CONFIG
from .paths import PATHS
from .providers import ProviderError, active_provider, audio_provider_readiness, host_voice_instruction, host_voice_selection, provider_by_id, study_generation_profile, synthesize, synthesize_sequence, transcribe_audio
from .audio_quality import assess_transcription, repair_turn_indexes
from .podcast import PODCAST_DURATION_CALIBRATION_VERSION, PODCAST_ENGINE_VERSION, PodcastQualityError, build_podcast_script
from .services import ingest_source, make_summary
from .study import generate_study_artifact
from .observability import LABELS, Reporter
from .cleanup import process_cleanup_operations, register_resource
from .context_budget import TokenLimits


def enqueue(kind: str, notebook_id: str | None, payload: dict[str, Any], parent_id: str | None = None) -> dict[str, Any]:
    payload = dict(payload)
    job_id, now = new_id("job"), utc_now()
    source_count = len(payload.get("source_ids") or []) or (1 if payload.get("source_id") else 0)
    workload = {"source_count": source_count, "count": payload.get("count"), "minutes": payload.get("minutes"), "bucket": "single" if source_count <= 1 else "small_multi" if source_count <= 4 else "large_multi"}
    provider = active_provider("audio" if kind == "podcast" else "main") if kind != "ingest" else None
    context_provider = active_provider("main") if kind not in {"ingest"} else None
    if kind == "podcast":
        payload["provider_ids"] = {
            "audio": provider.get("id") if provider else None,
            "main": context_provider.get("id") if context_provider else None,
        }
    profile = {"kind": provider.get("kind") if provider else "local", "model": provider.get("model") if provider else CONFIG.models.embedding, "device": (provider.get("config") or {}).get("compute_device") if provider else "local"}
    if kind == "podcast" and provider:
        audio_config = provider.get("config") or {}
        profile["asr"] = {
            "model": audio_config.get("asr_model"),
            "device": audio_config.get("asr_compute_device"),
        }
    if context_provider:
        limits = TokenLimits.from_provider(context_provider)
        profile["context"] = {
            "provider": context_provider.get("name"),
            "model": context_provider.get("model"),
            "effective_context_tokens": limits.effective_context_tokens,
            "max_output_tokens": limits.max_output_tokens,
            "context_source": limits.context_source,
        }
        if kind in {"quiz", "flashcard"}:
            profile["study_generation"] = study_generation_profile(context_provider)
    DB.execute("""INSERT INTO jobs
        (id,kind,state,stage,progress,notebook_id,parent_id,payload_json,result_json,error,retryable,cancel_requested,attempts,created_at,updated_at,started_at,finished_at,display_name,stage_code,stage_progress,progress_basis,activity_json,workload_json,execution_profile_json,processing_seconds)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (job_id, kind, "queued", "等待执行", 0.0, notebook_id, parent_id, json_dump(payload), None, None, 0, 0, 0, now, now, None, None, LABELS.get(kind, kind), "queued", 0.0, "observed", "{}", json_dump(workload), json_dump(profile), 0.0))
    Reporter(job_id).update("queued", "等待执行", 0.0, state="queued")
    return DB.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,)) or {}


def _update(job_id: str, *, state: str | None = None, stage: str | None = None, progress: float | None = None, result: Any = None, error: str | None = None) -> None:
    row = DB.fetchone("SELECT state,stage,stage_code,progress,started_at FROM jobs WHERE id=?", (job_id,)) or {}
    next_state = state or row.get("state", "queued")
    stage_code = {"queued":"queued","running":"running","cancelling":"cancelling","complete":"complete","failed":"failed","cancelled":"cancelled"}.get(next_state, row.get("stage_code", "running"))
    Reporter(job_id).update(stage_code, stage or row.get("stage", "处理中"), progress if progress is not None else float(row.get("progress", 0)), state=next_state)
    fields, values = ["updated_at=?"], [utc_now()]
    if error is not None:
        fields.append("error=?"); values.append(error)
    if result is not None:
        fields.append("result_json=?"); values.append(json_dump(result))
    if state == "running" and not row.get("started_at"):
        fields.extend(["started_at=?", "attempts=attempts+1"]); values.append(utc_now())
    if state in {"complete", "failed", "cancelled"}:
        fields.append("finished_at=?"); values.append(utc_now())
    values.append(job_id)
    DB.execute(f"UPDATE jobs SET {','.join(fields)} WHERE id=?", tuple(values))


async def _run_ffmpeg(arguments: list[str], timeout: float, cancel_check: Callable[[], bool]) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(*arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    started = asyncio.get_running_loop().time()
    while process.returncode is None:
        try:
            await asyncio.wait_for(process.wait(), timeout=0.4)
        except TimeoutError:
            if cancel_check() or asyncio.get_running_loop().time() - started > timeout:
                process.terminate()
                try: await asyncio.wait_for(process.wait(), timeout=3)
                except TimeoutError: process.kill(); await process.wait()
                if cancel_check(): raise RuntimeError("任务已取消")
                raise RuntimeError("FFmpeg 执行超时")
    _, stderr = await process.communicate()
    return process.returncode or 0, stderr.decode(errors="replace")


def _actual_duration_check(target_minutes: float, actual_seconds: float) -> dict[str, Any]:
    actual_minutes = actual_seconds / 60
    ratio = actual_minutes / max(0.001, target_minutes)
    return {
        "passed": 0.85 <= ratio <= 1.20,
        "target_minutes": target_minutes,
        "actual_minutes": round(actual_minutes, 2),
        "duration_ratio": round(ratio, 3),
        "accepted_range": [0.85, 1.20],
    }


def _podcast_overlap_safe(main_provider: dict[str, Any], audio_provider: dict[str, Any]) -> bool:
    """Avoid resource contention when MAIN is local or shares the AUDIO service host."""
    main = urlsplit(str(main_provider.get("base_url") or ""))
    audio = urlsplit(str(audio_provider.get("base_url") or ""))
    main_host = (main.hostname or "").lower()
    audio_host = (audio.hostname or "").lower()
    if main_host in {"", "localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return False
    return main_host != audio_host


async def _podcast(notebook_id: str, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
    podcast_started = time.perf_counter()
    snapshot = payload.get("provider_ids") or {}
    provider = provider_by_id(snapshot.get("audio")) if snapshot.get("audio") else active_provider("audio")
    ready, readiness_message = audio_provider_readiness(provider)
    if not ready or not provider:
        raise RuntimeError(readiness_message)
    main_provider = provider_by_id(snapshot.get("main")) if snapshot.get("main") else active_provider("main")
    if not main_provider:
        raise RuntimeError("请先配置并启用 MAIN provider")
    config = provider.get("config", {})
    suffix = job_id.removeprefix("job_")
    work_dir = PATHS.job_work / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    register_resource("job", job_id, notebook_id, "podcast-work", work_dir)
    out_dir = PATHS.artifacts / f"podcast_{suffix}"
    manifest_path = work_dir / "manifest.json"
    signature = hashlib.sha256(
        json_dump(
            {
                "payload": payload,
                "main": {"name": main_provider.get("name"), "model": main_provider.get("model"), "config": main_provider.get("config")},
                "audio": {
                    "name": provider.get("name"),
                    "tts_model": provider.get("model"),
                    "tts_device": config.get("compute_device"),
                    "asr_model": config.get("asr_model"),
                    "asr_device": config.get("asr_compute_device"),
                    "voices": {
                        key: config.get(key) for key in (
                            "host_a", "host_b", "host_a_en", "host_b_en",
                            "host_a_voice_mode", "host_b_voice_mode",
                            "host_a_voiceprint_person_id", "host_b_voiceprint_person_id",
                            "host_a_voiceprint_sample_id", "host_b_voiceprint_sample_id",
                            "host_a_instruct", "host_b_instruct",
                        )
                    },
                },
                "script_engine": PODCAST_ENGINE_VERSION,
                "duration_calibration": PODCAST_DURATION_CALIBRATION_VERSION,
            }
        ).encode()
    ).hexdigest()

    def save_manifest(value: dict[str, Any]) -> None:
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json_dump(value), encoding="utf-8")
        temporary.replace(manifest_path)

    manifest = json_load(manifest_path.read_text(encoding="utf-8"), {}) if manifest_path.exists() else {}
    selected_model_early = str(provider.get("model") or "")
    selected_device_early = str(config.get("compute_device") or "gpu")
    model_caps_early = next(
        (item for item in provider.get("capabilities", {}).get("models", []) if item.get("id") == selected_model_early), {}
    )
    checkpoint_revisions_early = {
        str(item.get("variant") or ""): str(item.get("revision") or "")
        for item in model_caps_early.get("checkpoints") or [] if isinstance(item, dict)
    }
    sequence_capability_early = (provider.get("capabilities") or {}).get("sequence_jobs") or {}
    overlap_enabled = bool(
        config.get("podcast_sequence_tts", True)
        and sequence_capability_early.get("supported")
        and int(sequence_capability_early.get("contract_version") or 0) == 1
        and _podcast_overlap_safe(main_provider, provider)
        and not manifest.get("generated")
    )
    parts_dir_early = work_dir / "parts"
    parts_dir_early.mkdir(parents=True, exist_ok=True)
    cancel_check_early = lambda: bool((DB.fetchone("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)) or {}).get("cancel_requested"))
    speculative_stats: dict[str, Any] = {
        "jobs": 0, "turns": 0, "started": None, "finished": None,
        "indexes": set(), "generation_batch_sizes": [], "oom_fallbacks": [],
    }
    speculative_queue: asyncio.Queue[dict[str, Any] | None] | None = asyncio.Queue() if overlap_enabled else None
    speculative_task: asyncio.Task[None] | None = None

    def speculative_digest(
        turn: dict[str, Any], language_code: str, voices_value: dict[str, Any], instructions_value: dict[str, Any],
    ) -> str:
        selection = voices_value[turn["speaker"]]
        return hashlib.sha256(json_dump({
            "contract": 1, "model": selected_model_early,
            "checkpoint_revisions": checkpoint_revisions_early, "device": selected_device_early,
            "language": "English" if language_code == "en" else "Chinese", "speaker": turn["speaker"],
            "voice": selection, "instruction": instructions_value[turn["speaker"]], "text": turn["text"],
        }).encode()).hexdigest()

    async def consume_speculative_acts() -> None:
        assert speculative_queue is not None
        stop_after_batch = False
        sequence_failed = False
        while not stop_after_batch:
            first = await speculative_queue.get()
            if first is None:
                break
            acts = [first]
            while True:
                try:
                    pending = speculative_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if pending is None:
                    stop_after_batch = True
                    break
                acts.append(pending)
            if sequence_failed:
                continue
            language_code = str(acts[0]["language"])
            voices_value = {
                "HOST_A": host_voice_selection(config, "host_a", language_code, preset_override=payload.get("host_a")),
                "HOST_B": host_voice_selection(config, "host_b", language_code, preset_override=payload.get("host_b")),
            }
            instruction_modes = (model_caps_early.get("controls") or {}).get("instruction_voice_modes") or []
            instructions_value = {
                "HOST_A": host_voice_instruction(config, "host_a", language_code, supported="preset" in instruction_modes),
                "HOST_B": host_voice_instruction(config, "host_b", language_code, supported="preset" in instruction_modes),
            }
            indexed = [
                (int(act["start_index"]) + offset, turn)
                for act in acts for offset, turn in enumerate(act["turns"])
            ]
            max_items = max(1, min(100, int(sequence_capability_early.get("max_items") or 100)))
            max_chars = max(1, int(sequence_capability_early.get("max_total_chars") or 100000))
            for mode in ("preset", "voiceprint"):
                candidates = [(index, turn) for index, turn in indexed if voices_value[turn["speaker"]]["mode"] == mode]
                while candidates:
                    batch: list[tuple[int, dict[str, Any]]] = []
                    chars = 0
                    while candidates and len(batch) < max_items:
                        length = len(str(candidates[0][1]["text"]))
                        if batch and chars + length > max_chars:
                            break
                        value = candidates.pop(0)
                        batch.append(value)
                        chars += length
                    items: list[dict[str, Any]] = []
                    outputs: dict[str, Path] = {}
                    digests: dict[int, str] = {}
                    part_hashes = manifest.setdefault("tts_parts", {})
                    for index, turn in batch:
                        digest = speculative_digest(turn, language_code, voices_value, instructions_value)
                        part = parts_dir_early / f"{index:04d}.wav"
                        if part_hashes.get(str(index)) == digest and part.exists() and part.stat().st_size >= 128:
                            continue
                        selection = voices_value[turn["speaker"]]
                        item_id = f"turn-{index:04d}"
                        items.append({
                            "id": item_id, "text": turn["text"], "speaker": selection.get("speaker"),
                            "voiceprint_sample_id": selection.get("sample_id"),
                            "instruct": instructions_value[turn["speaker"]],
                        })
                        outputs[item_id] = part
                        digests[index] = digest
                    if not items:
                        continue
                    if speculative_stats["started"] is None:
                        speculative_stats["started"] = time.perf_counter()
                    batch_key = hashlib.sha256("".join(digests.values()).encode()).hexdigest()[:24]
                    try:
                        execution: dict[str, Any] = {}
                        await synthesize_sequence(
                            items, outputs, language="English" if language_code == "en" else "Chinese",
                            cancel_check=cancel_check_early, model=selected_model_early,
                            compute_device=selected_device_early, voice_mode=mode,
                            idempotency_key=f"sread-spec-{suffix[:18]}-{batch_key}", provider=provider,
                            execution=execution,
                        )
                    except Exception as exc:
                        speculative_stats["failure"] = str(exc)[:300]
                        sequence_failed = True
                        break
                    speculative_stats["jobs"] += 1
                    speculative_stats["turns"] += len(items)
                    speculative_stats["indexes"].update(digests)
                    speculative_stats["generation_batch_sizes"].append(
                        int(execution.get("generation_batch_size") or 1)
                    )
                    speculative_stats["oom_fallbacks"].extend(execution.get("oom_fallbacks") or [])
                    speculative_stats["finished"] = time.perf_counter()
                    for index, digest in digests.items():
                        part_hashes[str(index)] = digest
                    save_manifest(manifest)

    def on_act_ready(value: dict[str, Any]) -> None:
        if speculative_queue is not None:
            speculative_queue.put_nowait(value)

    if speculative_queue is not None:
        speculative_task = asyncio.create_task(consume_speculative_acts())
    script_started = time.perf_counter()
    if manifest.get("signature") == signature and manifest.get("generated"):
        generated = manifest["generated"]
        script_finished = time.perf_counter()
    else:
        def report(stage: str, progress: float) -> None:
            Reporter(job_id).update("script", stage, progress, current=progress, total=1, unit="阶段")

        try:
            generated = await build_podcast_script(
                notebook_id, payload, progress=report, act_ready=on_act_ready if overlap_enabled else None,
            )
            script_finished = time.perf_counter()
        except PodcastQualityError as exc:
            if speculative_task:
                speculative_task.cancel()
                await asyncio.gather(speculative_task, return_exceptions=True)
            save_manifest({"version": PODCAST_ENGINE_VERSION, "signature": signature, "quality_failure": exc.report})
            raise RuntimeError(f"播客脚本未通过质量门槛：{exc}") from exc
        except Exception:
            if speculative_task:
                speculative_task.cancel()
                await asyncio.gather(speculative_task, return_exceptions=True)
            raise
        if speculative_queue is not None:
            speculative_queue.put_nowait(None)
        if speculative_task:
            await speculative_task
        manifest = {
            "version": PODCAST_ENGINE_VERSION, "signature": signature, "generated": generated,
            "tts_parts": manifest.get("tts_parts") or {},
        }
        save_manifest(manifest)
    script_seconds = script_finished - script_started

    language = generated["language"]
    voices = {
        "HOST_A": host_voice_selection(config, "host_a", language, preset_override=payload.get("host_a")),
        "HOST_B": host_voice_selection(config, "host_b", language, preset_override=payload.get("host_b")),
    }
    selected_model = provider.get("model") or ""
    selected_device = config.get("compute_device", "gpu")
    model_caps = next((item for item in provider.get("capabilities", {}).get("models", []) if item.get("id") == selected_model), {})
    instruction_modes = (model_caps.get("controls") or {}).get("instruction_voice_modes") or []
    instructions = {
        "HOST_A": host_voice_instruction(config, "host_a", language, supported="preset" in instruction_modes),
        "HOST_B": host_voice_instruction(config, "host_b", language, supported="preset" in instruction_modes),
    }
    parts_dir = work_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    cancel_check = lambda: bool((DB.fetchone("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)) or {}).get("cancel_requested"))
    turns = generated["turns"]
    sequence_capability = (provider.get("capabilities") or {}).get("sequence_jobs") or {}
    sequence_enabled = bool(
        config.get("podcast_sequence_tts", True)
        and sequence_capability.get("supported")
        and int(sequence_capability.get("contract_version") or 0) == 1
    )
    max_sequence_items = max(1, min(100, int(sequence_capability.get("max_items") or 100)))
    max_sequence_chars = max(1, int(sequence_capability.get("max_total_chars") or 100000))
    part_hashes = manifest.setdefault("tts_parts", {})
    tts_execution: dict[str, Any] = {
        "requested_device": selected_device, "compute_device": selected_device, "fallback_used": False,
        "sequence_supported": bool(sequence_capability.get("supported")), "sequence_enabled": sequence_enabled,
        "sequence_jobs": 0, "single_jobs": 0, "batch_sizes": [], "reused_turns": 0,
        "generation_batch_sizes": [], "oom_fallbacks": [],
        "speculative_jobs": speculative_stats["jobs"], "speculative_turns": speculative_stats["turns"],
        "speculative_generation_batch_sizes": speculative_stats["generation_batch_sizes"],
        "speculative_oom_fallbacks": speculative_stats["oom_fallbacks"],
    }

    checkpoint_revisions = {
        str(item.get("variant") or ""): str(item.get("revision") or "")
        for item in model_caps.get("checkpoints") or [] if isinstance(item, dict)
    }

    def turn_digest(index: int) -> str:
        turn = turns[index]
        selection = voices[turn["speaker"]]
        return hashlib.sha256(json_dump({
            "contract": 1,
            "model": selected_model,
            "checkpoint_revisions": checkpoint_revisions,
            "device": selected_device,
            "language": "English" if language == "en" else "Chinese",
            "speaker": turn["speaker"],
            "voice": selection,
            "instruction": instructions[turn["speaker"]],
            "text": turn["text"],
        }).encode()).hexdigest()

    def reusable_part(index: int, digest: str) -> bool:
        part = parts_dir / f"{index:04d}.wav"
        return part_hashes.get(str(index)) == digest and part.exists() and part.stat().st_size >= 128

    async def synthesize_turn(index: int, *, retry: bool = False) -> Path:
        turn = turns[index]
        if cancel_check():
            raise RuntimeError("任务已取消")
        part = parts_dir / f"{index:04d}.wav"
        digest = turn_digest(index)
        if retry or not reusable_part(index, digest):
            Reporter(job_id).update("tts", f"高质量语音合成 {index + 1}/{len(turns)}", 0.40 + 0.50 * index / max(1, len(turns)), current=index + 1, total=len(turns), unit="段")
            selection = voices[turn["speaker"]]
            instruction = instructions[turn["speaker"]]
            execution: dict[str, Any] = {}
            await synthesize(
                turn["text"],
                selection.get("speaker"),
                part,
                language="English" if language == "en" else "Chinese",
                cancel_check=cancel_check,
                model=selected_model,
                compute_device=selected_device,
                voice_mode=selection["mode"],
                voiceprint_sample_id=selection.get("sample_id"),
                instruct=instruction,
                idempotency_key=f"sread-{suffix[:18]}-{index:04d}-{digest[:20]}{'-r1' if retry else ''}",
                execution=execution,
                provider=provider,
            )
            tts_execution["single_jobs"] += 1
            if execution.get("fallback_used"):
                tts_execution.update(execution)
            part_hashes[str(index)] = digest
            save_manifest(manifest)
        else:
            tts_execution["reused_turns"] += 1
        return part

    def sequence_batches(indexes: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        for mode in ("preset", "voiceprint"):
            pending = [index for index in indexes if voices[turns[index]["speaker"]]["mode"] == mode]
            current: list[int] = []
            chars = 0
            for index in pending:
                length = len(str(turns[index]["text"]))
                if current and (len(current) >= max_sequence_items or chars + length > max_sequence_chars):
                    batches.append(current)
                    current, chars = [], 0
                current.append(index)
                chars += length
            if current:
                batches.append(current)
        return batches

    async def synthesize_indexes(indexes: list[int], *, retry: bool = False) -> None:
        missing: list[int] = []
        for index in indexes:
            if retry or not reusable_part(index, turn_digest(index)):
                missing.append(index)
            else:
                tts_execution["reused_turns"] += 1
        if not missing:
            return
        if not sequence_enabled:
            for index in missing:
                await synthesize_turn(index, retry=retry)
            return
        for batch in sequence_batches(missing):
            mode = voices[turns[batch[0]]["speaker"]]["mode"]
            items = []
            outputs: dict[str, Path] = {}
            for index in batch:
                selection = voices[turns[index]["speaker"]]
                item_id = f"turn-{index:04d}"
                items.append({
                    "id": item_id, "text": turns[index]["text"], "speaker": selection.get("speaker"),
                    "voiceprint_sample_id": selection.get("sample_id"), "instruct": instructions[turns[index]["speaker"]],
                })
                outputs[item_id] = parts_dir / f"{index:04d}.wav"
            Reporter(job_id).update(
                "tts", f"批量高质量语音合成 {batch[0] + 1}–{batch[-1] + 1}/{len(turns)}",
                0.40 + 0.50 * batch[0] / max(1, len(turns)), current=batch[-1] + 1, total=len(turns), unit="段",
            )
            execution: dict[str, Any] = {}
            batch_digest = hashlib.sha256("".join(turn_digest(index) for index in batch).encode()).hexdigest()[:24]
            try:
                await synthesize_sequence(
                    items, outputs, language="English" if language == "en" else "Chinese",
                    cancel_check=cancel_check, model=selected_model, compute_device=selected_device,
                    voice_mode=mode, idempotency_key=f"sread-seq-{suffix[:18]}-{batch_digest}{'-r1' if retry else ''}",
                    execution=execution, provider=provider,
                )
                tts_execution["sequence_jobs"] += 1
                tts_execution["batch_sizes"].append(len(batch))
                tts_execution["generation_batch_sizes"].append(
                    int(execution.get("generation_batch_size") or 1)
                )
                tts_execution["oom_fallbacks"].extend(execution.get("oom_fallbacks") or [])
                if execution.get("fallback_used"):
                    tts_execution.update(execution)
                for index in batch:
                    part_hashes[str(index)] = turn_digest(index)
                save_manifest(manifest)
            except ProviderError as exc:
                if exc.code == "cancelled" or cancel_check():
                    raise RuntimeError("任务已取消") from exc
                tts_execution["sequence_fallback_reason"] = str(exc)[:300]
                for index in batch:
                    await synthesize_turn(index, retry=retry)

    speculative_indexes = set(speculative_stats.get("indexes") or set())
    speculative_reused_turns = sum(
        reusable_part(index, turn_digest(index))
        for index in speculative_indexes if 0 <= index < len(turns)
    )
    tts_execution["speculative_reused_turns"] = speculative_reused_turns
    tts_execution["speculative_reuse_ratio"] = round(
        speculative_reused_turns / max(1, len(speculative_indexes)), 3,
    ) if speculative_indexes else 0.0

    tts_started = time.perf_counter()
    await synthesize_indexes(list(range(len(turns))))
    tts_seconds = time.perf_counter() - tts_started
    parts.extend(parts_dir / f"{index:04d}.wav" for index in range(len(turns)))
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = CONFIG.tools.ffmpeg_path

    async def render_audio(force_indexes: set[int] | None = None) -> tuple[Path, float]:
        force_indexes = force_indexes or set()
        destination = out_dir / "podcast.m4a"
        if not ffmpeg:
            destination = out_dir / "podcast.wav"
            chapter_ends = {int(chapter["turn_end"]) for chapter in generated["chapters"][:-1]}
            cursor = 0.0
            params = None
            rendered: list[bytes] = []
            for index, part in enumerate(parts):
                with wave.open(str(part), "rb") as audio:
                    if params is None:
                        params = audio.getparams()
                    elif audio.getparams()[:3] != params[:3]:
                        raise RuntimeError("TTS 输出音频参数不一致，请安装项目内 FFmpeg")
                    frames = audio.readframes(audio.getnframes())
                    duration = audio.getnframes() / max(1, audio.getframerate())
                pause = 0.60 if index in chapter_ends else 0.22
                silence = b"\x00" * int(params.framerate * params.sampwidth * params.nchannels * pause)
                turns[index]["start_seconds"] = round(cursor, 3)
                cursor += duration + pause
                turns[index]["end_seconds"] = round(cursor, 3)
                rendered.append(frames)
                rendered.append(silence)
            with wave.open(str(destination), "wb") as output:
                output.setparams(params)
                for chunk in rendered:
                    output.writeframes(chunk)
            for chapter in generated["chapters"]:
                chapter_turns = turns[chapter["turn_start"] : chapter["turn_end"] + 1]
                if chapter_turns:
                    chapter["start_seconds"] = chapter_turns[0]["start_seconds"]
                    chapter["end_seconds"] = chapter_turns[-1]["end_seconds"]
            return destination, cursor
        normalized_dir = work_dir / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        normalized: list[Path] = []
        chapter_ends = {int(chapter["turn_end"]) for chapter in generated["chapters"][:-1]}
        for index, part in enumerate(parts):
            target = normalized_dir / f"{index:03d}.wav"
            if index in force_indexes or not target.exists() or target.stat().st_size < 128:
                pause = 0.60 if index in chapter_ends else 0.22
                returncode, stderr = await _run_ffmpeg([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(part), "-af", f"apad=pad_dur={pause}", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(target)], 180, cancel_check)
                if returncode != 0:
                    raise RuntimeError(f"FFmpeg 音频标准化失败: {stderr[-1000:]}")
            normalized.append(target)
        cursor = 0.0
        for turn, part in zip(turns, normalized):
            with wave.open(str(part), "rb") as audio:
                duration = audio.getnframes() / max(1, audio.getframerate())
            turn["start_seconds"] = round(cursor, 3)
            cursor += duration
            turn["end_seconds"] = round(cursor, 3)
        for chapter in generated["chapters"]:
            chapter_turns = turns[chapter["turn_start"] : chapter["turn_end"] + 1]
            if chapter_turns:
                chapter["start_seconds"] = chapter_turns[0]["start_seconds"]
                chapter["end_seconds"] = chapter_turns[-1]["end_seconds"]
        concat_file = normalized_dir / "concat.txt"
        concat_file.write_text("".join(f"file '{part.as_posix()}'\n" for part in normalized), encoding="utf-8")
        returncode, stderr = await _run_ffmpeg([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "24000", "-ac", "1", "-c:a", "aac", "-b:a", "128k", str(destination)], 600, cancel_check)
        if returncode != 0:
            raise RuntimeError(f"FFmpeg 音频合并失败: {stderr[-1000:]}")
        return destination, cursor

    destination, cursor = await render_audio()
    duration_check = _actual_duration_check(float(generated["duration"]["target_minutes"]), cursor)
    if not duration_check["passed"]:
        failure = {"stage": "duration", **duration_check}
        manifest.update({"generated": generated, "media_path": str(destination.relative_to(PATHS.root)), "audio_quality_failure": failure})
        save_manifest(manifest)
        raise RuntimeError("成品音频实际时长未通过质量门槛")
    Reporter(job_id).update("asr", "本地 ASR 验收成品音频", 0.92, current=0, total=1, unit="次")
    try:
        asr_result = await transcribe_audio(
            destination,
            language="English" if language == "en" else "Chinese",
            cancel_check=cancel_check,
            idempotency_key=f"sread-asr-{suffix[:36]}",
        )
    except ProviderError as exc:
        manifest.update({"generated": generated, "media_path": str(destination.relative_to(PATHS.root)), "audio_quality_failure": {"stage": "asr", "error": str(exc), "code": exc.code}})
        save_manifest(manifest)
        raise RuntimeError(f"成品音频 ASR 验收失败：{exc}") from exc
    audio_quality = assess_transcription(turns, asr_result, language)
    retry_indexes = repair_turn_indexes(audio_quality)
    if retry_indexes and len(retry_indexes) <= 6:
        repair_started = time.perf_counter()
        await synthesize_indexes(retry_indexes, retry=True)
        tts_seconds += time.perf_counter() - repair_started
        destination, cursor = await render_audio(set(retry_indexes))
        duration_check = _actual_duration_check(float(generated["duration"]["target_minutes"]), cursor)
        if not duration_check["passed"]:
            failure = {"stage": "duration_after_audio_repair", **duration_check}
            manifest.update({"generated": generated, "media_path": str(destination.relative_to(PATHS.root)), "audio_quality_failure": failure})
            save_manifest(manifest)
            raise RuntimeError("修复后的成品音频实际时长未通过质量门槛")
        asr_result = await transcribe_audio(
            destination,
            language="English" if language == "en" else "Chinese",
            cancel_check=cancel_check,
            idempotency_key=f"sread-asr-{suffix[:32]}-verify",
        )
        audio_quality = assess_transcription(turns, asr_result, language)
        audio_quality["repaired_turns"] = retry_indexes
    audio_quality["duration"] = duration_check
    if not audio_quality.get("passed"):
        manifest.update({"generated": generated, "media_path": str(destination.relative_to(PATHS.root)), "audio_quality_failure": audio_quality})
        save_manifest(manifest)
        raise RuntimeError("成品音频未通过本地 ASR 质量门槛")
    generated["duration"]["actual_seconds"] = round(cursor, 3)
    generated["voices"] = {
        speaker: {"mode": selection["mode"], "label": selection["label"]}
        for speaker, selection in voices.items()
    }
    generated["provider"] = {
        "name": provider.get("name"),
        "kind": provider.get("kind"),
        "model": selected_model,
        **tts_execution,
        "asr": {
            "model": audio_quality.get("asr_model"),
            "compute_device": audio_quality.get("compute_device"),
            "fallback_used": audio_quality.get("device_fallback"),
            "fallback_reason": audio_quality.get("fallback_reason"),
        },
    }
    generated["audio_quality"] = audio_quality
    speculative_started = speculative_stats.get("started")
    speculative_finished = speculative_stats.get("finished")
    overlap_seconds = (
        max(0.0, min(script_finished, speculative_finished) - max(script_started, speculative_started))
        if speculative_started is not None and speculative_finished is not None else 0.0
    )
    speculative_seconds = (
        max(0.0, speculative_finished - speculative_started)
        if speculative_started is not None and speculative_finished is not None else 0.0
    )
    total_seconds = time.perf_counter() - podcast_started
    serial_estimate_seconds = total_seconds + overlap_seconds
    generated["performance"] = {
        "script_seconds": round(script_seconds, 3),
        "tts_seconds": round(tts_seconds + speculative_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "overlap_enabled": overlap_enabled,
        "overlap_seconds": round(overlap_seconds, 3),
        "serial_estimate_seconds": round(serial_estimate_seconds, 3),
        "overlap_gain_ratio": round(overlap_seconds / max(serial_estimate_seconds, 0.001), 4),
    }
    generated["quality"]["actual_minutes"] = round(cursor / 60, 2)
    generated["quality"]["actual_duration_ratio"] = duration_check["duration_ratio"]
    manifest["generated"] = generated
    manifest["media_path"] = str(destination.relative_to(PATHS.root))
    save_manifest(manifest)
    shutil.copy2(manifest_path, out_dir / "manifest.json")
    artifact_id, now = f"artifact_{suffix}", utc_now()
    DB.execute("INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (artifact_id, notebook_id, "podcast", "双人音频解读", json_dump(generated["source_ids"]), language, "ready", json_dump(generated), json_dump(generated["citations"]), str(destination.relative_to(PATHS.root)), now, now))
    register_resource("notebook", notebook_id, notebook_id, "podcast", out_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    DB.execute("UPDATE local_resources SET state='transferred',transferred_at=? WHERE owner_type='job' AND owner_id=?", (utc_now(), job_id))
    return {"id": artifact_id, "media_url": f"/api/artifacts/{artifact_id}/media", "context_usage": generated.get("context_usage", {})}


async def execute(job: dict[str, Any]) -> Any:
    payload = json_load(job["payload_json"], {})
    if job["kind"] == "ingest":
        reporter = Reporter(job["id"])
        def ingest_progress(stage: str, value: float) -> None:
            match = re.search(r"(\d+)/(\d+)", stage)
            code = "vision" if stage.startswith("视觉") else "embedding" if stage.startswith("生成本地向量") else "parse"
            reporter.update(code, stage, value, current=float(match.group(1)) if match else None, total=float(match.group(2)) if match else None, unit="页" if code == "vision" else "批" if code == "embedding" else None)
        return await ingest_source(
            payload["source_id"],
            progress=ingest_progress,
            cancel_check=lambda: bool((DB.fetchone("SELECT cancel_requested FROM jobs WHERE id=?", (job["id"],)) or {}).get("cancel_requested")),
            image_policy=payload.get("image_policy"),
            image_provider_ids=payload.get("image_provider_ids"),
        )
    if job["kind"] == "summary": return await make_summary(job["notebook_id"], payload.get("source_ids"), payload.get("language", "auto"), job["id"])
    if job["kind"] in {"quiz", "flashcard"}:
        return await generate_study_artifact(
            job["notebook_id"],
            job["kind"],
            int(payload.get("count", 10)),
            payload.get("source_ids"),
            payload.get("language", "auto"),
            payload.get("difficulty", "mixed"),
            payload.get("custom_prompt", ""),
            job["id"],
        )
    if job["kind"] == "podcast": return await _podcast(job["notebook_id"], payload, job["id"])
    raise RuntimeError(f"未知任务类型: {job['kind']}")


class JobWorker:
    task: asyncio.Task | None = None
    stopped = False

    async def run(self) -> None:
        while not self.stopped:
            job = DB.fetchone("SELECT * FROM jobs WHERE state='queued' ORDER BY created_at LIMIT 1")
            if not job:
                process_cleanup_operations()
                await asyncio.sleep(0.4); continue
            if job["cancel_requested"]:
                _update(job["id"], state="cancelled", stage="已取消", progress=1); continue
            _update(job["id"], state="running", stage="准备执行", progress=0.01)
            try:
                result = await execute(job)
                latest = DB.fetchone("SELECT cancel_requested FROM jobs WHERE id=?", (job["id"],)) or {}
                if latest.get("cancel_requested"):
                    _update(job["id"], state="cancelled", stage="已取消", progress=1)
                else:
                    _update(job["id"], state="complete", stage="完成", progress=1, result=result)
            except Exception as exc:
                cancelled = bool((DB.fetchone("SELECT cancel_requested FROM jobs WHERE id=?", (job["id"],)) or {}).get("cancel_requested"))
                _update(job["id"], state="cancelled" if cancelled else "failed", stage="已取消" if cancelled else "失败", progress=1 if cancelled else None, error=str(exc))
                if job["kind"] == "ingest":
                    payload = json_load(job["payload_json"], {})
                    DB.execute("UPDATE sources SET state='failed',error=?,updated_at=? WHERE id=?", (str(exc), utc_now(), payload.get("source_id")))
            finally:
                process_cleanup_operations()

    def start(self) -> None:
        self.stopped = False
        self.task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self.stopped = True
        if self.task:
            await self.task


WORKER = JobWorker()
