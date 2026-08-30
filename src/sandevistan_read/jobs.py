from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import wave
from pathlib import Path
from typing import Any, Awaitable, Callable

from .database import DB, json_dump, json_load, new_id, utc_now
from .config import CONFIG
from .paths import PATHS
from .providers import active_provider, synthesize
from .podcast import build_podcast_script
from .services import ingest_source, make_structured, make_summary
from .observability import LABELS, Reporter
from .cleanup import process_cleanup_operations, register_resource


def enqueue(kind: str, notebook_id: str | None, payload: dict[str, Any], parent_id: str | None = None) -> dict[str, Any]:
    job_id, now = new_id("job"), utc_now()
    source_count = len(payload.get("source_ids") or []) or (1 if payload.get("source_id") else 0)
    workload = {"source_count": source_count, "count": payload.get("count"), "minutes": payload.get("minutes"), "bucket": "single" if source_count <= 1 else "small_multi" if source_count <= 4 else "large_multi"}
    provider = active_provider("tts" if kind == "podcast" else "main") if kind != "ingest" else None
    profile = {"kind": provider.get("kind") if provider else "local", "model": provider.get("model") if provider else CONFIG.models.embedding, "device": (provider.get("config") or {}).get("compute_device") if provider else "local"}
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


async def _podcast(notebook_id: str, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
    provider = active_provider("tts")
    if not provider:
        raise RuntimeError("请先配置并启用 TTS provider")
    config = provider.get("config", {})
    suffix = job_id.removeprefix("job_")
    work_dir = PATHS.job_work / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    register_resource("job", job_id, notebook_id, "podcast-work", work_dir)
    out_dir = PATHS.artifacts / f"podcast_{suffix}"
    manifest_path = work_dir / "manifest.json"
    signature = hashlib.sha256(json_dump({"payload": payload, "model": provider.get("model"), "device": config.get("compute_device")}).encode()).hexdigest()

    def save_manifest(value: dict[str, Any]) -> None:
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json_dump(value), encoding="utf-8")
        temporary.replace(manifest_path)

    manifest = json_load(manifest_path.read_text(encoding="utf-8"), {}) if manifest_path.exists() else {}
    if manifest.get("signature") == signature and manifest.get("generated"):
        generated = manifest["generated"]
    else:
        def report(stage: str, progress: float) -> None:
            Reporter(job_id).update("script", stage, progress, current=progress, total=1, unit="阶段")

        generated = await build_podcast_script(notebook_id, payload, progress=report)
        manifest = {"version": 2, "signature": signature, "generated": generated}
        save_manifest(manifest)

    language = generated["language"]
    defaults = ("Ryan", "Aiden") if language == "en" else ("Vivian", "Dylan")
    voices = {
        "HOST_A": payload.get("host_a") or config.get("host_a_en" if language == "en" else "host_a", defaults[0]),
        "HOST_B": payload.get("host_b") or config.get("host_b_en" if language == "en" else "host_b", defaults[1]),
    }
    selected_model = provider.get("model") or ""
    selected_device = config.get("compute_device", "gpu")
    model_caps = next((item for item in provider.get("capabilities", {}).get("models", []) if item.get("id") == selected_model), {})
    instruction_modes = (model_caps.get("controls") or {}).get("instruction_voice_modes") or []
    instructions = {
        "HOST_A": config.get("host_a_instruct") if "preset" in instruction_modes else None,
        "HOST_B": config.get("host_b_instruct") if "preset" in instruction_modes else None,
    }
    parts_dir = work_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    cancel_check = lambda: bool((DB.fetchone("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)) or {}).get("cancel_requested"))
    turns = generated["turns"]
    for index, turn in enumerate(turns):
        if cancel_check():
            raise RuntimeError("任务已取消")
        part = parts_dir / f"{index:04d}.wav"
        if not part.exists() or part.stat().st_size < 128:
            Reporter(job_id).update("tts", f"高质量语音合成 {index + 1}/{len(turns)}", 0.40 + 0.50 * index / max(1, len(turns)), current=index + 1, total=len(turns), unit="段")
            digest = hashlib.sha256(f"{selected_model}|{selected_device}|{voices[turn['speaker']]}|{turn['text']}".encode()).hexdigest()[:20]
            await synthesize(
                turn["text"],
                voices[turn["speaker"]],
                part,
                language="English" if language == "en" else "Chinese",
                cancel_check=cancel_check,
                model=selected_model,
                compute_device=selected_device,
                instruct=instructions[turn["speaker"]],
                idempotency_key=f"sread-{suffix[:24]}-{index:04d}-{digest}",
            )
        parts.append(part)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "podcast.m4a"
    ffmpeg = CONFIG.tools.ffmpeg_path
    if ffmpeg:
        normalized_dir = work_dir / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        normalized: list[Path] = []
        for index, part in enumerate(parts):
            target = normalized_dir / f"{index:03d}.wav"
            if not target.exists() or target.stat().st_size < 128:
                returncode, stderr = await _run_ffmpeg([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(part), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(target)], 180, cancel_check)
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
    else:
        destination = out_dir / "podcast.wav"
        with wave.open(str(parts[0]), "rb") as first:
            params, frames = first.getparams(), [first.readframes(first.getnframes())]
        for part in parts[1:]:
            with wave.open(str(part), "rb") as audio:
                if audio.getparams()[:3] != params[:3]:
                    raise RuntimeError("TTS 输出音频参数不一致，请安装项目内 FFmpeg")
                frames.append(audio.readframes(audio.getnframes()))
        with wave.open(str(destination), "wb") as output:
            output.setparams(params)
            for frame in frames: output.writeframes(frame)
        with wave.open(str(destination), "rb") as audio:
            cursor = audio.getnframes() / max(1, audio.getframerate())
    generated["duration"]["actual_seconds"] = round(cursor, 3)
    generated["voices"] = voices
    generated["provider"] = {"name": provider.get("name"), "model": selected_model, "compute_device": selected_device}
    generated["quality"]["actual_minutes"] = round(cursor / 60, 2)
    manifest["generated"] = generated
    manifest["media_path"] = str(destination.relative_to(PATHS.root))
    save_manifest(manifest)
    shutil.copy2(manifest_path, out_dir / "manifest.json")
    artifact_id, now = f"artifact_{suffix}", utc_now()
    DB.execute("INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (artifact_id, notebook_id, "podcast", "双人音频解读", json_dump(generated["source_ids"]), language, "ready", json_dump(generated), json_dump(generated["citations"]), str(destination.relative_to(PATHS.root)), now, now))
    register_resource("notebook", notebook_id, notebook_id, "podcast", out_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    DB.execute("UPDATE local_resources SET state='transferred',transferred_at=? WHERE owner_type='job' AND owner_id=?", (utc_now(), job_id))
    return {"id": artifact_id, "media_url": f"/api/artifacts/{artifact_id}/media"}


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
        )
    if job["kind"] == "summary": return await make_summary(job["notebook_id"], payload.get("source_ids"), payload.get("language", "auto"), job["id"])
    if job["kind"] in {"quiz", "flashcard"}: return await make_structured(job["notebook_id"], job["kind"], int(payload.get("count", 10)), payload.get("source_ids"), payload.get("language", "auto"), payload.get("difficulty", "mixed"), job["id"])
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
