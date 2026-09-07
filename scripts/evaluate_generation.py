#!/usr/bin/env python3
"""Explicit live integration evaluation in an isolated, frozen application instance."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import time

import httpx


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("zh-CN", "en", "auto")


def cases(phase: str) -> list[dict]:
    result: list[dict] = []

    def add(kind: str, suffix: str, payload: dict, config: dict | None = None) -> None:
        result.append({"id": f"{kind}-{suffix}", "kind": kind, "payload": payload, "config": config or {}})

    for language in LANGUAGES[:1] if phase == "baseline" else LANGUAGES:
        add("summary", language, {"language": language})
        add("chat", language, {"language": language})
        for kind, default, maximum in (("quiz", 10, 30), ("flashcards", 20, 50)):
            for count in (default,) if phase == "baseline" else (1, default, maximum):
                for difficulty in ("mixed",) if phase == "baseline" else ("easy", "medium", "hard", "mixed"):
                    add(kind, f"{language}-{count}-{difficulty}", {"language": language, "count": count, "difficulty": difficulty})
        for duration in (5,) if phase == "baseline" else ("auto", 5, 10, 20, 30):
            add("podcasts", f"{language}-{duration}", {"language": language, "duration_mode": "auto" if duration == "auto" else "fixed", **({} if duration == "auto" else {"minutes": duration})})
    if phase != "extras":
        return result
    result = []
    for key, values in (("max_output_tokens", (1024, 1536, 8192)), ("temperature", (0, 1, 2))):
        for value in values:
            for case in cases("baseline"):
                add(case["kind"], f"{key}-{value}", case["payload"], {key: value})
    for tier in ("lite", "full"):
        for kind, count in (("quiz", 10), ("flashcards", 20)):
            for difficulty in ("mixed", "hard"):
                add(kind, f"{tier}-{difficulty}", {"count": count, "difficulty": difficulty, "language": "zh-CN"}, {"study_generation_tier": tier})
    for kind in ("quiz", "flashcards", "podcasts"):
        for length in ("focus", "limit"):
            instruction = "重点解释工作量证明如何防止双重支付。"
            if length == "limit":
                instruction = (instruction * 100)[:1000]
            payload = {"language": "zh-CN", "focus" if kind == "podcasts" else "custom_prompt": instruction}
            payload.update({"minutes": 5} if kind == "podcasts" else {"count": 10 if kind == "quiz" else 20})
            add(kind, length, payload)
    add("podcasts", "single", {"language": "zh-CN", "minutes": 5}, {"_sequence": False})
    return result


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def serve(args: argparse.Namespace) -> None:
    from sandevistan_read.database import DB, json_dump
    from sandevistan_read import providers
    import uvicorn

    DB.initialize()
    DB.seed(args.main_url, args.model, args.audio_url)
    DB.execute("UPDATE provider_profiles SET active=0,selected=0 WHERE role='vlm'")
    DB.execute("UPDATE provider_role_settings SET enabled=0 WHERE role='vlm'")
    DB.execute("UPDATE provider_profiles SET config_json=? WHERE role='main'", (json_dump({"context_window_tokens": 30720}),))
    DB.execute("UPDATE provider_profiles SET model='qwen3-tts-0.6b',config_json=? WHERE role='audio'", (json_dump({"auto_select": False, "compute_device": "gpu", "allow_device_fallback": True, "host_a": "Vivian", "host_b": "Dylan", "asr_model": "qwen3-asr-0.6b", "asr_auto_select": False, "asr_compute_device": "gpu", "asr_allow_device_fallback": True, "podcast_sequence_tts": True}),))
    original = providers._chat_once

    async def recorded(provider, messages, **kwargs):
        if provider["base_url"].rstrip("/") != args.main_url.rstrip("/") or provider["model"] != args.model:
            raise RuntimeError("Evaluation attempted to use a different MAIN provider")
        event = {"at": time.time(), "model": provider["model"], "messages": messages, "options": kwargs}
        lock = sqlite3.connect(ROOT / "runtime/evals/.generation-main-lock.sqlite", timeout=0)
        try:
            while True:
                try:
                    lock.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc):
                        raise
                    await asyncio.sleep(0.2)
            completion = await original(provider, messages, **kwargs)
            event["response"] = vars(completion)
            return completion
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            lock.close()
            await asyncio.sleep(0.25)
            with (args.output / "main-calls.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    providers._chat_once = recorded
    uvicorn.run("sandevistan_read.app:app", host="127.0.0.1", port=args.port, log_level="warning")


def evaluate(args: argparse.Namespace) -> int:
    # Refuse occupied ports before any API mutation can reach another instance.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    selected = [case for case in cases(args.phase) if not args.case or case["id"] in args.case]
    if args.case and set(args.case) - {case["id"] for case in selected}:
        raise ValueError("Unknown --case for this phase")
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    instance = output / "instance"
    identity = {"main_url": args.main_url, "model": args.model, "audio_url": args.audio_url, "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(), "phase": args.phase}
    if args.case:
        identity["cases"] = sorted(set(args.case))
    manifest = output / "manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != identity:
            raise ValueError("Existing run requires --resume and identical inputs")
    else:
        if args.resume:
            raise ValueError("Cannot resume a run without a manifest")
        instance.mkdir()
        shutil.copytree(ROOT / "src", instance / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (instance / "runtime").mkdir()
        (instance / "runtime/models").symlink_to(ROOT / "runtime/models", target_is_directory=True)
        (instance / ".tools").symlink_to(ROOT / ".tools", target_is_directory=True)
        (instance / "frontend").mkdir()
        (instance / "frontend/dist").symlink_to(ROOT / "frontend/dist", target_is_directory=True)
        (instance / "runtime/config.toml").write_text(f'[server]\nport = {args.port}\n[development]\nollama_url = {json.dumps(args.main_url)}\nollama_model = {json.dumps(args.model)}\naudio_url = {json.dumps(args.audio_url)}\n', encoding="utf-8")
        save(manifest, identity)
        save(output / "cases.json", selected)
        shutil.copy2(Path(__file__), output / "evaluator.py")
        save(output / "source-hashes.json", {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "src").rglob("*.py"))})
    environment = {**os.environ, "SANDEVISTAN_PROJECT_ROOT": str(instance), "PYTHONPATH": str(instance / "src"), "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
    command = [sys.executable, str(Path(__file__).resolve()), "--serve", "--output", str(output), "--port", str(args.port), "--main-url", args.main_url, "--model", args.model, "--audio-url", args.audio_url]
    results_path = output / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    with (output / "server.log").open("a") as log, httpx.Client(base_url=f"http://127.0.0.1:{args.port}/api", timeout=1200) as client:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError("Isolated server exited; see server.log")
                try:
                    response = client.get("/notebooks", timeout=2)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Isolated server did not start")

            def request(method: str, path: str, **kwargs):
                response = client.request(method, path, **kwargs)
                response.raise_for_status()
                return response.json()

            def wait_job(identifier: str) -> dict:
                deadline = time.monotonic() + args.job_timeout
                while time.monotonic() < deadline:
                    job = request("GET", f"/jobs/{identifier}")
                    if job["state"] in {"complete", "failed", "cancelled"}:
                        return job
                    time.sleep(2)
                request("POST", f"/jobs/{identifier}/cancel")
                raise TimeoutError(f"Job {identifier} exceeded evaluation deadline")

            notebook = request("GET", "/notebooks")[0]
            notebook_id = notebook["id"]
            notebook = request("GET", f"/notebooks/{notebook_id}")
            if not notebook["sources"]:
                with args.sample.open("rb") as handle:
                    uploaded = request("POST", f"/notebooks/{notebook_id}/sources", files={"files": (args.sample.name, handle, "application/pdf")}, data={"image_policy": '{"mode":"off","processors":[]}'})
                save(output / "ingest.json", uploaded)
            # Includes resumed ingestion jobs.
            for _ in range(300):
                notebook = request("GET", f"/notebooks/{notebook_id}")
                if any(s["state"] == "failed" for s in notebook["sources"]):
                    raise RuntimeError("Sample ingestion failed")
                if notebook["sources"] and all(s["state"] == "ready" for s in notebook["sources"]):
                    break
                time.sleep(2)
            else:
                raise TimeoutError("Sample ingestion did not finish")
            profiles = request("GET", "/providers")
            main = next(p for p in profiles if p["role"] == "main")
            audio = next(p for p in profiles if p["role"] == "audio")
            save(output / "providers.json", profiles)
            for case in selected:
                if any(item["id"] == case["id"] for item in results):
                    continue
                start = time.monotonic()
                result = {**case, "started_at": time.time()}
                print(f"START {case['id']}", flush=True)
                audio_lock = None
                try:
                    if case["kind"] == "podcasts":
                        audio_lock = sqlite3.connect(ROOT / "runtime/evals/.generation-audio-lock.sqlite", timeout=args.job_timeout)
                        audio_lock.execute("BEGIN IMMEDIATE")
                    config = {k: v for k, v in case["config"].items() if not k.startswith("_")}
                    request("PATCH", f"/providers/{main['id']}", json={"config": {"context_window_tokens": 30720, **config}})
                    request("PATCH", f"/providers/{audio['id']}", json={"config": {**audio["config"], "auto_select": False, "podcast_sequence_tts": case["config"].get("_sequence", True)}})
                    if case["kind"] == "chat":
                        questions = ["What problem does proof of work solve in this paper?", "Why is it necessary?", "Compare these approaches in this order: 1. proof of work; 2. relying on a trusted third party.", "Explain the second approach more simply.", "Does that mean an attacker can never catch up? Correct that assumption.", "What is the current market price? Is that in the source?"] if case["payload"]["language"] == "en" else ["这份资料中的工作量证明解决什么问题？", "它为什么是必要的？", "请按这个顺序比较两种方式：第一种是工作量证明，第二种是依赖可信第三方。", "把刚才提到的第二种方式讲得更简单些。", "这是否意味着攻击者永远不可能追上？请纠正这个假设。", "现在市场价格是多少？资料里有这个信息吗？"]
                        conversation = None
                        result["messages"] = []
                        saved_chat = output / f"{case['id']}.json"
                        if args.resume and saved_chat.exists():
                            result["messages"] = json.loads(saved_chat.read_text()).get("messages", [])
                            if result["messages"]:
                                conversation = result["messages"][-1]["conversation_id"]
                        for question in questions[len(result["messages"]):]:
                            answer = request("POST", f"/notebooks/{notebook_id}/chat", json={**case["payload"], "question": question, "conversation_id": conversation})
                            conversation = answer["conversation_id"]
                            result["messages"].append({"question": question, **answer})
                            save(output / f"{case['id']}.json", result)
                        result["status"] = "degraded" if any(m.get("degraded") for m in result["messages"]) else "passed"
                    else:
                        current_path = output / "current.json"
                        pending = json.loads(current_path.read_text()) if current_path.exists() else {}
                        if pending.get("id") == case["id"] and pending.get("job_id"):
                            job = request("GET", f"/jobs/{pending['job_id']}")
                        else:
                            job = request("POST", f"/notebooks/{notebook_id}/{case['kind']}", json=case["payload"])
                        result["job_id"] = job["id"]
                        save(output / "current.json", result)
                        job = wait_job(job["id"])
                        result["job"] = job
                        if case["kind"] == "quiz" and job["state"] == "complete":
                            listed = request("GET", "/jobs", params={"notebook_id": notebook_id, "kind": "quiz"})
                            public_jobs = [job, next(item for item in listed["items"] if item["id"] == job["id"])]
                            for public in public_jobs:
                                for item in (public.get("result") or {}).get("payload", {}).get("items", []):
                                    if any(key in item for key in ("answer", "answer_index", "explanation", "citations")):
                                        raise AssertionError("Job API exposes Quiz answers before submission")
                            result["job_privacy_verified"] = True
                        if job["state"] != "complete":
                            result["status"] = "failed"
                        else:
                            artifact_id = (job.get("result") or {}).get("artifact_id") or (job.get("result") or {}).get("id")
                            artifact = request("GET", f"/artifacts/{artifact_id}")
                            result["artifact"] = artifact
                            if case["kind"] == "quiz":
                                session = request("POST", f"/artifacts/{artifact_id}/study-sessions", json={"mode": "all"})
                                result["quiz_answers"] = []
                                for item in session["items"]:
                                    if any(key in item for key in ("answer_index", "explanation")):
                                        raise AssertionError("Quiz exposes answers before submission")
                                    result["quiz_answers"].append(request("POST", f"/study-sessions/{session['id']}/quiz-answer", json={"item_id": item["id"], "option_index": 0}))
                            result["status"] = "degraded" if artifact["status"] == "partial" or artifact.get("payload", {}).get("degraded") else "passed"
                except Exception as exc:
                    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                finally:
                    if audio_lock is not None:
                        audio_lock.close()
                result["seconds"] = round(time.monotonic() - start, 2)
                results.append(result)
                save(output / f"{case['id']}.json", result)
                save(results_path, results)
                print(f"END {case['id']}: {result['status']} ({result['seconds']}s)", flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return int(any(item["status"] == "failed" for item in results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=ROOT / ".experiment/samples/bitcoin/source/bitcoin.pdf")
    parser.add_argument("--main-url", default="http://100.80.59.126:11434")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--audio-url", default="http://127.0.0.1:20810")
    parser.add_argument("--port", type=int, default=20831)
    parser.add_argument("--phase", choices=("baseline", "matrix", "extras"), default="baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case", action="append", help="Run only this case ID; repeat to select several cases")
    parser.add_argument("--job-timeout", type=float, default=10800)
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.serve:
        serve(args)
    else:
        raise SystemExit(evaluate(args))


if __name__ == "__main__":
    main()
