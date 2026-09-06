from __future__ import annotations

import wave
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sandevistan_read import app as app_module, jobs, observability, providers, study_sessions
from sandevistan_read.database import Database, json_dump, json_load


@pytest.fixture
def database(tmp_path, monkeypatch):
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    database.execute("INSERT INTO notebooks(id,title,description,created_at,updated_at) VALUES('n','Audit','','now','now')")
    database.execute("""INSERT INTO sources(id,notebook_id,revision_id,filename,media_type,size_bytes,sha256,blob_path,state,selected,created_at,updated_at)
        VALUES('s','n','r','audit.txt','text/plain',1,'hash','fixture','queued',1,'now','now')""")
    database.execute("""INSERT INTO artifacts(id,notebook_id,type,title,scope_json,language,status,payload_json,citations_json,created_at,updated_at)
        VALUES('cards','n','flashcard','Cards','[]','en','ready',?,'[]','now','now')""",
        (json_dump({"items": [{"id": key, "front": key, "back": key, "citations": []} for key in ("f1", "f2")]}),))
    for module in (app_module, jobs, observability, study_sessions):
        monkeypatch.setattr(module, "DB", database)
    monkeypatch.setattr(jobs, "active_provider", lambda role: None)
    monkeypatch.setattr(app_module.CONFIG.security, "access_key", "")
    monkeypatch.setattr(jobs, "process_cleanup_operations", lambda: None)
    return database


def test_cancel_queued_ingest_updates_source_and_is_idempotent(database):
    job = jobs.enqueue("ingest", "n", {"source_id": "s"})
    client = TestClient(app_module.api)
    assert client.post(f"/jobs/{job['id']}/cancel").json() == {"ok": True}
    row = database.fetchone("SELECT state,selected,error FROM sources WHERE id='s'")
    assert row == {"state": "failed", "selected": 0, "error": "解析已取消"}
    first = database.fetchone("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert first["state"] == "cancelled" and first["finished_at"]
    events = database.fetchall("SELECT * FROM job_events WHERE job_id=?", (job["id"],))
    assert client.post(f"/jobs/{job['id']}/cancel").status_code == 200
    assert database.fetchone("SELECT * FROM jobs WHERE id=?", (job["id"],)) == first
    assert database.fetchall("SELECT * FROM job_events WHERE job_id=?", (job["id"],)) == events


@pytest.mark.asyncio
async def test_worker_pre_cancel_and_restart_do_not_execute(database, monkeypatch):
    job = jobs.enqueue("ingest", "n", {"source_id": "s"})
    database.execute("UPDATE jobs SET state='cancelling',cancel_requested=1 WHERE id=?", (job["id"],))
    database.reset_running_jobs()
    worker = jobs.JobWorker()
    monkeypatch.setattr(jobs, "process_cleanup_operations", lambda: setattr(worker, "stopped", True))

    async def unexpected_execute(job):
        pytest.fail("cancelled job executed")

    monkeypatch.setattr(jobs, "execute", unexpected_execute)
    await worker.run()
    assert database.fetchone("SELECT state FROM jobs WHERE id=?", (job["id"],))["state"] == "cancelled"
    assert database.fetchone("SELECT state FROM sources WHERE id='s'")["state"] == "failed"


def test_removed_flashcard_stays_removed_after_resume_and_review(database):
    client = TestClient(app_module.api)
    session = client.post("/artifacts/cards/study-sessions", json={"mode": "due"}).json()
    assert client.delete("/artifacts/cards/flashcards/f1").status_code == 204
    resumed = client.post("/artifacts/cards/study-sessions", json={"mode": "due"}).json()
    assert [item["id"] for item in resumed["items"]] == ["f2"]
    result = client.post(f"/study-sessions/{session['id']}/flashcard-review", json={"item_id": "f2", "rating": "good"}).json()
    assert result["session"]["progress"] == {"current": 1, "total": 1}
    assert result["session"]["status"] == "complete"
    assert client.post(f"/study-sessions/{session['id']}/flashcard-review", json={"item_id": "f1", "rating": "good"}).status_code == 409
    same = study_sessions.create_session("cards", "same", shuffle=True)
    assert [item["id"] for item in same["items"]] == ["f2"]


def test_removing_reviewed_and_last_card_preserves_history(database):
    session = study_sessions.create_session("cards", "all")
    study_sessions.review_flashcard(session["id"], "f1", "good")
    history = database.fetchall("SELECT * FROM flashcard_reviews")
    study_sessions.suspend_flashcard("cards", "f1")
    assert study_sessions.get_session(session["id"])["progress"] == {"current": 0, "total": 1}
    study_sessions.suspend_flashcard("cards", "f2")
    final = study_sessions.get_session(session["id"])
    assert final["items"] == [] and final["status"] == "complete"
    assert database.fetchall("SELECT * FROM flashcard_reviews") == history
    assert study_sessions.create_session("cards", "same")["items"] == []


def test_legacy_flashcard_session_reconciles_without_changing_completed_history(database):
    session = study_sessions.create_session("cards", "all")
    study_sessions.review_flashcard(session["id"], "f1", "good")
    study_sessions.suspend_flashcard("cards", "f2")
    # Simulate the old implementation leaving a suspended card in an active queue.
    database.execute("UPDATE study_sessions SET item_ids_json=?,status='active',completed_at=NULL WHERE id=?", (json_dump(["f1", "f2"]), session["id"]))
    resumed = study_sessions.create_session("cards", "all")
    assert resumed["status"] == "complete" and resumed["progress"] == {"current": 1, "total": 1}
    stored = database.fetchone("SELECT * FROM study_sessions WHERE id=?", (session["id"],))
    assert stored["status"] == "complete"
    study_sessions.suspend_flashcard("cards", "f1")
    assert study_sessions.get_session(session["id"])["items"] == []
    assert database.fetchone("SELECT * FROM study_sessions WHERE id=?", (session["id"],)) == stored


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [None, {"id": "new"}])
async def test_asr_explicit_provider_survives_role_change(tmp_path, monkeypatch, active):
    original = {"id": "original"}
    monkeypatch.setattr(providers, "active_provider", lambda role: active)
    monkeypatch.setattr(providers, "audio_provider_readiness", lambda value: (value is original, "wrong provider"))

    async def transcribe(provider, path, **kwargs):
        assert provider is original
        return {"text": "verified"}

    monkeypatch.setattr(providers, "_transcribe_with_provider", transcribe)
    result = await providers.transcribe_audio(tmp_path / "fixture.wav", language="English", provider=original)
    assert result["text"] == "verified"


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.asyncio
async def test_running_cancel_preserves_ready_sources_and_blocks_late_progress(database, monkeypatch, ready, raises):
    job = jobs.enqueue("ingest", "n", {"source_id": "s"})
    worker = jobs.JobWorker()

    async def execute(current):
        database.execute("UPDATE sources SET state=? WHERE id='s'", ("ready" if ready else "processing",))
        assert jobs.request_cancel(current["id"])
        observability.Reporter(current["id"]).update("parse", "late parse callback", 0.5)
        assert database.fetchone("SELECT state FROM jobs WHERE id=?", (job["id"],))["state"] == "cancelling"
        if raises:
            raise RuntimeError("cancel observed")
        return {"source_id": "s"}

    monkeypatch.setattr(jobs, "execute", execute)
    monkeypatch.setattr(jobs, "process_cleanup_operations", lambda: setattr(worker, "stopped", True))
    await worker.run()
    row = database.fetchone("SELECT state,finished_at FROM jobs WHERE id=?", (job["id"],))
    assert row["state"] == "cancelled" and row["finished_at"]
    assert database.fetchone("SELECT state FROM sources WHERE id='s'")["state"] == ("ready" if ready else "failed")


def test_claim_and_cancel_concurrently_never_resurrect_job(database):
    job = jobs.enqueue("ingest", "n", {"source_id": "s"})
    barrier = Barrier(2)

    def claim():
        barrier.wait(timeout=5)
        return jobs._claim_next_job()

    def cancel():
        barrier.wait(timeout=5)
        return jobs.request_cancel(job["id"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        claim_result = executor.submit(claim)
        cancel_result = executor.submit(cancel)
        claimed = claim_result.result(timeout=10)
        assert cancel_result.result(timeout=10)
    if claimed:
        jobs._finish_job(job["id"], result={})
    assert jobs._claim_next_job() is None
    assert database.fetchone("SELECT state FROM jobs WHERE id=?", (job["id"],))["state"] == "cancelled"
    assert database.fetchone("SELECT state FROM sources WHERE id='s'")["state"] == "failed"
    observability.Reporter(job["id"]).update("running", "stale", 0.2, state="running")
    assert database.fetchone("SELECT state FROM jobs WHERE id=?", (job["id"],))["state"] == "cancelled"


def test_cancel_transaction_rolls_back_job_and_events_with_source_failure(database, monkeypatch):
    job = jobs.enqueue("ingest", "n", {"source_id": "s"})

    def fail(*args):
        raise RuntimeError("fixture write failure")

    monkeypatch.setattr(jobs, "_cancel_ingest_source", fail)
    with pytest.raises(RuntimeError, match="fixture write failure"):
        jobs.request_cancel(job["id"])
    assert database.fetchone("SELECT state FROM jobs WHERE id=?", (job["id"],))["state"] == "queued"
    assert database.fetchone("SELECT state FROM sources WHERE id='s'")["state"] == "queued"
    assert len(database.fetchall("SELECT * FROM job_events WHERE job_id=?", (job["id"],))) == 1


@pytest.mark.parametrize("other_state", [None, "queued", "running", "complete"])
def test_legacy_cancel_reconciliation_only_repairs_attributable_sources(database, other_state):
    job = jobs.enqueue("ingest", "n", {"source_id": "s"})
    database.execute("UPDATE jobs SET state='cancelled' WHERE id=?", (job["id"],))
    if other_state:
        other = jobs.enqueue("ingest", "n", {"source_id": "s"})
        database.execute("UPDATE jobs SET state=? WHERE id=?", (other_state, other["id"]))
        if other_state == "running":
            # Even an older active job prevents the repair of its source.
            database.execute("UPDATE jobs SET created_at='2000' WHERE id=?", (other["id"],))
    jobs.reconcile_cancelled_ingests()
    first = database.fetchone("SELECT * FROM sources WHERE id='s'")
    assert first["state"] == ("failed" if other_state is None else "queued")
    jobs.reconcile_cancelled_ingests()
    assert database.fetchone("SELECT * FROM sources WHERE id='s'") == first


def test_cancel_other_jobs_and_ready_sources_leave_documents_intact(database):
    source = database.fetchone("SELECT * FROM sources WHERE id='s'")
    summary = jobs.enqueue("summary", "n", {"source_ids": ["s"]})
    assert jobs.request_cancel(summary["id"])
    assert database.fetchone("SELECT * FROM sources WHERE id='s'") == source
    database.execute("UPDATE sources SET state='ready' WHERE id='s'")
    ingest = jobs.enqueue("ingest", "n", {"source_id": "s"})
    assert jobs.request_cancel(ingest["id"])
    jobs.reconcile_cancelled_ingests()
    assert database.fetchone("SELECT state,selected FROM sources WHERE id='s'") == {"state": "ready", "selected": 1}
    assert TestClient(app_module.api).post("/jobs/missing/cancel").status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["queued", "tts", "legacy"])
@pytest.mark.parametrize("pause", [True, False])
async def test_podcast_tts_and_both_asr_passes_keep_enqueued_audio(database, tmp_path, monkeypatch, phase, pause):
    original = {"id": "original", "name": "Original AUDIO", "role": "audio", "kind": "sandevistan_audio", "model": "tts",
                "base_url": "http://audio.invalid", "api_key": "", "config": {"asr_auto_select": False, "asr_model": "asr", "asr_compute_device": "cpu"},
                "capabilities": {"models": [{"id": "tts", "installed": True}], "asr": {
                    "models": [{"id": "asr", "installed": True, "devices": [{"id": "cpu", "available": True}]}],
                    "diarization": True, "timestamp_precisions": ["segment"], "languages": ["Chinese", "English"], "aligner_languages": ["Chinese", "English"]}}}
    main = {"id": "main", "name": "Main", "kind": "ollama", "model": "fixture", "config": {}, "base_url": "http://localhost"}
    active = {"audio": original, "main": main}
    replacement = None if pause else {**original, "id": "new", "name": "New AUDIO"}
    monkeypatch.setattr(jobs, "active_provider", lambda role: active[role])
    monkeypatch.setattr(providers, "active_provider", lambda role: active[role])
    monkeypatch.setattr(jobs, "provider_by_id", lambda key: {"original": original, "main": main}.get(key))
    monkeypatch.setattr(jobs, "PATHS", SimpleNamespace(root=tmp_path, job_work=tmp_path / "work", artifacts=tmp_path / "artifacts"))
    monkeypatch.setattr(jobs, "CONFIG", SimpleNamespace(tools=SimpleNamespace(ffmpeg_path=None), models=jobs.CONFIG.models))
    monkeypatch.setattr(jobs, "register_resource", lambda *args: None)
    job = jobs.enqueue("podcast", "n", {"minutes": 5})
    payload = json_load(job["payload_json"], {})
    assert payload["provider_ids"]["audio"] == "original"
    if phase == "legacy":
        payload.pop("provider_ids")
    if phase == "queued":
        active["audio"] = replacement
    assert jobs._claim_next_job()["id"] == job["id"]
    tts_providers, asr_providers = [], []

    async def script(*args, **kwargs):
        return {"language": "en", "source_ids": ["s"], "citations": [], "quality": {}, "duration": {"target_minutes": 5},
                "turns": [{"speaker": "HOST_A", "text": "First turn"}, {"speaker": "HOST_B", "text": "Second turn"}],
                "chapters": [{"turn_start": 0, "turn_end": 1}]}

    async def synthesize(text, voice, output, **kwargs):
        tts_providers.append(kwargs["provider"])
        if phase in {"tts", "legacy"}:
            active["audio"] = replacement
        with wave.open(str(output), "wb") as audio:
            audio.setparams((1, 2, 1000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0" * 300_000)  # 150 seconds, generated entirely within this test.
        return output

    async def transcribe(provider, path, **kwargs):
        assert path.exists()
        asr_providers.append(provider)
        return {"text": "fixture"}

    monkeypatch.setattr(jobs, "build_podcast_script", script)
    monkeypatch.setattr(jobs, "synthesize", synthesize)
    monkeypatch.setattr(providers, "_transcribe_with_provider", transcribe)
    monkeypatch.setattr(jobs, "assess_transcription", lambda *args: {"passed": len(asr_providers) > 1, "turn_errors": [0] if len(asr_providers) == 1 else []})
    result = await jobs._podcast("n", payload, job["id"])
    assert len(tts_providers) == 3 and all(value is original for value in tts_providers)
    assert len(asr_providers) == 2 and all(value is original for value in asr_providers)
    assert database.fetchone("SELECT id FROM artifacts WHERE id=?", (result["id"],))
    later = jobs.enqueue("podcast", "n", {})
    assert json_load(later["payload_json"], {})["provider_ids"]["audio"] == (None if pause else "new")


@pytest.mark.asyncio
async def test_missing_bound_audio_never_falls_back(database, monkeypatch):
    monkeypatch.setattr(jobs, "provider_by_id", lambda key: None)
    monkeypatch.setattr(jobs, "active_provider", lambda role: pytest.fail("must not resolve a different AUDIO"))
    with pytest.raises(RuntimeError, match="任务绑定的 AUDIO Provider 不存在"):
        await jobs._podcast("n", {"provider_ids": {"audio": "deleted"}}, "job_missing")
