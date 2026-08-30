from __future__ import annotations

import asyncio
import hashlib
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import CONFIG
from .database import DB, json_dump, json_load, new_id, utc_now
from .documents import SUPPORTED_EXTENSIONS, sanitize_filename
from .jobs import WORKER, enqueue
from .cleanup import backfill_resources, process_cleanup_operations, purge_job, reconcile_legacy_podcast_temps, register_resource, request_notebook_delete, request_notebook_deletes
from .observability import Reporter, present_job
from .paths import PATHS
from .providers import ProviderError, active_provider, health, inspect_provider, normalize_provider_base_url, probe_tts_provider, provider_by_id
from .retrieval import EMBEDDINGS
from .schemas import ChatRequest, FlashcardRequest, FlashcardReview, LoginRequest, NotebookBatchDelete, NotebookCreate, NotebookUpdate, PodcastRequest, ProviderCreate, ProviderInspectionRequest, ProviderUpdate, QuizRequest, QuizSubmission, SourceSelection, SummaryRequest
from .security import VAULT
from .services import grounded_generate, source_scope


@asynccontextmanager
async def lifespan(app: FastAPI):
    DB.initialize(); DB.seed(CONFIG.development.ollama_url, CONFIG.development.ollama_model, CONFIG.development.tts_url); DB.reset_running_jobs()
    backfill_resources(); reconcile_legacy_podcast_temps(); process_cleanup_operations()
    tts = active_provider("tts")
    if tts:
        try:
            await probe_tts_provider(tts["id"], apply_defaults=True)
        except Exception:
            pass
    WORKER.start()
    yield
    await WORKER.stop()


app = FastAPI(title="Sandevistan-Read", version="0.4.0", lifespan=lifespan)


def request_token(request: Request, authorization: str | None = None) -> str:
    return (authorization or "").removeprefix("Bearer ") or request.cookies.get("sread_session", "")


def require_access(request: Request, authorization: str | None = Header(default=None)) -> None:
    if not CONFIG.security.access_key: return
    if not VAULT.verify_session(request_token(request, authorization), CONFIG.security.access_key): raise HTTPException(401, "需要访问密钥")


api = FastAPI(dependencies=[Depends(require_access)])


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    for field in list(row):
        if field.endswith("_json"):
            row[field[:-5]] = json_load(row.pop(field), [] if field == "citations_json" else {})
    return row


@app.post("/auth/login")
def login(body: LoginRequest, response: Response):
    if not CONFIG.security.access_key or body.access_key == CONFIG.security.access_key:
        token = VAULT.session_token(body.access_key)
        response.set_cookie("sread_session", token, httponly=True, samesite="strict", max_age=86400, path="/")
        return {"token": token}
    raise HTTPException(401, "访问密钥错误")


@app.get("/auth/status")
def auth_status(request: Request, authorization: str | None = Header(default=None)):
    required = bool(CONFIG.security.access_key)
    authenticated = not required or VAULT.verify_session(request_token(request, authorization), CONFIG.security.access_key)
    return {"required": required, "authenticated": authenticated}


@api.get("/status")
async def status():
    ffmpeg = CONFIG.tools.ffmpeg_path
    libreoffice = CONFIG.tools.libreoffice_path
    tool_status = {
        "ffmpeg": {"available": bool(ffmpeg), "scope": CONFIG.tools.scope(ffmpeg), "version": CONFIG.tools.version(ffmpeg), "path": str(Path(ffmpeg).resolve().relative_to(PATHS.root)) if ffmpeg and CONFIG.tools.scope(ffmpeg) == "project" else ffmpeg},
        "libreoffice": {"available": bool(libreoffice), "scope": CONFIG.tools.scope(libreoffice), "version": CONFIG.tools.version(libreoffice), "path": str(Path(libreoffice).resolve().relative_to(PATHS.root)) if libreoffice and CONFIG.tools.scope(libreoffice) == "project" else libreoffice},
    }
    roles = ("main", "vlm", "tts")
    health_results = await asyncio.gather(*(health(role) for role in roles))
    return {"name": "Sandevistan-Read", "version": "0.4.0", "host": CONFIG.server.host, "port": CONFIG.server.port, "providers": dict(zip(roles, health_results)), "tools": tool_status, "retrieval": {"embedding_mode": EMBEDDINGS.mode, "model": CONFIG.models.embedding, "offline": CONFIG.models.offline}, "runtime_root": str(PATHS.runtime)}


@api.get("/notebooks")
def notebooks(include_deleting: bool = False):
    rows = DB.fetchall("SELECT * FROM notebooks WHERE (? OR state='active') ORDER BY updated_at DESC", (int(include_deleting),))
    return [normalize(row) for row in rows]


@api.get("/notebook-management")
def notebook_management(q: str = "", state: str = "all", page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    filters, values = ["(n.title LIKE ? OR n.description LIKE ? OR n.id LIKE ?)"], [f"%{q}%", f"%{q}%", f"%{q}%"]
    if state != "all": filters.append("n.state=?"); values.append(state)
    where = " AND ".join(filters)
    total = (DB.fetchone(f"SELECT COUNT(*) AS count FROM notebooks n WHERE {where}", tuple(values)) or {"count": 0})["count"]
    rows = DB.fetchall(f"""SELECT n.*,
        (SELECT COUNT(*) FROM sources s WHERE s.notebook_id=n.id) source_count,
        (SELECT COALESCE(SUM(size_bytes),0) FROM sources s WHERE s.notebook_id=n.id) source_bytes,
        (SELECT COUNT(*) FROM artifacts a WHERE a.notebook_id=n.id) artifact_count,
        (SELECT COUNT(*) FROM jobs j WHERE j.notebook_id=n.id AND j.state IN ('queued','running','cancelling')) active_jobs
        FROM notebooks n WHERE {where} ORDER BY n.updated_at DESC LIMIT ? OFFSET ?""", tuple(values + [page_size, (page - 1) * page_size]))
    return {"items": [normalize(row) for row in rows], "page": page, "page_size": page_size, "total": total, "pages": max(1, (total + page_size - 1) // page_size)}


@api.post("/notebooks")
def create_notebook(body: NotebookCreate):
    identifier, now = new_id("nb"), utc_now(); DB.execute("INSERT INTO notebooks(id,title,description,created_at,updated_at) VALUES(?,?,?,?,?)", (identifier, body.title, body.description, now, now)); return normalize(DB.fetchone("SELECT * FROM notebooks WHERE id=?", (identifier,)) or {})


@api.post("/notebooks/batch-delete", status_code=202)
def batch_delete_notebooks(body: NotebookBatchDelete):
    return {"items": request_notebook_deletes(body.notebook_ids)}


@api.patch("/notebooks/{notebook_id}")
def update_notebook(notebook_id: str, body: NotebookUpdate):
    row = DB.fetchone("SELECT * FROM notebooks WHERE id=?", (notebook_id,));
    if not row: raise HTTPException(404, "笔记本不存在")
    DB.execute("UPDATE notebooks SET title=?,description=?,updated_at=? WHERE id=?", (body.title or row["title"], body.description if body.description is not None else row["description"], utc_now(), notebook_id)); return normalize(DB.fetchone("SELECT * FROM notebooks WHERE id=?", (notebook_id,)) or {})


def _remove_project_path(relative: str | None, allowed_root: Path) -> None:
    if not relative:
        return
    path = (PATHS.root / relative).resolve()
    try:
        path.relative_to(allowed_root.resolve())
    except ValueError:
        return
    target = path if path.is_dir() else path.parent
    shutil.rmtree(target, ignore_errors=True)


def _require_notebook(notebook_id: str) -> None:
    if not DB.fetchone("SELECT 1 FROM notebooks WHERE id=?", (notebook_id,)):
        raise HTTPException(404, "笔记本不存在")


@api.delete("/notebooks/{notebook_id}", status_code=202)
def delete_notebook(notebook_id: str):
    _require_notebook(notebook_id)
    operation_id = request_notebook_delete(notebook_id)
    return {"accepted": True, "operation_id": operation_id}


@api.post("/notebooks/{notebook_id}/cleanup/retry")
def retry_notebook_cleanup(notebook_id: str):
    _require_notebook(notebook_id)
    DB.execute("UPDATE notebooks SET state='deleting',cleanup_error=NULL,updated_at=? WHERE id=?", (utc_now(), notebook_id))
    operation_id = request_notebook_delete(notebook_id)
    return {"accepted": True, "operation_id": operation_id}


@api.get("/notebooks/{notebook_id}")
def notebook(notebook_id: str):
    row = DB.fetchone("SELECT * FROM notebooks WHERE id=?", (notebook_id,));
    if not row: raise HTTPException(404, "笔记本不存在")
    result = normalize(row); result["sources"] = [normalize(item) for item in DB.fetchall("SELECT * FROM sources WHERE notebook_id=? ORDER BY created_at DESC", (notebook_id,))]; return result


@api.post("/notebooks/{notebook_id}/sources")
async def upload_sources(notebook_id: str, files: list[UploadFile] = File(...)):
    _require_notebook(notebook_id)
    named = [(upload, sanitize_filename(upload.filename or "document")) for upload in files]
    unsupported = [name for _, name in named if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS]
    if unsupported:
        raise HTTPException(415, "不支持 " + "、".join(unsupported))
    staged: list[dict[str, Any]] = []
    created_dirs: list[Path] = []
    temporary_paths: list[Path] = []
    try:
        for upload, name in named:
            source_id, revision, now = new_id("source"), new_id("rev"), utc_now()
            temporary = PATHS.temp / f"{source_id}.upload"
            temporary_paths.append(temporary)
            size, digest = 0, hashlib.sha256()
            with temporary.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > CONFIG.runtime.max_upload_mib * 1024 * 1024:
                        raise HTTPException(413, f"{name} 文件过大")
                    free = shutil.disk_usage(PATHS.root).free
                    if free - len(chunk) < CONFIG.runtime.minimum_free_mib * 1024 * 1024:
                        raise HTTPException(507, "项目磁盘剩余空间低于安全阈值")
                    digest.update(chunk); handle.write(chunk)
            target_dir = PATHS.blobs / source_id
            target_dir.mkdir(parents=True, exist_ok=False)
            created_dirs.append(target_dir)
            target = target_dir / name
            temporary.replace(target)
            register_resource("notebook", notebook_id, notebook_id, "source", target_dir)
            staged.append({"id": source_id, "revision": revision, "name": name, "media_type": upload.content_type or "application/octet-stream", "size": size, "sha256": digest.hexdigest(), "target": target, "now": now})
        with DB.transaction() as connection:
            for item in staged:
                connection.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (item["id"], notebook_id, item["revision"], item["name"], item["media_type"], item["size"], item["sha256"], str(item["target"].relative_to(PATHS.root)), None, "queued", 1, 0, None, None, "{}", item["now"], item["now"]))
        return [{"source_id": item["id"], "job": enqueue("ingest", notebook_id, {"source_id": item["id"]})} for item in staged]
    except Exception:
        for item in staged:
            DB.execute("DELETE FROM sources WHERE id=?", (item["id"],))
        for directory in created_dirs:
            shutil.rmtree(directory, ignore_errors=True)
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        for upload, _ in named:
            await upload.close()


@api.patch("/sources/{source_id}/selection")
def select_source(source_id: str, body: SourceSelection):
    if not DB.fetchone("SELECT 1 FROM sources WHERE id=?", (source_id,)):
        raise HTTPException(404, "资料不存在")
    DB.execute("UPDATE sources SET selected=?,updated_at=? WHERE id=?", (int(body.selected), utc_now(), source_id)); return {"ok": True}


@api.delete("/sources/{source_id}", status_code=204)
def delete_source(source_id: str):
    row = DB.fetchone("SELECT blob_path FROM sources WHERE id=?", (source_id,))
    if not row:
        raise HTTPException(404, "资料不存在")
    DB.execute("DELETE FROM chunks_fts WHERE source_id=?", (source_id,))
    DB.execute("DELETE FROM sources WHERE id=?", (source_id,))
    _remove_project_path(row.get("blob_path"), PATHS.blobs)
    shutil.rmtree(PATHS.renders / source_id, ignore_errors=True)
    return Response(status_code=204)


@api.post("/notebooks/{notebook_id}/chat")
async def ask(notebook_id: str, body: ChatRequest):
    _require_notebook(notebook_id)
    ids = source_scope(notebook_id, body.source_ids)
    if not ids:
        raise HTTPException(409, "当前范围没有已就绪的文档")
    conversation_id = body.conversation_id
    now = utc_now()
    if conversation_id:
        conversation = DB.fetchone("SELECT notebook_id FROM conversations WHERE id=?", (conversation_id,))
        if not conversation or conversation["notebook_id"] != notebook_id:
            raise HTTPException(404, "当前笔记本中不存在该对话")
    else:
        conversation_id = new_id("conversation"); DB.execute("INSERT INTO conversations VALUES(?,?,?,?,?)", (conversation_id, notebook_id, body.question[:80], now, now))
    DB.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)", (new_id("message"), conversation_id, "user", body.question, "[]", None, "complete", now))
    result = await grounded_generate(notebook_id, "直接、清楚地回答问题。", body.question, ids, body.language)
    message_id = new_id("message"); DB.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)", (message_id, conversation_id, "assistant", result["content"], json_dump(result["citations"]), result["scope_hash"], "complete", utc_now()))
    DB.execute("UPDATE conversations SET updated_at=? WHERE id=?", (utc_now(), conversation_id))
    return {"id": message_id, "conversation_id": conversation_id, **result}


@api.get("/notebooks/{notebook_id}/conversations")
def conversations(notebook_id: str): return [normalize(row) for row in DB.fetchall("SELECT * FROM conversations WHERE notebook_id=? ORDER BY updated_at DESC", (notebook_id,))]


@api.get("/conversations/{conversation_id}/messages")
def messages(conversation_id: str): return [normalize(row) for row in DB.fetchall("SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at", (conversation_id,))]


@api.post("/notebooks/{notebook_id}/summary")
def summary(notebook_id: str, body: SummaryRequest): _require_notebook(notebook_id); return enqueue("summary", notebook_id, body.model_dump())


@api.get("/notebooks/{notebook_id}/summary")
def latest_summary(notebook_id: str):
    row = DB.fetchone("SELECT * FROM summaries WHERE notebook_id=? ORDER BY created_at DESC LIMIT 1", (notebook_id,)); return normalize(row) if row else None


@api.post("/notebooks/{notebook_id}/quiz")
def quiz(notebook_id: str, body: QuizRequest): _require_notebook(notebook_id); return enqueue("quiz", notebook_id, body.model_dump())


@api.post("/notebooks/{notebook_id}/flashcards")
def flashcards(notebook_id: str, body: FlashcardRequest): _require_notebook(notebook_id); return enqueue("flashcard", notebook_id, body.model_dump())


@api.post("/notebooks/{notebook_id}/podcasts")
def podcast(notebook_id: str, body: PodcastRequest): _require_notebook(notebook_id); return enqueue("podcast", notebook_id, body.model_dump())


@api.get("/notebooks/{notebook_id}/artifacts")
def artifacts(notebook_id: str, type: str | None = None):
    rows = DB.fetchall("SELECT * FROM artifacts WHERE notebook_id=? AND (? IS NULL OR type=?) ORDER BY created_at DESC", (notebook_id, type, type))
    output = [normalize(row) for row in rows]
    for item in output:
        if item.get("media_path"):
            item["media_url"] = f"/api/artifacts/{item['id']}/media"
    return output


@api.get("/artifacts/{artifact_id}")
def artifact(artifact_id: str):
    row = DB.fetchone("SELECT * FROM artifacts WHERE id=?", (artifact_id,))
    if not row:
        raise HTTPException(404, "产物不存在")
    item = normalize(row)
    if item.get("media_path"):
        item["media_url"] = f"/api/artifacts/{item['id']}/media"
    return item


@api.get("/artifacts/{artifact_id}/media")
def artifact_media(artifact_id: str):
    row = DB.fetchone("SELECT media_path FROM artifacts WHERE id=?", (artifact_id,)); path = PATHS.root / row["media_path"] if row and row["media_path"] else None
    if not path or not path.exists(): raise HTTPException(404, "音频不存在")
    media_type = "audio/mp4" if path.suffix.lower() in {".m4a", ".mp4"} else "audio/wav"
    return FileResponse(path, media_type=media_type, filename=f"sandevistan-podcast{path.suffix.lower()}")


@api.post("/artifacts/{artifact_id}/quiz/submit")
def submit_quiz(artifact_id: str, body: QuizSubmission):
    row = DB.fetchone("SELECT payload_json FROM artifacts WHERE id=? AND type='quiz'", (artifact_id,));
    if not row: raise HTTPException(404, "题库不存在")
    items = json_load(row["payload_json"], {}).get("items", []); correct = sum(1 for item in items if body.answers.get(item["id"]) == item["answer"]); score = correct / max(1, len(items)); DB.execute("INSERT INTO quiz_attempts VALUES(?,?,?,?,?)", (new_id("attempt"), artifact_id, json_dump(body.answers), score, utc_now())); return {"score": score, "correct": correct, "total": len(items)}


@api.post("/artifacts/{artifact_id}/flashcards/review")
def review(artifact_id: str, body: FlashcardReview):
    row = DB.fetchone("SELECT payload_json FROM artifacts WHERE id=? AND type='flashcard'", (artifact_id,))
    cards = json_load(row["payload_json"], {}).get("items", []) if row else []
    if not row or body.card_id not in {card.get("id") for card in cards}:
        raise HTTPException(404, "闪卡不存在")
    DB.execute("INSERT INTO flashcard_reviews VALUES(?,?,?,?,?)", (new_id("review"), artifact_id, body.card_id, body.rating, utc_now())); return {"ok": True}


@api.get("/jobs")
def jobs(notebook_id: str | None = None, q: str = "", kind: str = "all", state: str = "all", page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    filters, values = ["(j.display_name LIKE ? OR j.stage LIKE ? OR j.id LIKE ? OR COALESCE(n.title,'') LIKE ?)", "(? IS NULL OR j.notebook_id=?)"], [f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%", notebook_id, notebook_id]
    if kind != "all": filters.append("j.kind=?"); values.append(kind)
    if state != "all": filters.append("j.state=?"); values.append(state)
    where = " AND ".join(filters)
    total = (DB.fetchone(f"SELECT COUNT(*) count FROM jobs j LEFT JOIN notebooks n ON n.id=j.notebook_id WHERE {where}", tuple(values)) or {"count": 0})["count"]
    rows = DB.fetchall(f"SELECT j.*,n.title notebook_title FROM jobs j LEFT JOIN notebooks n ON n.id=j.notebook_id WHERE {where} ORDER BY j.created_at DESC LIMIT ? OFFSET ?", tuple(values + [page_size, (page - 1) * page_size]))
    queued = [item["id"] for item in DB.fetchall("SELECT id FROM jobs WHERE state='queued' ORDER BY created_at")]
    items = [present_job(row, queued.index(row["id"]) + 1 if row["id"] in queued else 0) for row in rows]
    return {"items": items, "page": page, "page_size": page_size, "total": total, "pages": max(1, (total + page_size - 1) // page_size)}


@api.get("/jobs/{job_id}")
def job(job_id: str):
    row = DB.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,));
    if not row: raise HTTPException(404, "任务不存在")
    return present_job(row)


@api.get("/jobs/{job_id}/events")
def job_events(job_id: str):
    if not DB.fetchone("SELECT 1 FROM jobs WHERE id=?", (job_id,)): raise HTTPException(404, "任务不存在")
    return [normalize(row) for row in DB.fetchall("SELECT * FROM job_events WHERE job_id=? ORDER BY id DESC LIMIT 200", (job_id,))]


@api.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    row = DB.fetchone("SELECT state FROM jobs WHERE id=?", (job_id,))
    if not row: raise HTTPException(404, "任务不存在")
    if row["state"] == "queued":
        Reporter(job_id).update("cancelled", "已取消", 1, state="cancelled")
        DB.execute("UPDATE jobs SET cancel_requested=1,finished_at=?,updated_at=? WHERE id=?", (utc_now(), utc_now(), job_id))
    elif row["state"] == "running":
        DB.execute("UPDATE jobs SET cancel_requested=1,updated_at=? WHERE id=?", (utc_now(), job_id))
        Reporter(job_id).update("cancelling", "正在安全终止", float((DB.fetchone("SELECT progress FROM jobs WHERE id=?", (job_id,)) or {"progress": 0})["progress"]), state="cancelling")
    return {"ok": True}


@api.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    try: return purge_job(job_id)
    except RuntimeError as exc: raise HTTPException(409, str(exc)) from exc


@api.post("/jobs/batch-purge")
def batch_purge(job_ids: list[str] = Body(..., embed=True)):
    results = []
    for job_id in job_ids:
        try: results.append({"id": job_id, **purge_job(job_id)})
        except RuntimeError as exc: results.append({"id": job_id, "deleted": False, "error": str(exc)})
    return {"items": results}


@api.get("/providers")
def providers():
    rows = []
    for row in DB.fetchall("SELECT * FROM provider_profiles ORDER BY role,name"):
        has_api_key = bool(row.get("secret_enc")); item = normalize(row); item.pop("secret_enc", None); item["has_api_key"] = has_api_key; rows.append(item)
    return rows


def _provider_candidate(body: ProviderCreate | ProviderInspectionRequest, *, api_key: str | None = None) -> dict[str, Any]:
    return {
        "name": getattr(body, "name", "Provider"),
        "role": body.role,
        "kind": body.kind,
        "base_url": body.base_url,
        "model": body.model,
        "api_key": body.api_key if api_key is None else api_key,
        "config": dict(body.config),
    }


def _persisted_capabilities(kind: str, inspection: dict[str, Any]) -> dict[str, Any]:
    capabilities = dict(inspection.get("capabilities") or {})
    if kind == "sandevistan_tts":
        capabilities["models"] = inspection.get("models") or []
        capabilities["recommended"] = inspection.get("recommended")
    return capabilities


def _apply_recommendation(candidate: dict[str, Any], inspection: dict[str, Any]) -> None:
    recommended = inspection.get("recommended") or {}
    config = candidate.get("config") or {}
    if candidate.get("kind") == "sandevistan_tts" and config.get("auto_select") and recommended.get("model"):
        candidate["model"] = recommended["model"]
        config["compute_device"] = recommended.get("compute_device")
        candidate["config"] = config


def _inspection_conflict(inspection: dict[str, Any]) -> HTTPException:
    error = inspection.get("error") or {}
    message = error.get("message") or inspection.get("warning") or "Provider 验证未通过"
    return HTTPException(409, {"message": message, "inspection": inspection})


@api.post("/providers/inspect")
async def inspect_provider_configuration(body: ProviderInspectionRequest):
    key = body.api_key
    if body.provider_id and key is None:
        stored = provider_by_id(body.provider_id)
        if not stored:
            raise HTTPException(404, "Provider 不存在")
        key = stored.get("api_key", "")
    return await inspect_provider(_provider_candidate(body, api_key=key or ""), body.mode)


@api.post("/providers")
async def create_provider(body: ProviderCreate):
    identifier, now = new_id("provider"), utc_now()
    candidate = _provider_candidate(body)
    try:
        candidate["base_url"] = normalize_provider_base_url(body.kind, body.base_url)
    except ProviderError as exc:
        raise HTTPException(422, str(exc)) from exc
    inspection = None
    if body.active:
        inspection = await inspect_provider(candidate, body.validation_mode)
        if not inspection.get("activation_eligible"):
            raise _inspection_conflict(inspection)
        _apply_recommendation(candidate, inspection)
    capabilities = _persisted_capabilities(body.kind, inspection) if inspection else body.capabilities
    with DB.transaction() as connection:
        if body.active:
            connection.execute("UPDATE provider_profiles SET active=0 WHERE role=?", (body.role,))
        connection.execute(
            """INSERT INTO provider_profiles
            (id,name,role,kind,base_url,model,secret_enc,capabilities_json,config_json,active,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (identifier, body.name, body.role, body.kind, candidate["base_url"], candidate["model"], VAULT.encrypt(body.api_key), json_dump(capabilities), json_dump(candidate["config"]), int(body.active), now, now),
        )
    return {"id": identifier, "active": body.active, "inspection": inspection}


@api.patch("/providers/{provider_id}")
async def update_provider(provider_id: str, body: ProviderUpdate):
    row = provider_by_id(provider_id)
    if not row:
        raise HTTPException(404, "Provider 不存在")
    values = body.model_dump(exclude_none=True, exclude={"validation_mode"})
    if "base_url" in values:
        try:
            values["base_url"] = normalize_provider_base_url(row["kind"], values["base_url"])
        except ProviderError as exc:
            raise HTTPException(422, str(exc)) from exc
    material = {"base_url", "model", "api_key", "config"}
    target_active = bool(values.get("active", row["active"]))
    requires_validation = target_active and (not row["active"] or bool(material.intersection(values)))
    if requires_validation:
        candidate = {
            "name": values.get("name", row["name"]),
            "role": row["role"],
            "kind": row["kind"],
            "base_url": values.get("base_url", row["base_url"]),
            "model": values.get("model", row["model"]),
            "api_key": values.get("api_key", row.get("api_key", "")),
            "config": dict(values.get("config", row.get("config") or {})),
        }
        inspection = await inspect_provider(candidate, body.validation_mode)
        if not inspection.get("activation_eligible"):
            raise _inspection_conflict(inspection)
        _apply_recommendation(candidate, inspection)
        values["model"] = candidate["model"]
        values["config"] = candidate["config"]
        values["capabilities"] = _persisted_capabilities(row["kind"], inspection)
    mapping = {"capabilities": "capabilities_json", "config": "config_json", "api_key": "secret_enc"}
    sets, params = [], []
    for key, value in values.items():
        column = mapping.get(key, key); value = VAULT.encrypt(value) if key == "api_key" else json_dump(value) if key in {"capabilities", "config"} else int(value) if key == "active" else value
        sets.append(f"{column}=?"); params.append(value)
    sets.append("updated_at=?"); params.extend([utc_now(), provider_id])
    with DB.transaction() as connection:
        if values.get("active"):
            connection.execute("UPDATE provider_profiles SET active=0 WHERE role=?", (row["role"],))
        connection.execute(f"UPDATE provider_profiles SET {','.join(sets)} WHERE id=?", tuple(params))
    return {"ok": True, "active": target_active}


@api.post("/providers/{provider_id}/test")
async def test_provider(provider_id: str):
    provider = provider_by_id(provider_id)
    if not provider:
        raise HTTPException(404, "Provider 不存在")
    inspection = await inspect_provider(provider, "catalog")
    return {"ok": inspection.get("activation_eligible", False), **inspection}


@api.post("/providers/{provider_id}/probe")
async def probe_provider(provider_id: str):
    row = DB.fetchone("SELECT role FROM provider_profiles WHERE id=?", (provider_id,))
    if not row:
        raise HTTPException(404, "Provider 不存在")
    if row["role"] != "tts":
        raise HTTPException(409, "仅 TTS Provider 支持能力探测")
    try:
        return await probe_tts_provider(provider_id, apply_defaults=True)
    except Exception as exc:
        raise HTTPException(502, f"TTS 能力探测失败：{exc}") from exc


app.mount("/api", api)
frontend = PATHS.root / "frontend" / "dist"
if frontend.exists():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")


def create_app() -> FastAPI: return app
