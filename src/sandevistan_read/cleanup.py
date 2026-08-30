from __future__ import annotations

import shutil
import re
from pathlib import Path
from typing import Any

from .database import DB, new_id, utc_now
from .paths import PATHS


ALLOWED_ROOTS = (PATHS.blobs, PATHS.renders, PATHS.artifacts, PATHS.job_work)


def safe_remove(relative_path: str | None) -> int:
    if not relative_path:
        return 0
    target = (PATHS.root / relative_path).resolve()
    if not any(_within(target, root) for root in ALLOWED_ROOTS):
        raise ValueError(f"拒绝清理未授权路径: {relative_path}")
    if not target.exists():
        return 0
    size = allocated_bytes(target)
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return size


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def allocated_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def register_resource(owner_type: str, owner_id: str, notebook_id: str | None, kind: str, path: Path) -> None:
    relative = str(path.resolve().relative_to(PATHS.root.resolve()))
    DB.execute(
        """INSERT OR REPLACE INTO local_resources
        (id,owner_type,owner_id,notebook_id,kind,relative_path,state,size_bytes,created_at,transferred_at)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (new_id("resource"), owner_type, owner_id, notebook_id, kind, relative, "active", allocated_bytes(path), utc_now(), None),
    )


def purge_job(job_id: str) -> dict[str, Any]:
    job = DB.fetchone("SELECT state FROM jobs WHERE id=?", (job_id,))
    if not job:
        return {"deleted": False, "bytes_freed": 0}
    if job["state"] in {"queued", "running", "cancelling"}:
        raise RuntimeError("进行中的任务必须先终止")
    resources = DB.fetchall("SELECT * FROM local_resources WHERE owner_type='job' AND owner_id=? AND state='active'", (job_id,))
    freed = 0
    for resource in resources:
        freed += safe_remove(resource["relative_path"])
    with DB.transaction() as connection:
        connection.execute("DELETE FROM local_resources WHERE owner_type='job' AND owner_id=?", (job_id,))
        connection.execute("DELETE FROM job_events WHERE job_id=?", (job_id,))
        connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    return {"deleted": True, "bytes_freed": freed}


def request_notebook_delete(notebook_id: str) -> str:
    row = DB.fetchone("SELECT id FROM notebooks WHERE id=?", (notebook_id,))
    if not row:
        raise KeyError(notebook_id)
    operation_id, now = new_id("cleanup"), utc_now()
    with DB.transaction() as connection:
        connection.execute("UPDATE jobs SET cancel_requested=1,state=CASE WHEN state='running' THEN 'cancelling' ELSE state END,updated_at=? WHERE notebook_id=? AND state IN ('queued','running','cancelling')", (now, notebook_id))
        connection.execute("UPDATE notebooks SET state='deleting',deletion_requested_at=?,cleanup_error=NULL,updated_at=? WHERE id=?", (now, now, notebook_id))
        connection.execute("INSERT INTO cleanup_operations VALUES(?,?,?,?,?,?,?,?,?)", (operation_id, "notebook", notebook_id, "queued", "waiting_for_jobs", None, now, now, None))
    return operation_id


def request_notebook_deletes(notebook_ids: list[str]) -> list[dict[str, Any]]:
    """Queue deletions for active notebooks and report each accepted target."""
    unique_ids = list(dict.fromkeys(notebook_ids))
    results: list[dict[str, Any]] = []
    with DB.transaction() as connection:
        for notebook_id in unique_ids:
            row = connection.execute("SELECT state FROM notebooks WHERE id=?", (notebook_id,)).fetchone()
            if not row:
                results.append({"id": notebook_id, "accepted": False, "error": "Notebook 不存在"})
                continue
            if row["state"] != "active":
                results.append({"id": notebook_id, "accepted": False, "error": "Notebook 当前状态不可删除"})
                continue
            operation_id, now = new_id("cleanup"), utc_now()
            connection.execute(
                "UPDATE jobs SET cancel_requested=1,state=CASE WHEN state='running' THEN 'cancelling' ELSE state END,updated_at=? WHERE notebook_id=? AND state IN ('queued','running','cancelling')",
                (now, notebook_id),
            )
            connection.execute(
                "UPDATE notebooks SET state='deleting',deletion_requested_at=?,cleanup_error=NULL,updated_at=? WHERE id=?",
                (now, now, notebook_id),
            )
            connection.execute(
                "INSERT INTO cleanup_operations VALUES(?,?,?,?,?,?,?,?,?)",
                (operation_id, "notebook", notebook_id, "queued", "waiting_for_jobs", None, now, now, None),
            )
            results.append({"id": notebook_id, "accepted": True, "operation_id": operation_id})
    return results


def process_cleanup_operations() -> None:
    operations = DB.fetchall("SELECT * FROM cleanup_operations WHERE state IN ('queued','running') ORDER BY created_at")
    for operation in operations:
        notebook_id = operation["target_id"]
        active = DB.fetchone("SELECT COUNT(*) AS count FROM jobs WHERE notebook_id=? AND state IN ('running','cancelling')", (notebook_id,))
        if active and active["count"]:
            continue
        now = utc_now()
        try:
            DB.execute("UPDATE cleanup_operations SET state='running',phase='files',error=NULL,updated_at=? WHERE id=?", (now, operation["id"]))
            sources = DB.fetchall("SELECT id,blob_path FROM sources WHERE notebook_id=?", (notebook_id,))
            media = DB.fetchall("SELECT media_path FROM artifacts WHERE notebook_id=? AND media_path IS NOT NULL", (notebook_id,))
            jobs = DB.fetchall("SELECT id FROM jobs WHERE notebook_id=?", (notebook_id,))
            resources = DB.fetchall("SELECT relative_path FROM local_resources WHERE notebook_id=? AND state='active'", (notebook_id,))
            for resource in resources:
                safe_remove(resource["relative_path"])
            for source in sources:
                safe_remove(source.get("blob_path")); safe_remove(str((PATHS.renders / source["id"]).relative_to(PATHS.root)))
            for item in media:
                safe_remove(item.get("media_path"))
            for job in jobs:
                safe_remove(str((PATHS.job_work / job["id"]).relative_to(PATHS.root)))
            with DB.transaction() as connection:
                for source in sources:
                    connection.execute("DELETE FROM chunks_fts WHERE source_id=?", (source["id"],))
                connection.execute("DELETE FROM local_resources WHERE notebook_id=?", (notebook_id,))
                connection.execute("DELETE FROM job_events WHERE job_id IN (SELECT id FROM jobs WHERE notebook_id=?)", (notebook_id,))
                connection.execute("DELETE FROM jobs WHERE notebook_id=?", (notebook_id,))
                connection.execute("DELETE FROM notebooks WHERE id=?", (notebook_id,))
                connection.execute("UPDATE cleanup_operations SET state='complete',phase='complete',updated_at=?,finished_at=? WHERE id=?", (now, now, operation["id"]))
        except Exception as exc:
            DB.execute("UPDATE cleanup_operations SET state='failed',error=?,updated_at=? WHERE id=?", (str(exc), now, operation["id"]))
            DB.execute("UPDATE notebooks SET state='cleanup_failed',cleanup_error=?,updated_at=? WHERE id=?", (str(exc), now, notebook_id))


def backfill_resources() -> None:
    for source in DB.fetchall("SELECT id,notebook_id,blob_path FROM sources"):
        path = PATHS.root / source["blob_path"]
        parent_relative = str(path.parent.relative_to(PATHS.root))
        existing_file = DB.fetchone("SELECT id FROM local_resources WHERE relative_path=?", (source["blob_path"],))
        if path.exists() and existing_file:
            DB.execute("UPDATE local_resources SET relative_path=?,size_bytes=? WHERE id=?", (parent_relative, allocated_bytes(path.parent), existing_file["id"]))
        elif path.exists() and not DB.fetchone("SELECT 1 FROM local_resources WHERE relative_path=?", (parent_relative,)):
            register_resource("notebook", source["notebook_id"], source["notebook_id"], "source", path.parent)
    for artifact in DB.fetchall("SELECT id,notebook_id,media_path FROM artifacts WHERE media_path IS NOT NULL"):
        path = PATHS.root / artifact["media_path"]
        owner_path = path.parent
        relative = str(owner_path.relative_to(PATHS.root))
        if owner_path.exists() and not DB.fetchone("SELECT 1 FROM local_resources WHERE relative_path=?", (relative,)):
            register_resource("notebook", artifact["notebook_id"], artifact["notebook_id"], "artifact", owner_path)


def reconcile_legacy_podcast_temps() -> int:
    """Remove only legacy per-turn copies after a final podcast.wav is present."""
    freed = 0
    for directory in PATHS.artifacts.glob("podcast_*"):
        if not directory.is_dir() or not (directory / "podcast.wav").is_file():
            continue
        normalized = directory / "normalized"
        if normalized.exists():
            freed += safe_remove(str(normalized.relative_to(PATHS.root)))
        for item in directory.iterdir():
            if item.is_file() and re.fullmatch(r"\d{3,4}\.wav", item.name):
                freed += safe_remove(str(item.relative_to(PATHS.root)))
    return freed
