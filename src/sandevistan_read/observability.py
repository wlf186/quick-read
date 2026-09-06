from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from .database import DB, json_dump, json_load, utc_now


LABELS = {
    "ingest": "文档解析", "summary": "生成摘要", "quiz": "Quiz 题库",
    "flashcard": "Flashcard 闪卡", "podcast": "双人音频播客",
}


def _seconds(started: str | None, finished: str | None = None) -> float:
    if not started:
        return 0.0
    try:
        start = datetime.fromisoformat(started)
        end = datetime.fromisoformat(finished) if finished else datetime.now(UTC)
        return max(0.0, (end - start).total_seconds())
    except (TypeError, ValueError):
        return 0.0


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * ratio
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


class Reporter:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def cancelled(self) -> bool:
        row = DB.fetchone("SELECT cancel_requested FROM jobs WHERE id=?", (self.job_id,)) or {}
        return bool(row.get("cancel_requested"))

    def update(
        self, stage_code: str, stage: str, progress: float, *, state: str | None = None,
        current: float | None = None, total: float | None = None, unit: str | None = None,
        basis: str = "observed", detail: dict[str, Any] | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if connection is None:
            with DB.transaction() as transaction:
                self.update(stage_code, stage, progress, state=state, current=current, total=total,
                            unit=unit, basis=basis, detail=detail, connection=transaction)
            return
        row = connection.execute("SELECT progress,state,started_at FROM jobs WHERE id=?", (self.job_id,)).fetchone()
        if not row or row["state"] in {"complete", "failed", "cancelled"}:
            return
        next_state = state or row["state"]
        # A late progress callback cannot undo a cancellation request.
        if row["state"] == "cancelling" and next_state != "cancelled":
            return
        progress = min(1.0, max(float(row["progress"]), float(progress)))
        now = utc_now()
        processing = _seconds(row["started_at"], now if next_state in {"complete", "failed", "cancelled"} else None)
        connection.execute(
            """UPDATE jobs SET state=?,stage_code=?,stage=?,progress=?,stage_progress=?,progress_basis=?,
            stage_current=?,stage_total=?,stage_unit=?,activity_json=?,processing_seconds=?,updated_at=? WHERE id=?""",
            (next_state, stage_code, stage, progress, progress, basis, current, total, unit,
             json_dump(detail or {}), processing, now, self.job_id),
        )
        connection.execute(
            """INSERT INTO job_events(job_id,state,stage_code,stage,progress,stage_current,stage_total,stage_unit,detail_json,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (self.job_id, next_state, stage_code, stage, progress, current, total, unit, json_dump(detail or {}), now),
        )


def estimate(job: dict[str, Any], queue_position: int = 0) -> dict[str, Any]:
    samples = DB.fetchall(
        "SELECT processing_seconds,workload_json,execution_profile_json FROM jobs WHERE kind=? AND state='complete' AND processing_seconds>0 ORDER BY finished_at DESC LIMIT 100",
        (job["kind"],),
    )
    profile = job.get("execution_profile") or json_load(job.get("execution_profile_json"), {})
    workload = job.get("workload") or json_load(job.get("workload_json"), {})
    cohort = [item for item in samples if json_load(item.get("execution_profile_json"), {}) == profile and json_load(item.get("workload_json"), {}).get("bucket") == workload.get("bucket")]
    chosen = cohort if len(cohort) >= 5 else samples
    durations = [float(item["processing_seconds"]) for item in chosen]
    if len(durations) < 5:
        return {"status": "learning", "sample_count": len(durations), "queue_position": queue_position}
    p10, median, p90 = (_percentile(durations, q) for q in (0.1, 0.5, 0.9))
    progress = max(0.01, float(job.get("progress") or 0.0))
    elapsed = _seconds(job.get("started_at")) if job.get("state") in {"running", "cancelling"} else 0.0
    total_low = max(p10, elapsed / progress if elapsed else p10)
    total_high = max(p90, elapsed / progress if elapsed else p90)
    remaining_low = max(0.0, total_low - elapsed)
    remaining_high = max(0.0, total_high - elapsed)
    running = (DB.fetchone("SELECT COUNT(*) AS count FROM jobs WHERE state IN ('running','cancelling')") or {"count": 0})["count"]
    queue_wait = (max(0, queue_position - 1) + (1 if queue_position and running else 0)) * median
    now = datetime.now(UTC)
    start_at = now + timedelta(seconds=queue_wait)
    completion_low = now + timedelta(seconds=remaining_low + queue_wait)
    completion_high = now + timedelta(seconds=remaining_high + queue_wait)
    return {
        "status": "ready", "sample_count": len(durations),
        "confidence": "high" if len(durations) >= 50 else "medium" if len(durations) >= 20 else "low",
        "queue_position": queue_position,
        "start_in_seconds": round(queue_wait),
        "estimated_start_at": start_at.isoformat(),
        "remaining_seconds": round((remaining_low + remaining_high) / 2 + queue_wait),
        "remaining_range": [round(remaining_low + queue_wait), round(remaining_high + queue_wait)],
        "completion_range": [completion_low.isoformat(), completion_high.isoformat()],
    }


def present_job(row: dict[str, Any], queue_position: int = 0) -> dict[str, Any]:
    result = dict(row)
    for field in ("payload_json", "result_json", "activity_json", "workload_json", "execution_profile_json"):
        result[field.removesuffix("_json")] = json_load(result.pop(field, None), {})
    result["eta"] = estimate(result, queue_position)
    result["display_name"] = result.get("display_name") or LABELS.get(result["kind"], result["kind"])
    return result
