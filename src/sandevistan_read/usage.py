"""Durable request accounting. No prompts, responses or credentials are stored."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from .context_budget import estimate_messages_tokens
from .database import Database, json_dump, json_load, new_id, utc_now


@dataclass
class Run:
    db: Database
    id: str
    token_limit: int | None = None
    kind: str = "chat"
    job_id: str | None = None
    quota_denial: dict[str, Any] | None = None


CURRENT: ContextVar[Run | None] = ContextVar("usage_run", default=None)
ACCOUNTING_DB: ContextVar[Database | None] = ContextVar("accounting_db", default=None)
STAGE: ContextVar[str] = ContextVar("usage_stage", default="generation")


@contextmanager
def running(db: Database, notebook_id: str, kind: str, *, job_id: str | None = None,
            target_id: str | None = None, token_limit: int | None = None) -> Iterator[Run]:
    run = Run(db, new_id("run"), token_limit, kind, job_id)
    db.execute("INSERT INTO generation_runs(id,notebook_id,kind,job_id,target_id,state,token_limit,created_at) VALUES(?,?,?,?,?,'running',?,?)",
               (run.id, notebook_id, kind, job_id, target_id, token_limit, utc_now()))
    token = CURRENT.set(run)
    state = "complete"
    try:
        yield run
    except BaseException:
        state = "interrupted"
        raise
    finally:
        if job_id and (db.fetchone("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)) or {}).get("cancel_requested"):
            state = "cancelled"
        db.execute("UPDATE generation_runs SET state=?,finished_at=? WHERE id=?", (state, utc_now(), run.id))
        CURRENT.reset(token)


def attach(run: Run, target_id: str) -> None:
    run.db.execute("UPDATE generation_runs SET target_id=? WHERE id=?", (target_id, run.id))


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


async def post(client: Any, url: str, *, provider: dict[str, Any], payload: dict[str, Any],
               headers: dict[str, str]) -> Any:
    """Record every actual HTTP attempt, including compatibility retries."""
    run = CURRENT.get()
    db = run.db if run else ACCOUNTING_DB.get()
    if db is None:
        return await client.post(url, json=payload, headers=headers)
    from .context_budget import TokenLimits
    estimated = estimate_messages_tokens(payload.get("messages", []), TokenLimits.from_provider(provider).image_tokens_per_image)
    output = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or (payload.get("options") or {}).get("num_predict") or 0)
    call_id = new_id("call")
    # Transactions serialize reservations made by concurrent child tasks.
    from .weekly_budget import reserve as reserve_weekly
    with db.transaction() as conn:
        try:
            reserve_weekly(conn, call_id=call_id, provider=provider, estimated=estimated, output=output, created_at=utc_now())
        except RuntimeError as exc:
            if run and getattr(exc, "detail", {}).get("code") == "quota_exhausted":
                run.quota_denial = exc.detail
            raise
        if run and run.job_id:
            spent = conn.execute("SELECT COALESCE(SUM(c.accounted_tokens),0) FROM provider_calls c JOIN generation_runs r ON r.id=c.run_id WHERE r.job_id=?", (run.job_id,)).fetchone()[0]
        elif run:
            spent = conn.execute("SELECT COALESCE(SUM(accounted_tokens),0) FROM provider_calls WHERE run_id=?", (run.id,)).fetchone()[0]
        if run:
            # Keep a quarter of a user-specified budget for the single final audit.
            audit_started = conn.execute("SELECT 1 FROM provider_calls WHERE run_id=? AND stage LIKE '%audit%' LIMIT 1", (run.id,)).fetchone()
            reserve = int(run.token_limit * .25) if run.token_limit and run.kind not in {"ingest", "review"} and "audit" not in STAGE.get() and not audit_started else 0
            if run.token_limit is not None and spent + estimated + output + reserve > run.token_limit:
                raise RuntimeError("已达到本次自定义用量上限；未发送下一次模型请求")
            controls = {key: payload[key] for key in ("think", "thinking", "reasoning_effort", "max_tokens") if key in payload}
            conn.execute("INSERT INTO provider_calls(id,run_id,provider_id,model,role,stage,state,estimated_input_tokens,output_limit,accounted_tokens,controls_json,created_at) VALUES(?,?,?,?,?,?,'pending',?,?,?,?,?)",
                         (call_id, run.id, provider.get("id"), provider.get("model", ""), provider.get("role", "main"), STAGE.get(), estimated, output, estimated + output, json_dump(controls), utc_now()))
    try:
        response = await client.post(url, json=payload, headers=headers)
        try:
            result = response.json()
            result = result if isinstance(result, dict) else {}
        except ValueError:
            result = {}
        usage = result.get("usage") or {}
        usage = usage if isinstance(usage, dict) else {}
        prompt = _count(result.get("prompt_eval_count")) if provider.get("kind") == "ollama" else _count(usage.get("prompt_tokens"))
        completion = _count(result.get("eval_count")) if provider.get("kind") == "ollama" else _count(usage.get("completion_tokens"))
        details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        reasoning = _count(details.get("reasoning_tokens")) if isinstance(details, dict) else None
        cached = _count(prompt_details.get("cached_tokens")) if isinstance(prompt_details, dict) else None
        if cached is None:
            cached = _count(usage.get("prompt_cache_hit_tokens", usage.get("cached_tokens")))
        accounted = (prompt if prompt is not None else estimated) + (completion if completion is not None else output)
        with db.transaction() as conn:
            conn.execute("UPDATE usage_events SET state=?,prompt_tokens=?,completion_tokens=?,accounted_tokens=? WHERE id=?", ("complete" if response.is_success else "failed", prompt, completion, accounted, call_id))
            if run:
                conn.execute("UPDATE provider_calls SET state=?,prompt_tokens=?,completion_tokens=?,reasoning_tokens=?,cached_tokens=?,accounted_tokens=?,finished_at=? WHERE id=?",
                           ("complete" if response.is_success else "failed", prompt, completion, reasoning, cached, accounted, utc_now(), call_id))
        return response
    except BaseException:
        # An interrupted request might have run remotely: retain its reservation.
        db.execute("UPDATE usage_events SET state='unknown' WHERE id=?", (call_id,))
        if run:
            run.db.execute("UPDATE provider_calls SET state='unknown',finished_at=? WHERE id=?", (utc_now(), call_id))
        raise


def summarize(db: Database, *, run_id: str | None = None, notebook_id: str | None = None,
              target_id: str | None = None, job_id: str | None = None, include_reviews: bool = True) -> dict[str, Any]:
    where, params = [], []
    for key, value in (("id", run_id), ("notebook_id", notebook_id), ("target_id", target_id), ("job_id", job_id)):
        if value is not None:
            if key == "target_id":
                where.append("(r.target_id=? OR r.job_id IN (SELECT job_id FROM generation_runs WHERE target_id=? AND job_id IS NOT NULL))")
                params.extend([value, value])
            else:
                where.append(f"r.{key}=?")
                params.append(value)
    if not include_reviews:
        where.append("r.kind != 'review'")
    condition = " AND ".join(where) or "1=0"
    runs = db.fetchall(f"SELECT r.* FROM generation_runs r WHERE {condition} ORDER BY created_at DESC", tuple(params))
    calls = db.fetchall(f"SELECT c.* FROM provider_calls c JOIN generation_runs r ON r.id=c.run_id WHERE {condition} ORDER BY c.created_at", tuple(params))
    media = db.fetchall(f"SELECT m.stage,m.model,m.state,m.submitted_chars,m.audio_seconds FROM media_calls m JOIN generation_runs r ON r.id=m.run_id WHERE {condition}", tuple(params))
    stages: dict[str, dict[str, int]] = {}
    for call in calls:
        group = stages.setdefault(call["stage"], {"calls": 0, "input_tokens": 0, "output_tokens": 0, "unknown_calls": 0})
        group["calls"] += 1
        group["input_tokens"] += call["prompt_tokens"] or 0
        group["output_tokens"] += call["completion_tokens"] or 0
        group["unknown_calls"] += int(call["prompt_tokens"] is None or call["completion_tokens"] is None)
    missing = sum(c["prompt_tokens"] is None or c["completion_tokens"] is None for c in calls)
    return {"version": 1, "media": media, "recorded": bool(runs), "run_ids": [r["id"] for r in runs],
            "calls": len(calls), "unknown_calls": missing,
            "input_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
            "output_tokens": sum(c["completion_tokens"] or 0 for c in calls),
            "reasoning_tokens": sum(c["reasoning_tokens"] or 0 for c in calls),
            "cached_tokens": sum(c["cached_tokens"] or 0 for c in calls),
            "estimated_input_tokens": sum(c["estimated_input_tokens"] for c in calls if c["prompt_tokens"] is None),
            "accounted_tokens": sum(c["accounted_tokens"] for c in calls), "stages": stages,
            "complete": bool(runs) and not missing and all(r["state"] != "running" for r in runs),
            "scope": "本次运行的语言与视觉模型请求；语音用量单列，历史未记录部分不计入。",
            "requests": [{k: c[k] for k in ("id", "model", "role", "stage", "state", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens")} | {"requested_controls": json_load(c["controls_json"], {})} for c in calls[-100:]], "requests_truncated": len(calls) > 100}


def metered_media(stage: str):
    """Measure submitted text/audio separately from language-model tokens."""
    import inspect
    from functools import wraps
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        async def wrapped(*args, **kwargs):
            run = CURRENT.get()
            if run is None:
                return await function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs).arguments
            provider = bound.get("provider") or {}
            characters = len(bound.get("text") or "") + sum(len(item.get("text") or "") for item in bound.get("items", []))
            seconds = None
            path = bound.get("path")
            if path:
                try:
                    import soundfile
                    seconds = soundfile.info(str(path)).duration
                except (ImportError, RuntimeError, OSError):
                    pass
            identifier = new_id("media_call")
            run.db.execute("INSERT INTO media_calls(id,run_id,stage,model,state,submitted_chars,audio_seconds,created_at) VALUES(?,?,?,?,'pending',?,?,?)", (identifier, run.id, stage, bound.get("model") or provider.get("model", ""), characters, seconds, utc_now()))
            try:
                result = await function(*args, **kwargs)
            except BaseException:
                run.db.execute("UPDATE media_calls SET state='unknown' WHERE id=?", (identifier,))
                raise
            run.db.execute("UPDATE media_calls SET state='complete' WHERE id=?", (identifier,))
            return result
        return wrapped
    return decorate
