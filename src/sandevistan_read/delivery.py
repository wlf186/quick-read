"""Product delivery policy, independent of evidence selection and qualification."""
from __future__ import annotations
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import inspect
import copy
from typing import Any, Callable

@dataclass
class DeliveryBudget:
    audits: int = 0
    recoveries: int = 0
    provider: dict[str, Any] | None = None
    stage_output_tokens: dict[str, int] = field(default_factory=dict)

CURRENT: ContextVar[DeliveryBudget | None] = ContextVar("delivery_budget", default=None)

def claim_recovery() -> bool:
    state = CURRENT.get()
    if state is None:
        return True
    if state.recoveries:
        return False
    state.recoveries += 1
    return True

def claim_audit() -> bool:
    state = CURRENT.get()
    if state is None:
        return True
    if state.audits:
        return False
    state.audits += 1
    return True

def delivery_task(function: Callable) -> Callable:
    @wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        if CURRENT.get() is not None:
            return await function(*args, **kwargs)
        values = inspect.signature(function).bind(*args, **kwargs).arguments
        owner = inspect.unwrap(function).__globals__
        pinned = ((values.get("payload") or {}).get("provider_ids") or {}).get("main")
        lookup = owner.get("provider_by_id") if pinned else owner.get("active_provider")
        if pinned and lookup is None:
            from .providers import provider_by_id
            lookup = provider_by_id
        provider = lookup(pinned if pinned else "main") if lookup else None
        if pinned and (not provider or provider.get("role") != "main"):
            raise RuntimeError("任务绑定的 MAIN Provider 不存在")
        token = CURRENT.set(DeliveryBudget(provider=copy.deepcopy(provider)))
        try:
            result = await function(*args, **kwargs)
            values = inspect.signature(function).bind(*args, **kwargs).arguments
            cancel = values.get("cancel_check")
            db = function.__globals__.get("DB")
            if cancel and cancel() or db and values.get("job_id") and (db.fetchone("SELECT cancel_requested FROM jobs WHERE id=?", (values["job_id"],)) or {}).get("cancel_requested"):
                raise RuntimeError("任务已取消")
            return result
        finally:
            CURRENT.reset(token)
    return wrapped

def assessment(total: int, reviewed: int = 0, issues: list[dict[str, Any]] | None = None,
               *, method: str = "local", supported: int = 0) -> dict[str, Any]:
    issues = issues or []
    reviewed = max(0, min(total, reviewed))
    supported = max(0, min(reviewed, supported))
    level = "needs_review" if any(i.get("severity") == "suspect" for i in issues) else "fair" if issues else "good" if reviewed == total and total else "unrated"
    return {"version": 1, "level": level, "method": method, "total_units": total,
            "reviewed_units": reviewed, "supported_units": supported, "issues": issues,
            "notice": "自动检查仅覆盖标明的内容，不保证事实正确；未检查部分不视为通过。"}
