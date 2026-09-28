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
    audit_reason: str | None = None
    quota_denial: dict[str, Any] | None = None
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
    reason = CURRENT.get().audit_reason if CURRENT.get() else None
    if method in {"local_excerpt", "not_applicable"}:
        reason = "not_applicable"
    elif reviewed < total and not reason:
        reason = "sampled" if reviewed else "invalid_response" if method == "model_sample" else "unknown"
    status = "not_applicable" if reason == "not_applicable" else "complete" if total and reviewed == total else "partial" if reviewed else "unavailable"
    return {"review_status": status, "reason_code": reason,
            "reason": (CURRENT.get().quota_denial["message"] if reason == "weekly_quota_exhausted" and CURRENT.get() and CURRENT.get().quota_denial else REVIEW_REASONS.get(reason, "") if reason else ""),
            "quota_denial": CURRENT.get().quota_denial if CURRENT.get() else None,
            "version": 1, "level": level, "method": method, "total_units": total,
            "reviewed_units": reviewed, "supported_units": supported, "issues": issues,
            "notice": "自动检查仅覆盖标明的内容，不保证事实正确；未检查部分不视为通过。"}


REVIEW_REASONS = {
    "budget_exhausted": "本次用量或调用额度不足，未完成原文核查。可缩小范围，或调整预算后重新审校。",
    "provider_unavailable": "审校服务未能完成请求。可以稍后仅重新审校，已有内容已保留。",
    "invalid_response": "模型没有返回可验证的审校结论。可以查看引用原文，或仅重新审校。",
    "output_truncated": "审校输出被截断；仅统计已完整返回并验证的结论。",
    "sampled": "本次采用抽样核查；未检查的内容不算通过。",
    "not_applicable": "本次为资料不足说明或原文摘录，未进行额外事实核查。",
    "unknown": "未记录未完成原因，不能据此判断内容有误。",
}


def audit_reason(reason: str | BaseException) -> None:
    state = CURRENT.get()
    if state is None:
        return
    if isinstance(reason, BaseException) and getattr(reason, "detail", {}).get("code") == "quota_exhausted":
        state.quota_denial = reason.detail
        state.audit_reason = "weekly_quota_exhausted"
        return
    if isinstance(reason, BaseException):
        message = str(reason)
        reason = ("budget_exhausted" if any(word in message for word in ("预算", "上限", "额度"))
                  else "invalid_response" if isinstance(reason, (ValueError, TypeError))
                  else "provider_unavailable")
    state.audit_reason = reason
