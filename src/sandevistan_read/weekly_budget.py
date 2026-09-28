"""Instance-wide weekly budgets; independent of deletable document history."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .database import Database, json_dump, json_load, utc_now

DEFAULT_LIMIT = {"tokens": 1_000_000, "calls": 500, "mode": "warn"}


class QuotaExceeded(RuntimeError):
    def __init__(self, scope: str, metric: str, reset_at: str):
        self.detail = {"code": "quota_exhausted", "scope": scope, "metric": metric, "reset_at": reset_at,
                       "message": f"本周{'总' if scope == 'global' else 'Provider '}额度不足（{'token' if metric == 'tokens' else '调用次数'}）；未发送请求。可调整额度，或等待 {reset_at} 重置。"}
        super().__init__(self.detail["message"])


def settings_from(conn: Any) -> dict[str, Any]:
    row = conn.execute("SELECT value_json FROM app_settings WHERE key='usage_budget'").fetchone()
    return json_load(row[0], {}) if row else {"timezone": "UTC", "initialized": False, "global_limit": dict(DEFAULT_LIMIT), "providers": {}}


def settings(db: Database) -> dict[str, Any]:
    with db.transaction() as conn:
        return settings_from(conn)


def save_settings(db: Database, value: dict[str, Any], *, initialize_only: bool = False) -> dict[str, Any]:
    with db.transaction() as conn:
        current = settings_from(conn)
        if initialize_only and current.get("initialized"):
            return current
        if initialize_only:
            value = {**current, "timezone": value["timezone"]}
        value = {**value, "initialized": True}
        conn.execute("INSERT INTO app_settings VALUES('usage_budget',?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at", (json_dump(value), utc_now()))
        return value


def period(zone: str, now: datetime | None = None) -> tuple[str, str]:
    local = (now or datetime.now(UTC)).astimezone(ZoneInfo(zone))
    start = (local - timedelta(days=local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(UTC).isoformat(), (start + timedelta(days=7)).astimezone(UTC).isoformat()


def totals(conn: Any, start: str, end: str, provider_id: str | None = None) -> dict[str, int]:
    clause = " AND provider_id=?" if provider_id is not None else ""
    args = (start, end, provider_id) if provider_id is not None else (start, end)
    row = conn.execute("SELECT COUNT(*) AS calls,COALESCE(SUM(COALESCE(prompt_tokens,0)+COALESCE(completion_tokens,0)),0) AS known_tokens,COALESCE(SUM(accounted_tokens),0) AS occupied_tokens,COALESCE(SUM(prompt_tokens IS NULL OR completion_tokens IS NULL),0) AS unknown_calls FROM usage_events WHERE created_at>=? AND created_at<?" + clause, args).fetchone()
    result = dict(row)
    result["unconfirmed_tokens"] = max(0, result["occupied_tokens"] - result["known_tokens"])
    return result


def reserve(conn: Any, *, call_id: str, provider: dict[str, Any], estimated: int, output: int, created_at: str) -> None:
    config = settings_from(conn)
    start, end = period(config["timezone"], datetime.fromisoformat(created_at))
    provider_id = provider.get("id") or "__connection_test__"
    for scope, limit in (("global", config["global_limit"]), (provider_id, config["providers"].get(provider_id))):
        if not limit or limit["mode"] != "block":
            continue
        spent = totals(conn, start, end, None if scope == "global" else scope)
        for metric, used, additional in (("tokens", spent["occupied_tokens"], estimated + output), ("calls", spent["calls"], 1)):
            if limit.get(metric) is not None and used + additional > limit[metric]:
                raise QuotaExceeded(scope, metric, end)
    conn.execute("INSERT INTO usage_events(id,provider_id,provider_name,kind,created_at,state,accounted_tokens) VALUES(?,?,?,?,?,'pending',?)",
                 (call_id, provider_id, provider.get("name") or provider.get("model") or "连接测试", provider.get("kind", ""), created_at, estimated + output))


def overview(db: Database, now: datetime | None = None) -> dict[str, Any]:
    with db.transaction() as conn:
        config = settings_from(conn)
        start, end = period(config["timezone"], now)
        def view(provider_id: str | None, limit: dict[str, Any] | None) -> dict[str, Any]:
            values = totals(conn, start, end, provider_id)
            values["limit"] = limit
            values["remaining_tokens"] = max(0, limit["tokens"] - values["occupied_tokens"]) if limit and limit.get("tokens") is not None else None
            values["remaining_calls"] = max(0, limit["calls"] - values["calls"]) if limit and limit.get("calls") is not None else None
            return values
        ids = {r[0] for r in conn.execute("SELECT DISTINCT provider_id FROM usage_events WHERE created_at>=? AND created_at<?", (start, end))} | set(config["providers"])
        providers = []
        for identifier in sorted(ids):
            profile = conn.execute("SELECT name,kind FROM provider_profiles WHERE id=?", (identifier,)).fetchone()
            historic = conn.execute("SELECT provider_name AS name,kind FROM usage_events WHERE provider_id=? ORDER BY created_at DESC LIMIT 1", (identifier,)).fetchone()
            identity = dict(profile or historic or {"name": "已删除的 Provider", "kind": ""})
            if identifier == "__connection_test__":
                identity["name"] = "未保存的连接测试"
            providers.append({"id": identifier, **identity, **view(identifier, config["providers"].get(identifier))})
        return {"has_active_work": bool(conn.execute("SELECT EXISTS(SELECT 1 FROM jobs WHERE state IN ('queued','running','cancelling')) OR EXISTS(SELECT 1 FROM generation_runs WHERE state='running') OR EXISTS(SELECT 1 FROM usage_events WHERE state='pending')").fetchone()[0]), "period_start": start, "reset_at": end, "timezone": config["timezone"], "global_usage": view(None, config["global_limit"]), "providers": providers,
                "notice": "仅统计本应用的文字/视觉模型请求，包含本地 Ollama；不是厂商套餐余额或账单。历史仅含已有计量记录，语音另计。"}
