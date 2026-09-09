"""Local, configuration-bound qualification for adaptive summary and chat."""
from __future__ import annotations

import hashlib
import json
from typing import Any
from itertools import product

from .context_budget import CONTEXT_STRATEGY_VERSION, TokenLimits


def fingerprint(provider: dict[str, Any]) -> str:
    config = {key: value for key, value in (provider.get("config") or {}).items()
              if key != "context_strategy"}
    caps = provider.get("capabilities") or {}
    identity = {"provider_id": provider.get("id"), "kind": provider.get("kind"),
                "base_url": str(provider.get("base_url") or "").rstrip("/"),
                "model": provider.get("model"), "config": config,
                "limits": vars(TokenLimits.from_provider(provider)),
                "revision": caps.get("revision") or caps.get("model_revision"),
                "strategy_version": CONTEXT_STRATEGY_VERSION}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def qualified(provider: dict[str, Any], kind: str) -> bool:
    if kind not in {"summary", "chat"}:
        return False
    from .database import DB, json_load
    row = DB.fetchone("SELECT value_json FROM app_settings WHERE key='context_qualifications'")
    records = json_load(row["value_json"], []) if row else []
    if not isinstance(records, list):
        return False
    key = fingerprint(provider)
    return any(isinstance(record, dict) and record.get("fingerprint") == key
               and record.get("kind") == kind and record.get("passed") is True
               and bool(record.get("report_sha256")) for record in records)


def strategy_reason(provider: dict[str, Any], kind: str) -> str:
    explicit = (provider.get("config") or {}).get("context_strategy")
    if explicit == "conservative":
        return "显式保守配置优先"
    if explicit == "balanced":
        return "显式启用均衡策略；不代表取得质量资格"
    if provider.get("id") and qualified(provider, kind):
        return "当前 Provider、配置与功能已通过对照验收"
    return "当前 Provider、配置与功能尚无匹配的质量资格"


def assess(report: dict[str, Any], provider: dict[str, Any], kind: str) -> dict[str, Any]:
    """Fail closed on incomplete, mixed-revision or regressing paired reviews.

    Reviews must be source-grounded assessments, not keyword/coverage scores.
    The CLI additionally verifies the immutable artifacts referenced by the report.
    This is an observed-sample qualification, not a statistical guarantee.
    """
    reasons: list[str] = []
    if kind not in {"summary", "chat"}:
        reasons.append("Only summary and chat can qualify")
    if report.get("provider_fingerprint") != fingerprint(provider) or report.get("kind") != kind:
        reasons.append("Provider/configuration/feature fingerprint mismatch")
    if report.get("strategy_version") != CONTEXT_STRATEGY_VERSION:
        reasons.append("Strategy revision mismatch")
    if report.get("audit_mode") != "native" or report.get("review_method") != "source_grounded":
        reasons.append("Native behavior and source-grounded review are required")
    for field in ("reference_sha256", "baseline_sha256", "candidate_sha256", "fixture_sha256"):
        value = report.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            reasons.append(f"Missing immutable identity: {field}")
    pairs = report.get("pairs", [])
    expected = set(product(("bitcoin", "geb", "multi"), ("zh-CN", "en"), (1, 2)))
    actual = [(p.get("corpus"), p.get("language"), p.get("repeat")) for p in pairs if isinstance(p, dict)] if isinstance(pairs, list) else []
    if len(actual) != len(expected) or set(actual) != expected:
        reasons.append("Expected exactly 12 distinct paired scenarios")
    totals = {arm: {key: 0 for key in ("missing", "critical", "failed", "degraded")} for arm in ("baseline", "candidate")}
    if not reasons:
        for pair in pairs:
            long = pair["corpus"] != "bitcoin"
            before, after = pair.get("baseline", {}), pair.get("candidate", {})
            for arm, values in (("baseline", before), ("candidate", after)):
                required = ("missing", "critical", "failed", "degraded", "fact_errors", "citation_errors")
                if any(type(values.get(key)) is not int or values[key] < 0 for key in required):
                    reasons.append("Incomplete review metrics")
                    continue
                if values["missing"] > values["critical"]:
                    reasons.append("Invalid omission denominator")
                for key in totals[arm]:
                    if key not in {"missing", "critical"} or long:
                        totals[arm][key] += values[key]
            if before.get("critical") != after.get("critical"):
                reasons.append("The two arms must use the same critical propositions")
            if after.get("fact_errors", 1) or after.get("citation_errors", 1):
                reasons.append("Candidate contains factual or citation errors")
            if not long and after.get("missing", 1) > before.get("missing", 0):
                reasons.append("Short material regressed")
        for key in ("failed", "degraded"):
            if totals["candidate"][key] > totals["baseline"][key]:
                reasons.append(f"Observed {key} results increased")
        old, new = totals["baseline"]["missing"], totals["candidate"]["missing"]
        if not totals["baseline"]["critical"] or old <= 0 or new > old * .8:
            reasons.append("Long-material critical omissions did not decrease by at least 20%")
    return {"passed": not reasons, "reasons": list(dict.fromkeys(reasons)), "totals": totals,
            "limitation": "Finite paired samples; no guarantee for untested documents or configurations"}
