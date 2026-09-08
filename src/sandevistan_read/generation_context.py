"""Task-local context planning, immutable evidence snapshots and coverage."""
from __future__ import annotations

import copy
import inspect
import json
import math
import re
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from functools import wraps
from typing import Any, Callable

from .context_budget import ContextPlan, ContextUsage, TokenLimits, context_strategy, estimate_text_tokens, plan_context


def region(row: dict[str, Any]) -> str:
    locator = row.get("locator") or json.loads(row.get("locator_json") or "{}")
    if locator.get("sheet"):
        return "工作表：" + str(locator["sheet"])
    if locator.get("spine") is not None:
        return "EPUB 区段：" + str(locator["spine"])
    if locator.get("section"):
        return "章节：" + str(locator["section"])
    if locator.get("page") is not None:
        start = max(0, (int(locator["page"]) - 1) // 10) * 10 + 1
        return f"页区间：{start}–{start + 9}"
    start = int(row.get("ordinal") or 0) // 10 * 10
    return f"片段区间：{start + 1}–{start + 10}"


@dataclass
class GenerationContext:
    provider: dict[str, Any]
    plan: ContextPlan
    rows: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    trace: ContextUsage
    selected: set[str] = field(default_factory=set)
    sent: set[str] = field(default_factory=set)
    partial: set[str] = field(default_factory=set)
    notes: list[dict[str, Any]] = field(default_factory=list)
    preparation_attempted: bool = False
    cancel_check: Callable[[], bool] | None = None


CURRENT: ContextVar[GenerationContext | None] = ContextVar("generation_context", default=None)


def current() -> GenerationContext | None:
    return CURRENT.get()


def generation_trace() -> ContextUsage:
    state = current()
    return state.trace if state else ContextUsage()


def restore_trace(trace: ContextUsage, saved: dict[str, Any]) -> None:
    """Restore charged work from a checkpoint without widening its token ceiling."""
    ceiling = trace.total_token_limit
    aliases = {"accounted_tokens": "accounted_total_tokens"}
    for item in fields(trace):
        key = aliases.get(item.name, item.name)
        if key in saved:
            value = saved[key]
            if value is None and isinstance(getattr(trace, item.name), int):
                value = 0
            setattr(trace, item.name, copy.deepcopy(value))
    trace.reserved_tokens = 0
    if ceiling is not None:
        trace.total_token_limit = min(ceiling, trace.total_token_limit or ceiling)


def report(citations: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    state = current()
    if not state:
        return None
    cited = {str(item.get("chunk_id")) for item in citations or [] if item.get("chunk_id")}
    details = []
    for source in state.sources:
        rows = [row for row in state.rows if row["source_id"] == source["id"]]
        groups = list(dict.fromkeys(region(row) for row in rows))
        sent_groups = {region(row) for row in rows if row["id"] in state.sent}
        details.append({"source_id": source["id"], "filename": source["filename"],
                        "revision_id": source["revision_id"], "candidate_segments": len(rows),
                        "selected_segments": sum(row["id"] in state.selected for row in rows),
                        "sent_segments": sum(row["id"] in state.sent for row in rows),
                        "cited_segments": sum(row["id"] in cited for row in rows),
                        "regions_total": len(groups), "regions_sent": len(sent_groups),
                        "unsent_regions": [group for group in groups if group not in sent_groups]})
    return {"plan": state.plan.as_dict(), "candidate_segments": len(state.rows),
            "selected_segments": len(state.selected), "sent_segments": len(state.sent),
            "partially_sent_segments": len(state.partial), "cited_segments": len(cited),
            "sources": details, "meaning": "原文选材与调用覆盖，不代表重要信息完整覆盖"}


def mark_selected(rows: list[dict[str, Any]]) -> None:
    state = current()
    if state:
        state.selected.update(str(row["id"]) for row in rows)


def mark_sent(build: Any) -> None:
    state = current()
    if not state:
        return
    text = "\n".join(str(message.get("content") or "") for message in build.messages)
    candidates = build.metadata.get("chunks") or build.metadata.get("items") or []
    by_id = {row["id"]: row for row in state.rows}
    # Podcast builders often wrap several source cards in one scene item.
    # Match only selected raw text actually present in the sent prompt.
    candidates = list(candidates) + [by_id[key] for key in state.selected if key in by_id]
    normalized = re.sub(r"\s+", " ", text)
    for item in candidates:
        if not isinstance(item, dict):
            continue
        identifier = item.get("chunk_id") or item.get("id")
        row = by_id.get(identifier)
        if not row:
            continue
        content = str(row["content"])
        compact = re.sub(r"\s+", " ", content).strip()
        if compact and compact in normalized:
            state.sent.add(identifier)
            state.partial.discard(identifier)
        elif len(compact) >= 40 and compact[:40] in normalized:
            if identifier not in state.sent:
                state.partial.add(identifier)


async def prepare_evidence(rows: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    """Reduce bounded batches to source-linked notes; quoted text must be exact."""
    state = current()
    if not state or not state.plan.preparation_batches or state.preparation_attempted:
        return state.notes if state else []
    state.preparation_attempted = True
    from .context_budget import estimate_messages_tokens, pack_items
    from .providers import PromptBuild, budgeted_chat
    batches: list[list[dict[str, Any]]] = [[]]
    cost = 0
    for row in rows:
        size = estimate_text_tokens(row["content"]) + 80
        if cost + size > state.plan.batch_evidence_tokens and batches[-1]:
            batches.append([])
            cost = 0
        batches[-1].append(row)
        cost += size
    by_id = {row["id"]: row for row in rows}
    for batch in batches[:4]:
        if state.cancel_check and state.cancel_check():
            raise RuntimeError("任务已取消")
        if state.trace.request_limit is not None and state.trace.requests >= state.trace.request_limit - 4:
            break
        def build(budget: Any) -> Any:
            note_slots = max(1, min(24, (budget.output_tokens - 512) // 400))
            prefix = (f"Read every supplied region. Extract up to {note_slots} distinct central claims, "
                      "including qualifications, exceptions and disagreements. Do not invent facts. "
                      "Prioritize the central argument and mechanisms of each region over isolated anecdotes. "
                      "Represent every supplied source and spread notes across its regions; do not merge different authors' claims. "
                      f"Write claims in {'English' if language == 'en' else 'Simplified Chinese'}. "
                      "Preserve who makes each statement and whether it is a hypothesis, fictional dialogue, example, or established assertion. "
                      'Return JSON {"notes":[{"chunk_id":"exact id","claim":"","qualification":"","statement_kind":"","quote":"verbatim supporting substring"}]}. '
                      "Keep each claim under 100 characters and the qualification under 60 characters. "
                      "The quote must be copied exactly from that chunk, 30 to 160 characters; do not use ellipses.\n")
            packed = pack_items(batch, lambda row: f"[{row['id']}|{row['source_id']}|{region(row)}] {row['content']}",
                                max(0, budget.input_tokens - estimate_messages_tokens([{"role": "user", "content": prefix}]) - 16),
                                group_key=lambda row: row["source_id"])
            return PromptBuild([{"role": "user", "content": prefix + "\n".join(packed.texts)}],
                               total_segments=packed.total, included_segments=len(packed.items),
                               truncated_segments=packed.truncated, metadata={"chunks": packed.items})
        try:
            result = await budgeted_chat(build, json_mode=True, max_tokens=state.plan.output_tokens,
                                         trace=state.trace, stage="context_prepare")
            try:
                parsed = json.loads(result.content[result.content.find("{"):result.content.rfind("}") + 1])
            except (ValueError, json.JSONDecodeError):
                match = re.search(r'"notes"\s*:\s*\[', result.content)
                tail = result.content[match.end():] if match else ""
                complete = []
                while tail.strip():
                    try:
                        note, end = json.JSONDecoder().raw_decode(tail.lstrip())
                    except ValueError:
                        break
                    complete.append(note)
                    tail = tail.lstrip()[end:].lstrip().removeprefix(',')
                parsed = {"notes": complete}
            if not isinstance(parsed, dict):
                parsed = {"notes": []}
            visible = {row["id"] for row in result.build.metadata["chunks"]}
            for note in parsed.get("notes") or []:
                if not isinstance(note, dict) or note.get("chunk_id") not in visible:
                    continue
                original = by_id[note["chunk_id"]]
                quote = re.sub(r"\s+", " ", str(note.get("quote") or "")).strip()
                if len(quote) < 30 or quote not in re.sub(r"\s+", " ", original["content"]):
                    continue
                if str(note.get("claim") or "").strip():
                    state.notes.append({"chunk_id": original["id"], "claim": str(note["claim"])[:600],
                                        "statement_kind": str(note.get("statement_kind") or "unspecified")[:120],
                                        "qualification": str(note.get("qualification") or "")[:360], "quote": quote[:1200]})
        except Exception:
            state.trace.mark_fallback()
    return state.notes


def adaptive_generation(kind: str) -> Callable:
    """Pin provider/evidence for one invocation; strict podcast stays independent."""
    def decorate(function: Callable) -> Callable:
        signature = inspect.signature(function)

        @wraps(function)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            if current():
                return await function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            if kind == "podcast" and "allow_partial" in values and not values["allow_partial"]:
                return await function(*args, **kwargs)
            from . import providers, retrieval
            # Use the function's database and provider hook so isolated fixtures
            # and imported application snapshots never consult production data.
            db = function.__globals__["DB"]
            payload = values.get("payload") or {}
            lookup = function.__globals__.get("provider_by_id", providers.provider_by_id)
            audio_id = (payload.get("provider_ids") or {}).get("audio")
            if kind == "podcast" and values.get("job_id") and audio_id and not lookup(audio_id):
                # Let the job report its bound AUDIO error before resolving MAIN.
                return await function(*args, **kwargs)
            pinned_id = (payload.get("provider_ids") or {}).get("main")
            provider = lookup(pinned_id) if pinned_id else function.__globals__.get("active_provider", providers.active_provider)("main")
            if pinned_id and not provider:
                raise RuntimeError("任务绑定的 MAIN Provider 不存在")
            actual_kind = values.get("kind", kind)
            if not provider or context_strategy(provider, actual_kind) != "balanced":
                return await function(*args, **kwargs)
            ids = values.get("source_ids", values.get("ids", payload.get("source_ids")))
            if ids is not None and not ids:
                return await function(*args, **kwargs)
            scope = "s.selected=1" if ids is None else "s.id IN (" + ",".join("?" for _ in ids) + ")"
            snapshot = db.fetchall(f"""SELECT c.*, s.id AS snapshot_source_id,
                s.filename AS snapshot_filename, s.revision_id AS snapshot_revision,
                s.selected AS snapshot_selected FROM sources s LEFT JOIN chunks c ON c.source_id=s.id
                WHERE s.notebook_id=? AND s.state='ready' AND {scope} ORDER BY s.created_at,s.id,c.ordinal""",
                (values["notebook_id"], *(ids or [])))
            sources_by_id: dict[str, dict[str, Any]] = {}
            rows = []
            for row in snapshot:
                source_id = row.pop("snapshot_source_id")
                source = {"id": source_id, "filename": row.pop("snapshot_filename"),
                          "revision_id": row.pop("snapshot_revision"), "selected": row.pop("snapshot_selected")}
                sources_by_id[source_id] = source
                if row.get("id"):
                    rows.append(row)
            sources = list(sources_by_id.values())
            if not sources:
                return await function(*args, **kwargs)
            rows = [row for row in rows if retrieval.is_quality_chunk(row)]
            if not rows:
                return await function(*args, **kwargs)
            costs = [estimate_text_tokens(row["content"]) + 80 for row in rows]
            actual_kind = values.get("kind", kind)
            plan = plan_context(TokenLimits.from_provider(provider), actual_kind,
                                material_tokens=sum(costs), segment_tokens=math.ceil(sum(costs) / len(costs)),
                                count=values.get("count", 10), minutes=payload.get("minutes") or 20)
            if not any(cost <= plan.evidence_tokens for cost in costs):
                # Existing builders can clip individual passages and preserve
                # local fallbacks when the window cannot hold one full chunk.
                return await function(*args, **kwargs)
            trace = ContextUsage(total_token_limit=plan.total_token_limit, request_limit=80)
            state = GenerationContext(copy.deepcopy(provider), plan, rows, sources, trace)
            state.cancel_check = values.get("cancel_check")
            if values.get("job_id"):
                job_id = values["job_id"]
                state.cancel_check = lambda: bool((db.fetchone("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)) or {}).get("cancel_requested"))
            token = CURRENT.set(state)
            try:
                result = await function(*args, **kwargs)
                if state.cancel_check and state.cancel_check():
                    raise RuntimeError("任务已取消")
                if isinstance(result, dict) and "context_usage" in result:
                    result["context_usage"]["coverage"] = report(result.get("citations"))
                return result
            finally:
                CURRENT.reset(token)
        return wrapped
    return decorate
