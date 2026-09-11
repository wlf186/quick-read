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
    sent_by_stage: dict[str, set[str]] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    preparation: dict[str, Any] = field(default_factory=dict)


CURRENT: ContextVar[GenerationContext | None] = ContextVar("generation_context", default=None)


def evidence_cost(row: dict[str, Any]) -> int:
    locator = row.get("locator") or json.loads(row.get("locator_json") or "{}")
    location = str(locator.get("section") or "") + str(locator.get("sheet") or "") + str(locator.get("cell_range") or "")
    return estimate_text_tokens(str(row["content"])) + estimate_text_tokens(str(row.get("filename") or "") + location) + 80


def context_material(kind: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    """Use identical eligibility and source costs in preview and execution."""
    from . import retrieval
    if kind == "podcast":
        eligible = retrieval.podcast_candidates(retrieval.context_candidates(rows))
    elif kind in {"summary", "chat"}:
        eligible = retrieval.context_candidates(rows)
    else:
        eligible = [row for row in rows if retrieval.is_quality_chunk(row)]
    costs = [evidence_cost(row) if kind in {"summary", "chat", "podcast"}
             else estimate_text_tokens(row["content"]) + 80 for row in eligible]
    return eligible, costs


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
            "sources": details, "stages": {stage: {"sent_segments": len(ids),
                "source_tokens": sum(estimate_text_tokens(row["content"]) for row in state.rows if row["id"] in ids)}
                for stage, ids in state.sent_by_stage.items()},
            "audit": state.audit, "preparation": {**state.preparation, "accepted_notes": len(state.notes)},
            "meaning": "原文选材与调用覆盖，不代表重要信息完整覆盖；预读不等于最终综合读取"}


def mark_selected(rows: list[dict[str, Any]]) -> None:
    state = current()
    if state:
        state.selected.update(str(row["id"]) for row in rows)


def mark_sent(build: Any, stage: str = "generation") -> None:
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
            state.sent_by_stage.setdefault(stage, set()).add(identifier)
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
    state.preparation = {"planned_batches": state.plan.preparation_batches, "attempted_batches": 0, "failed_batches": 0}
    podcast = state.plan.kind == "podcast"
    preparation_start = state.trace.accounted_tokens
    rejected: dict[str, int] = {}
    state.preparation["rejected"] = rejected
    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1
    from .context_budget import estimate_messages_tokens, pack_items
    from .providers import PromptBuild, budgeted_chat
    batches: list[list[dict[str, Any]]] = [[]]
    cost = 0
    # Spread sources across batches; a long first book must not consume all preparation slots.
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row['source_id'], []).append(row)
    ordered = [values[i] for i in range(max(map(len, buckets.values()), default=0)) for values in buckets.values() if i < len(values)]
    for row in ordered:
        size = estimate_text_tokens(row["content"]) + 80
        if cost + size > state.plan.batch_evidence_tokens and batches[-1]:
            batches.append([])
            cost = 0
        batches[-1].append(row)
        cost += size
    by_id = {row["id"]: row for row in rows}
    for batch in batches[:state.plan.preparation_batches]:
        if state.cancel_check and state.cancel_check():
            raise RuntimeError("任务已取消")
        from .delivery import CURRENT as DELIVERY
        delivery = DELIVERY.get()
        reserved_calls = 2 + int(delivery is None or not delivery.recoveries)
        if state.trace.request_limit is not None and state.trace.requests >= state.trace.request_limit - reserved_calls:
            state.preparation["stop_reason"] = "request_limit"
            break
        output = state.plan.preparation_output_tokens or state.plan.output_tokens
        remaining = state.plan.preparation_token_limit - (state.trace.accounted_tokens - preparation_start)
        if podcast and remaining < output + 1024:
            state.preparation["stop_reason"] = "preparation_budget"
            break
        before_notes = len(state.notes)
        def build(budget: Any) -> Any:
            note_slots = max(1, min(len(batch), (budget.output_tokens - 512) // 400))
            prefix = (f"Read every supplied region. Extract up to {note_slots} distinct central claims, "
                      "including qualifications, exceptions and disagreements. Do not invent facts. "
                      "Organize each note as the question or premise, the mechanism that answers it, and its necessary qualification. Prefer central arguments and source examples over isolated remarks. "
                      "Represent every supplied source and spread notes across its regions; do not merge different authors' claims. "
                      f"Write claims in {'English' if language == 'en' else 'Simplified Chinese'}. "
                      "Preserve who makes each statement and whether it is a hypothesis, fictional dialogue, example, or established assertion. "
                      'Return JSON {"notes":[{"chunk_id":"exact id","claim":"","qualification":"","statement_kind":"","quote":"verbatim supporting substring"}]}. '
                      "Keep each claim under 100 characters and the qualification under 60 characters. "
                      "The quote must be copied exactly from that chunk, 30 to 160 characters; do not use ellipses.\n")
            input_limit = min(budget.input_tokens, remaining - output) if podcast else budget.input_tokens
            packed = pack_items(batch, lambda row: f"[{row['id']}|{row['source_id']}|{region(row)}] {row['content']}",
                                max(0, input_limit - estimate_messages_tokens([{"role": "user", "content": prefix}]) - 16),
                                group_key=lambda row: row["source_id"])
            return PromptBuild([{"role": "user", "content": prefix + "\n".join(packed.texts)}],
                               total_segments=packed.total, included_segments=len(packed.items),
                               truncated_segments=packed.truncated, metadata={"chunks": packed.items})
        try:
            state.preparation["attempted_batches"] += 1
            from .podcast_contracts import notes_schema
            result = await budgeted_chat(build, json_mode=True, max_tokens=output,
                                         **({"response_schema": notes_schema()} if podcast else {}),
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
                identifier = note.get("chunk_id") if isinstance(note, dict) else None
                # Some models omit the storage prefix. Resolve only an exact visible
                # identifier; the verbatim quote check below remains mandatory.
                if podcast and isinstance(identifier, str) and identifier not in visible and f"chunk_{identifier}" in visible:
                    identifier = f"chunk_{identifier}"
                if not isinstance(identifier, str) or identifier not in visible:
                    reject("invalid_source_or_shape")
                    continue
                original = by_id[identifier]
                quote = re.sub(r"\s+", " ", str(note.get("quote") or "")).strip()
                if len(quote) < 30 or quote not in re.sub(r"\s+", " ", original["content"]):
                    reject("quote_not_found_or_short")
                    continue
                if str(note.get("claim") or "").strip():
                    state.notes.append({"chunk_id": original["id"], "claim": str(note["claim"])[:600],
                                        "statement_kind": str(note.get("statement_kind") or "unspecified")[:120],
                                        "qualification": str(note.get("qualification") or "")[:360], "quote": quote[:1200]})
                else:
                    reject("empty_claim")
        except Exception:
            state.preparation["failed_batches"] += 1
            state.trace.mark_fallback()
        if podcast and len(state.notes) == before_notes:
            state.preparation["stop_reason"] = "no_accepted_notes"
            break
    state.preparation["accounted_tokens"] = state.trace.accounted_tokens - preparation_start
    if state.cancel_check and state.cancel_check():
        raise RuntimeError("任务已取消")
    return state.notes


def synthesis_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select original passages, retaining source fallbacks when notes are missing."""
    state = current()
    if not state or not state.preparation_attempted:
        return rows
    noted = {note['chunk_id'] for note in state.notes}
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row['source_id'], []).append(row)
    for values in buckets.values():
        values.sort(key=lambda row: row['id'] not in noted)
    ordered = [values[i] for i in range(max(map(len, buckets.values()), default=0)) for values in buckets.values() if i < len(values)]
    remaining = state.plan.final_evidence_tokens or state.plan.evidence_tokens
    selected = []
    for row in ordered:
        cost = evidence_cost(row)
        if cost <= remaining:
            selected.append(row)
            remaining -= cost
    return selected or rows[:1]


def synthesis_hints(rows: list[dict[str, Any]], labels: list[str]) -> str:
    """Attach bounded notes only when their original passage accompanies synthesis."""
    state = current()
    if not state or not state.notes:
        return ""
    visible = {row['id']: label for row, label in zip(rows, labels)}
    remaining = min(state.plan.output_tokens, (state.plan.final_evidence_tokens or state.plan.evidence_tokens) // 4)
    lines = []
    for note in state.notes:
        if note['chunk_id'] not in visible:
            continue
        line = json.dumps({'citation': visible[note['chunk_id']], 'claim': note['claim'],
                           'qualification': note.get('qualification', ''), 'statement_kind': note.get('statement_kind', '')}, ensure_ascii=False)
        cost = estimate_text_tokens(line) + 2
        if cost <= remaining:
            lines.append(line)
            remaining -= cost
    return ("\nPre-reading hints (fallible; verify attribution, conditions and every claim against the ORIGINAL passages below):\n" + "\n".join(lines) + "\n") if lines else ""


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
                    row["filename"] = source["filename"]
                    rows.append(row)
            sources = list(sources_by_id.values())
            if not sources:
                return await function(*args, **kwargs)
            rows, costs = context_material(actual_kind, rows)
            if not rows:
                return await function(*args, **kwargs)
            actual_kind = values.get("kind", kind)
            plan = plan_context(TokenLimits.from_provider(provider), actual_kind,
                                material_tokens=sum(costs), segment_tokens=math.ceil(sum(costs) / len(costs)),
                                count=values.get("count", 10), minutes=payload.get("minutes") or 20,
                                source_count=len(sources), broad_query=bool(re.search(r"每份|各份|分别|所有|对比|比较|\b(?:each|all|across|compare)\b", str(values.get('query') or ''), re.I)))
            if actual_kind not in {"summary", "chat"} and not any(cost <= plan.evidence_tokens for cost in costs):
                # Existing builders can clip individual passages and preserve
                # local fallbacks when the window cannot hold one full chunk.
                return await function(*args, **kwargs)
            trace = ContextUsage(total_token_limit=plan.total_token_limit, request_limit=80)
            from .delivery import CURRENT as DELIVERY
            if actual_kind == "podcast" and DELIVERY.get():
                from .context_budget import reserve_podcast_audit, high_reasoning
                reserve_podcast_audit(trace, TokenLimits.from_provider(provider), reasoning=high_reasoning(provider))
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
