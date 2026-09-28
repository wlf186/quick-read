"""Explicit, one-pass review of an existing result against its original chunks."""
from __future__ import annotations

import re
from typing import Any

from .database import Database, json_load
from .delivery import assessment, delivery_task
from .context_budget import ContextUsage


def load_target(db: Database, target_type: str, target_id: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    if target_type == "message":
        row = db.fetchone("SELECT m.*,c.notebook_id FROM messages m JOIN conversations c ON c.id=m.conversation_id WHERE m.id=? AND m.role='assistant'", (target_id,))
        if not row:
            raise ValueError("回答不存在")
        units = [{"claim": part, "citations": re.findall(r"\[(S\d+)\]", part)} for part in row["content"].split("\n\n") if part.strip()]
    else:
        row = db.fetchone("SELECT * FROM artifacts WHERE id=?", (target_id,))
        if not row:
            raise ValueError("产物不存在")
        payload = json_load(row["payload_json"], {})
        if row["type"] == "summary":
            units = payload.get("points") or [{"claim": payload.get("content", ""), "citations": []}]
        elif row["type"] in {"quiz", "flashcard"}:
            units = [{"claim": "\n".join(f"{key}: {item[key]}" for key in ("question", "options", "answer_index", "front", "back", "explanation") if key in item), "citations": item.get("citations", [])} for item in payload.get("items", [])]
        else:
            units = [{"claim": item.get("text", ""), "citations": []} for item in payload.get("turns", [])]
    citations = json_load(row.get("citations_json"), [])
    chunks, labels = [], []
    for citation in citations:
        chunk = db.fetchone("SELECT c.*,s.filename FROM chunks c JOIN sources s ON s.id=c.source_id WHERE c.id=? AND s.notebook_id=? AND c.source_id=? AND c.source_revision_id=s.revision_id", (citation.get("chunk_id"), row["notebook_id"], citation.get("source_id")))
        if not chunk or not citation.get("quote") or citation["quote"] not in chunk["content"]:
            raise ValueError("原始资料修订或引用片段不可用，无法重新审校；请重新生成内容。")
        chunk["locator"] = json_load(chunk.get("locator_json"), {})
        chunks.append(chunk)
        labels.append(citation["id"])
    if not chunks or not units:
        raise ValueError("没有可核对的原始引用或内容，无法重新审校。")
    for unit in units:
        unit.setdefault("why_it_matters", "")
        unit.setdefault("qualification", "")
        if not unit.get("citations"):
            unit["citations"] = labels
    return row["notebook_id"], units, [dict(chunk, review_label=label) for chunk, label in zip(chunks, labels)]


@delivery_task
async def review_units(units: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    from .services import _audit_summary_batch
    verdicts = await _audit_summary_batch(units[:12], chunks, [c["review_label"] for c in chunks], ContextUsage(request_limit=1))
    issues = [{"unit": str(i + 1), "code": "evidence_unconfirmed", "severity": "suspect", "message": "该项原文支持待核实。"} for i, verdict in verdicts.items() if verdict == "unsupported"]
    return assessment(len(units), len(verdicts), issues, method="model_sample", supported=sum(v == "supported" for v in verdicts.values()))
