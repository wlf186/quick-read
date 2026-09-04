from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from typing import Any, Callable

from .database import DB, json_dump, json_load, new_id, utc_now
from .documents import chunk_blocks, parse_document
from .paths import PATHS
from .context_budget import ContextUsage, PromptBudget, estimate_messages_tokens, pack_items, structured_output_tokens, truncate_text_tokens
from .providers import PromptBuild, ProviderError, budgeted_chat, describe_image, provider_by_id
from .retrieval import EMBEDDINGS, retrieve, select_quality_evidence
from .observability import Reporter
from .languages import resolve_output_language


def source_scope(notebook_id: str, requested: list[str] | None) -> list[str]:
    if requested is None:
        rows = DB.fetchall("SELECT id FROM sources WHERE notebook_id=? AND selected=1 AND state='ready' ORDER BY created_at", (notebook_id,))
    else:
        if not requested:
            return []
        marks = ",".join("?" for _ in requested)
        rows = DB.fetchall(f"SELECT id FROM sources WHERE notebook_id=? AND state='ready' AND id IN ({marks})", (notebook_id, *requested))
    return [row["id"] for row in rows]


def scope_hash(source_ids: list[str]) -> str:
    revisions = []
    for source_id in sorted(source_ids):
        row = DB.fetchone("SELECT revision_id FROM sources WHERE id=?", (source_id,))
        if row:
            revisions.append(row["revision_id"])
    return hashlib.sha256("|".join(revisions).encode()).hexdigest()


async def ingest_source(
    source_id: str,
    progress: Callable[[str, float], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    image_policy: dict[str, Any] | None = None,
    image_provider_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    source = DB.fetchone("SELECT * FROM sources WHERE id=?", (source_id,))
    if not source:
        raise ValueError("source not found")
    DB.execute("UPDATE sources SET state='processing',error=NULL,updated_at=? WHERE id=?", (utc_now(), source_id))
    path = PATHS.root / source["blob_path"]
    if progress:
        progress("解析文档结构", 0.08)
    parsed = await asyncio.to_thread(parse_document, path, source_id)
    if cancel_check and cancel_check():
        raise RuntimeError("任务已取消")
    policy = image_policy or {"mode": "process", "processors": ["vlm", "main", "ocr"]}
    processors = list(policy.get("processors") or []) if policy.get("mode") == "process" else []
    visual_blocks = [block for block in parsed.blocks if block.visual_needed and block.image_path]
    successful_visuals = 0
    processor_counts: dict[str, int] = {}
    visual_rows: list[dict[str, Any]] = []
    derived_blocks = []
    for visual_index, block in enumerate(visual_blocks, start=1):
        attempts: list[dict[str, Any]] = []
        if cancel_check and cancel_check():
            raise RuntimeError("任务已取消")
        if progress:
            progress(f"视觉解析 {visual_index}/{len(visual_blocks)}", 0.12 + 0.38 * (visual_index - 1) / max(1, len(visual_blocks)))
        description, used = "", None
        for processor in processors:
            try:
                if processor in {"vlm", "main"}:
                    provider_id = (image_provider_ids or {}).get(processor)
                    provider = provider_by_id(provider_id) if provider_id else None
                    if not provider:
                        attempts.append({"processor": processor, "status": "unavailable"})
                        continue
                    if not provider.get("capabilities", {}).get("vision"):
                        attempts.append({"processor": processor, "status": "unsupported"})
                        continue
                    description = (await describe_image(PATHS.root / block.image_path, block.text, provider)).strip()
                else:
                    from rapidocr import RapidOCR
                    result = await asyncio.to_thread(RapidOCR(), str(PATHS.root / block.image_path))
                    lines = [item if isinstance(item, str) else getattr(item, "txt", "") for item in (getattr(result, "txts", []) or [])]
                    description = "\n".join(line.strip() for line in lines if line and line.strip())
                if len(description) >= 2:
                    used = processor
                    attempts.append({"processor": processor, "status": "success"})
                    break
                attempts.append({"processor": processor, "status": "empty"})
            except Exception as exc:
                attempts.append({"processor": processor, "status": "failed", "error": str(exc)[:240]})
                description = ""
        visual_id = new_id("visual")
        locator = dict(block.locator)
        locator.update({"visual_id": visual_id, "derived_visual": True})
        if description:
            label = "本地 OCR" if used == "ocr" else f"{str(used).upper()} 视觉解析"
            derived_blocks.append(type(block)(f"[{label}]\n{description}", locator))
            successful_visuals += 1
            processor_counts[used or "unknown"] = processor_counts.get(used or "unknown", 0) + 1
        image_bytes = (PATHS.root / block.image_path).read_bytes()
        visual_rows.append({
            "id": visual_id, "ordinal": visual_index, "kind": str(block.locator.get("kind") or "image"),
            "locator": locator, "path": block.image_path, "status": "ready" if description else "skipped" if not processors else "unresolved",
            "processor": used, "description": description, "attempts": attempts,
            "checksum": hashlib.sha256(image_bytes).hexdigest(),
        })
    parsed.blocks.extend(derived_blocks)
    chunks = chunk_blocks(parsed.blocks)
    vectors: list[list[float]] = []
    batch_size = 32
    total_batches = math.ceil(len(chunks) / batch_size) if chunks else 0
    for start in range(0, len(chunks), batch_size):
        if cancel_check and cancel_check():
            raise RuntimeError("任务已取消")
        batch_number = start // batch_size + 1
        if progress:
            progress(f"生成本地向量索引 {batch_number}/{total_batches}", 0.58 + 0.34 * (batch_number - 1) / max(1, total_batches))
        vectors.extend(await asyncio.to_thread(EMBEDDINGS.encode, [chunk.text for chunk in chunks[start:start + batch_size]]))
    now = utc_now()
    with DB.transaction() as connection:
        connection.execute("DELETE FROM chunks_fts WHERE source_id=?", (source_id,))
        connection.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
        connection.execute("DELETE FROM source_visuals WHERE source_id=?", (source_id,))
        for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
            chunk_id = new_id("chunk")
            checksum = hashlib.sha256(chunk.text.encode()).hexdigest()
            connection.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)", (chunk_id, source_id, source["revision_id"], ordinal, chunk.text, json_dump(chunk.locator), json_dump(vector), checksum, now))
            connection.execute("INSERT INTO chunks_fts(chunk_id,source_id,content) VALUES(?,?,?)", (chunk_id, source_id, chunk.text))
        for item in visual_rows:
            connection.execute(
                """INSERT INTO source_visuals
                (id,source_id,ordinal,kind,locator_json,relative_path,mime_type,width,height,status,processor,description,attempts_json,checksum,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (item["id"], source_id, item["ordinal"], item["kind"], json_dump(item["locator"]), item["path"], "image/png", None, None,
                 item["status"], item["processor"], item["description"], json_dump(item["attempts"]), item["checksum"], now, now),
            )
        metadata = dict(parsed.metadata)
        metadata.update({
            "vision_pages": successful_visuals, "vision_candidates": len(visual_blocks), "vision_budget": len(visual_blocks),
            "chunk_count": len(chunks), "embedding_mode": EMBEDDINGS.mode,
            "image_processing": {"policy": policy, "processed": successful_visuals, "processors": processor_counts},
            "indexable": bool(chunks),
        })
        connection.execute("UPDATE sources SET state='ready',selected=?,page_count=?,parser=?,preview_path=?,metadata_json=?,updated_at=? WHERE id=?", (int(bool(chunks)), parsed.page_count, parsed.parser, parsed.preview_path, json_dump(metadata), now, source_id))
    return {"source_id": source_id, "chunks": len(chunks), "vision_pages": successful_visuals, "visuals": len(visual_blocks)}


def _context(chunks: list[dict[str, Any]], labels: list[str] | None = None) -> tuple[str, list[dict[str, Any]]]:
    lines, citations = [], []
    for index, chunk in enumerate(chunks, start=1):
        label = labels[index - 1] if labels else f"S{index}"
        source = DB.fetchone("SELECT filename FROM sources WHERE id=?", (chunk["source_id"],)) or {"filename": "未知来源"}
        locator = chunk["locator"]
        loc = f"第{locator['page']}页" if locator.get("page") else f"第{locator['slide']}张" if locator.get("slide") else locator.get("section") or "文档位置"
        lines.append(f"[{label}] {source['filename']} · {loc}\n{chunk['content']}")
        citations.append({"id": label, "source_id": chunk["source_id"], "chunk_id": chunk["id"], "filename": source["filename"], "locator": locator, "quote": chunk["content"][:260]})
    return "\n\n".join(lines), citations


def _context_entry(chunk: dict[str, Any], label: str) -> str:
    source = DB.fetchone("SELECT filename FROM sources WHERE id=?", (chunk["source_id"],)) or {"filename": "未知来源"}
    locator = chunk["locator"]
    loc = f"第{locator['page']}页" if locator.get("page") else f"第{locator['slide']}张" if locator.get("slide") else locator.get("section") or "文档位置"
    return f"[{label}] {source['filename']} · {loc}\n{chunk['content']}"


def _context_citation(chunk: dict[str, Any], label: str) -> dict[str, Any]:
    source = DB.fetchone("SELECT filename FROM sources WHERE id=?", (chunk["source_id"],)) or {"filename": "未知来源"}
    return {
        "id": label,
        "source_id": chunk["source_id"],
        "chunk_id": chunk["id"],
        "filename": source["filename"],
        "locator": chunk["locator"],
        "quote": chunk["content"][:260],
    }


def _evidence_prompt_build(
    budget: PromptBudget,
    *,
    chunks: list[dict[str, Any]],
    labels: list[str],
    prefix: str,
    suffix: str = "",
    system: str | None = None,
    ensure_source_coverage: bool = False,
) -> PromptBuild:
    empty_messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prefix + suffix}]
    available = max(0, budget.input_tokens - estimate_messages_tokens(empty_messages, budget.image_tokens_per_image) - 8)
    labeled = list(zip(labels, chunks))
    packed = pack_items(
        labeled,
        lambda item: _context_entry(item[1], item[0]),
        available,
        group_key=(lambda item: str(item[1]["source_id"])) if ensure_source_coverage else None,
    )
    context = "\n\n".join(packed.texts)
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prefix + context + suffix}]
    citations = [_context_citation(chunk, label) for label, chunk in packed.items]
    return PromptBuild(
        messages,
        total_segments=packed.total,
        included_segments=len(packed.items),
        truncated_segments=packed.truncated,
        metadata={"citations": citations, "chunks": [chunk for _, chunk in packed.items], "labels": [label for label, _ in packed.items], "context": context},
    )


def _grounding_issues(answer: str, valid_ids: set[str], prefix: str = "S") -> list[str]:
    issues: list[str] = []
    for marker in re.findall(rf"\[({prefix}\d+)\]", answer):
        if marker not in valid_ids:
            issues.append(f"未知引用 {marker}")
    for line in answer.splitlines():
        plain = re.sub(r"^[#>*\-\d.\s]+", "", line).strip()
        if len(plain) < 16 or plain.endswith(("：", ":")):
            continue
        if any(term in plain for term in ("无法从资料", "资料不足", "cannot be confirmed", "not enough information")):
            continue
        if not re.search(rf"\[{prefix}\d+\]", line):
            issues.append("缺少引用: " + plain[:80])
    return issues


def _grounding_quality(answer: str, valid_ids: set[str], prefix: str = "S") -> tuple[bool, float]:
    markers = re.findall(rf"\[({prefix}\d+)\]", answer)
    unknown = any(marker not in valid_ids for marker in markers)
    claims = cited = 0
    for line in answer.splitlines():
        plain = re.sub(r"^[#>*\-\d.\s]+", "", line).strip()
        if len(plain) < 16 or plain.endswith(("：", ":")):
            continue
        claims += 1
        cited += bool(re.search(rf"\[{prefix}\d+\]", line))
    return (not unknown and bool(markers), cited / max(1, claims))


def _remove_unsupported_lines(answer: str, valid_ids: set[str], prefix: str = "S") -> str:
    cleaned: list[str] = []
    for line in answer.splitlines():
        line = re.sub(rf"\[({prefix}\d+)\]", lambda match: match.group(0) if match.group(1) in valid_ids else "", line)
        plain = re.sub(r"^[#>*\-\d.\s]+", "", line).strip()
        if len(plain) >= 16 and not plain.endswith(("：", ":")) and not re.search(rf"\[{prefix}\d+\]", line):
            if not any(term in plain for term in ("无法从资料", "资料不足", "cannot be confirmed", "not enough information")):
                continue
        cleaned.append(line)
    return "\n".join(cleaned).strip() or "资料不足，无法形成可核验回答。"


def _extractive_fallback(citations: list[dict[str, Any]], language: str) -> str:
    if language == "en":
        heading = "The model could not produce a fully grounded synthesis. Here are the most relevant source excerpts:"
    else:
        heading = "模型未能形成完全符合引用约束的综合回答。以下是资料中最相关、可直接核验的原文摘录："
    representatives: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for item in citations:
        if item["source_id"] not in seen_sources:
            representatives.append(item)
            seen_sources.add(item["source_id"])
    for item in citations:
        if item not in representatives:
            representatives.append(item)
        if len(representatives) >= max(5, len(seen_sources)):
            break
    return heading + "\n\n" + "\n\n".join(f"[{item['id']}] {item['quote']}" for item in representatives)


def _is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(term in lowered for term in ("无法从资料", "资料不足", "无法确认", "cannot be confirmed", "not enough information", "not provided in the"))


async def grounded_generate(notebook_id: str, instruction: str, query: str, source_ids: list[str] | None, language: str, max_tokens: int = 1800) -> dict[str, Any]:
    ids = source_scope(notebook_id, source_ids)
    if not ids:
        raise ValueError("当前范围没有已就绪的文档")
    chunks = retrieve(notebook_id, query, ids, limit=max(12, min(20, len(ids) * 4)), ensure_source_coverage=len(ids) > 1)
    if not chunks:
        raise ValueError("当前范围没有可检索的内容")
    language_rule = "使用中文" if language == "zh-CN" else "Use English" if language == "en" else "跟随用户问题及资料的主要语言"
    prompt_prefix = f"""你是严格依据资料的研究助手。{language_rule}。
规则：只允许使用下方资料；每个事实陈述后必须标注一个或多个 [S1] 格式引用；资料不足就明确说无法从资料确认；禁止使用外部知识；不要编造引用。
任务：{instruction}
问题/主题：{query}

资料：
"""
    labels = [f"S{index}" for index in range(1, len(chunks) + 1)]
    trace = ContextUsage()
    try:
        generated = await budgeted_chat(
            lambda budget: _evidence_prompt_build(
                budget,
                chunks=chunks,
                labels=labels,
                prefix=prompt_prefix,
                system="Ground every claim in supplied sources.",
                ensure_source_coverage=len(ids) > 1,
            ),
            max_tokens=max_tokens,
            minimum_output_tokens=256,
            trace=trace,
        )
        answer = generated.content
        citations = list(generated.build.metadata["citations"])
        context = str(generated.build.metadata["context"])
        valid_ids = {citation["id"] for citation in citations}
        issues = _grounding_issues(answer, valid_ids)
        if issues:
            repair_prefix = f"""仅依据原资料重写下方回答。输出 3 到 6 个完整、简洁的句子，不要标题、编号或项目符号。每句话末尾必须紧跟一个或多个已有 [S数字] 引用。删除无法支持的陈述，不得把引用集中放到末尾，不得创造新引用。仅输出重写后的回答。
问题：{query}
问题列表：{'；'.join(issues[:12])}
原回答：
"""

            def repair_build(budget: PromptBudget) -> PromptBuild:
                answer_budget = max(64, budget.input_tokens // 4)
                clipped_answer, answer_clipped = truncate_text_tokens(answer, answer_budget)
                return_build = _evidence_prompt_build(
                    budget,
                    chunks=list(generated.build.metadata["chunks"]),
                    labels=list(generated.build.metadata["labels"]),
                    prefix=repair_prefix + clipped_answer + "\n原资料：\n",
                )
                return_build.truncated_segments += int(answer_clipped)
                return return_build

            repaired = await budgeted_chat(repair_build, max_tokens=max_tokens, minimum_output_tokens=256, trace=trace)
            answer = repaired.content
            citations = list(repaired.build.metadata["citations"])
            valid_ids = {citation["id"] for citation in citations}
            context = str(repaired.build.metadata["context"])
        remaining = _grounding_issues(answer, valid_ids)
        used_sources = {citation["source_id"] for citation in citations if citation["id"] in set(re.findall(r"\[(S\d+)\]", answer))}
        coverage_requested = len(ids) > 1 and bool(re.search(r"每份|各份|分别|四份|所有文档|each (?:source|document)|all (?:sources|documents)", query, re.I))
        if remaining and _is_refusal(answer):
            answer = "资料不足，无法从已选文档确认该问题。" if language != "en" else "The selected sources do not provide enough information to confirm this."
            degraded = False
        elif remaining or (coverage_requested and used_sources != set(ids)):
            answer = _extractive_fallback(citations, language)
            degraded = True
        else:
            degraded = False
    except Exception:
        trace.mark_fallback()
        context, citations = _context(chunks[: min(3, len(chunks))])
        answer = _extractive_fallback(citations, language)
        degraded = True
    used = {item for item in re.findall(r"\[(S\d+)\]", answer)}
    valid = [citation for citation in citations if citation["id"] in used]
    if not valid and not _is_refusal(answer):
        valid = citations[:3]
        if valid:
            answer += "\n\n" + " ".join(f"[{item['id']}]" for item in valid)
    return {
        "content": answer,
        "citations": valid,
        "scope_hash": scope_hash(ids),
        "source_ids": ids,
        "degraded": degraded,
        "context_usage": trace.as_dict(),
    }


def _evenly_spaced(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    if limit <= 1:
        return rows[:1]
    indexes = {round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)}
    return [row for index, row in enumerate(rows) if index in indexes]


async def _hierarchical_summary(notebook_id: str, ids: list[str], language: str, reporter: Reporter | None = None) -> dict[str, Any]:
    representatives = select_quality_evidence(notebook_id, ids, limit=min(36, max(24, len(ids) * 12)))
    if not representatives:
        raise ValueError("当前范围没有可摘要的内容")
    labels = [f"S{index}" for index in range(1, len(representatives) + 1)]
    citations_by_id = {item["id"]: item for item in _context(representatives, labels)[1]}
    trace = ContextUsage()
    trace.request_limit = 2
    trace.total_token_limit = 24_000

    def parse_points(raw: str, valid_labels: set[str]) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return []
        points: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in parsed.get("points") or []:
            if not isinstance(value, dict):
                continue
            claim = re.sub(r"\s+", " ", str(value.get("claim") or "")).strip()
            why = re.sub(r"\s+", " ", str(value.get("why_it_matters") or "")).strip()
            refs = list(dict.fromkeys(str(item) for item in value.get("citations") or [] if str(item) in valid_labels))[:3]
            key = re.sub(r"\W+", "", claim).lower()
            if len(claim) < 24 or len(why) < 16 or not refs or key in seen:
                continue
            seen.add(key)
            points.append(
                {
                    "claim": claim[:600],
                    "why_it_matters": why[:600],
                    "qualification": re.sub(r"\s+", " ", str(value.get("qualification") or "")).strip()[:360],
                    "citations": refs,
                }
            )
        return points[:10]

    language_rule = "自然简体中文" if language != "en" else "natural English"
    prefix = f"""你是严谨的研究编辑。只依据资料，用{language_rule}提炼 6–10 个相互独立、覆盖全文主线的高信息密度要点。不要复述封面、版权、目录、书目或索引。每点只能引用真正支持该点的 1–3 个编号；不得给每点附整批编号。仅输出 JSON：{{"points":[{{"claim":"完整核心判断","why_it_matters":"为何重要或如何作用","qualification":"资料中的限制或空字符串","citations":["S1"]}}]}}。\n资料：\n"""
    points: list[dict[str, Any]] = []
    valid_labels: set[str] = set()
    if reporter:
        reporter.update("summarize", "一次性综合全文要点", 0.30, current=0, total=1, unit="次")
    try:
        generated = await budgeted_chat(
            lambda budget: _evidence_prompt_build(
                budget, chunks=representatives, labels=labels, prefix=prefix, ensure_source_coverage=len(ids) > 1
            ),
            json_mode=True,
            max_tokens=structured_output_tokens(2200),
            minimum_output_tokens=700,
            trace=trace,
            stage="summary",
        )
        valid_labels = set(generated.build.metadata["labels"])
        points = parse_points(generated.content, valid_labels)
        if len(points) < 6:
            repair_prefix = f"""上次摘要只有 {len(points)} 个有效要点。只依据资料补足到 6–10 个，避免与现有要点重复，并保持每点 1–3 个精确引用。现有有效要点：{json.dumps(points, ensure_ascii=False)}。仅输出同一 JSON points 结构。\n资料：\n"""
            repaired = await budgeted_chat(
                lambda budget: _evidence_prompt_build(
                    budget, chunks=representatives, labels=labels, prefix=repair_prefix, ensure_source_coverage=len(ids) > 1
                ),
                json_mode=True,
                max_tokens=structured_output_tokens(1600),
                minimum_output_tokens=500,
                trace=trace,
                stage="summary_repair",
            )
            extra = parse_points(repaired.content, set(repaired.build.metadata["labels"]))
            known = {re.sub(r"\W+", "", item["claim"]).lower() for item in points}
            points.extend(item for item in extra if re.sub(r"\W+", "", item["claim"]).lower() not in known)
            points = points[:10]
    except Exception:
        trace.mark_fallback()
    degraded = len(points) < 6
    if degraded and not points:
        points = [
            {"claim": str(chunk["content"])[:300], "why_it_matters": "可直接核验的资料摘录" if language != "en" else "A directly verifiable source excerpt", "qualification": "", "citations": [label]}
            for chunk, label in zip(representatives[:6], labels[:6])
        ]
        valid_labels = set(labels[:6])
    if degraded:
        trace.mark_fallback()
    heading = "## Evidence-bound summary" if language == "en" else "## 可追溯摘要"
    lines = []
    for point in points:
        qualification = f" {point['qualification']}" if point["qualification"] else ""
        markers = " ".join(f"[{label}]" for label in point["citations"])
        lines.append(f"- {point['claim']} — {point['why_it_matters']}{qualification} {markers}")
    answer = heading + "\n\n" + "\n".join(lines)
    used_labels = list(dict.fromkeys(label for point in points for label in point["citations"]))
    output_citations = [citations_by_id[label] for label in used_labels if label in citations_by_id]
    return {"version": 2, "content": answer, "points": points, "citations": output_citations, "scope_hash": scope_hash(ids), "source_ids": ids, "degraded": degraded, "context_usage": trace.as_dict()}


async def make_summary(notebook_id: str, source_ids: list[str] | None, language: str, job_id: str | None = None) -> dict[str, Any]:
    ids = source_scope(notebook_id, source_ids)
    if not ids:
        raise ValueError("当前范围没有已就绪的文档")
    language, language_selection = resolve_output_language(DB, ids, language)
    reporter = Reporter(job_id) if job_id else None
    if reporter:
        reporter.update("collect", "构建全文证据采样", 0.06, current=0, total=len(ids), unit="份")
    result = await _hierarchical_summary(notebook_id, ids, language, reporter)
    if reporter:
        reporter.update("persist", "保存可追溯摘要", 0.94, current=1, total=1, unit="项")
    suffix = job_id.removeprefix("job_") if job_id else None
    summary_id = f"summary_{suffix}" if suffix else new_id("summary")
    artifact_id, now = (f"artifact_{suffix}" if suffix else new_id("artifact")), utc_now()
    DB.execute("INSERT OR REPLACE INTO summaries VALUES(?,?,?,?,?,?)", (summary_id, notebook_id, result["scope_hash"], result["content"], json_dump(result["citations"]), now))
    DB.execute("INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (artifact_id, notebook_id, "summary", "资料摘要", json_dump(ids), language, "ready", json_dump({"version": result["version"], "content": result["content"], "points": result["points"], "degraded": result["degraded"], "context_usage": result["context_usage"], "language_selection": language_selection}), json_dump(result["citations"]), None, now, now))
    result["language"] = language
    result["language_selection"] = language_selection
    result["id"] = summary_id
    result["artifact_id"] = artifact_id
    return result


async def make_structured(notebook_id: str, kind: str, count: int, source_ids: list[str] | None, language: str, difficulty: str = "mixed", job_id: str | None = None) -> dict[str, Any]:
    ids = source_scope(notebook_id, source_ids)
    if not ids:
        raise ValueError("当前范围没有已就绪的文档")
    target_chunks = min(24, max(10, count))
    per_source = max(1, math.ceil(target_chunks / len(ids)))
    chunks: list[dict[str, Any]] = []
    for source_id in ids:
        rows = DB.fetchall("SELECT * FROM chunks WHERE source_id=? ORDER BY ordinal", (source_id,))
        for row in _evenly_spaced(rows, per_source):
            row["locator"] = json_load(row.get("locator_json"), {})
            chunks.append(row)
    _, citations = _context(chunks)
    label_by_chunk = {citation["chunk_id"]: citation["id"] for citation in citations}
    buckets = {source_id: [chunk for chunk in chunks if chunk["source_id"] == source_id] for source_id in ids}
    interleaved: list[dict[str, Any]] = []
    while any(buckets.values()):
        for source_id in ids:
            if buckets[source_id]:
                interleaved.append(buckets[source_id].pop(0))
    total_batches = (count + 4) // 5
    reporter = Reporter(job_id) if job_id else None
    items: list[dict[str, Any]] = []
    degraded = False
    trace = ContextUsage()
    citation_by_label = {citation["id"]: citation for citation in citations}
    while len(items) < count:
        batch_count = min(5, count - len(items))
        start = len(items) + 1
        batch_index = len(items) // 5
        if reporter:
            if reporter.cancelled():
                raise RuntimeError("任务已取消")
            reporter.update("generate", f"生成并核验第 {batch_index + 1}/{total_batches} 批", 0.10 + 0.78 * batch_index / max(1, total_batches), current=batch_index + 1, total=total_batches, unit="批")
        window = interleaved[batch_index::total_batches] or interleaved
        window = window[:12]
        window_labels = [label_by_chunk[chunk["id"]] for chunk in window]
        citation_suffix = " ".join(f"[{label}]" for label in window_labels)
        if kind == "quiz":
            schema = '{"items":[{"question":"...","options":["A","B","C","D"],"answer":0,"explanation":"..."}]}'
            task = f"生成恰好{batch_count}道{difficulty}难度单选题，四个互斥选项，answer为0-3。"
        else:
            schema = '{"items":[{"front":"...","back":"..."}]}'
            task = f"生成恰好{batch_count}张不重复的高质量闪卡。"
        accepted: list[dict[str, Any]] = []
        if kind == "quiz":
            for index in range(batch_count):
                target = window[index % len(window)]
                label = label_by_chunk[target["id"]]
                citation = citation_by_label[label]
                locator = citation["locator"]
                location = f"第{locator['page']}页" if locator.get("page") else f"第{locator['slide']}张" if locator.get("slide") else locator.get("section") or "该引用位置"
                correct = re.sub(r"\s+", " ", citation["quote"]).strip()[:140]
                distractors: list[str] = []
                for other in window:
                    if other["id"] == target["id"]:
                        continue
                    option = re.sub(r"\s+", " ", str(other["content"])).strip()[:140]
                    if option and option != correct and option not in distractors:
                        distractors.append(option)
                    if len(distractors) == 3:
                        break
                while len(distractors) < 3:
                    distractors.append(f"资料的其它位置未给出这一表述（干扰项 {len(distractors) + 1}）。")
                answer = (start + index - 1) % 4
                options = distractors[:]
                options.insert(answer, correct)
                question = f"以下哪项内容直接出现在《{citation['filename']}》{location}？"
                accepted.append({"question": question, "options": options, "answer": answer, "explanation": f"正确选项直接摘自该位置原文。 [{label}]", "citations": [label]})
        for _ in ([] if kind == "quiz" else range(3)):
            try:
                def flashcard_build(budget: PromptBudget) -> PromptBuild:
                    requested_count = max(1, min(batch_count, (budget.output_tokens - 100) // 180))
                    build = _evidence_prompt_build(
                        budget,
                        chunks=window,
                        labels=window_labels,
                        prefix=f"只根据资料生成内容，不要输出引用标记。生成恰好{requested_count}张不重复的高质量闪卡。仅输出合法JSON：{schema}\n资料：\n",
                        ensure_source_coverage=len(ids) > 1,
                    )
                    build.metadata["requested_count"] = requested_count
                    return build

                generated = await budgeted_chat(
                    flashcard_build,
                    json_mode=True,
                    max_tokens=1800,
                    minimum_output_tokens=256,
                    trace=trace,
                )
                raw = generated.content
                used_labels = list(generated.build.metadata["labels"])
                citation_suffix = " ".join(f"[{label}]" for label in used_labels)
                candidate = json.loads(raw[raw.find("{"):raw.rfind("}") + 1]).get("items", [])
            except Exception:
                candidate = []
            for item in candidate:
                if not isinstance(item, dict):
                    continue
                identity_field = "question" if kind == "quiz" else "front"
                identity = re.sub(r"\W+", "", str(item.get(identity_field, ""))).lower()
                existing = {re.sub(r"\W+", "", str(entry.get(identity_field, ""))).lower() for entry in (*items, *accepted)}
                if not identity or identity in existing:
                    continue
                if not isinstance(item.get("front"), str) or not isinstance(item.get("back"), str):
                    continue
                item["back"] = re.sub(r"\[S\d+\]", "", item["back"]).strip() + " " + citation_suffix
                item["citations"] = used_labels
                accepted.append(item)
                if len(accepted) == batch_count:
                    break
            if len(accepted) == batch_count:
                break
        if len(accepted) != batch_count:
            degraded = True
            trace.mark_fallback()
            for index in range(len(accepted), batch_count):
                label = window_labels[index % len(window_labels)]
                citation = citation_by_label[label]
                excerpt = re.sub(r"\s+", " ", citation["quote"]).strip()[:180]
                if kind == "quiz":
                    correct = excerpt or "资料包含该位置的明确原文。"
                    distractors = ["资料明确否定了该表述。", "资料没有讨论该主题。", "该表述仅来自系统外部知识。"]
                    answer = (start + index - 1) % 4
                    options = distractors[:]
                    options.insert(answer, correct)
                    accepted.append({"question": f"以下哪一项是资料在该引用位置明确陈述的内容？（{start + index}）", "options": options, "answer": answer, "explanation": f"正确选项直接摘自该位置的原文。 [{label}]", "citations": [label]})
                else:
                    accepted.append({"front": f"资料要点 {start + index}：{citation['filename']} 的该引用位置包含什么核心内容？", "back": f"{excerpt} [{label}]", "citations": [label]})
        for offset, item in enumerate(accepted, start=start):
            item["id"] = f"q{offset}" if kind == "quiz" else f"c{offset}"
            items.append(item)
    payload = {"items": items, "degraded": degraded, "context_usage": trace.as_dict()}
    if reporter:
        reporter.update("persist", "保存题库" if kind == "quiz" else "保存闪卡", 0.94, current=count, total=count, unit="题" if kind == "quiz" else "张")
    artifact_id = f"artifact_{job_id.removeprefix('job_')}" if job_id else new_id("artifact")
    now = utc_now()
    DB.execute("INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (artifact_id, notebook_id, kind, "单选题库" if kind == "quiz" else "闪卡组", json_dump(ids), language, "ready", json_dump(payload), json_dump(citations), None, now, now))
    return {"id": artifact_id, "type": kind, "payload": payload, "citations": citations}
