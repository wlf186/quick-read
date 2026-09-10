from __future__ import annotations

from .delivery import delivery_task, assessment, claim_recovery

import asyncio
import hashlib
import json
import math
import re
import threading
import time
from typing import Any, Callable

from .database import DB, json_dump, json_load, new_id, utc_now
from .documents import chunk_blocks, parse_document
from .paths import PATHS
from .context_budget import ContextUsage, PromptBudget, TokenLimits, chat_history_budget, estimate_messages_tokens, pack_items, structured_output_tokens, truncate_text_tokens, plan_context
from .providers import PromptBuild, ProviderError, active_provider, budgeted_chat, describe_image, provider_by_id
from .retrieval import EMBEDDINGS, retrieve, select_quality_evidence
from .observability import Reporter
from .languages import resolve_output_language, text_matches_language
from .generation_context import adaptive_generation, current, generation_trace, prepare_evidence, synthesis_evidence, synthesis_hints, evidence_cost


_OCR_ENGINE: Any = None
_OCR_LOCK = threading.Lock()


def _read_ocr(image_path: str) -> Any:
    """Initialize and run the shared engine entirely off the asyncio event loop."""
    global _OCR_ENGINE
    with _OCR_LOCK:
        if _OCR_ENGINE is None:
            from rapidocr import RapidOCR
            _OCR_ENGINE = RapidOCR()
        return _OCR_ENGINE(image_path)


def source_scope(notebook_id: str, requested: list[str] | None) -> list[str]:
    if current():
        ids = [source["id"] for source in current().sources]
        return ids if requested is None else [identifier for identifier in ids if identifier in requested]
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
        row = next((source for source in current().sources if source["id"] == source_id), None) if current() else DB.fetchone("SELECT revision_id FROM sources WHERE id=?", (source_id,))
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
    started = time.perf_counter()
    source = DB.fetchone("SELECT * FROM sources WHERE id=?", (source_id,))
    if not source:
        raise ValueError("source not found")
    DB.execute("UPDATE sources SET state='processing',error=NULL,updated_at=? WHERE id=?", (utc_now(), source_id))
    path = PATHS.root / source["blob_path"]
    if progress:
        progress("解析文档结构", 0.08)
    parsed = await asyncio.to_thread(parse_document, path, source_id)
    timings = parsed.metadata.setdefault("ingest_timings", {})
    timings["parse_seconds"] = round(time.perf_counter() - started, 4)
    visual_started = time.perf_counter()
    if cancel_check and cancel_check():
        raise RuntimeError("任务已取消")
    policy = image_policy or {"mode": "process", "processors": ["vlm", "main", "ocr"]}
    processors = list(policy.get("processors") or []) if policy.get("mode") == "process" else []
    visual_blocks = [block for block in parsed.blocks if block.visual_needed and block.image_path]
    successful_visuals = 0
    processor_counts: dict[str, int] = {}
    visual_rows: list[dict[str, Any]] = []
    derived_blocks = []
    unavailable: dict[str, str] = {}
    consecutive_failures: dict[str, int] = {}
    for visual_index, block in enumerate(visual_blocks, start=1):
        attempts: list[dict[str, Any]] = []
        if cancel_check and cancel_check():
            raise RuntimeError("任务已取消")
        if progress:
            progress(f"视觉解析 {visual_index}/{len(visual_blocks)}", 0.12 + 0.38 * (visual_index - 1) / max(1, len(visual_blocks)))
        description, used = "", None
        for processor in processors:
            if processor in unavailable:
                attempts.append({"processor": processor, "status": "unavailable", "reason": unavailable[processor]})
                continue
            processor_started = time.perf_counter()
            try:
                if processor in {"vlm", "main"}:
                    provider_id = (image_provider_ids or {}).get(processor)
                    provider = provider_by_id(provider_id) if provider_id else None
                    if not provider:
                        unavailable[processor] = "provider_unavailable"
                        attempts.append({"processor": processor, "status": "unavailable"})
                        continue
                    if not provider.get("capabilities", {}).get("vision"):
                        unavailable[processor] = "vision_unsupported"
                        attempts.append({"processor": processor, "status": "unsupported"})
                        continue
                    description = (await describe_image(PATHS.root / block.image_path, block.text, provider)).strip()
                else:
                    result = await asyncio.to_thread(_read_ocr, str(PATHS.root / block.image_path))
                    lines = [item if isinstance(item, str) else getattr(item, "txt", "") for item in (getattr(result, "txts", []) or [])]
                    description = "\n".join(line.strip() for line in lines if line and line.strip())
                if len(description) >= 2:
                    consecutive_failures[processor] = 0
                    used = processor
                    attempts.append({"processor": processor, "status": "success"})
                    break
                attempts.append({"processor": processor, "status": "empty"})
            except Exception as exc:
                consecutive_failures[processor] = consecutive_failures.get(processor, 0) + 1
                if (isinstance(exc, (ImportError, FileNotFoundError))
                        or isinstance(exc, ProviderError) and getattr(exc, "status", None) in {401, 403, 404, 405}
                        or consecutive_failures[processor] >= 2):
                    unavailable[processor] = type(exc).__name__
                attempts.append({"processor": processor, "status": "failed", "error": str(exc)[:240]})
                description = ""
            finally:
                key = f"{processor}_seconds"
                timings[key] = round(timings.get(key, 0) + time.perf_counter() - processor_started, 4)
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
    timings["visual_seconds"] = round(time.perf_counter() - visual_started, 4)
    indexing_started = time.perf_counter()
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
    timings["index_seconds"] = round(time.perf_counter() - indexing_started, 4)
    persist_started = time.perf_counter()
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
        unresolved = sum(item["status"] == "unresolved" for item in visual_rows)
        if unresolved:
            metadata.setdefault("warnings", []).append({"code": "visuals_unresolved", "count": unresolved,
                "message": "部分图片或图表未完成识别，正文索引已保留。"})
        timings["persist_seconds"] = round(time.perf_counter() - persist_started, 4)
        timings["total_seconds"] = round(time.perf_counter() - started, 4)
        connection.execute("UPDATE sources SET state='ready',selected=?,page_count=?,parser=?,preview_path=?,metadata_json=?,updated_at=? WHERE id=?", (int(bool(chunks)), parsed.page_count, parsed.parser, parsed.preview_path, json_dump(metadata), now, source_id))
    return {"source_id": source_id, "chunks": len(chunks), "vision_pages": successful_visuals, "visuals": len(visual_blocks)}


def _context_source(chunk: dict[str, Any]) -> dict[str, Any]:
    if current():
        return next((source for source in current().sources if source['id'] == chunk['source_id']), {"filename": chunk.get("filename", "未知来源")})
    return DB.fetchone("SELECT filename FROM sources WHERE id=?", (chunk["source_id"],)) or {"filename": "未知来源"}


def _context(chunks: list[dict[str, Any]], labels: list[str] | None = None) -> tuple[str, list[dict[str, Any]]]:
    lines, citations = [], []
    for index, chunk in enumerate(chunks, start=1):
        label = labels[index - 1] if labels else f"S{index}"
        source = _context_source(chunk)
        locator = chunk["locator"]
        loc = f"第{locator['page']}页" if locator.get("page") else f"第{locator['slide']}张" if locator.get("slide") else f"工作表 {locator['sheet']} · {locator.get('cell_range', '')}" if locator.get("sheet") else locator.get("section") or "文档位置"
        lines.append(f"[{label}] {source['filename']} · {loc}\n{chunk['content']}")
        citations.append({"id": label, "source_id": chunk["source_id"], "chunk_id": chunk["id"], "filename": source["filename"], "locator": locator, "quote": chunk["content"][:260]})
    return "\n\n".join(lines), citations


def _context_entry(chunk: dict[str, Any], label: str) -> str:
    source = _context_source(chunk)
    locator = chunk["locator"]
    loc = f"第{locator['page']}页" if locator.get("page") else f"第{locator['slide']}张" if locator.get("slide") else f"工作表 {locator['sheet']} · {locator.get('cell_range', '')}" if locator.get("sheet") else locator.get("section") or "文档位置"
    return f"[{label}] {source['filename']} · {loc}\n{chunk['content']}"


def _context_citation(chunk: dict[str, Any], label: str) -> dict[str, Any]:
    source = _context_source(chunk)
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
    include_notes: bool = False,
) -> PromptBuild:
    empty_messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prefix + suffix}]
    available = max(0, budget.input_tokens - estimate_messages_tokens(empty_messages, budget.image_tokens_per_image) - 8)
    if current() and current().plan.kind in {"summary", "chat"}:
        available = min(available, current().plan.evidence_tokens)
    labeled = list(zip(labels, chunks))
    packed = pack_items(
        labeled,
        lambda item: _context_entry(item[1], item[0]) + (synthesis_hints([item[1]], [item[0]]) if include_notes else ""),
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
    return any(term in lowered for term in ("无法从资料", "资料不足", "无法确认", "资料中没有提供", "资料没有提供", "cannot be confirmed", "not enough information", "not provided in the", "does not provide", "do not provide", "do not contain", "does not contain"))


def normalize_citation_markers(text: str) -> str:
    """Expand grouped labels so every reference is validated and clickable."""
    return re.sub(r"\[(?:S\d+[\s,，、;；]*)+\]", lambda match: " ".join(f"[{label}]" for label in re.findall(r"S\d+", match.group())), text)


def conversation_context(notebook_id: str, conversation_id: str | None, ids: list[str]) -> str:
    """Use complete same-scope exchanges as dialogue context, never as evidence."""
    if not conversation_id:
        return ""
    owner = DB.fetchone("SELECT notebook_id FROM conversations WHERE id=?", (conversation_id,))
    if not owner or owner["notebook_id"] != notebook_id:
        return ""
    rows = DB.fetchall("SELECT role,content,scope_hash FROM messages WHERE conversation_id=? AND state='complete' ORDER BY created_at DESC,rowid DESC LIMIT 25", (conversation_id,))
    expected_scope = scope_hash(ids)
    pairs: list[str] = []
    pending: str | None = None
    for row in reversed(rows):
        if row["role"] == "user":
            pending = str(row["content"])
        elif row["role"] == "assistant" and pending is not None:
            if row["scope_hash"] != expected_scope:
                pairs = []
            else:
                # Citation labels are local to each answer and cannot be reused.
                answer = re.sub(r"\[S\d+\]", "", normalize_citation_markers(str(row["content"])))
                question = pending
                if not current():
                    question, _ = truncate_text_tokens(pending, 160)
                    answer, _ = truncate_text_tokens(answer, 340)
                pair = f"用户：{question}\n助手：{answer}"
                pairs.append(pair)
            pending = None
    return "\n\n".join(pairs[-6:])


def _bounded_dialogue(history: str, token_budget: int) -> tuple[str, bool]:
    pairs = re.split(r"\n\n(?=用户：)", history) if history else []
    if current() and current().plan.kind == "chat":
        retained = []
        for pair in reversed(pairs):
            candidate = "\n\n".join(reversed(retained + [pair]))
            if estimate_messages_tokens([{"role": "user", "content": candidate}]) > token_budget:
                break
            retained.append(pair)
        return "\n\n".join(reversed(retained)), len(retained) != len(pairs)
    packed = pack_items(reversed(pairs), lambda pair: pair, token_budget)
    return "\n\n".join(reversed(packed.texts)), bool(packed.truncated or len(packed.items) < len(pairs))


def _has_formula(text: str) -> bool:
    return bool(re.search(r"[\ue000-\uf8ff]|\\(?:begin|frac|le|ge|ne|eq)|[A-Za-z}\]]\s*[<>=≤≥]", text))


def _cited_answer_lines(answer: str, valid_ids: set[str], language: str, *, omit_formulas: bool = False) -> str:
    retained = []
    for line in answer.splitlines():
        if omit_formulas and _has_formula(line):
            continue
        markers = set(re.findall(r"\[(S\d+)\]", line))
        plain = re.sub(r"\[S\d+\]|[#>*]", "", line).strip()
        if markers and markers <= valid_ids and len(plain) >= 16 and not plain.endswith((":", "：")) and text_matches_language(plain, language):
            retained.append(line)
    return "\n\n".join(retained)


@delivery_task
@adaptive_generation("chat")
async def grounded_generate(notebook_id: str, instruction: str, query: str, source_ids: list[str] | None, language: str, max_tokens: int = 1800, *, conversation_id: str | None = None) -> dict[str, Any]:
    ids = source_scope(notebook_id, source_ids)
    if not ids:
        raise ValueError("当前范围没有已就绪的文档")
    history = conversation_context(notebook_id, conversation_id, ids)
    retrieval_history, _ = truncate_text_tokens(history[-1800:], 600)
    limit = max(12, min(20, len(ids) * 4))
    if current():
        limit = max(1, current().plan.estimated_segments)
        max_tokens = current().plan.output_tokens
    require_all = len(ids) > 1 and bool(re.search(r"每份|各份|分别|所有|\b(?:each|all|across|compare)\b|对比|比较", query, re.I))
    if current() and len(ids) > 1 and history and re.search(r"这些|上述|它们|各自|\b(?:these|those|their)\b", query, re.I):
        require_all = require_all or bool(re.search(r"每份|各份|分别|所有|对比|比较|\b(?:each|all|across|compare)\b", history, re.I))
    if current() and require_all:
        state = current()
        state.plan = plan_context(TokenLimits.from_provider(state.provider), 'chat',
                                  material_tokens=sum(evidence_cost(row) for row in state.rows),
                                  source_count=len(ids), broad_query=True)
        state.trace.total_token_limit = state.plan.total_token_limit
        limit = max(1, state.plan.estimated_segments)
        max_tokens = state.plan.output_tokens
    chunks = retrieve(notebook_id, query, ids, limit=limit, ensure_source_coverage=require_all if current() else len(ids) > 1)
    if history:
        contextual = retrieve(notebook_id, f"{query}\n对话主题：{retrieval_history}", ids, limit=limit, ensure_source_coverage=require_all if current() else len(ids) > 1)
        # History helps resolve pronouns, but must not bury a new, specific query.
        merged = list(chunks[: limit // 2])
        seen = {chunk["id"] for chunk in merged}
        for chunk in contextual + chunks:
            if chunk["id"] not in seen:
                merged.append(chunk)
                seen.add(chunk["id"])
            if len(merged) >= limit:
                break
        chunks = merged
    if not chunks:
        raise ValueError("当前范围没有可检索的内容")
    trace = generation_trace()
    if current() and require_all and current().plan.preparation_batches:
        trace.request_limit = current().plan.preparation_batches + 3
        await prepare_evidence(chunks, language)
        chunks = synthesis_evidence(chunks)
    language_rule = "使用中文" if language == "zh-CN" else "Use English" if language == "en" else "跟随用户问题及资料的主要语言"
    unreadable_math = any(re.search(r"[\ue000-\uf8ff]", str(chunk["content"])) for chunk in chunks)
    formula_rule = "资料提取含无法识别的特殊字体字符，公式可能残缺。禁止复写无法核对的公式或猜测数学符号；依据清楚可读的原文文字解释。若用户需要精确公式，明确说明当前提取不足以确认。" if unreadable_math else ""
    prompt_prefix = f"""你是严格依据资料的研究助手。{language_rule}。
规则：只允许使用下方资料；每个事实陈述后必须标注一个或多个 [S1] 格式引用；资料不足就明确说无法从资料确认；禁止使用外部知识；不要编造引用。
当前问题优先于历史话题；历史仅用于理解指代。若当前问题无法从资料确认，直接说明资料没有提供该信息并停止，不要附加无关的历史结论或“无引用”标记。多个引用分别写成 [S1] [S2]。
先直接回答用户的问题或纠正其假设，再给必要的资料依据。比较前后文时，先确认前文是否已经包含该条件，不要把重述说成新增加的限制。准确保留原文的限定条件、否定词及不等号方向；不要把概率低说成绝不可能。
用户没有要求公式时，优先使用资料中清楚可读的文字解释。资料中的公式若包含乱码或断裂排版，不要猜测或主动复写，改用附近可核验的文字结论。
任务：{instruction}
问题/主题：{query}

{formula_rule}
资料：
"""
    labels = [f"S{index}" for index in range(1, len(chunks) + 1)]
    trace = generation_trace()
    partial_answer = False
    retained_answer = ""
    retained_citations: list[dict[str, Any]] = []
    formula_issue = False

    def build_answer(budget: PromptBudget) -> PromptBuild:
        history_budget = chat_history_budget(budget) if current() else min(2000, budget.input_tokens // 5)
        clipped, truncated = _bounded_dialogue(history, history_budget)
        history_prefix = f"对话历史（仅用于理解指代，不是证据，旧回答可能有错）：\n{clipped}\n\n" if clipped else ""
        built = _evidence_prompt_build(budget, chunks=chunks, labels=labels, prefix=history_prefix + prompt_prefix, system="Ground every claim in supplied sources. Treat dialogue history as context, not factual evidence.", ensure_source_coverage=require_all if current() else len(ids) > 1, include_notes=True)
        built.truncated_segments += int(truncated)
        return built

    try:
        generated = await budgeted_chat(
            build_answer,
            max_tokens=max_tokens,
            minimum_output_tokens=256,
            trace=trace,
        )
        answer = normalize_citation_markers(generated.content)
        if not answer.strip():
            raise ValueError("模型返回空回答")
        citations = list(generated.build.metadata["citations"])
        context = str(generated.build.metadata["context"])
        valid_ids = {citation["id"] for citation in citations}
        retained_answer = _cited_answer_lines(answer, valid_ids, language, omit_formulas=unreadable_math)
        retained_citations = citations
        issues = _grounding_issues(answer, valid_ids)
        if unreadable_math and _has_formula(answer):
            formula_issue = True
            issues.append("原文公式字符提取不完整，只保留原文可读的文字解释，不要复写公式或变量关系")
        if language in {"zh-CN", "en"} and not text_matches_language(answer, language):
            issues.append("输出语言不符合要求")
        if getattr(generated, "finish_reason", None) in {"length", "max_tokens"}:
            issues.append("输出被截断")
        answer = re.sub(r"\[(S\d+)\]", lambda m: m.group(0) if m.group(1) in valid_ids else "", answer)
        degraded = bool(issues)
        partial_answer = False
    except Exception:
        trace.mark_fallback()
        if retained_answer:
            answer, citations = retained_answer, retained_citations
            partial_answer = True
        else:
            context, citations = _context(chunks[: min(3, len(chunks))])
            answer = _extractive_fallback(citations, language)
        degraded = True
    if formula_issue:
        degraded = True
        trace.mark_fallback()
    used = {item for item in re.findall(r"\[(S\d+)\]", answer)}
    valid = [citation for citation in citations if citation["id"] in used]
    local_issues = [{"unit": "answer", "code": "answer_check", "severity": "suspect", "message": issue} for issue in locals().get("issues", [])]
    paragraphs = [part.strip() for part in answer.split("\n\n") if part.strip()]
    audit_points = [{"claim": part, "why_it_matters": "", "qualification": "", "citations": list(dict.fromkeys(re.findall(r"\[(S\d+)\]", part)))} for part in paragraphs[:12]]
    verdicts = {} if _is_refusal(answer) else await _audit_summary_batch(audit_points, chunks, labels, trace)
    for index, value in verdicts.items():
        if value == "unsupported":
            local_issues.append({"unit": f"paragraph_{index + 1}", "code": "evidence_unconfirmed", "severity": "suspect", "message": "该段落的原文支持待核实。"})
    quality_assessment = assessment(len(paragraphs), len(verdicts), local_issues, method="model_sample", supported=sum(v == "supported" for v in verdicts.values()))
    return {
        "content": answer,
        "quality_assessment": quality_assessment,
        "delivery_status": "partial" if degraded else "full",
        "citations": valid,
        "scope_hash": scope_hash(ids),
        "source_ids": ids,
        "degraded": degraded,
        "warnings": ([{"code": "source_formula_unreadable" if formula_issue else "answer_partial" if partial_answer else "extractive_fallback", "stage": "answer", "message": "原文公式提取含无法识别的字符，已保留可读文字解释或原文摘录，请通过原文件核对精确公式。" if formula_issue else "部分回答需要核对，已保留模型输出。" if partial_answer else "回答包含待核实内容或使用了原文摘录，请查看质量说明。"}] if degraded else []) + ([{"code":"retrieval_scope","stage":"answer","message":"本次回答基于检索到的原文片段；资料不足不代表整本书不存在相关信息。"}] if current() and _is_refusal(answer) else []),
        "context_usage": trace.as_dict(),
    }


def _evenly_spaced(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    if limit <= 1:
        return rows[:1]
    indexes = {round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)}
    return [row for index, row in enumerate(rows) if index in indexes]


async def _audit_summary_batch(
    points: list[dict[str, Any]], chunks: list[dict[str, Any]], labels: list[str], trace: ContextUsage,
) -> dict[int, str]:
    """Audit against original evidence, rejecting missing or unverifiable verdicts."""
    requested = {label for point in points for label in point["citations"]}
    evidence = [(label, chunk) for label, chunk in zip(labels, chunks) if label in requested]
    prefix = (
        "逐点核验摘要的 claim、why_it_matters 和 qualification 是否全部得到原文支持。"
        "中间笔记不是证据。核对陈述者、虚构对话/示例/假设与事实的区别、否定、概率、条件及因果方向。"
        "任何部分夸大、限定丢失或原文未显示都不能通过。不要凭外部知识认可。原文 I/my 等第一人称没有明确归属时，不得认可将其说成作者观点的要点。"
        '仅输出 JSON {"verdicts":[{"index":0,"supported":true,"evidence":[{"id":"S1","quote":"原文连续摘录"}]}]}。'
        '每个要点都必须有判定，索引从 0 开始；不支持也返回 supported:false，不要用空数组表示检查完成。'
        f"本次恰好 {len(points)} 个要点，合法 index 仅为 {list(range(len(points)))}；不要把一个要点拆成多个判定。"
        "evidence 必须为对象数组，不能用字符串数组。每个判定附 reason，简述具体不受支持的子句或通过原因。"
        "逐点检查全部引用，但每个通过的要点只返回一段 20–80 字符的原文定位摘录，不复写整段，不加省略号。\n待核验摘要："
        + json.dumps(points, ensure_ascii=False) + "\n原文：\n"
    )
    try:
        result = await budgeted_chat(
            lambda budget: _evidence_prompt_build(budget, chunks=[chunk for _, chunk in evidence],
                labels=[label for label, _ in evidence], prefix=prefix),
            json_mode=True, max_tokens=max(structured_output_tokens(1200 + len(points) * 100),
                current().plan.output_tokens * 2 if current() else 4096),
            minimum_output_tokens=512, trace=trace, stage="summary_grounding_audit",
        )
        try:
            parsed = json.loads(result.content[result.content.find("{"):result.content.rfind("}") + 1])
        except ValueError:
            match = re.search(r'"verdicts"\s*:\s*\[', result.content)
            tail = result.content[match.end():] if match else ""
            verdicts = []
            while tail.strip():
                try:
                    value, end = json.JSONDecoder().raw_decode(tail.lstrip())
                except ValueError:
                    break
                verdicts.append(value)
                tail = tail.lstrip()[end:].lstrip().removeprefix(',')
            parsed = {"verdicts": verdicts}
        visible = {label: re.sub(r"\s+", " ", chunk["content"]).strip()
                   for label, chunk in zip(result.build.metadata["labels"], result.build.metadata["chunks"])}
        # A clipped final passage is not complete evidence of its qualifications.
        if result.build.truncated_segments and result.build.metadata["labels"]:
            visible.pop(result.build.metadata["labels"][-1], None)
        accepted: set[int] = set()
        rejected: set[int] = set()
        for verdict in parsed.get("verdicts", []) if isinstance(parsed, dict) else []:
            if not isinstance(verdict, dict):
                continue
            index = verdict.get("index")
            if type(index) is not int or not 0 <= index < len(points):
                continue
            supported = set()
            for item in verdict.get("evidence", []) if isinstance(verdict.get("evidence"), list) else []:
                if not isinstance(item, dict):
                    continue
                quote = re.sub(r"\s+", " ", str(item.get("quote") or "")).strip()
                label = item.get("id")
                if isinstance(label, str) and label in visible and len(quote) >= 20 and quote in visible[label]:
                    supported.add(label)
            if (verdict.get("supported") is True and set(points[index]["citations"]) <= visible.keys()
                    and set(points[index]["citations"]) & supported):
                accepted.add(index)
            elif verdict.get("supported") is False and set(points[index]["citations"]) <= visible.keys():
                rejected.add(index)
        return {index: "unsupported" if index in rejected else "supported" for index in accepted | rejected}
    except (ProviderError, RuntimeError, ValueError, TypeError):
        trace.mark_fallback()
        return {}


async def _audit_summary_points(
    points: list[dict[str, Any]], chunks: list[dict[str, Any]], labels: list[str], trace: ContextUsage,
) -> list[dict[str, Any]]:
    """One sampled pass; preserve supported, suspect and unreviewed points."""
    groups: dict[str | None, list[int]] = {}
    for index, point in enumerate(points):
        groups.setdefault(point.get('source_id'), []).append(index)
    indexes = [values[i] for i in range(max(map(len, groups.values()), default=0)) for values in groups.values() if i < len(values)][:12]
    sampled = [points[i] for i in indexes]
    local_verdicts = await _audit_summary_batch(sampled, chunks, labels, trace)
    verdicts = {indexes[i]: verdict for i, verdict in local_verdicts.items() if 0 <= i < len(indexes)}
    status = {"total": len(points), "supported": sum(v == "supported" for v in verdicts.values()),
              "unsupported": sum(v == "unsupported" for v in verdicts.values()),
              "unreviewed": len(points) - len(verdicts), "supplement_attempted": False}
    if current():
        current().audit = status
    for index, point in enumerate(points):
        point["review_status"] = verdicts.get(index, "unreviewed")
    return points


def summary_point_quotas(source_ids: list[str], total: int, overview: int) -> dict[str | None, int]:
    """Allocate bounded per-source sections without starving the overview."""
    if len(source_ids) <= 1:
        return {None: total}
    result: dict[str | None, int] = {None: min(overview, max(1, total - min(len(source_ids), total - 1)))}
    remaining = max(0, total - result[None])
    for _ in range(4):
        for source_id in source_ids:
            if remaining:
                result[source_id] = result.get(source_id, 0) + 1
                remaining -= 1
    return result


def allocate_summary_points(
    points: list[dict[str, Any]], quotas: dict[str | None, int], citation_sources: dict[str, str]
) -> list[dict[str, Any]]:
    """Use unambiguous citations to preserve source coverage when labels drift."""
    retained: list[dict[str, Any]] = []
    counts: dict[str | None, int] = {}
    for point in points:
        group = point.get("source_id")
        sources = {citation_sources[label] for label in point.get("citations", []) if label in citation_sources}
        if group is None and len(sources) == 1:
            source = next(iter(sources))
            if counts.get(source, 0) < quotas.get(source, 0):
                group = source
        if counts.get(group, 0) >= quotas.get(group, 0):
            group = None
        if counts.get(group, 0) < quotas.get(group, 0):
            retained.append({**point, "source_id": group})
            counts[group] = counts.get(group, 0) + 1
    return retained


@delivery_task
@adaptive_generation("summary")
async def _hierarchical_summary(notebook_id: str, ids: list[str], language: str, reporter: Reporter | None = None) -> dict[str, Any]:
    representatives = select_quality_evidence(notebook_id, ids, limit=min(36, max(24, len(ids) * 12)))
    if not representatives:
        raise ValueError("当前范围没有可摘要的内容")
    labels = [f"S{index}" for index in range(1, len(representatives) + 1)]
    citations_by_id = {item["id"]: item for item in _context(representatives, labels)[1]}
    original_representatives = representatives
    original_labels = labels
    trace = generation_trace()
    trace.request_limit = (current().plan.preparation_batches if current() else 0) + 3
    if not current():
        trace.total_token_limit = 300_000
    if current() and current().plan.preparation_batches:
        if reporter:
            reporter.update("summarize", "按资料区间提炼证据", 0.20)
        await prepare_evidence(representatives, language)
    representatives = synthesis_evidence(representatives)
    label_by_chunk = {citation["chunk_id"]: label for label, citation in citations_by_id.items()}
    labels = [label_by_chunk[row["id"]] for row in representatives]
    target_points = current().plan.output_items if current() else 6
    overview_count = current().plan.overview_items if current() else 6
    quotas = summary_point_quotas(ids, target_points, overview_count)
    filenames = {row['source_id']: row.get('filename', row['source_id']) for row in original_representatives}

    def parse_points(raw: str, valid_labels: set[str]) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        except (ValueError, json.JSONDecodeError):
            # Preserve complete points when the output cap truncates the array.
            match = re.search(r'"points"\s*:\s*\[', raw)
            values = []
            tail = raw[match.end():] if match else ""
            while tail.strip():
                try:
                    value, end = json.JSONDecoder().raw_decode(tail.lstrip())
                except ValueError:
                    break
                values.append(value)
                tail = tail.lstrip()[end:].lstrip().removeprefix(",")
            parsed = {"points": values}
        if not isinstance(parsed, dict) or not isinstance(parsed.get("points"), list):
            return []
        points: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in parsed.get("points") or []:
            if not isinstance(value, dict):
                continue
            claim = re.sub(r"\s+", " ", str(value.get("claim") or "")).strip()
            why = re.sub(r"\s+", " ", str(value.get("why_it_matters") or "")).strip()
            refs = list(dict.fromkeys(str(item).strip().strip("[]").upper() for item in value.get("citations") or [] if str(item).strip().strip("[]").upper() in valid_labels))[:3]
            key = re.sub(r"\W+", "", claim).lower()
            if not claim or key in seen:
                continue
            seen.add(key)
            points.append(
                {
                    "source_id": value.get("source_id") if value.get("source_id") in ids else None,
                    "claim": claim[:600],
                    "why_it_matters": why[:600],
                    "qualification": re.sub(r"\s+", " ", str(value.get("qualification") or "")).strip()[:360],
                    "citations": refs,
                }
            )
        if len(ids) == 1:
            return points[:target_points]
        return allocate_summary_points(points, quotas, {label: citation['source_id'] for label, citation in citations_by_id.items()})

    language_rule = "自然简体中文" if language != "en" else "natural English"
    output_limit = TokenLimits.from_provider(active_provider("main") or {}).max_output_tokens
    batch_points = min(target_points, max(1, (output_limit - 512) // 350))
    prefix = f"""你是严谨的研究编辑。只依据资料，用{language_rule}提炼 {batch_points} 个相互独立、覆盖全文主线的高信息密度要点。先通读所有提供的原文区段，再决定要点的主题分配，不能只依次概括开篇内容。避免用多个要点重复介绍同一背景；为核心机制、后部结论及重要限制保留空间。紧密相关的背景与机制可以合并，但每点只表达一个可由所引段落直接支持的判断，claim 最多两句；不要串联多个例子、独立结论或不必要的数字、音程等细节。why_it_matters 简述原文明确解释的作用，禁止添加“全书共同地基”等概括性评价。写出结论前核对其成立条件，将必要前提直接写入 claim；不要把概率性、条件性结论改写为无条件保证。why_it_matters 也必须有原文支持，不补充原文没有的意义或评价。先概括资料的主要论题、核心机制及其关系，再解释必要的例证；不要让孤立轶事取代全书主线。保留虚构对话、假设和例子的性质；说话人不明确时不要擅自归因给作者。不要复述封面、版权、目录、书目或索引。每点只能引用真正支持该点的 1–3 个编号；不得给每点附整批编号。仅输出 JSON：{{"points":[{{"claim":"完整核心判断","why_it_matters":"为何重要或如何作用","qualification":"","citations":["S1"]}}]}}。\n资料：\n"""
    if language == 'en':
        prefix = f"""Write {batch_points} distinct source-grounded summary points in natural English. Cover the central argument, mechanism, representative example and important qualifications, including later conclusions. Preserve attribution, fiction, hypotheses, conditions, negations and probabilities. Do not turn conditional claims into guarantees or invent significance. Each point contains one defensible claim, its source-supported significance and necessary qualification, with 1–3 precise citations. Do not summarize covers, indexes or bibliographies. Return JSON {{"points":[{{"claim":"","why_it_matters":"","qualification":"","citations":["S1"]}}]}}. Evidence follows.\n"""
    if len(ids) > 1:
        prefix = prefix.replace('{"claim":', '{"source_id":null,"claim":', 1)
        allocation = [{'source_id': source_id, 'filename': filenames.get(source_id, ''), 'points': quota} for source_id, quota in quotas.items() if source_id]
        prefix += (f"First write up to {quotas.get(None, 0)} overview points with source_id:null, then per-source points with exact source_id. "
                   "Each source section must discuss that author's own argument and conditions, not assert that all sources prove the same thesis. "
                   "Include source_id on EVERY point. These are maxima; omit unsupported points without inventing replacements. Allocation: "
                   if language == 'en' else
                   f"先写最多{quotas.get(None, 0)}个总览要点，source_id填null；再按每份资料分配写其作者的论点及条件，不能声称所有资料共同证明一个结论。每点必须包含source_id字段。下列数量是上限，证据不足可少写，不编造补齐。分配：")
        prefix += json.dumps(allocation, ensure_ascii=False) + "\n"
        prefix += "Source IDs for evidence labels: " + json.dumps({label: row['source_id'] for label, row in zip(labels, representatives)}) + "\n"
    points: list[dict[str, Any]] = []
    valid_labels: set[str] = set()
    if reporter:
        reporter.update("summarize", "一次性综合全文要点", 0.30, current=0, total=1, unit="次")
    try:
        generated = await budgeted_chat(
            lambda budget: _evidence_prompt_build(
                budget, chunks=representatives, labels=labels, prefix=prefix, ensure_source_coverage=len(ids) > 1, include_notes=True
            ),
            json_mode=True,
            max_tokens=current().plan.output_tokens if current() else structured_output_tokens(2200),
            minimum_output_tokens=700,
            trace=trace,
            stage="summary",
        )
        valid_labels = set(generated.build.metadata["labels"])
        points = parse_points(generated.content, valid_labels)
        for _ in range(1):
            if points or not claim_recovery():
                break
            repair_prefix = f"""上次摘要只有 {len(points)} 个有效要点。用{language_rule}只依据资料补充 {min(batch_points, target_points - len(points))} 个简洁要点，避免与现有要点重复，并保持每点 1–3 个精确引用。现有有效要点：{json.dumps(points, ensure_ascii=False)}。仅输出同一 JSON points 结构。\n资料：\n"""
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
            additions = [item for item in extra if re.sub(r"\W+", "", item["claim"]).lower() not in known]
            points.extend(additions)
            points = points[:max(10, target_points)]
            if not additions:
                break
    except Exception:
        trace.mark_fallback()
    audit_removed = False
    if points:
        audited = await _audit_summary_points(points, original_representatives, original_labels, trace)
        audit_removed = len(audited) != len(points)
        points = audited
    quality_issues = [{"unit": f"point_{i + 1}", "code": "evidence_unconfirmed", "severity": "suspect", "message": "该要点的原文支持待核实。"}
                      for i, point in enumerate(points) if point.get("review_status") == "unsupported" or not point["citations"]]
    quality_assessment = assessment(len(points), sum(p.get("review_status") in {"supported", "unsupported"} for p in points), quality_issues, method="model_sample", supported=sum(p.get("review_status") == "supported" for p in points))
    degraded = audit_removed or len(points) < target_points
    if degraded and not points:
        points = [
            {"claim": str(chunk["content"])[:300], "why_it_matters": "可直接核验的资料摘录" if language != "en" else "A directly verifiable source excerpt", "qualification": "", "citations": [label]}
            for chunk, label in zip(original_representatives[:6], original_labels[:6])
        ]
        valid_labels = set(original_labels[:6])
        quality_assessment = assessment(len(points), method="local_excerpt")
    if degraded:
        trace.mark_fallback()
    heading = "## Evidence-bound summary" if language == "en" else "## 可追溯摘要"
    def point_line(point):
        qualification = f" {point['qualification']}" if point['qualification'] else ""
        markers = " ".join(f"[{label}]" for label in point['citations'])
        return f"- {point['claim']} — {point['why_it_matters']}{qualification} {markers}"
    overview = [point for point in points if not point.get('source_id')] if len(ids) > 1 else points
    answer = heading + "\n\n" + "\n".join(point_line(point) for point in overview)
    source_summaries = []
    if len(ids) > 1:
        for source_id in ids:
            selected = [point for point in points if point.get('source_id') == source_id]
            source_summaries.append({'source_id': source_id, 'filename': filenames.get(source_id, source_id), 'points': selected, 'covered': bool(selected)})
            answer += "\n\n### " + filenames.get(source_id, source_id).replace('\n', ' ') + "\n\n"
            answer += "\n".join(point_line(point) for point in selected) if selected else ('No separate source points were produced; coverage is incomplete.' if language == 'en' else '本次未生成该资料的独立要点，覆盖不完整。')
    used_labels = list(dict.fromkeys(label for point in points for label in point["citations"]))
    output_citations = [citations_by_id[label] for label in used_labels if label in citations_by_id]
    warnings = [{"code": "summary_partial", "stage": "summary", "message": "摘要综合未完整完成，已保留有效要点或可核验的原文摘录。"}] if degraded else []
    if any(not item['covered'] for item in source_summaries):
        degraded = True
        warnings.append({'code': 'source_summary_incomplete', 'stage': 'summary', 'message': '部分资料尚无独立要点，已保留其他内容。'})
    if current() and current().audit.get("unreviewed"):
        warnings.append({"code": "summary_audit_incomplete", "stage": "audit", "message": "部分要点的核验未完成，不等同于已认定事实错误。", "count": current().audit["unreviewed"]})
    if current() and current().audit.get("unsupported"):
        warnings.append({"code": "summary_unsupported", "stage": "audit", "message": "已保留原文支持待核实的要点，请核对引用。", "count": current().audit["unsupported"]})
    return {"version": 3, "source_summaries": source_summaries, "quality_assessment": quality_assessment, "delivery_status": "partial" if degraded else "full", "content": answer, "points": points, "citations": output_citations, "scope_hash": scope_hash(ids), "source_ids": ids, "degraded": degraded, "warnings": warnings, "context_usage": trace.as_dict()}


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
    DB.execute("INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (artifact_id, notebook_id, "summary", "资料摘要", json_dump(ids), language, "partial" if result["degraded"] else "ready", json_dump({"quality_assessment": result.get("quality_assessment"), "delivery_status": result.get("delivery_status"), "version": result["version"], "content": result["content"], "points": result["points"], "source_summaries": result.get("source_summaries", []), "degraded": result["degraded"], "warnings": result["warnings"], "context_usage": result["context_usage"], "language_selection": language_selection}), json_dump(result["citations"]), None, now, now))
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
