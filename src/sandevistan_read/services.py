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
from .providers import ProviderError, chat, describe_image
from .retrieval import EMBEDDINGS, retrieve, tokenize
from .observability import Reporter


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


async def ingest_source(source_id: str, progress: Callable[[str, float], None] | None = None, cancel_check: Callable[[], bool] | None = None) -> dict[str, Any]:
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
    vision_count = 0
    visual_indexes = [index for index, block in enumerate(parsed.blocks) if block.visual_needed and block.image_path]
    selected_visual = set(visual_indexes)
    if len(visual_indexes) > 48:
        low_text = [index for index in visual_indexes if len(parsed.blocks[index].text.strip()) < 200][:24]
        remaining = [index for index in visual_indexes if index not in set(low_text)]
        slots = max(0, 48 - len(low_text))
        if slots and remaining:
            positions = {round(position * (len(remaining) - 1) / max(1, slots - 1)) for position in range(slots)}
            sampled = [remaining[position] for position in sorted(positions)]
        else:
            sampled = []
        selected_visual = set(low_text + sampled)
    processed_visual = 0
    for block_index, block in enumerate(parsed.blocks):
        if block.visual_needed and block.image_path:
            if block_index not in selected_visual:
                continue
            if cancel_check and cancel_check():
                raise RuntimeError("任务已取消")
            if progress:
                progress(f"视觉解析 {processed_visual + 1}/{len(selected_visual)}", 0.12 + 0.38 * processed_visual / max(1, len(selected_visual)))
            visual = ""
            try:
                visual = await describe_image(PATHS.root / block.image_path, block.text)
                if visual:
                    block.text = (block.text + "\n\n[视觉解析]\n" + visual).strip()
                    vision_count += 1
            except Exception:
                pass
            if not visual:
                try:
                    from rapidocr import RapidOCR
                    result = await asyncio.to_thread(RapidOCR(), str(PATHS.root / block.image_path))
                    lines = [item if isinstance(item, str) else getattr(item, "txt", "") for item in (getattr(result, "txts", []) or [])]
                    lines = [line for line in lines if line]
                    if lines:
                        block.text = (block.text + "\n\n[本地OCR]\n" + "\n".join(lines)).strip()
                except Exception:
                    pass
            processed_visual += 1
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
        for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
            chunk_id = new_id("chunk")
            checksum = hashlib.sha256(chunk.text.encode()).hexdigest()
            connection.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)", (chunk_id, source_id, source["revision_id"], ordinal, chunk.text, json_dump(chunk.locator), json_dump(vector), checksum, now))
            connection.execute("INSERT INTO chunks_fts(chunk_id,source_id,content) VALUES(?,?,?)", (chunk_id, source_id, chunk.text))
        metadata = dict(parsed.metadata)
        metadata.update({"vision_pages": vision_count, "vision_candidates": len(visual_indexes), "vision_budget": len(selected_visual), "chunk_count": len(chunks), "embedding_mode": EMBEDDINGS.mode})
        connection.execute("UPDATE sources SET state='ready',page_count=?,parser=?,preview_path=?,metadata_json=?,updated_at=? WHERE id=?", (parsed.page_count, parsed.parser, parsed.preview_path, json_dump(metadata), now, source_id))
    return {"source_id": source_id, "chunks": len(chunks), "vision_pages": vision_count}


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
    context, citations = _context(chunks)
    language_rule = "使用中文" if language == "zh-CN" else "Use English" if language == "en" else "跟随用户问题及资料的主要语言"
    prompt = f"""你是严格依据资料的研究助手。{language_rule}。
规则：只允许使用下方资料；每个事实陈述后必须标注一个或多个 [S1] 格式引用；资料不足就明确说无法从资料确认；禁止使用外部知识；不要编造引用。
任务：{instruction}
问题/主题：{query}

资料：
{context}
"""
    try:
        answer = await chat([{"role": "system", "content": "Ground every claim in supplied sources."}, {"role": "user", "content": prompt}], max_tokens=max_tokens)
        valid_ids = {citation["id"] for citation in citations}
        issues = _grounding_issues(answer, valid_ids)
        if issues:
            repair = f"""仅依据原资料重写下方回答。输出 3 到 6 个完整、简洁的句子，不要标题、编号或项目符号。每句话末尾必须紧跟一个或多个已有 [S数字] 引用。删除无法支持的陈述，不得把引用集中放到末尾，不得创造新引用。仅输出重写后的回答。
问题：{query}
问题列表：{'；'.join(issues[:12])}
原回答：
{answer}
原资料：
{context}"""
            answer = await chat([{"role": "user", "content": repair}], max_tokens=max_tokens)
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
        answer = _extractive_fallback(citations, language)
        degraded = True
    used = {item for item in re.findall(r"\[(S\d+)\]", answer)}
    valid = [citation for citation in citations if citation["id"] in used]
    if not valid and not _is_refusal(answer):
        valid = citations[:3]
        if valid:
            answer += "\n\n" + " ".join(f"[{item['id']}]" for item in valid)
    return {"content": answer, "citations": valid, "scope_hash": scope_hash(ids), "source_ids": ids, "degraded": degraded}


def _evenly_spaced(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    if limit <= 1:
        return rows[:1]
    indexes = {round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)}
    return [row for index, row in enumerate(rows) if index in indexes]


async def _hierarchical_summary(notebook_id: str, ids: list[str], language: str, reporter: Reporter | None = None) -> dict[str, Any]:
    representatives: list[dict[str, Any]] = []
    for source_id in ids:
        rows = DB.fetchall("SELECT * FROM chunks WHERE source_id=? ORDER BY ordinal", (source_id,))
        for row in rows:
            row["locator"] = json_load(row.get("locator_json"), {})
        representatives.extend(_evenly_spaced(rows, 18))
    if not representatives:
        raise ValueError("当前范围没有可摘要的内容")
    labels = [f"C{index}" for index in range(1, len(representatives) + 1)]
    _, citations = _context(representatives, labels)
    notes: list[str] = []
    degraded = False
    total_batches = math.ceil(len(representatives) / 6)
    for start in range(0, len(representatives), 6):
        batch_number = start // 6 + 1
        if reporter:
            if reporter.cancelled():
                raise RuntimeError("任务已取消")
            reporter.update("summarize", f"分段提炼 {batch_number}/{total_batches}", 0.12 + 0.72 * (batch_number - 1) / max(1, total_batches), current=batch_number, total=total_batches, unit="批")
        batch = representatives[start:start + 6]
        batch_labels = labels[start:start + len(batch)]
        context, _ = _context(batch, batch_labels)
        prompt = f"只根据以下资料提炼 2 条有实质信息、可独立阅读的完整要点。不要使用外部知识，不要输出标题或引用标记。语言：{language}。\n{context}"
        try:
            raw = await chat([{"role": "user", "content": prompt}], max_tokens=650)
            candidates: list[str] = []
            for line in raw.splitlines():
                clean = re.sub(r"\[C\d+\]", "", line)
                clean = re.sub(r"^[#>*\-\d.\s]+", "", clean).strip()
                if len(clean) >= 24 and not clean.endswith(("：", ":")):
                    candidates.append(clean[:700])
            if not candidates:
                clean = re.sub(r"\[C\d+\]", "", raw).strip()
                candidates = [clean[:900]] if clean else []
            suffix = " ".join(f"[{label}]" for label in batch_labels)
            notes.extend(f"- {line} {suffix}" for line in candidates[:2])
        except Exception:
            degraded = True
            notes.extend(f"- {chunk['content'][:300]} [{label}]" for label, chunk in zip(batch_labels[:2], batch[:2]))
    heading = "## Evidence-bound summary" if language == "en" else "## 可追溯摘要"
    answer = heading + "\n\n" + "\n".join(notes)
    used_c = [citation for citation in citations if citation["id"] in set(re.findall(r"\[(C\d+)\]", answer))]
    if not used_c:
        used_c = citations[: min(8, len(citations))]
        answer += "\n\n" + " ".join(f"[{item['id']}]" for item in used_c)
    remap = {citation["id"]: f"S{index}" for index, citation in enumerate(used_c, start=1)}
    for old, new in remap.items():
        answer = answer.replace(f"[{old}]", f"[{new}]")
    output_citations = [{**citation, "id": remap[citation["id"]]} for citation in used_c]
    return {"content": answer, "citations": output_citations, "scope_hash": scope_hash(ids), "source_ids": ids, "degraded": degraded}


async def make_summary(notebook_id: str, source_ids: list[str] | None, language: str, job_id: str | None = None) -> dict[str, Any]:
    ids = source_scope(notebook_id, source_ids)
    if not ids:
        raise ValueError("当前范围没有已就绪的文档")
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
    DB.execute("INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (artifact_id, notebook_id, "summary", "资料摘要", json_dump(ids), language, "ready", json_dump({"content": result["content"], "degraded": result["degraded"]}), json_dump(result["citations"]), None, now, now))
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
        context, _ = _context(window, window_labels)
        citation_suffix = " ".join(f"[{label}]" for label in window_labels)
        if kind == "quiz":
            schema = '{"items":[{"question":"...","options":["A","B","C","D"],"answer":0,"explanation":"..."}]}'
            task = f"生成恰好{batch_count}道{difficulty}难度单选题，四个互斥选项，answer为0-3。"
        else:
            schema = '{"items":[{"front":"...","back":"..."}]}'
            task = f"生成恰好{batch_count}张不重复的高质量闪卡。"
        prompt = f"只根据资料生成内容，不要输出引用标记。{task} 仅输出合法JSON：{schema}\n资料：\n{context}"
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
                raw = await chat([{"role": "user", "content": prompt}], json_mode=True, max_tokens=1800)
                candidate = json.loads(raw[raw.find("{"):raw.rfind("}") + 1]).get("items", [])
                if kind == "quiz":
                    audit = f"""仅根据资料审核并修正题目。保证每题四个选项互不重复且只有一个正确答案；answer 必须准确指向正确选项；explanation 必须明确说明该选项为什么正确。保留恰好 {batch_count} 题，仅输出同结构合法 JSON。
资料：
{context}
待审核：
{json.dumps({'items': candidate}, ensure_ascii=False)}"""
                    checked = await chat([{"role": "user", "content": audit}], json_mode=True, max_tokens=1800)
                    candidate = json.loads(checked[checked.find("{"):checked.rfind("}") + 1]).get("items", [])
                    key_prompt = f"""只根据资料为下列 {len(candidate)} 道单选题选择唯一最佳答案。按题目顺序仅输出 JSON：{{"answers":[0,1,2]}}，数组值只能是 0 到 3。
资料：
{context}
题目：
{json.dumps([{'question': item.get('question'), 'options': item.get('options')} for item in candidate], ensure_ascii=False)}"""
                    try:
                        keyed = await chat([{"role": "user", "content": key_prompt}], json_mode=True, max_tokens=500)
                        answers = json.loads(keyed[keyed.find("{"):keyed.rfind("}") + 1]).get("answers", [])
                        if len(answers) == len(candidate) and all(answer in range(4) for answer in answers):
                            for item, answer in zip(candidate, answers):
                                item["answer"] = answer
                    except Exception:
                        pass
                    for item in candidate:
                        if not isinstance(item.get("options"), list) or item.get("answer") not in range(4):
                            continue
                        evidence_tokens = {token for token in tokenize(str(item.get("explanation", ""))) if len(token) > 1}
                        scores = [len(evidence_tokens & {token for token in tokenize(str(option)) if len(token) > 1}) for option in item["options"]]
                        best = max(range(4), key=scores.__getitem__)
                        if scores[best] >= 2 and scores[best] > scores[item["answer"]]:
                            item["answer"] = best
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
                if kind == "quiz":
                    if not isinstance(item.get("question"), str) or not isinstance(item.get("options"), list) or len(item["options"]) != 4 or not all(isinstance(option, str) and option.strip() for option in item["options"]) or item.get("answer") not in range(4):
                        continue
                    option_keys = [re.sub(r"^[A-Da-d][.、:)：\s-]*|\W+", "", option).lower() for option in item["options"]]
                    if len(set(option_keys)) != 4:
                        continue
                    if not isinstance(item.get("explanation"), str):
                        continue
                    item["explanation"] = re.sub(r"\[S\d+\]", "", item["explanation"]).strip() + " " + citation_suffix
                elif not isinstance(item.get("front"), str) or not isinstance(item.get("back"), str):
                    continue
                else:
                    item["back"] = re.sub(r"\[S\d+\]", "", item["back"]).strip() + " " + citation_suffix
                item["citations"] = window_labels
                accepted.append(item)
                if len(accepted) == batch_count:
                    break
            if len(accepted) == batch_count:
                break
        if len(accepted) != batch_count:
            degraded = True
            accepted = []
            for index in range(batch_count):
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
    payload = {"items": items, "degraded": degraded}
    if reporter:
        reporter.update("persist", "保存题库" if kind == "quiz" else "保存闪卡", 0.94, current=count, total=count, unit="题" if kind == "quiz" else "张")
    artifact_id = f"artifact_{job_id.removeprefix('job_')}" if job_id else new_id("artifact")
    now = utc_now()
    DB.execute("INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (artifact_id, notebook_id, kind, "单选题库" if kind == "quiz" else "闪卡组", json_dump(ids), language, "ready", json_dump(payload), json_dump(citations), None, now, now))
    return {"id": artifact_id, "type": kind, "payload": payload, "citations": citations}
