from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Callable

import numpy as np

from .context_budget import ContextUsage, PromptBudget, estimate_messages_tokens, estimate_text_tokens, pack_items, structured_output_tokens
from .database import DB, json_dump, json_load, new_id, utc_now
from .observability import Reporter
from .providers import PromptBuild, active_provider, budgeted_chat, study_generation_profile
from .retrieval import EMBEDDINGS, retrieve, tokenize
from .languages import resolve_output_language, text_matches_language


BANNED_QUIZ_STEMS = ("直接出现在", "资料位置", "引用位置", "第几页", "哪一页")
BANNED_DISTRACTORS = ("资料没有讨论", "资料明确否定", "系统外部知识", "其它位置未给出", "以上皆是", "以上都不是")
COGNITIVE_LEVELS = {"recall", "understand", "apply", "analyze"}
CARD_TYPES = {"fact", "concept", "relationship", "comparison", "application"}


def _source_scope(notebook_id: str, requested: list[str] | None) -> list[str]:
    if requested is None:
        rows = DB.fetchall("SELECT id FROM sources WHERE notebook_id=? AND selected=1 AND state='ready' ORDER BY created_at", (notebook_id,))
    elif requested:
        marks = ",".join("?" for _ in requested)
        rows = DB.fetchall(f"SELECT id FROM sources WHERE notebook_id=? AND state='ready' AND id IN ({marks})", (notebook_id, *requested))
    else:
        rows = []
    return [row["id"] for row in rows]


def _evenly_spaced(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    if limit <= 1:
        return rows[:1]
    indexes = {round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)}
    return [row for index, row in enumerate(rows) if index in indexes]


def _collect_evidence(notebook_id: str, source_ids: list[str], count: int, custom_prompt: str) -> list[dict[str, Any]]:
    target = min(80, max(16, count * 2))
    selected: list[dict[str, Any]] = []
    if custom_prompt.strip():
        selected.extend(retrieve(notebook_id, custom_prompt, source_ids, limit=min(30, target), ensure_source_coverage=len(source_ids) > 1))
    per_source = max(2, math.ceil(target / max(1, len(source_ids))))
    for source_id in source_ids:
        rows = DB.fetchall("SELECT * FROM chunks WHERE source_id=? ORDER BY ordinal", (source_id,))
        for row in _evenly_spaced(rows, per_source):
            row["locator"] = json_load(row.pop("locator_json", None), {})
            row.pop("embedding_json", None)
            selected.append(row)
    unique: dict[str, dict[str, Any]] = {}
    for row in selected:
        unique.setdefault(row["id"], row)
    return list(unique.values())[:target]


def _source_name(source_id: str) -> str:
    row = DB.fetchone("SELECT filename FROM sources WHERE id=?", (source_id,))
    return str((row or {}).get("filename") or "未知来源")


def _entry(chunk: dict[str, Any], label: str) -> str:
    locator = chunk.get("locator") or {}
    location = f"第{locator['page']}页" if locator.get("page") else f"第{locator['slide']}张" if locator.get("slide") else locator.get("section") or "文档位置"
    return f"[{label}] {_source_name(chunk['source_id'])} · {location}\n{chunk['content']}"


def _citation(chunk: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "id": label,
        "source_id": chunk["source_id"],
        "chunk_id": chunk["id"],
        "filename": _source_name(chunk["source_id"]),
        "locator": chunk.get("locator") or {},
        "quote": str(chunk["content"])[:320],
    }


def _evidence_build(
    budget: PromptBudget,
    chunks: list[dict[str, Any]],
    labels: list[str],
    prefix: str,
    suffix: str = "",
) -> PromptBuild:
    empty = [{"role": "user", "content": prefix + suffix}]
    available = max(0, budget.input_tokens - estimate_messages_tokens(empty, budget.image_tokens_per_image) - 8)
    packed = pack_items(
        list(zip(labels, chunks)),
        lambda pair: _entry(pair[1], pair[0]),
        available,
        group_key=lambda pair: str(pair[1]["source_id"]),
    )
    context = "\n\n".join(packed.texts)
    return PromptBuild(
        [{"role": "user", "content": prefix + context + suffix}],
        total_segments=packed.total,
        included_segments=len(packed.items),
        truncated_segments=packed.truncated,
        metadata={
            "labels": [label for label, _ in packed.items],
            "chunks": [chunk for _, chunk in packed.items],
            "citations": [_citation(chunk, label) for label, chunk in packed.items],
        },
    )


def _json_object(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型没有返回 JSON 对象")
    value = json.loads(raw[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型返回的 JSON 顶层必须是对象")
    return value


def _json_array_items(raw: str, key: str) -> list[Any]:
    try:
        value = _json_object(raw).get(key)
        if isinstance(value, list):
            return value
    except (ValueError, json.JSONDecodeError):
        pass
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', raw)
    if not match:
        return []
    decoder, position, recovered = json.JSONDecoder(), match.end(), []
    while position < len(raw):
        while position < len(raw) and (raw[position].isspace() or raw[position] == ","):
            position += 1
        try:
            value, consumed = decoder.raw_decode(raw[position:])
        except json.JSONDecodeError:
            break
        recovered.append(value)
        position += consumed
    return recovered


def _local_blueprint(chunks: list[dict[str, Any]], labels: list[str], count: int, difficulty: str, tier: str) -> list[dict[str, Any]]:
    levels = _difficulty_sequence(count, difficulty, tier)
    blueprint = []
    for index in range(max(count, min(len(chunks), count * 2))):
        chunk = chunks[index % len(chunks)]
        locator = chunk.get("locator") or {}
        title = str(locator.get("section") or _source_name(chunk["source_id"]))[:100]
        excerpt = re.sub(r"\s+", " ", str(chunk["content"])).strip()[:180]
        blueprint.append({"title": title, "objective": excerpt, "difficulty": levels[index % len(levels)], "citations": [labels[index % len(labels)]]})
    return blueprint


def _difficulty_sequence(count: int, difficulty: str, tier: str) -> list[str]:
    if difficulty != "mixed":
        return [difficulty] * max(1, count)
    weights = (("easy", 0.5), ("medium", 0.5)) if tier == "lite" else (("easy", 0.3), ("medium", 0.5), ("hard", 0.2))
    values: list[str] = []
    for name, weight in weights:
        values.extend([name] * max(1, round(count * weight)))
    return (values * math.ceil(max(1, count) / len(values)))[: max(1, count)]


async def _build_blueprint(
    chunks: list[dict[str, Any]],
    labels: list[str],
    count: int,
    difficulty: str,
    language: str,
    custom_prompt: str,
    tier: str,
    trace: ContextUsage,
) -> tuple[list[dict[str, Any]], bool]:
    fallback = _local_blueprint(chunks, labels, count, difficulty, tier)
    if tier == "lite":
        return fallback, False
    prefix = f"""你是学习设计师。只依据资料建立知识蓝图，不生成题目。选择至少 {count} 个跨章节、跨来源、值得主动回忆的核心概念，避免只记页码或原句位置。语言：{language}。难度：{difficulty}。定制要求：{custom_prompt or '无'}。
每个概念只能引用真正支持它的 1 到 3 个 [S数字]。仅输出 JSON：{{"concepts":[{{"title":"...","objective":"学习者应能...","difficulty":"easy|medium|hard","citations":["S1"]}}]}}
资料：
"""
    try:
        generated = await budgeted_chat(
            lambda budget: _evidence_build(budget, chunks, labels, prefix),
            json_mode=True,
            max_tokens=structured_output_tokens(min(1800, max(700, count * 110))),
            minimum_output_tokens=384,
            trace=trace,
        )
        valid_labels = set(generated.build.metadata["labels"])
        concepts = []
        for item in _json_array_items(generated.content, "concepts"):
            if not isinstance(item, dict):
                continue
            refs = [str(value) for value in item.get("citations", []) if str(value) in valid_labels][:3]
            title, objective = str(item.get("title") or "").strip(), str(item.get("objective") or "").strip()
            item_difficulty = str(item.get("difficulty") or "medium")
            if title and objective and refs and item_difficulty in {"easy", "medium", "hard"}:
                concepts.append({"title": title[:120], "objective": objective[:300], "difficulty": item_difficulty, "citations": refs})
        if len(concepts) >= min(3, count):
            return concepts, False
    except Exception:
        pass
    trace.mark_fallback()
    return fallback, True


def _normalize(value: str) -> str:
    return re.sub(r"\W+", "", value).lower()


def _evidence_overlap(text: str, citations: list[str], evidence_by_label: dict[str, dict[str, Any]]) -> int:
    target = {token for token in tokenize(text) if len(token) > 1}
    support: set[str] = set()
    for label in citations:
        support.update(token for token in tokenize(str(evidence_by_label[label]["content"])) if len(token) > 1)
    return len(target & support)


def validate_quiz_item(item: Any, valid_labels: set[str], evidence_by_label: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(item, dict):
        return None, "not_object"
    question = str(item.get("question") or "").strip()
    options = item.get("options")
    answer = item.get("answer_index", item.get("answer"))
    explanation = str(item.get("explanation") or "").strip()
    hint = str(item.get("hint") or "").strip()
    citations = list(dict.fromkeys(str(value) for value in item.get("citations", []) if str(value) in valid_labels))[:3]
    if not question or not isinstance(options, list) or len(options) != 4 or answer not in range(4):
        return None, "invalid_shape"
    options = [str(option).strip() for option in options]
    if any(not option for option in options) or len({_normalize(option) for option in options}) != 4:
        return None, "duplicate_options"
    if any(term in question for term in BANNED_QUIZ_STEMS) or re.search(r"第\s*\d+\s*页", question):
        return None, "source_location_stem"
    if any(term in option for option in options for term in BANNED_DISTRACTORS):
        return None, "generic_distractor"
    lengths = [estimate_text_tokens(option) for option in options]
    if max(lengths) > 60 or min(lengths) < 1 or max(lengths) > max(12, min(lengths) * 3):
        return None, "option_length_cue"
    if not explanation or not hint or not citations:
        return None, "missing_feedback_or_citation"
    if _normalize(options[answer]) and _normalize(options[answer]) in _normalize(hint):
        return None, "hint_leaks_answer"
    if _evidence_overlap(options[answer] + " " + explanation, citations, evidence_by_label) < 2:
        return None, "weak_evidence_overlap"
    difficulty = str(item.get("difficulty") or "medium")
    level = str(item.get("cognitive_level") or "understand")
    return {
        "question": question[:800],
        "options": options,
        "answer_index": int(answer),
        "hint": hint[:400],
        "explanation": explanation[:1200],
        "citations": citations,
        "learning_objective": str(item.get("learning_objective") or question)[:400],
        "difficulty": difficulty if difficulty in {"easy", "medium", "hard"} else "medium",
        "cognitive_level": level if level in COGNITIVE_LEVELS else "understand",
    }, None


def validate_flashcard_item(item: Any, valid_labels: set[str], evidence_by_label: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(item, dict):
        return None, "not_object"
    front, back = str(item.get("front") or "").strip(), str(item.get("back") or "").strip()
    explanation = str(item.get("explanation") or "").strip()
    citations = list(dict.fromkeys(str(value) for value in item.get("citations", []) if str(value) in valid_labels))[:3]
    if not front or not back or not citations:
        return None, "invalid_shape"
    if front.startswith("资料要点") or any(term in front for term in ("引用位置", "该位置包含什么")):
        return None, "vague_prompt"
    if estimate_text_tokens(front) > 80 or estimate_text_tokens(back) > 120 or estimate_text_tokens(explanation) > 180:
        return None, "too_long"
    if _evidence_overlap(back + " " + explanation, citations, evidence_by_label) < 2:
        return None, "weak_evidence_overlap"
    difficulty = str(item.get("difficulty") or "medium")
    card_type = str(item.get("card_type") or "concept")
    return {
        "front": front,
        "back": back,
        "explanation": explanation,
        "citations": citations,
        "learning_objective": str(item.get("learning_objective") or front)[:400],
        "difficulty": difficulty if difficulty in {"easy", "medium", "hard"} else "medium",
        "card_type": card_type if card_type in CARD_TYPES else "concept",
    }, None


def _candidate_prompt(kind: str, concepts: list[dict[str, Any]], count: int, language: str, custom_prompt: str) -> str:
    concept_json = json.dumps(concepts, ensure_ascii=False)
    common = f"语言：{language}。定制要求：{custom_prompt or '无'}。目标概念：{concept_json}。只可使用随后资料；每项 citations 必须是实际支持该项的 1–3 个 [S数字]。"
    if kind == "quiz":
        return f"""你是严谨的测验设计师。{common}
生成恰好 {count} 道四选一理解题。选项必须同类、语法平行且只有一个正确答案；干扰项应合理但能被资料排除。禁止询问页码、原句位置，禁止“资料未提及/外部知识/以上皆是”等偷懒选项。Hint 帮助思考但不得透露答案；explanation 解释正确项并说明干扰项为何不成立。仅输出 JSON：{{"items":[{{"learning_objective":"...","difficulty":"easy|medium|hard","cognitive_level":"recall|understand|apply|analyze","question":"...","options":["...","...","...","..."],"answer_index":0,"hint":"...","explanation":"...","citations":["S1"]}}]}}
资料：
"""
    return f"""你是严谨的闪卡设计师。{common}
生成恰好 {count} 张原子化闪卡：一张只测一个知识点，正面必须能独立理解，背面简洁准确；不要写“资料要点”或“该位置包含什么”。中文 front/back/explanation 分别不超过 50/80/120 字；英文分别不超过 20/35/50 words。可覆盖事实、概念、关系、比较与资料内应用。explanation 给出简短理解说明。仅输出 JSON：{{"items":[{{"learning_objective":"...","difficulty":"easy|medium|hard","card_type":"fact|concept|relationship|comparison|application","front":"...","back":"...","explanation":"...","citations":["S1"]}}]}}
资料：
"""


async def _audit_candidates(
    kind: str,
    candidates: list[dict[str, Any]],
    evidence_by_label: dict[str, dict[str, Any]],
    trace: ContextUsage,
) -> tuple[list[int], list[str]]:
    if not candidates:
        return [], []
    schema = "question/options/answer_index/hint/explanation/citations" if kind == "quiz" else "front/back/explanation/citations"
    prefix = f"""你是独立证据审校员。逐项检查下列候选是否完全由资料支持、清晰、无歧义且满足 {schema}。Quiz 必须恰有一个正确答案且干扰项合理；闪卡必须只测一个知识点。不得改写候选，只返回可发布项的零基索引。仅输出 JSON：{{"accepted_indexes":[0],"issues":["1: 原因"]}}。
候选：{json.dumps(candidates, ensure_ascii=False)}
资料：
"""
    labels = list(dict.fromkeys(label for item in candidates for label in item.get("citations", []) if label in evidence_by_label))
    chunks = [evidence_by_label[label] for label in labels]
    generated = await budgeted_chat(
        lambda budget: _evidence_build(budget, chunks, labels, prefix),
        json_mode=True,
        max_tokens=structured_output_tokens(max(300, len(candidates) * 80)),
        minimum_output_tokens=384,
        trace=trace,
    )
    parsed = _json_object(generated.content)
    indexes = [int(value) for value in parsed.get("accepted_indexes", []) if str(value).isdigit() and 0 <= int(value) < len(candidates)]
    return list(dict.fromkeys(indexes)), [str(value)[:240] for value in parsed.get("issues", [])][:20]


def _cosine(left: list[float], right: list[float]) -> float:
    a, b = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _semantic_unique(candidate: dict[str, Any], accepted: list[dict[str, Any]], kind: str) -> bool:
    field = "question" if kind == "quiz" else "front"
    text = str(candidate[field])
    if any(_normalize(text) == _normalize(str(item[field])) for item in accepted):
        return False
    if not accepted:
        return True
    vectors = EMBEDDINGS.encode([text, *[str(item[field]) for item in accepted]])
    return all(_cosine(vectors[0], vector) < 0.92 for vector in vectors[1:])


def _balance_answer(item: dict[str, Any], target: int) -> dict[str, Any]:
    current = item["answer_index"]
    if current == target:
        return item
    options = list(item["options"])
    options[current], options[target] = options[target], options[current]
    return {**item, "options": options, "answer_index": target}


async def generate_study_artifact(
    notebook_id: str,
    kind: str,
    count: int,
    source_ids: list[str] | None,
    language: str,
    difficulty: str = "mixed",
    custom_prompt: str = "",
    job_id: str | None = None,
) -> dict[str, Any]:
    if kind not in {"quiz", "flashcard"}:
        raise ValueError("不支持的学习产物类型")
    ids = _source_scope(notebook_id, source_ids)
    if not ids:
        raise ValueError("当前范围没有已就绪的文档")
    language, language_selection = resolve_output_language(DB, ids, language)
    provider = active_provider("main")
    if not provider:
        raise ValueError("请先启用 MAIN Provider")
    profile = study_generation_profile(provider)
    tier = profile["tier"]
    if tier == "lite" and difficulty == "hard":
        raise ValueError("当前模型处于兼容档，不支持困难难度；请改用中等难度或更强模型")
    chunks = _collect_evidence(notebook_id, ids, count, custom_prompt)
    if not chunks:
        raise ValueError("当前范围没有可生成学习内容的资料")
    labels = [f"S{index}" for index in range(1, len(chunks) + 1)]
    evidence_by_label = dict(zip(labels, chunks))
    all_citations = {_citation(chunk, label)["id"]: _citation(chunk, label) for label, chunk in zip(labels, chunks)}
    trace, reporter = ContextUsage(), Reporter(job_id) if job_id else None
    if reporter:
        reporter.update("plan", "构建知识蓝图", 0.08, current=0, total=count, unit="项")
    blueprint, blueprint_fallback = await _build_blueprint(chunks, labels, count, difficulty, language, custom_prompt, tier, trace)
    provisional: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    batch_size = 1 if tier == "lite" else 3
    max_candidate_rounds = math.ceil(count / batch_size) + 2
    cursor = 0
    candidate_rounds = 0
    audit_rounds = 0
    audit_fallback = False
    consecutive_zero = 0
    stop_reason = "candidate_round_limit"
    validator: Callable[[Any, set[str], dict[str, dict[str, Any]]], tuple[dict[str, Any] | None, str | None]] = validate_quiz_item if kind == "quiz" else validate_flashcard_item
    while len(provisional) < count and candidate_rounds < max_candidate_rounds and consecutive_zero < 2:
        batch_count = min(batch_size, count - len(provisional))
        concepts = [blueprint[(cursor + offset) % len(blueprint)] for offset in range(batch_count)]
        concept_labels = list(dict.fromkeys(label for concept in concepts for label in concept.get("citations", [])))
        selected_pairs = [(label, evidence_by_label[label]) for label in concept_labels if label in evidence_by_label]
        for label, chunk in zip(labels, chunks):
            if len(selected_pairs) >= max(4, batch_size * 2):
                break
            if label not in {item[0] for item in selected_pairs}:
                selected_pairs.append((label, chunk))
        selected_labels = [label for label, _ in selected_pairs]
        selected_chunks = [chunk for _, chunk in selected_pairs]
        if reporter:
            reporter.update("generate", f"生成候选 {len(provisional)}/{count}", 0.15 + 0.58 * len(provisional) / max(1, count), current=len(provisional), total=count, unit="项")
        try:
            generated = await budgeted_chat(
                lambda budget: _evidence_build(budget, selected_chunks, selected_labels, _candidate_prompt(kind, concepts, batch_count, language, custom_prompt)),
                json_mode=True,
                max_tokens=structured_output_tokens(max(900, batch_count * (900 if kind == "quiz" else 500))),
                minimum_output_tokens=320,
                trace=trace,
            )
            candidates = _json_array_items(generated.content, "items")
            used_labels = list(generated.build.metadata["labels"])
            used_chunks = list(generated.build.metadata["chunks"])
        except Exception:
            candidates, used_labels, used_chunks = [], selected_labels, selected_chunks
            rejected["provider_or_json_error"] += batch_count
        valid_labels = set(used_labels)
        used_evidence = dict(zip(used_labels, used_chunks))
        yielded = 0
        for candidate in candidates:
            item, reason = validator(candidate, valid_labels, used_evidence)
            if not item:
                rejected[reason or "invalid"] += 1
                continue
            text = " ".join(str(item.get(field) or "") for field in ("question", "front", "back", "explanation"))
            if not text_matches_language(text, language):
                rejected["language_mismatch"] += 1
                continue
            if not _semantic_unique(item, provisional, kind):
                rejected["semantic_duplicate"] += 1
                continue
            provisional.append(item)
            yielded += 1
            if len(provisional) >= count:
                break
        cursor += batch_count
        candidate_rounds += 1
        consecutive_zero = consecutive_zero + 1 if yielded == 0 else 0
        if yielded < batch_count and batch_size > 1:
            batch_size = 1
            remaining_items = max(0, count - len(provisional))
            max_candidate_rounds = max(max_candidate_rounds, candidate_rounds + remaining_items + 2)
    stop_reason = "target_met" if len(provisional) >= count else "consecutive_zero_yield" if consecutive_zero >= 2 else stop_reason
    if tier == "full" and provisional:
        if reporter:
            reporter.update("audit", "聚合审校全部候选", 0.76, current=len(provisional), total=count, unit="项")
        try:
            indexes, audit_issues = await _audit_candidates(kind, provisional, evidence_by_label, trace)
            audit_rounds += 1
            accepted = [provisional[index] for index in indexes]
            rejected["aggregate_audit"] += len(provisional) - len(accepted)
            if audit_issues:
                rejected["audit_reported_issues"] += len(audit_issues)
        except Exception:
            accepted = provisional[:count]
            audit_fallback = True
            trace.mark_fallback()
            rejected["audit_provider_or_json_error"] += 1
    else:
        accepted = provisional[:count]
    if tier == "full" and len(accepted) < count:
        deficit = min(batch_size, count - len(accepted))
        concepts = [blueprint[(cursor + offset) % len(blueprint)] for offset in range(deficit)]
        try:
            generated = await budgeted_chat(
                lambda budget: _evidence_build(budget, chunks, labels, _candidate_prompt(kind, concepts, deficit, language, custom_prompt) + "\n此前聚合审校淘汰了一些候选；只补充新的、证据更直接的项目。"),
                json_mode=True,
                max_tokens=structured_output_tokens(max(900, deficit * (900 if kind == "quiz" else 500))),
                minimum_output_tokens=320,
                trace=trace,
            )
            recovery: list[dict[str, Any]] = []
            for candidate in _json_array_items(generated.content, "items"):
                item, reason = validator(candidate, set(generated.build.metadata["labels"]), dict(zip(generated.build.metadata["labels"], generated.build.metadata["chunks"])))
                if item and text_matches_language(" ".join(str(item.get(field) or "") for field in ("question", "front", "back", "explanation")), language) and _semantic_unique(item, accepted + recovery, kind):
                    recovery.append(item)
                elif not item:
                    rejected[reason or "recovery_invalid"] += 1
            if recovery:
                indexes, _ = await _audit_candidates(kind, recovery, evidence_by_label, trace)
                audit_rounds += 1
                accepted.extend(recovery[index] for index in indexes)
            stop_reason = "recovery_completed" if len(accepted) >= count else "recovery_exhausted"
        except Exception:
            rejected["recovery_provider_or_json_error"] += deficit
            stop_reason = "recovery_exhausted"
    accepted = accepted[:count]
    if kind == "quiz":
        accepted = [_balance_answer(item, index % 4) for index, item in enumerate(accepted)]
    minimum = min(3, count)
    if len(accepted) < minimum:
        raise ValueError(f"模型只生成了 {len(accepted)} 个通过证据校验的内容；请降低难度或切换更强的 MAIN Provider")
    for index, item in enumerate(accepted, start=1):
        item["id"] = f"q{index}" if kind == "quiz" else f"c{index}"
    used_labels = list(dict.fromkeys(label for item in accepted for label in item["citations"]))
    citations = [all_citations[label] for label in used_labels if label in all_citations]
    partial = len(accepted) < count or audit_fallback
    if partial or blueprint_fallback:
        trace.mark_fallback()
    quality_report = {
        "pipeline_tier": tier,
        "tier_reason": profile["reason"],
        "requested_count": count,
        "generated_count": len(accepted),
        "partial": partial,
        "blueprint_fallback": blueprint_fallback,
        "rejected": dict(rejected),
        "source_coverage": len({citation["source_id"] for citation in citations}),
        "selected_sources": len(ids),
        "difficulty_requested": difficulty,
        "difficulty_effective": "easy+medium" if tier == "lite" and difficulty == "mixed" else difficulty,
        "candidate_rounds": candidate_rounds,
        "audit_rounds": audit_rounds,
        "audit_fallback": audit_fallback,
        "stop_reason": stop_reason,
    }
    payload = {"version": 2, "items": accepted, "quality_report": quality_report, "degraded": partial or blueprint_fallback, "context_usage": trace.as_dict(), "language_selection": language_selection}
    artifact_id = f"artifact_{job_id.removeprefix('job_')}" if job_id else new_id("artifact")
    now = utc_now()
    status = "partial" if partial else "ready"
    DB.execute(
        "INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (artifact_id, notebook_id, kind, "单选题库" if kind == "quiz" else "闪卡组", json_dump(ids), language, status, json_dump(payload), json_dump(citations), None, now, now),
    )
    if reporter:
        reporter.update("persist", "保存题库" if kind == "quiz" else "保存闪卡", 0.94, current=len(accepted), total=count, unit="题" if kind == "quiz" else "张")
    return {"id": artifact_id, "type": kind, "status": status, "payload": payload, "citations": citations}
