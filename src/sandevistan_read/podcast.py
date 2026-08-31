from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from typing import Any, Callable

from .context_budget import ContextUsage, PromptBudget, estimate_messages_tokens, pack_items
from .database import DB, json_load
from .providers import PromptBuild, budgeted_chat
from .retrieval import retrieve
from .services import _evenly_spaced, scope_hash, source_scope


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?:%|％)?")
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+|\n+")


def _segment_prompt_build(
    budget: PromptBudget,
    *,
    prefix: str,
    items: list[dict[str, Any]],
    renderer: Callable[[dict[str, Any]], str],
    group_key: Callable[[dict[str, Any]], str] | None = None,
) -> PromptBuild:
    empty = [{"role": "user", "content": prefix}]
    available = max(0, budget.input_tokens - estimate_messages_tokens(empty, budget.image_tokens_per_image) - 8)
    packed = pack_items(items, renderer, available, group_key=group_key)
    return PromptBuild(
        [{"role": "user", "content": prefix + "\n".join(packed.texts)}],
        packed.total,
        len(packed.items),
        packed.truncated,
        {"items": packed.items},
    )


def resolve_podcast_language(source_ids: list[str], requested: str) -> str:
    if requested != "auto":
        return requested
    chinese = 0
    english = 0
    for source_id in source_ids:
        row = DB.fetchone("SELECT filename,metadata_json FROM sources WHERE id=?", (source_id,)) or {}
        metadata = json_load(row.get("metadata_json"), {})
        declared = str(metadata.get("language", "")).lower()
        if declared.startswith("en"):
            english += 1
        elif declared.startswith("zh") or re.search(r"[\u3400-\u9fff]", row.get("filename", "")):
            chinese += 1
    return "en" if english > chinese else "zh-CN"


def estimate_auto_minutes(chapter_count: int, evidence_count: int) -> int:
    if evidence_count < 4:
        return max(5, evidence_count * 2)
    # Complexity grows with both thematic breadth and the evidence map, while
    # the square root prevents very long books from automatically becoming
    # unwieldy.  A compact but dense paper still receives enough room.
    return max(12, min(25, round(12 + chapter_count + math.sqrt(evidence_count) / 2)))


def target_turn_count(minutes: int) -> int:
    # Reference NotebookLM episodes average roughly 6–8 short speaker turns per
    # minute.  Shorter turns sound conversational and avoid dense TTS monologues.
    return max(30, min(200, round(minutes * 8)))


def _with_locator(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["locator"] = json_load(result.pop("locator_json", None), {})
    return result


def _podcast_chunk_quality(row: dict[str, Any]) -> bool:
    content = re.sub(r"\s+", " ", str(row.get("content") or "")).strip()
    lowered = content.lower()
    if len(content) < 120:
        return False
    if lowered.startswith(("references ", "bibliography ", "index ")):
        return False
    if lowered.count("http://") + lowered.count("https://") >= 2:
        return False
    if any(marker in lowered for marker in ("table of contents", "words of thanks", "deep gratitude", "acknowledgments")):
        return False
    locator = row.get("locator") if isinstance(row.get("locator"), dict) else json_load(row.get("locator_json"), {})
    if locator.get("kind") == "epub" and isinstance(locator.get("spine"), int):
        source = DB.fetchone("SELECT metadata_json FROM sources WHERE id=?", (row.get("source_id"),)) or {}
        spine_items = int(json_load(source.get("metadata_json"), {}).get("spine_items") or 0)
        if spine_items:
            lower = max(2, round(spine_items * 0.18))
            upper = max(lower + 1, round(spine_items * 0.88))
            if locator["spine"] < lower or locator["spine"] >= upper:
                return False
    return True


def select_podcast_evidence(notebook_id: str, source_ids: list[str], focus: str, per_source: int = 20) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    focus_ids: set[str] = set()
    queries = [
        focus.strip(),
        "central thesis core argument key mechanism",
        "important example evidence causal explanation",
        "conclusion implication limitation",
    ]
    for query in dict.fromkeys(value for value in queries if value):
        focus_rows = retrieve(
            notebook_id,
            query,
            source_ids,
            limit=min(12, max(6, len(source_ids) * 4)),
            ensure_source_coverage=True,
        )
        focus_rows = [row for row in focus_rows if _podcast_chunk_quality(row)]
        focus_ids.update(row["id"] for row in focus_rows)
        selected.extend(focus_rows)
    for source_id in source_ids:
        rows = DB.fetchall("SELECT * FROM chunks WHERE source_id=? ORDER BY ordinal", (source_id,))
        trim = max(1, round(len(rows) * 0.02)) if len(rows) > 10 else 0
        structural_pool = rows[trim:-trim] if trim and len(rows) > trim * 2 else rows
        structural = [
            row for row in _evenly_spaced(structural_pool, max(8, per_source - 4)) if _podcast_chunk_quality(row)
        ]
        for row in structural:
            if row["id"] not in focus_ids:
                selected.append(_with_locator(row))
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_source_counts: dict[str, int] = {}
    for row in selected:
        if row["id"] in seen or per_source_counts.get(row["source_id"], 0) >= per_source:
            continue
        if "locator" not in row:
            row = _with_locator(row)
        seen.add(row["id"])
        per_source_counts[row["source_id"]] = per_source_counts.get(row["source_id"], 0) + 1
        unique.append(row)
    return unique[:64]


def build_evidence_cards(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cards: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    source_names = {row["id"]: row["filename"] for row in DB.fetchall("SELECT id,filename FROM sources")}
    for index, row in enumerate(rows, start=1):
        evidence_id = f"E{index}"
        content = re.sub(r"\s+", " ", row["content"]).strip()
        card = {
            "id": evidence_id,
            "chunk_id": row["id"],
            "source_id": row["source_id"],
            "filename": source_names.get(row["source_id"], "未知来源"),
            "locator": row.get("locator") or {},
            "content": content,
        }
        cards.append(card)
        citations.append(
            {
                "id": evidence_id,
                "source_id": row["source_id"],
                "chunk_id": row["id"],
                "filename": card["filename"],
                "locator": card["locator"],
                "quote": content[:360],
            }
        )
    return cards, citations


def _extract_json(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _fallback_outline(cards: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    chapter_count = max(4, min(8, round(math.sqrt(max(1, len(cards))))))
    chapters: list[dict[str, Any]] = []
    for index in range(chapter_count):
        group = cards[index::chapter_count]
        if not group:
            continue
        locator = group[0]["locator"]
        position = locator.get("section") or (f"第{locator.get('page')}页" if locator.get("page") else "资料线索")
        title = f"核心线索 {index + 1} · {position}" if language != "en" else f"Core thread {index + 1} · {position}"
        chapters.append({"id": f"chapter_{index + 1}", "title": title, "purpose": title, "evidence_ids": [item["id"] for item in group[:8]]})
    return chapters


async def create_podcast_outline(cards: list[dict[str, Any]], language: str, focus: str, trace: ContextUsage | None = None) -> tuple[list[dict[str, Any]], bool]:
    language_rule = "只使用简体中文" if language != "en" else "Use English only"
    prompt_prefix = f"""你是严格依据资料的深度播客主编。{language_rule}。
只规划结构，不写脚本。把证据组织成 4 到 8 个逻辑递进的主题；优先解释机制、因果、反直觉点和资料内案例。禁止加入资料外背景。
输出严格 JSON：{{"chapters":[{{"title":"","purpose":"","evidence_ids":["E1"]}}]}}。
每章使用 3 到 8 个证据编号；只能使用已有 E 编号；多份资料时必须覆盖每一份。用户关注：{focus or '整体深度解读'}

证据：
"""
    valid_ids = {card["id"] for card in cards}
    source_by_id = {card["id"]: card["source_id"] for card in cards}
    try:
        generated = await budgeted_chat(
            lambda budget: _segment_prompt_build(
                budget,
                prefix=prompt_prefix,
                items=cards,
                renderer=lambda card: f"[{card['id']}] {card['filename']} · {card['content'][:520]}",
                group_key=lambda card: str(card["source_id"]),
            ),
            json_mode=True,
            max_tokens=2500,
            minimum_output_tokens=256,
            temperature=0.2,
            trace=trace,
        )
        raw = generated.content
        valid_ids = {card["id"] for card in generated.build.metadata["items"]}
        candidate = _extract_json(raw).get("chapters") or []
    except Exception:
        if trace:
            trace.mark_fallback()
        candidate = []
    chapters: list[dict[str, Any]] = []
    for item in candidate[:8]:
        evidence_ids = [value for value in item.get("evidence_ids", []) if value in valid_ids]
        title = str(item.get("title", "")).strip()
        if title and evidence_ids:
            chapters.append(
                {
                    "id": f"chapter_{len(chapters) + 1}",
                    "title": title[:120],
                    "purpose": str(item.get("purpose") or title)[:300],
                    "evidence_ids": list(dict.fromkeys(evidence_ids))[:8],
                }
            )
    degraded = len(chapters) < 4
    if degraded:
        chapters = _fallback_outline(cards, language)
    # Small local models sometimes return attractive chapter titles backed by a
    # single excerpt.  A chapter needs multiple independent passages to sustain
    # a grounded conversation, so deterministically widen sparse chapters.
    for chapter_index, chapter in enumerate(chapters):
        required = min(4, len(cards))
        existing = list(chapter["evidence_ids"])
        for offset in range(len(cards)):
            candidate = cards[(chapter_index * 3 + offset) % len(cards)]["id"]
            if candidate not in existing:
                existing.append(candidate)
            if len(existing) >= required:
                break
        chapter["evidence_ids"] = existing[:8]
    covered_sources = {source_by_id[evidence_id] for chapter in chapters for evidence_id in chapter["evidence_ids"]}
    all_sources = {card["source_id"] for card in cards}
    missing_sources = list(all_sources - covered_sources)
    for source_id in missing_sources:
        card = next(card for card in cards if card["source_id"] == source_id)
        supplement = {
            "id": f"chapter_{min(len(chapters) + 1, 8)}",
            "title": f"补充资料 · {card['filename']}" if language != "en" else f"Additional source · {card['filename']}",
            "purpose": "确保所选资料均进入节目主线",
            "evidence_ids": [item["id"] for item in cards if item["source_id"] == source_id][:6],
        }
        if len(chapters) < 8:
            chapters.append(supplement)
        else:
            supplement["id"] = chapters[-1]["id"]
            chapters[-1] = supplement
    return chapters[:8], degraded


def _normalize_text(value: str) -> str:
    value = re.sub(r"\[(?:E|S)\d+\]", "", value)
    value = re.sub(r"^(?:HOST_)?[AB]\s*[:：]\s*", "", value.strip(), flags=re.I)
    return re.sub(r"\s+", " ", value).strip(" -—")


def _similar(left: str, right: str) -> float:
    a = re.sub(r"[\W_]+", "", left).lower()
    b = re.sub(r"[\W_]+", "", right).lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _numbers_supported(text: str, evidence: str) -> bool:
    return all(number.replace(",", "") in evidence.replace(",", "") for number in NUMBER_PATTERN.findall(text))


def _sentences(content: str) -> list[str]:
    values = [item.strip() for item in SENTENCE_PATTERN.split(content) if len(item.strip()) >= 10]
    if not values and content:
        values = [content[:140]]
    return values


def _grounded_question(chapter_title: str, index: int, language: str) -> str:
    topic = chapter_title[:24]
    cycle = index // 7
    if language == "en":
        templates = [
            f"On {topic}, what does this passage establish?",
            f"The next source passage adds a mechanism to our account of {topic}.",
            f"Without going beyond the text, what can we conclude about {topic}?",
            f"How does the source make {topic} more concrete?",
            f"One source detail is especially important to {topic} at this step.",
            f"Let's continue with the sequence the source gives for {topic}.",
            f"How does this evidence refine our view of {topic}?",
        ]
        prefixes = ["", "Going one level deeper, ", "From another source passage, "]
    else:
        templates = [
            f"围绕“{topic}”，先把论证落到原文：这条材料明确说明了什么？",
            f"顺着“{topic}”这条主线，下一段证据补上了一个具体机制。",
            f"如果不做资料外推演，我们从“{topic}”这里能确定什么？",
            f"原文是怎样把“{topic}”进一步说具体的？",
            f"走到“{topic}”这一步，先保留材料中的一个关键细节。",
            f"沿着原文对“{topic}”的说明，我们继续看实际发生的过程。",
            f"回到“{topic}”这个主题，这条证据补充了什么？",
        ]
        prefixes = ["", "再往前推进一层，", "换一段原文来看，"]
    return prefixes[min(cycle, len(prefixes) - 1)] + templates[index % len(templates)]


def _safe_chapter_turns(cards: list[dict[str, Any]], target: int, language: str) -> list[dict[str, Any]]:
    facts: list[tuple[str, str]] = []
    for card in cards:
        for sentence in _sentences(card["content"])[:4]:
            if language != "en" and not re.search(r"[\u3400-\u9fff]", sentence):
                continue
            facts.append((sentence[:95] if language != "en" else " ".join(sentence.split()[:42]), card["id"]))
    turns: list[dict[str, Any]] = []
    prompts_zh = ["这里先抓住一个关键问题：这条线索究竟说明了什么？", "换个角度追问：它为什么会成为整套论证的关键？", "如果继续往下推，这条机制会带来什么结果？", "先停一下：资料为这个判断提供了什么依据？", "这和前面的线索如何连接起来？", "真正值得追问的是：这个变化解决了哪一个难题？"]
    prompts_en = ["What is the central question behind this evidence?", "Why does this point matter to the larger argument?", "What follows if we carry this mechanism forward?", "What support does the source give for that conclusion?", "How does this connect to the previous thread?", "Which problem is this mechanism designed to solve?"]
    prompts = prompts_en if language == "en" else prompts_zh
    for index, (fact, evidence_id) in enumerate(facts):
        if len(turns) >= target:
            break
        if len(turns) % 2 == 0:
            turns.append({"text": prompts[(index // 2) % len(prompts)], "citation_ids": [evidence_id], "safe": True})
        if len(turns) < target:
            prefix = "The source states: " if language == "en" else "资料给出的直接线索是："
            turns.append({"text": prefix + fact, "citation_ids": [evidence_id], "safe": True})
    return turns[:target]


async def _critic_invalid_indexes(turns: list[dict[str, Any]], cards: list[dict[str, Any]], language: str, trace: ContextUsage | None = None) -> set[int]:
    transcript = "\n".join(f"{index}: {turn['text']} ({','.join(turn['citation_ids'])})" for index, turn in enumerate(turns))
    prompt_prefix = f"""你是事实审校器。逐条判断播客文本是否能被它标注的证据直接支持。反问或过渡可以通过；逻辑矛盾、资料外数字/实体、错误因果必须判为不支持。
只输出 JSON：{{"invalid_indexes":[0]}}。不要改写文本。语言={language}。
文本：
{transcript}
证据：
"""
    try:
        raw = (await budgeted_chat(
            lambda budget: _segment_prompt_build(
                budget,
                prefix=prompt_prefix,
                items=cards,
                renderer=lambda card: f"[{card['id']}] {card['content'][:900]}",
            ),
            json_mode=True,
            max_tokens=800,
            minimum_output_tokens=128,
            temperature=0.0,
            trace=trace,
        )).content
        values = _extract_json(raw).get("invalid_indexes") or []
        return {int(value) for value in values if isinstance(value, int) or str(value).isdigit()}
    except Exception:
        return set()


async def _critic_grounded_pairs(pairs: list[dict[str, Any]], language: str, trace: ContextUsage | None = None) -> set[int]:
    answers = "\n".join(f"{index}: {pair['answer']}" for index, pair in enumerate(pairs))
    prompt_prefix = f"""你是严格的翻译忠实度审校器。逐项比较原文摘录与回答。回答必须只是摘录的忠实翻译或压缩改述；若新增因果、绝对化结论、实体、数字或摘录没有的判断，就判为不支持。
只输出 JSON：{{"invalid_indexes":[0]}}。语言={language}。
回答：
{answers}
原文摘录：
"""
    indexed_pairs = [{**pair, "index": index} for index, pair in enumerate(pairs)]
    try:
        raw = (await budgeted_chat(
            lambda budget: _segment_prompt_build(
                budget,
                prefix=prompt_prefix,
                items=indexed_pairs,
                renderer=lambda pair: f"{pair['index']}: {pair['support_quote']}",
            ),
            json_mode=True,
            max_tokens=700,
            minimum_output_tokens=128,
            temperature=0.0,
            trace=trace,
        )).content
        values = _extract_json(raw).get("invalid_indexes") or []
        return {int(value) for value in values if isinstance(value, int) or str(value).isdigit()}
    except Exception:
        return set()


async def create_chapter_turns(
    chapter: dict[str, Any],
    cards_by_id: dict[str, dict[str, Any]],
    target: int,
    language: str,
    episode_context: str,
    trace: ContextUsage | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    cards = [cards_by_id[evidence_id] for evidence_id in chapter["evidence_ids"] if evidence_id in cards_by_id]
    language_rule = "只用自然的简体中文口语" if language != "en" else "Use natural spoken English only"
    pair_target = max(2, math.ceil(target / 2))
    quote_bank: list[dict[str, str]] = []
    quotes_by_evidence: list[tuple[str, list[str]]] = []
    for card in cards:
        card_quotes: list[str] = []
        for sentence in _sentences(card["content"]):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if len(sentence) < 32:
                continue
            if sentence[:1].islower():
                continue
            if len(sentence) > 260:
                sentence = sentence[:260].rsplit(" ", 1)[0]
            lowered = sentence.lower()
            if "references [" in lowered or "http://" in lowered or "https://" in lowered:
                continue
            if any("\ue000" <= character <= "\uf8ff" for character in sentence) or "∑" in sentence or "#include" in lowered:
                continue
            visible = [character for character in sentence if not character.isspace()]
            if visible and sum(character.isalpha() for character in visible) / len(visible) < 0.52:
                continue
            if sentence.count("{") + sentence.count("}") + sentence.count(";") >= 4:
                continue
            card_quotes.append(sentence)
            if len(card_quotes) >= 8:
                break
        quotes_by_evidence.append((card["id"], card_quotes))
    for quote_index in range(8):
        for evidence_id, values in quotes_by_evidence:
            if quote_index < len(values):
                quote_bank.append(
                    {"id": f"Q{len(quote_bank) + 1}", "evidence_id": evidence_id, "text": values[quote_index]}
                )
    quote_by_id = {item["id"]: item for item in quote_bank}
    pairs: list[dict[str, Any]] = []
    attempted_ids: set[str] = set()

    for attempt in range(max(3, pair_target * 2)):
        remaining = pair_target - len(pairs)
        if remaining <= 0:
            break
        available = [item for item in quote_bank if item["id"] not in attempted_ids]
        if not available:
            break
        used = ", ".join(sorted(attempted_ids)) or "（无）"
        prompt_prefix = f"""你是严格依据资料的播客事实编辑。{language_rule}。只依据下列证据，不得补充常识、联想、评价或资料外因果。
输出严格 JSON：{{"pairs":[{{"quote_id":"Q1","answer":""}}]}}。
每项选择一个尚未使用的 Q 编号。answer 只能忠实翻译该 Q 摘录中的一个明确事实，原文过长时才压缩；不能提出问题。中文播客的 answer 必须是简体中文，不能直接复制英文原句；中文 25–90 字，英文 8–40 词。技术术语宁可保留英文也不要猜译；nonce 译为“随机数（nonce）”或“计数值（nonce）”，不得译为“非空值”。不要报幕、不要念编号。
本章：{chapter['title']}；目的：{chapter['purpose']}；前文：{episode_context}
不要再使用这些摘录：
{used}
预切分的逐字原文摘录（Q 编号 | 引用编号）：
"""
        try:
            def build(budget: PromptBudget) -> PromptBuild:
                requested = max(1, min(remaining, max(1, (budget.output_tokens - 120) // 160)))
                prefix = prompt_prefix.replace(
                    "每项选择一个尚未使用的 Q 编号。",
                    f"生成 {requested} 项，每项选择一个尚未使用的 Q 编号。",
                )
                return _segment_prompt_build(
                    budget,
                    prefix=prefix,
                    items=available,
                    renderer=lambda item: f"[{item['id']}|{item['evidence_id']}] {item['text']}",
                )

            generated = await budgeted_chat(
                build,
                json_mode=True,
                max_tokens=min(4200, max(1600, remaining * 240)),
                minimum_output_tokens=256,
                temperature=0.25,
                trace=trace,
            )
            candidates = _extract_json(generated.content).get("pairs") or []
        except Exception:
            candidates = []
        new_pairs: list[dict[str, Any]] = []
        for item in candidates:
            answer = _normalize_text(str(item.get("answer", "")))
            quote_id = str(item.get("quote_id", "")).strip().upper()
            quote = quote_by_id.get(quote_id)
            was_attempted = quote_id in attempted_ids
            if quote_id:
                attempted_ids.add(quote_id)
            if not answer or not quote or was_attempted:
                continue
            if language != "en" and len(re.findall(r"[\u3400-\u9fff]", answer)) < 12:
                continue
            if any(pair["quote_id"] == quote_id for pair in pairs + new_pairs):
                continue
            if not _numbers_supported(answer, quote["text"]):
                continue
            if answer.count("?") + answer.count("？") > 0:
                continue
            if any(_similar(answer, previous["answer"]) > 0.78 for previous in pairs + new_pairs):
                continue
            new_pairs.append(
                {
                    "answer": answer[:110] if language != "en" else " ".join(answer.split()[:45]),
                    "support_quote": quote["text"],
                    "quote_id": quote_id,
                    "citation_ids": [quote["evidence_id"]],
                }
            )
            if len(pairs) + len(new_pairs) >= pair_target:
                break
        pairs.extend(new_pairs)

    selected_pairs = pairs[:pair_target]
    accepted: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(selected_pairs):
        accepted.extend(
            [
                {
                    "text": _grounded_question(chapter["title"], pair_index, language),
                    "citation_ids": pair["citation_ids"],
                    "safe": False,
                },
                {"text": pair["answer"], "citation_ids": pair["citation_ids"], "safe": False},
            ]
        )

    generated_count = len(accepted)
    # Emergency fallback remains fully traceable.  It is mainly for unavailable
    # providers; normal local-model operation should fill the chapter above.
    if len(accepted) < target:
        for item in _safe_chapter_turns(cards, target * 3, language):
            if len(accepted) >= target:
                break
            if any(_similar(item["text"], previous["text"]) > 0.78 for previous in accepted):
                continue
            accepted.append(item)
    degraded = generated_count < max(4, round(target * 0.8))
    if degraded and trace:
        trace.mark_fallback()
    return accepted[:target], degraded


def _quality_metrics(
    turns: list[dict[str, Any]], citations: list[dict[str, Any]], target_minutes: int, requested_turns: int | None = None
) -> dict[str, Any]:
    a_chars = sum(len(turn["text"]) for turn in turns if turn["speaker"] == "HOST_A")
    b_chars = sum(len(turn["text"]) for turn in turns if turn["speaker"] == "HOST_B")
    total = max(1, a_chars + b_chars)
    valid_ids = {citation["id"] for citation in citations}
    joined = " ".join(turn["text"] for turn in turns)
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", joined))
    latin_words = len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", joined))
    estimated_minutes = cjk_chars / 270 + latin_words / 150 + len(turns) * 0.45 / 60
    requested_turns = requested_turns or target_turn_count(target_minutes)
    return {
        "target_minutes": target_minutes,
        "estimated_minutes": round(estimated_minutes, 1),
        "target_turn_count": requested_turns,
        "completion_ratio": round(min(1.0, len(turns) / max(1, requested_turns)), 3),
        "turn_count": len(turns),
        "host_a_ratio": round(a_chars / total, 3),
        "host_b_ratio": round(b_chars / total, 3),
        "questions": sum("?" in turn["text"] or "？" in turn["text"] for turn in turns),
        "all_turns_cited": all(turn["citation_ids"] and set(turn["citation_ids"]) <= valid_ids for turn in turns),
        "safe_fallback_turns": sum(bool(turn.get("safe")) for turn in turns),
    }


async def build_podcast_script(
    notebook_id: str,
    payload: dict[str, Any],
    *,
    progress: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    ids = source_scope(notebook_id, payload.get("source_ids"))
    if not ids:
        raise ValueError("当前范围没有已就绪的文档")
    language = resolve_podcast_language(ids, payload.get("language", "zh-CN"))
    focus = str(payload.get("focus") or "").strip()
    if progress:
        progress("构建全篇证据地图", 0.08)
    rows = select_podcast_evidence(notebook_id, ids, focus)
    cards, all_citations = build_evidence_cards(rows)
    if len(cards) < 2:
        raise ValueError("资料内容不足，无法生成深度播客")
    if progress:
        progress("规划节目主题", 0.14)
    context_usage = ContextUsage()
    chapters, outline_degraded = await create_podcast_outline(cards, language, focus, context_usage)
    duration_mode = payload.get("duration_mode") or ("fixed" if payload.get("minutes") else "auto")
    requested_minutes = int(payload.get("minutes") or 0) or None
    target_minutes = requested_minutes if duration_mode == "fixed" and requested_minutes else estimate_auto_minutes(len(chapters), len(cards))
    total_target = (
        max(24, round(target_minutes * 6.5)) if duration_mode == "fixed" else target_turn_count(target_minutes)
    )
    chapter_targets = [4 for _ in chapters]
    cursor = 0
    while sum(chapter_targets) < total_target:
        chapter_targets[cursor % len(chapter_targets)] += 2
        cursor += 1
    cards_by_id = {card["id"]: card for card in cards}
    turns: list[dict[str, Any]] = []
    degraded = outline_degraded
    covered_titles: list[str] = []
    chapter_payloads: list[dict[str, Any]] = []
    for chapter_index, chapter in enumerate(chapters):
        if progress:
            progress(f"编写章节 {chapter_index + 1}/{len(chapters)}", 0.17 + 0.20 * chapter_index / max(1, len(chapters)))
        episode_context = "；".join(covered_titles[-4:]) or ("这是节目开篇" if language != "en" else "This opens the episode")
        chapter_turns, chapter_degraded = await create_chapter_turns(
            chapter, cards_by_id, chapter_targets[chapter_index], language, episode_context, context_usage
        )
        start_index = len(turns)
        for item in chapter_turns:
            item["speaker"] = "HOST_A" if len(turns) % 2 == 0 else "HOST_B"
            item["id"] = f"turn_{len(turns) + 1}"
            item["chapter_id"] = chapter["id"]
            turns.append(item)
        chapter_payloads.append({**chapter, "turn_start": start_index, "turn_end": len(turns) - 1})
        covered_titles.append(chapter["title"])
        degraded = degraded or chapter_degraded
    used_evidence = {evidence_id for turn in turns for evidence_id in turn["citation_ids"]}
    used_citations = [citation for citation in all_citations if citation["id"] in used_evidence]
    remap = {citation["id"]: f"S{index}" for index, citation in enumerate(used_citations, start=1)}
    citations = [{**citation, "id": remap[citation["id"]]} for citation in used_citations]
    for turn in turns:
        turn["citation_ids"] = [remap[value] for value in turn["citation_ids"] if value in remap]
    if degraded:
        context_usage.mark_fallback()
    script = "\n".join(f"{turn['speaker']}: {turn['text']} {' '.join(f'[{value}]' for value in turn['citation_ids'])}" for turn in turns)
    return {
        "version": 2,
        "language": language,
        "source_ids": ids,
        "scope_hash": scope_hash(ids),
        "duration": {"mode": duration_mode, "requested_minutes": requested_minutes, "target_minutes": target_minutes},
        "chapters": chapter_payloads,
        "turns": turns,
        "script": script,
        "citations": citations,
        "degraded": degraded,
        "context_usage": context_usage.as_dict(),
        "quality": _quality_metrics(turns, citations, target_minutes, total_target),
    }
