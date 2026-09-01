from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

from .context_budget import ContextUsage, PromptBudget, TokenLimits, estimate_messages_tokens, pack_items, structured_output_tokens
from .database import DB, json_load
from .providers import PromptBuild, active_provider, budgeted_chat, study_generation_profile
from .retrieval import retrieve, tokenize
from .services import _evenly_spaced, scope_hash, source_scope
from .languages import resolve_output_language, text_matches_language


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?:%|％)?")
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+|\n+")
PODCAST_ENGINE_VERSION = 3
NONFACTUAL_ACTS = {"intro", "bridge", "question", "acknowledgement", "outro"}
FACTUAL_ACTS = {"frame", "explain", "evidence", "example", "challenge", "synthesis"}
ALLOWED_DIALOGUE_ACTS = NONFACTUAL_ACTS | FACTUAL_ACTS
GENERIC_STEMS = (
    "这条材料明确说明了什么",
    "如果不做资料外推演",
    "原文是怎样把",
    "资料给出的直接线索是",
    "what does this passage establish",
    "without going beyond the text",
)


class PodcastQualityError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report or {"passed": False, "reason": message}


@dataclass
class EpisodeMemory:
    thesis: str
    covered_claim_ids: list[str] = field(default_factory=list)
    chapter_summaries: list[dict[str, str]] = field(default_factory=list)
    open_hook: str = ""
    last_turns: list[dict[str, Any]] = field(default_factory=list)
    last_speaker: str | None = None

    def prompt_payload(self, recent_limit: int) -> dict[str, Any]:
        return {
            "episode_thesis": self.thesis,
            "covered_claim_ids": self.covered_claim_ids[-24:],
            "chapter_summaries": self.chapter_summaries[-6:],
            "open_hook": self.open_hook,
            "recent_dialogue": [
                {"speaker": turn["speaker"], "text": turn["text"], "dialogue_act": turn["dialogue_act"]}
                for turn in self.last_turns[-recent_limit:]
            ],
            "last_speaker": self.last_speaker,
        }


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
    return resolve_output_language(DB, source_ids, requested)[0]


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
    return max(24, min(160, round(minutes * 6.5)))


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


def _extract_array(raw: str, key: str) -> list[dict[str, Any]] | None:
    """Recover only complete objects from a possibly truncated JSON list."""
    parsed = _extract_json(raw)
    if isinstance(parsed.get(key), list):
        return parsed[key]
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', raw)
    if not match:
        return None
    decoder = json.JSONDecoder()
    position = match.end()
    recovered: list[dict[str, Any]] = []
    while position < len(raw):
        while position < len(raw) and (raw[position].isspace() or raw[position] == ","):
            position += 1
        if position >= len(raw) or raw[position] == "]":
            break
        try:
            value, consumed = decoder.raw_decode(raw[position:])
        except json.JSONDecodeError:
            break
        if not isinstance(value, dict):
            break
        recovered.append(value)
        position += consumed
    return recovered or None


def _extract_turns(raw: str) -> list[dict[str, Any]] | None:
    return _extract_array(raw, "turns")


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


def build_claim_ledger(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create compact, source-addressable claims without asking the model to invent facts."""
    claims: list[dict[str, Any]] = []
    for card in cards:
        candidates = []
        for sentence in _sentences(card["content"]):
            text = re.sub(r"\s+", " ", sentence).strip()
            latin = re.findall(r"[A-Za-z]", text)
            starts_with_fragment = bool(latin) and len(latin) >= len(text.replace(" ", "")) * 0.6 and text[:1].islower()
            if 24 <= len(text) <= 320 and not starts_with_fragment and not text.lower().startswith(("references ", "bibliography ")):
                candidates.append(text)
            if len(candidates) >= 2:
                break
        if not candidates and card["content"]:
            candidates = [str(card["content"])[:260]]
        for text in candidates:
            claims.append(
                {
                    "id": f"C{len(claims) + 1}",
                    "text": text,
                    "evidence_ids": [card["id"]],
                    "source_id": card["source_id"],
                    "filename": card["filename"],
                    "locator": card.get("locator") or {},
                }
            )
            if len(claims) >= 48:
                return claims
    return claims


def _fallback_episode_plan(claims: list[dict[str, Any]], language: str) -> dict[str, Any]:
    chapter_count = max(2, min(6, round(math.sqrt(max(1, len(claims))))))
    size = max(1, math.ceil(len(claims) / chapter_count))
    chapters = []
    for start in range(0, len(claims), size):
        group = claims[start : start + size]
        first = group[0]
        locator = first.get("locator") or {}
        topic = str(locator.get("section") or first.get("filename") or "资料主线")[:64]
        number = len(chapters) + 1
        title = f"{number}. {topic}" if language != "en" else f"{number}. {topic}"
        chapters.append(
            {
                "id": f"chapter_{number}",
                "title": title,
                "purpose": group[0]["text"][:180],
                "claim_ids": [claim["id"] for claim in group],
                "bridge_in": "承接上一部分的结论" if number > 1 and language != "en" else "Build on the previous conclusion" if number > 1 else "",
                "bridge_out": "由当前结论引出下一层问题" if language != "en" else "Use this conclusion to open the next question",
            }
        )
        if len(chapters) >= 6:
            break
    thesis = claims[0]["text"][:220] if claims else ("资料深度解读" if language != "en" else "A grounded deep dive")
    return {"episode_thesis": thesis, "chapters": chapters, "fallback": True}


async def create_episode_plan(
    claims: list[dict[str, Any]], language: str, focus: str, trace: ContextUsage | None = None
) -> tuple[dict[str, Any], bool]:
    language_rule = "只使用自然的简体中文" if language != "en" else "Use natural spoken English only"
    target = max(3, min(6, round(math.sqrt(max(1, len(claims))))))
    prompt_prefix = f"""你是资料型播客的总编。{language_rule}。只规划一条能从问题逐步走向结论的叙事主线，不写对话，不补充资料外事实。
角色固定：HOST_A 负责综合和解释，HOST_B 负责追问、澄清和检验推论。规划 {target} 个逻辑递进章节；每个 claim 只在真正相关的章节使用，不要为了覆盖来源而塞入无关内容。
输出 JSON：{{"episode_thesis":"","chapters":[{{"title":"","purpose":"","claim_ids":["C1"],"bridge_in":"如何承接上一章","bridge_out":"留给下一章的问题"}}]}}。
用户关注：{focus or '整体深度解读'}
可用主张：
"""
    try:
        generated = await budgeted_chat(
            lambda budget: _segment_prompt_build(
                budget,
                prefix=prompt_prefix,
                items=claims,
                renderer=lambda claim: f"[{claim['id']}|{claim['filename']}] {claim['text']}",
                group_key=lambda claim: str(claim["source_id"]),
            ),
            json_mode=True,
            max_tokens=structured_output_tokens(1800),
            minimum_output_tokens=384,
            temperature=0.15,
            trace=trace,
        )
        parsed = _extract_json(generated.content)
        available = {claim["id"] for claim in generated.build.metadata["items"]}
        chapters = []
        used: set[str] = set()
        for item in _extract_array(generated.content, "chapters") or []:
            if not isinstance(item, dict):
                continue
            claim_ids = [str(value) for value in item.get("claim_ids") or [] if str(value) in available and str(value) not in used]
            title = str(item.get("title") or "").strip()
            purpose = str(item.get("purpose") or "").strip()
            if not title or not purpose or not claim_ids:
                continue
            used.update(claim_ids)
            chapters.append(
                {
                    "id": f"chapter_{len(chapters) + 1}",
                    "title": title[:120],
                    "purpose": purpose[:300],
                    "claim_ids": claim_ids[:10],
                    "bridge_in": str(item.get("bridge_in") or "")[:220],
                    "bridge_out": str(item.get("bridge_out") or "")[:220],
                }
            )
            if len(chapters) >= 8:
                break
        thesis = str(parsed.get("episode_thesis") or "").strip()
        if len(chapters) >= 2 and thesis:
            return {"episode_thesis": thesis[:400], "chapters": chapters, "fallback": False}, False
    except Exception:
        pass
    if trace:
        trace.mark_fallback()
    return _fallback_episode_plan(claims, language), True


def podcast_generation_profile() -> dict[str, Any]:
    provider = active_provider("main")
    if not provider:
        raise ValueError("请先启用 MAIN Provider")
    limits = TokenLimits.from_provider(provider)
    study_profile = study_generation_profile(provider)
    tier = "lite" if limits.effective_context_tokens < 8192 or limits.max_output_tokens < 1536 or study_profile["tier"] == "lite" else "full"
    return {
        "tier": tier,
        "provider": provider.get("name"),
        "model": provider.get("model"),
        "effective_context_tokens": limits.effective_context_tokens,
        "max_output_tokens": limits.max_output_tokens,
        "scene_turns": 4 if tier == "lite" else 7,
        "recent_turns": 4 if tier == "lite" else 8,
    }


def _speaker(value: Any) -> str:
    normalized = str(value or "").upper().replace("PERSON", "HOST_").replace("HOST__", "HOST_")
    return {"A": "HOST_A", "B": "HOST_B", "1": "HOST_A", "2": "HOST_B", "HOST_1": "HOST_A", "HOST_2": "HOST_B"}.get(normalized, normalized)


def _claim_evidence_text(claim_ids: list[str], claims_by_id: dict[str, dict[str, Any]], cards_by_id: dict[str, dict[str, Any]]) -> str:
    values = []
    for claim_id in claim_ids:
        claim = claims_by_id[claim_id]
        values.append(claim["text"])
        values.extend(cards_by_id[evidence_id]["content"] for evidence_id in claim["evidence_ids"] if evidence_id in cards_by_id)
    return " ".join(values)


def _is_duplicate(text: str, turns: list[dict[str, Any]]) -> bool:
    return any(_similar(text, turn["text"]) >= 0.84 for turn in turns)


def _infer_claim_id(text: str, claims_by_id: dict[str, dict[str, Any]]) -> str | None:
    target = set(tokenize(text))
    if not target:
        return None
    ranked = []
    for claim_id, claim in claims_by_id.items():
        support = set(tokenize(str(claim["text"])))
        score = len(target & support) / max(1, min(len(target), len(support)))
        ranked.append((score, claim_id))
    ranked.sort(reverse=True)
    if not ranked or ranked[0][0] < 0.25:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.04:
        return None
    return ranked[0][1]


def validate_scene_turns(
    raw_turns: Any,
    claims_by_id: dict[str, dict[str, Any]],
    cards_by_id: dict[str, dict[str, Any]],
    *,
    last_speaker: str | None,
    existing_turns: list[dict[str, Any]],
    language: str,
    expected_count: int,
    scene_kind: str = "chapter",
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(raw_turns, list):
        return [], ["模型没有返回 turns 数组"]
    accepted: list[dict[str, Any]] = []
    issues: list[str] = []
    previous = last_speaker
    for index, source in enumerate(raw_turns[: expected_count + 2]):
        if len(accepted) >= expected_count:
            break
        if not isinstance(source, dict):
            issues.append(f"第 {index + 1} 轮不是对象")
            continue
        supplied_speaker = _speaker(source.get("speaker"))
        text = _normalize_text(str(source.get("text") or ""))
        act = str(source.get("dialogue_act") or "").lower()
        claim_ids = list(dict.fromkeys(str(value).upper() for value in source.get("claim_ids") or [] if str(value).upper() in claims_by_id))
        expected_speaker = "HOST_B" if previous == "HOST_A" else "HOST_A"
        if supplied_speaker not in {"HOST_A", "HOST_B"}:
            issues.append(f"第 {index + 1} 轮说话人无效")
            continue
        speaker = expected_speaker
        if act not in ALLOWED_DIALOGUE_ACTS:
            issues.append(f"第 {index + 1} 轮 dialogue_act 无效")
            continue
        if (scene_kind == "intro" and act == "outro") or (scene_kind in {"chapter", "boundary_repair"} and act in {"intro", "outro"}):
            issues.append(f"第 {index + 1} 轮 dialogue_act 不适合 {scene_kind}")
            continue
        minimum = 8 if language != "en" else 4
        maximum = 150 if language != "en" else 320
        if len(text) < minimum or len(text) > maximum:
            issues.append(f"第 {index + 1} 轮长度不合格")
            continue
        if not text_matches_language(text, language):
            issues.append(f"第 {index + 1} 轮语言不符合输出要求")
            continue
        lowered = text.lower()
        if any(stem in lowered for stem in GENERIC_STEMS):
            issues.append(f"第 {index + 1} 轮使用机械模板")
            continue
        if not claim_ids:
            inferred = _infer_claim_id(text, claims_by_id)
            if inferred:
                claim_ids = [inferred]
        factual = act in FACTUAL_ACTS or bool(claim_ids) or bool(NUMBER_PATTERN.search(text))
        if factual and not claim_ids:
            issues.append(f"第 {index + 1} 轮包含事实但没有 claim_id")
            continue
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for claim_id in claim_ids
                for evidence_id in claims_by_id[claim_id]["evidence_ids"]
                if evidence_id in cards_by_id
            )
        )
        if claim_ids and not _numbers_supported(text, _claim_evidence_text(claim_ids, claims_by_id, cards_by_id)):
            issues.append(f"第 {index + 1} 轮包含资料不支持的数字")
            continue
        if _is_duplicate(text, existing_turns + accepted):
            issues.append(f"第 {index + 1} 轮与已有内容重复")
            continue
        accepted.append(
            {
                "speaker": speaker,
                "text": text,
                "dialogue_act": act,
                "claim_ids": claim_ids,
                "citation_ids": evidence_ids,
                "safe": False,
            }
        )
        previous = speaker
    required = max(1, expected_count - 1)
    if len(accepted) < required:
        issues.append(f"有效轮次不足：{len(accepted)}/{required}")
    return accepted[:expected_count], issues


def _scene_instruction(scene_kind: str, language: str) -> str:
    if language == "en":
        return {
            "intro": "Open with the episode's central question and the two hosts' complementary perspectives; do not preview every answer.",
            "chapter": "Advance one coherent line of reasoning. Each turn must directly answer, qualify, or build on the immediately previous turn.",
            "outro": "Resolve the central question using only claims already discussed, then close naturally without introducing new facts.",
            "boundary_repair": "Rewrite this chapter opening so it responds directly to the preceding exchange and then enters the planned topic.",
        }[scene_kind]
    return {
        "intro": "用核心问题开场，让两位主持人的互补视角自然出现；不要提前罗列所有答案。",
        "chapter": "沿一条推理主线推进；每一轮必须直接回应、修正或承接紧邻的上一轮。",
        "outro": "只用已经讨论过的主张回应开场问题，自然收束，不引入任何新事实。",
        "boundary_repair": "重写本章开头，使其先回应上一段真实对话，再自然进入规划主题。",
    }[scene_kind]


async def _draft_scene(
    *,
    scene_kind: str,
    chapter: dict[str, Any],
    claims: list[dict[str, Any]],
    cards_by_id: dict[str, dict[str, Any]],
    memory: EpisodeMemory,
    existing_turns: list[dict[str, Any]],
    target: int,
    language: str,
    profile: dict[str, Any],
    trace: ContextUsage,
    repair_feedback: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    language_rule = "只输出自然的简体中文口语" if language != "en" else "Use natural spoken English only"
    start_speaker = "HOST_B" if memory.last_speaker == "HOST_A" else "HOST_A"
    memory_json = json.dumps(memory.prompt_payload(profile["recent_turns"]), ensure_ascii=False)
    feedback = "；".join(repair_feedback or [])
    prompt_prefix = f"""你是严格资料内的双人播客编剧。{language_rule}。HOST_A 是主讲与综合者，HOST_B 是敏锐的追问与澄清者；双方都要听见并回应上一轮，使用自然的“接住并推进”方式，禁止机械采访和孤立事实罗列。
{_scene_instruction(scene_kind, language)}
生成 {target} 轮，从 {start_speaker} 开始并严格交替。中文每轮约 35–95 字；英文每轮约 12–35 词。事实、数字、案例、判断必须填写真正支持它的 claim_ids；问句、开场或回应只要复述了资料事实，也必须填 claim_ids。只有“那这意味着什么？”这类完全不含事实的纯承接才可以为空。不得使用资料外常识、轶事或类比，不得念出编号，不得提前结束节目。
输出 JSON：{{"turns":[{{"speaker":"HOST_A|HOST_B","dialogue_act":"intro|frame|bridge|question|acknowledgement|explain|evidence|example|challenge|synthesis|outro","text":"","claim_ids":["C1"]}}]}}。claim_ids 只能从下方允许列表逐字复制，不能省略事实轮的编号。
剧集记忆：{memory_json}
当前部分：{chapter.get('title')}；目的：{chapter.get('purpose')}；承接：{chapter.get('bridge_in')}；后续钩子：{chapter.get('bridge_out')}。
{f'上次草稿问题，必须修复：{feedback}' if feedback else ''}
允许使用的主张：
"""
    generated = await budgeted_chat(
        lambda budget: _segment_prompt_build(
            budget,
            prefix=prompt_prefix,
            items=claims,
            renderer=lambda claim: f"[{claim['id']}|{','.join(claim['evidence_ids'])}] {claim['text']}",
            group_key=lambda claim: str(claim["source_id"]),
        ),
        json_mode=True,
        max_tokens=structured_output_tokens(min(3200, max(900, target * 400))),
        minimum_output_tokens=min(768, max(256, target * 160)),
        temperature=0.45,
        trace=trace,
    )
    available_claims = {claim["id"]: claim for claim in generated.build.metadata["items"]}
    raw_turns = _extract_turns(generated.content)
    if not isinstance(raw_turns, list):
        reason = f"（结束原因：{generated.finish_reason}）" if generated.finish_reason else ""
        return [], [f"模型没有返回可解析的 turns 数组{reason}"]
    validated, issues = validate_scene_turns(
        raw_turns,
        available_claims,
        cards_by_id,
        last_speaker=memory.last_speaker,
        existing_turns=existing_turns,
        language=language,
        expected_count=target,
        scene_kind=scene_kind,
    )
    return validated, issues


async def _audit_scene(
    turns: list[dict[str, Any]],
    claims_by_id: dict[str, dict[str, Any]],
    memory: EpisodeMemory,
    language: str,
    trace: ContextUsage,
) -> dict[str, Any]:
    used = list(dict.fromkeys(claim_id for turn in turns for claim_id in turn["claim_ids"]))
    transcript = "\n".join(
        f"{index}: {turn['speaker']} [{turn['dialogue_act']}] {turn['text']} claims={','.join(turn['claim_ids']) or '-'}"
        for index, turn in enumerate(turns)
    )
    previous = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in memory.last_turns[-4:]) or "（节目开篇）"
    prompt_prefix = f"""你是严格的播客场景审校员。检查：每个事实是否只来自其 claim；第一轮是否自然回应前文；轮次之间是否前言搭后语；HOST_A/HOST_B 角色是否稳定；是否有重复或机械套话。纯过渡可以无 claim。
只输出 JSON：{{"verdict":"pass|fail","invalid_indexes":[],"scores":{{"grounding":5,"continuity":5,"roles":5,"repetition":5}},"issues":[]}}。5=优秀、4=可发布、1=严重失败；没有问题时必须给 pass 和 4–5 分，不能在 issues 为空时给低分。纯承接问句或寒暄没有事实时可以不带 claim，不能仅因此判错。语言={language}。
前文：
{previous}
待审场景：
{transcript}
主张：
"""
    try:
        generated = await budgeted_chat(
            lambda budget: _segment_prompt_build(
                budget,
                prefix=prompt_prefix,
                items=[claims_by_id[claim_id] for claim_id in used if claim_id in claims_by_id],
                renderer=lambda claim: f"[{claim['id']}] {claim['text']}",
            ),
            json_mode=True,
            max_tokens=700,
            minimum_output_tokens=160,
            temperature=0.0,
            trace=trace,
        )
        result = _extract_json(generated.content)
        scores = result.get("scores") or {}
        invalid = [int(value) for value in result.get("invalid_indexes") or [] if str(value).isdigit() and 0 <= int(value) < len(turns)]
        issues = [str(value)[:240] for value in result.get("issues") or []][:8]
        verdict = str(result.get("verdict") or "").lower()
        deterministic_defaults = {"grounding": 5 if not invalid else 2, "continuity": 4, "roles": 5, "repetition": 5}
        normalized_scores = {
            name: int(scores[name]) if str(scores.get(name, "")).isdigit() and 1 <= int(scores[name]) <= 5 else deterministic_defaults[name]
            for name in deterministic_defaults
        }
        if not invalid and not issues and verdict != "fail" and min(normalized_scores.values(), default=0) < 4:
            normalized_scores = deterministic_defaults
        passed = verdict != "fail" and not invalid and not issues and min(normalized_scores.values(), default=0) >= 4
        return {"passed": passed, "invalid_indexes": invalid, "scores": normalized_scores, "issues": issues}
    except Exception as exc:
        return {"passed": False, "invalid_indexes": [], "scores": {}, "issues": [f"审校调用失败：{type(exc).__name__}"]}


async def create_linked_scene(
    *,
    scene_kind: str,
    chapter: dict[str, Any],
    claims: list[dict[str, Any]],
    cards_by_id: dict[str, dict[str, Any]],
    memory: EpisodeMemory,
    existing_turns: list[dict[str, Any]],
    target: int,
    language: str,
    profile: dict[str, Any],
    trace: ContextUsage,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    draft: list[dict[str, Any]] = []
    deterministic_issues: list[str] = []
    try:
        draft, deterministic_issues = await _draft_scene(
            scene_kind=scene_kind,
            chapter=chapter,
            claims=claims,
            cards_by_id=cards_by_id,
            memory=memory,
            existing_turns=existing_turns,
            target=target,
            language=language,
            profile=profile,
            trace=trace,
        )
    except Exception as exc:
        deterministic_issues = [f"场景生成失败：{type(exc).__name__}"]
    if draft:
        return draft, {"passed": not deterministic_issues, "partial": len(draft) < target, "deterministic_issues": deterministic_issues, "repaired": False}
    if deterministic_issues:
        feedback = deterministic_issues
        try:
            repaired, repair_issues = await _draft_scene(
                scene_kind=scene_kind,
                chapter=chapter,
                claims=claims,
                cards_by_id=cards_by_id,
                memory=memory,
                existing_turns=existing_turns,
                target=target,
                language=language,
                profile=profile,
                trace=trace,
                repair_feedback=feedback or ["提升事实忠实度、上下文承接和角色稳定性"],
            )
        except Exception as exc:
            repaired, repair_issues = [], [f"场景修复失败：{type(exc).__name__}"]
        if repaired:
            return repaired, {"passed": not repair_issues, "partial": len(repaired) < target, "deterministic_issues": repair_issues, "repaired": True}
        report = {
            "passed": False,
            "stage": scene_kind,
            "deterministic_issues": repair_issues or deterministic_issues,
        }
        raise PodcastQualityError(f"{chapter.get('title') or scene_kind} 未通过场景质量检查", report)
    raise PodcastQualityError(f"{chapter.get('title') or scene_kind} 未返回可用内容", {"passed": False, "stage": scene_kind})


def _update_memory(memory: EpisodeMemory, turns: list[dict[str, Any]], chapter: dict[str, Any], recent_limit: int) -> None:
    for turn in turns:
        for claim_id in turn["claim_ids"]:
            if claim_id not in memory.covered_claim_ids:
                memory.covered_claim_ids.append(claim_id)
    substantive = [turn["text"] for turn in turns if turn["claim_ids"]]
    if substantive:
        memory.chapter_summaries.append({"title": str(chapter.get("title") or ""), "summary": " ".join(substantive[-2:])[:360]})
    memory.open_hook = str(chapter.get("bridge_out") or "")
    memory.last_turns = (memory.last_turns + turns)[-recent_limit:]
    memory.last_speaker = turns[-1]["speaker"] if turns else memory.last_speaker


async def _audit_episode(
    turns: list[dict[str, Any]], chapters: list[dict[str, Any]], thesis: str, language: str, trace: ContextUsage
) -> dict[str, Any]:
    boundaries = []
    for index in range(1, len(chapters)):
        previous, current = chapters[index - 1], chapters[index]
        left = turns[max(previous["turn_start"], previous["turn_end"] - 1) : previous["turn_end"] + 1]
        right = turns[current["turn_start"] : min(current["turn_end"] + 1, current["turn_start"] + 2)]
        boundaries.append(
            {
                "index": index,
                "from": previous["title"],
                "to": current["title"],
                "dialogue": " | ".join(f"{turn['speaker']}: {turn['text']}" for turn in left + right),
            }
        )
    prompt = f"""你是整集播客主编。根据章节边界判断整集是否围绕同一核心问题递进，跨章是否自然，主持人角色是否一致，是否反复重启话题或重复套话。只输出 JSON：{{"verdict":"pass|fail","scores":{{"grounding":5,"coherence":5,"roles":5,"repetition":5,"completeness":5}},"invalid_boundaries":[],"issues":[]}}。5=优秀、4=可发布、1=严重失败；没有问题时必须给 pass 和 4–5 分。语言={language}。核心命题：{thesis}
章节边界：{json.dumps(boundaries, ensure_ascii=False)}
"""
    try:
        generated = await budgeted_chat(
            lambda budget: PromptBuild([{"role": "user", "content": prompt}], len(boundaries), len(boundaries), 0),
            json_mode=True,
            max_tokens=structured_output_tokens(650),
            minimum_output_tokens=160,
            temperature=0.0,
            trace=trace,
        )
        parsed = _extract_json(generated.content)
        scores = parsed.get("scores") or {}
        invalid = [int(value) for value in parsed.get("invalid_boundaries") or [] if str(value).isdigit() and 0 < int(value) < len(chapters)]
        issues = [str(value)[:240] for value in parsed.get("issues") or []][:8]
        verdict = str(parsed.get("verdict") or "").lower()
        names = ("grounding", "coherence", "roles", "repetition", "completeness")
        normalized = {name: int(scores[name]) if str(scores.get(name, "")).isdigit() and 1 <= int(scores[name]) <= 5 else 4 for name in names}
        if not invalid and not issues and verdict != "fail" and min(normalized.values(), default=0) < 4:
            normalized = {name: 4 for name in names}
        return {
            "passed": not invalid and min(normalized.values(), default=0) >= 4,
            "verdict": verdict or "unspecified",
            "scores": normalized,
            "invalid_boundaries": invalid,
            "issues": issues,
        }
    except Exception as exc:
        return {"passed": False, "scores": {}, "invalid_boundaries": [], "issues": [f"整集审校失败：{type(exc).__name__}"]}


async def _repair_episode_boundaries(
    turns: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    claims_by_id: dict[str, dict[str, Any]],
    cards_by_id: dict[str, dict[str, Any]],
    episode_audit: dict[str, Any],
    thesis: str,
    language: str,
    profile: dict[str, Any],
    trace: ContextUsage,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired = list(turns)
    audits: list[dict[str, Any]] = []
    for boundary_index in list(episode_audit.get("invalid_boundaries") or [])[:2]:
        if boundary_index <= 0 or boundary_index >= len(chapters):
            continue
        previous, chapter = chapters[boundary_index - 1], chapters[boundary_index]
        start = int(chapter["turn_start"])
        replace_count = min(2, int(chapter["turn_end"]) - start + 1)
        if replace_count < 2:
            continue
        previous_turns = repaired[max(0, int(previous["turn_end"]) - 3) : int(previous["turn_end"]) + 1]
        memory = EpisodeMemory(
            thesis,
            open_hook=str(previous.get("bridge_out") or ""),
            last_turns=previous_turns,
            last_speaker=previous_turns[-1]["speaker"] if previous_turns else None,
        )
        chapter_claims = [claims_by_id[value] for value in chapter.get("claim_ids") or [] if value in claims_by_id]
        existing = repaired[:start] + repaired[start + replace_count :]
        boundary_turns, boundary_audit = await create_linked_scene(
            scene_kind="boundary_repair",
            chapter=chapter,
            claims=chapter_claims,
            cards_by_id=cards_by_id,
            memory=memory,
            existing_turns=existing,
            target=replace_count,
            language=language,
            profile=profile,
            trace=trace,
        )
        repaired[start : start + replace_count] = boundary_turns
        audits.append(boundary_audit)
    return repaired, audits


def _content_minutes(turns: list[dict[str, Any]]) -> float:
    joined = " ".join(turn["text"] for turn in turns)
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", joined))
    latin_words = len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", joined))
    return cjk_chars / 270 + latin_words / 150 + len(turns) * 0.45 / 60


def _repeated_stem_ratio(turns: list[dict[str, Any]]) -> float:
    stems = [re.sub(r"[“\"].*", "", turn["text"])[:24] for turn in turns if turn["dialogue_act"] == "question"]
    if not stems:
        return 0.0
    repeated = sum(count - 1 for count in Counter(stems).values() if count > 1)
    return repeated / len(stems)


def _quality_metrics_v3(
    turns: list[dict[str, Any]], citations: list[dict[str, Any]], target_minutes: int, requested_turns: int,
    episode_audit: dict[str, Any], scene_audits: list[dict[str, Any]], selected_source_ids: list[str]
) -> dict[str, Any]:
    a_chars = sum(len(turn["text"]) for turn in turns if turn["speaker"] == "HOST_A")
    b_chars = sum(len(turn["text"]) for turn in turns if turn["speaker"] == "HOST_B")
    total = max(1, a_chars + b_chars)
    estimated = _content_minutes(turns)
    factual = [turn for turn in turns if turn["dialogue_act"] in FACTUAL_ACTS or turn["claim_ids"]]
    source_coverage = len({citation["source_id"] for citation in citations})
    duplicate_pairs = sum(_similar(left["text"], right["text"]) >= 0.84 for index, left in enumerate(turns) for right in turns[index + 1 :])
    report = {
        "passed": False,
        "target_minutes": target_minutes,
        "estimated_minutes": round(estimated, 2),
        "duration_ratio": round(estimated / max(1, target_minutes), 3),
        "target_turn_count": requested_turns,
        "turn_count": len(turns),
        "host_a_ratio": round(a_chars / total, 3),
        "host_b_ratio": round(b_chars / total, 3),
        "factual_turns": len(factual),
        "uncited_factual_turns": sum(not turn["citation_ids"] for turn in factual),
        "bridge_turns": sum(turn["dialogue_act"] in NONFACTUAL_ACTS and not turn["claim_ids"] for turn in turns),
        "all_factual_turns_cited": all(turn["citation_ids"] for turn in factual),
        "duplicate_pairs": duplicate_pairs,
        "repeated_stem_ratio": round(_repeated_stem_ratio(turns), 3),
        "source_coverage": source_coverage,
        "selected_sources": len(selected_source_ids),
        "scene_repairs": sum(bool(audit.get("repaired")) for audit in scene_audits),
        "scene_scores": [audit.get("scores") for audit in scene_audits],
        "episode_audit": episode_audit,
        "safe_fallback_turns": 0,
    }
    report["passed"] = bool(
        0.85 <= report["duration_ratio"] <= 1.20
        and 0.30 <= report["host_a_ratio"] <= 0.70
        and report["uncited_factual_turns"] == 0
        and report["duplicate_pairs"] == 0
        and report["repeated_stem_ratio"] <= 0.10
        and episode_audit.get("passed")
    )
    return report


async def build_podcast_script(
    notebook_id: str,
    payload: dict[str, Any],
    *,
    progress: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    ids = source_scope(notebook_id, payload.get("source_ids"))
    if not ids:
        raise ValueError("当前范围没有已就绪的文档")
    language, language_selection = resolve_output_language(DB, ids, payload.get("language", "zh-CN"))
    focus = str(payload.get("focus") or "").strip()
    if progress:
        progress("构建全篇证据地图", 0.08)
    rows = select_podcast_evidence(notebook_id, ids, focus)
    cards, all_citations = build_evidence_cards(rows)
    if len(cards) < 2:
        raise ValueError("资料内容不足，无法生成深度播客")
    if progress:
        progress("提取可引用主张", 0.12)
    context_usage = ContextUsage()
    claims = build_claim_ledger(cards)
    if len(claims) < 2:
        raise ValueError("资料中缺少足够的可验证主张")
    if progress:
        progress("规划递进式剧集结构", 0.16)
    episode_plan, outline_degraded = await create_episode_plan(claims, language, focus, context_usage)
    chapters = episode_plan["chapters"]
    duration_mode = payload.get("duration_mode") or ("fixed" if payload.get("minutes") else "auto")
    requested_minutes = int(payload.get("minutes") or 0) or None
    target_minutes = requested_minutes if duration_mode == "fixed" and requested_minutes else estimate_auto_minutes(len(chapters), len(cards))
    total_target = target_turn_count(target_minutes)
    intro_target, outro_target = (3, 3) if total_target < 60 else (4, 4)
    body_target = max(len(chapters) * 3, total_target - intro_target - outro_target)
    chapter_targets = [body_target // len(chapters) for _ in chapters]
    for index in range(body_target % len(chapters)):
        chapter_targets[index] += 1
    cards_by_id = {card["id"]: card for card in cards}
    claims_by_id = {claim["id"]: claim for claim in claims}
    profile = podcast_generation_profile()
    planned_scene_blocks = 2 + sum(math.ceil(target / profile["scene_turns"]) for target in chapter_targets) + 1
    context_usage.request_limit = 1 + 2 * planned_scene_blocks + 4
    turns: list[dict[str, Any]] = []
    memory = EpisodeMemory(episode_plan["episode_thesis"])
    chapter_payloads: list[dict[str, Any]] = []
    scene_audits: list[dict[str, Any]] = []

    intro_claims = [claims_by_id[value] for value in chapters[0]["claim_ids"][:3] if value in claims_by_id]
    intro_chapter = {"title": "节目开场" if language != "en" else "Introduction", "purpose": episode_plan["episode_thesis"], "bridge_in": "", "bridge_out": chapters[0].get("bridge_in") or chapters[0]["purpose"]}
    intro_turns, intro_audit = await create_linked_scene(
        scene_kind="intro", chapter=intro_chapter, claims=intro_claims, cards_by_id=cards_by_id, memory=memory,
        existing_turns=turns, target=intro_target, language=language, profile=profile, trace=context_usage,
    )
    turns.extend(intro_turns)
    scene_audits.append(intro_audit)
    _update_memory(memory, intro_turns, intro_chapter, profile["recent_turns"])

    for chapter_index, chapter in enumerate(chapters):
        if progress:
            progress(f"连贯续写章节 {chapter_index + 1}/{len(chapters)}", 0.20 + 0.34 * chapter_index / max(1, len(chapters)))
        start_index = len(turns)
        chapter_claims = [claims_by_id[value] for value in chapter["claim_ids"] if value in claims_by_id]
        remaining = chapter_targets[chapter_index]
        consecutive_zero = 0
        while remaining > 0:
            batch = min(profile["scene_turns"], remaining)
            try:
                scene_turns, scene_audit = await create_linked_scene(
                    scene_kind="chapter", chapter=chapter, claims=chapter_claims, cards_by_id=cards_by_id, memory=memory,
                    existing_turns=turns, target=batch, language=language, profile=profile, trace=context_usage,
                )
            except PodcastQualityError:
                consecutive_zero += 1
                if consecutive_zero >= 2:
                    raise PodcastQualityError(
                        f"{chapter['title']} 连续两次没有可用内容",
                        {"passed": False, "stage": "chapter_zero_yield", "context_usage": context_usage.as_dict()},
                    )
                continue
            if not scene_turns:
                consecutive_zero += 1
                if consecutive_zero >= 2:
                    raise PodcastQualityError(
                        f"{chapter['title']} 连续两次没有可用内容",
                        {"passed": False, "stage": "chapter_zero_yield", "context_usage": context_usage.as_dict()},
                    )
                continue
            consecutive_zero = 0
            turns.extend(scene_turns)
            scene_audits.append(scene_audit)
            _update_memory(memory, scene_turns, chapter, profile["recent_turns"])
            remaining -= len(scene_turns)
        chapter_payloads.append({**chapter, "turn_start": start_index, "turn_end": len(turns) - 1})

    # A short draft may be expanded once, but only with claims the episode has
    # not covered yet. This preserves the requested duration without falling
    # back to generic banter or repeating an earlier explanation.
    if _content_minutes(turns) < target_minutes * 0.72:
        unused_claims = [claim for claim in claims if claim["id"] not in memory.covered_claim_ids]
        if unused_claims:
            expansion_claims = unused_claims[: max(2, profile["scene_turns"])]
            expansion_chapter = {
                **chapters[-1],
                "purpose": (
                    f"{chapters[-1]['purpose']}；补齐尚未讨论但与核心命题直接相关的证据"
                    if language != "en"
                    else f"{chapters[-1]['purpose']}; cover remaining evidence directly relevant to the thesis"
                ),
                "claim_ids": [claim["id"] for claim in expansion_claims],
            }
            expansion_turns, expansion_audit = await create_linked_scene(
                scene_kind="chapter", chapter=expansion_chapter, claims=expansion_claims,
                cards_by_id=cards_by_id, memory=memory, existing_turns=turns,
                target=max(2, profile["scene_turns"]), language=language, profile=profile,
                trace=context_usage,
            )
            turns.extend(expansion_turns)
            scene_audits.append(expansion_audit)
            _update_memory(memory, expansion_turns, expansion_chapter, profile["recent_turns"])
            chapter_payloads[-1]["turn_end"] = len(turns) - 1

    covered_claims = [claims_by_id[value] for value in memory.covered_claim_ids if value in claims_by_id]
    outro_chapter = {"title": "节目结语" if language != "en" else "Conclusion", "purpose": episode_plan["episode_thesis"], "bridge_in": memory.open_hook, "bridge_out": ""}
    outro_turns, outro_audit = await create_linked_scene(
        scene_kind="outro", chapter=outro_chapter, claims=covered_claims[-8:], cards_by_id=cards_by_id, memory=memory,
        existing_turns=turns, target=outro_target, language=language, profile=profile, trace=context_usage,
    )
    turns.extend(outro_turns)
    scene_audits.append(outro_audit)
    if progress:
        progress("执行整集连贯性审校", 0.58)
    episode_audit = await _audit_episode(turns, chapter_payloads, episode_plan["episode_thesis"], language, context_usage)
    if not episode_audit.get("passed") and episode_audit.get("invalid_boundaries"):
        turns, boundary_audits = await _repair_episode_boundaries(
            turns, chapter_payloads, claims_by_id, cards_by_id, episode_audit, episode_plan["episode_thesis"],
            language, profile, context_usage,
        )
        scene_audits.extend(boundary_audits)
        episode_audit = await _audit_episode(turns, chapter_payloads, episode_plan["episode_thesis"], language, context_usage)
    used_evidence = {evidence_id for turn in turns for evidence_id in turn["citation_ids"]}
    used_citations = [citation for citation in all_citations if citation["id"] in used_evidence]
    remap = {citation["id"]: f"S{index}" for index, citation in enumerate(used_citations, start=1)}
    citations = [{**citation, "id": remap[citation["id"]]} for citation in used_citations]
    for turn in turns:
        turn["citation_ids"] = [remap[value] for value in turn["citation_ids"] if value in remap]
    for index, turn in enumerate(turns, start=1):
        turn["id"] = f"turn_{index}"
        turn["chapter_id"] = next((chapter["id"] for chapter in chapter_payloads if chapter["turn_start"] <= index - 1 <= chapter["turn_end"]), "intro" if index <= intro_target else "outro")
    quality = _quality_metrics_v3(turns, citations, target_minutes, total_target, episode_audit, scene_audits, ids)
    if not quality["passed"]:
        quality["context_usage"] = context_usage.as_dict()
        raise PodcastQualityError("整集脚本未达到发布门槛", quality)
    if outline_degraded:
        context_usage.mark_fallback()
    script = "\n".join(f"{turn['speaker']}: {turn['text']} {' '.join(f'[{value}]' for value in turn['citation_ids'])}" for turn in turns)
    return {
        "version": PODCAST_ENGINE_VERSION,
        "engine": {**profile, "strategy": "linked_scenes", "version": PODCAST_ENGINE_VERSION},
        "language": language,
        "language_selection": language_selection,
        "source_ids": ids,
        "scope_hash": scope_hash(ids),
        "duration": {"mode": duration_mode, "requested_minutes": requested_minutes, "target_minutes": target_minutes},
        "chapters": chapter_payloads,
        "episode_plan": episode_plan,
        "turns": turns,
        "script": script,
        "citations": citations,
        "degraded": outline_degraded,
        "context_usage": context_usage.as_dict(),
        "quality": quality,
        "quality_report": quality,
    }
