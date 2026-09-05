from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

import httpx

from .context_budget import ContextUsage, PromptBudget, TokenLimits, estimate_messages_tokens, pack_items, structured_output_tokens
from .database import DB, json_load
from .providers import PromptBuild, active_provider, budgeted_chat, study_generation_profile
from .retrieval import select_quality_evidence, tokenize
from .services import _evenly_spaced, scope_hash, source_scope
from .languages import resolve_output_language, text_matches_language


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?:%|％)?")
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+|\n+")
PODCAST_ENGINE_VERSION = 4
PODCAST_DURATION_CALIBRATION_VERSION = 4
GENERATION_DURATION_TARGET_RATIO = 0.95
CJK_CHARS_PER_MINUTE = 225
LATIN_WORDS_PER_MINUTE = 150
TURN_PAUSE_SECONDS = 0.45
EPISODE_AUDIT_RECOVERY_RESERVE_TOKENS = 8000
MAX_DURATION_EXPANSION_UNITS = {"en": 1350, "zh-CN": 2400}
NONFACTUAL_ACTS = {"intro", "bridge", "question", "acknowledgement", "outro"}
FACTUAL_ACTS = {"frame", "explain", "evidence", "example", "challenge", "synthesis"}
ALLOWED_DIALOGUE_ACTS = NONFACTUAL_ACTS | FACTUAL_ACTS
COMPACT_ACT_CODES = {
    "I": "intro", "F": "frame", "B": "bridge", "Q": "question", "A": "acknowledgement",
    "X": "explain", "E": "evidence", "M": "example", "C": "challenge", "S": "synthesis", "O": "outro",
}
GENERIC_STEMS = (
    "这条材料明确说明了什么",
    "如果不做资料外推演",
    "原文是怎样把",
    "资料给出的直接线索是",
    "what does this passage establish",
    "without going beyond the text",
)
# 审计式套话语义族。硬族只含不可能是自然口语内容的元话语，进入整集确定性门禁；
# 软族（边界/门槛/范围）在技术、哲学题材中有正当内容用法（离线校准：历史胜者 0.14/轮 ≈
# 失败样本 0.148/轮，无区分度），只统计进报告、不参与判定。
CLICHE_FAMILIES = {
    "audit_negation": ("不能推出", "无法推出", "推不出", "只支持到这里", "只能支持到", "不能直接得出", "不能得出"),
    "recap_meta": ("回扣", "压实", "收数"),
    "next_layer": ("下一层",),
}
CLICHE_SOFT_FAMILIES = {
    "boundary_meta": ("边界", "门槛", "范围之内", "范围之外"),
}
# 阈值取自离线校准中点（历史胜者 vs 匿名评审失败样本）：整集胜者 0.02/轮、失败 0.188/轮；
# 单族胜者最大 1、失败最大 6；单 Act 胜者最差 0.062、失败最差 0.333。
CLICHE_EPISODE_DENSITY_LIMIT = 0.10
CLICHE_FAMILY_COUNT_LIMIT = 3
CLICHE_ACT_DENSITY_LIMIT = 0.20
# 防护句式（“别把它读成/夸成 X”“A 不等于 B”式防误读提醒）用正则统计：纯子串会把
# “把这个读成工程上的顺手”等正当用法误算。V6 离线校准（生产正则口径）：历史胜者 4 次
# （0.080/轮）、参考 0 次、V5 盲评通过样本 12 次（0.194/轮，重复控制 4 压线）、
# V5 失败样本 14 次（0.200/轮，重复控制 3）。
GUARD_FAMILIES = {
    "guard_disclaimer": re.compile(r"别[^，。；]{0,16}?(?:读成|夸成|说成|想成|当成|看成|理解成|误认为|误以为|推向|拧成|急着)"),
    "neq_disclaimer": re.compile(r"不等于|并不等同|不意味着|未必是|谈不上"),
}
# 阈值宁松勿紧：整集密度 0.18 卡在胜者 0.080 与 V5 两轮 0.194/0.200 之间（盲评对两轮
# 都点名了防护鼓点，故两者都应被拦）；单族计数 10 取通过 8 与失败 12 之间的宽口径。
# Act 级区分度弱（通过样本最差 Act 达 0.38），只在报告中统计、不参与判定。
GUARD_FAMILY_COUNT_LIMIT = 10
GUARD_EPISODE_DENSITY_LIMIT = 0.18


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


@dataclass
class EpisodeGenerationState:
    continuation_used: bool = False
    duration_expansion_used: bool = False
    duration_compression_used: bool = False
    empty_response_retry_used: bool = False

    @property
    def recovery_kind(self) -> str | None:
        if self.continuation_used:
            return "length_continuation"
        if self.duration_expansion_used:
            return "duration_expansion"
        if self.duration_compression_used:
            return "duration_compression"
        return None


@dataclass
class SceneDraftResult:
    turns: list[dict[str, Any]]
    issues: list[str]
    finish_reason: str | None = None


def _coerce_scene_draft(value: Any) -> SceneDraftResult:
    if isinstance(value, SceneDraftResult):
        return value
    turns, issues = value
    return SceneDraftResult(turns, issues)


def _reserve_episode_audit_after_recovery(trace: ContextUsage) -> None:
    """Keep the mandatory final audit reachable after one bounded recovery call."""
    if trace.total_token_limit is not None:
        trace.total_token_limit = min(45_000, trace.total_token_limit + EPISODE_AUDIT_RECOVERY_RESERVE_TOKENS)


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
    # Strong podcast references use fewer, more substantial turns than chatty
    # interview templates. This also gives the model enough room to complete an
    # act in one bounded call instead of paying for continuation calls.
    return max(18, min(90, round(minutes * 2.8)))


def _spoken_unit_count(text: str, language: str) -> float:
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_words = len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", text))
    if language == "en":
        return latin_words + cjk_chars * LATIN_WORDS_PER_MINUTE / CJK_CHARS_PER_MINUTE
    return cjk_chars + latin_words * CJK_CHARS_PER_MINUTE / LATIN_WORDS_PER_MINUTE


def _scene_duration_budget(language: str, target_minutes: float, turn_count: int, carry_in_minutes: float) -> dict[str, Any]:
    rate = LATIN_WORDS_PER_MINUTE if language == "en" else CJK_CHARS_PER_MINUTE
    pause_minutes = turn_count * TURN_PAUSE_SECONDS / 60
    target_units = max(0, (target_minutes - pause_minutes) * rate)
    return {
        "target_minutes": round(target_minutes, 3),
        "turn_count": turn_count,
        "minimum_units": max(1, round(target_units)),
        "maximum_units": max(1, round(target_units * 1.10)),
        "unit": "words" if language == "en" else "cjk_equivalent_chars",
        "carry_in_minutes": round(carry_in_minutes, 3),
    }


def _remaining_scene_duration_budget(
    language: str,
    duration_goal: float,
    current_minutes: float,
    total_turns: int,
    chapter_targets: list[int],
    chapter_index: int,
) -> dict[str, Any]:
    remaining_turns = sum(chapter_targets[chapter_index:])
    buffered_goal = duration_goal * 1.05
    remaining_minutes = max(0.0, buffered_goal - current_minutes)
    nominal_scene = duration_goal * chapter_targets[chapter_index] / max(1, total_turns)
    proportional_target = remaining_minutes * chapter_targets[chapter_index] / max(1, remaining_turns)
    scene_target_minutes = min(nominal_scene * 1.20, max(nominal_scene * 0.80, proportional_target))
    nominal_remaining = duration_goal * remaining_turns / max(1, total_turns)
    return _scene_duration_budget(
        language,
        scene_target_minutes,
        chapter_targets[chapter_index],
        remaining_minutes - nominal_remaining,
    )


def _turn_slot_plan(
    target: int,
    minimum_units: int,
    language: str,
    claim_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Allocate an Act's spoken-content floor without asking the model to count a total."""
    if target <= 0:
        return []
    question_slots = max(1, min(target - 1, round(target * 0.28)))
    short_slots = min(target - 1, question_slots + max(1, round(target * 0.10)))
    short_floor = 18 if language == "en" else 35
    short_positions: set[int] = set()
    for index in range(short_slots):
        position = round((index + 1) * (target - 1) / (short_slots + 1))
        short_positions.add(max(1, min(target - 2 if target > 2 else target - 1, position)))
    for position in range(1, max(1, target - 1)):
        if len(short_positions) >= short_slots:
            break
        short_positions.add(position)
    deep_count = max(1, target - len(short_positions))
    deep_floor = max(short_floor + 1, math.ceil((minimum_units - len(short_positions) * short_floor) / deep_count))
    plan = [
        {
            "index": index + 1,
            "kind": "short" if index in short_positions else "deep",
            "minimum_units": short_floor if index in short_positions else deep_floor,
            "default_claim_id": claim_ids[index % len(claim_ids)] if claim_ids else None,
        }
        for index in range(target)
    ]
    return plan


def _slot_plan_instruction(plan: list[dict[str, Any]], language: str) -> str:
    encoded = ",".join(
        f"{item['index']}:{'S' if item['kind'] == 'short' else 'D'}@{item['default_claim_id'] or '-'}"
        for item in plan
    )
    if language == "en":
        return (
            f"Follow this ordered slot plan: {encoded}. S slots use 1–2 natural sentences for concise questions, "
            "acknowledgements, or bridges; D slots use 3–5 complete sentences to explain, probe, qualify, or synthesize the @ claim. "
            "Within the 3–5 sentence range, alternate compact and expansive D turns instead of writing them all at one length, "
            "but do not lower the act's overall density below the slot plan. "
            "The @ claim is also the only default support when an S slot states a fact. "
            "Do not strengthen association into causation, or a supporting argument into the only, final, or definitive one unless the claim says so. "
            "Write directly without counting words or reporting statistics; use useful spoken content, not filler or repeated summaries."
        )
    return (
        f"严格执行按轮次排列的槽位计划：{encoded}。S 槽用 1–2 个自然句完成简洁追问、回应或承接；"
        "D 槽用 3–5 个完整但紧凑的句子解释、追问、辨析或综合 @ 后的主张；在 3–5 句范围内让紧凑轮与展开轮长短交替，"
        "不要所有 D 槽写成同一长度，但整 Act 的总篇幅不得低于槽位计划的密度；"
        "S 槽一旦陈述事实，也只能使用该槽的 @ 主张作为默认支持。"
        "除非主张本身明说，不得把相关性强化为因果，也不得把支持性论据说成‘唯一、最终、根本、证明’。"
        "直接写正文，不要在思考中逐字计数或输出统计；禁止填充语和重复总结。"
    )


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
    limit = min(64, max(32, per_source * max(1, len(source_ids))))
    return select_quality_evidence(notebook_id, source_ids, limit=limit, focus=focus)


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


def _extract_array(raw: str, key: str) -> list[Any] | None:
    """Recover only complete objects from a possibly truncated JSON list."""
    parsed = _extract_json(raw)
    if isinstance(parsed.get(key), list):
        return parsed[key]
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', raw)
    if not match:
        return None
    decoder = json.JSONDecoder()
    position = match.end()
    recovered: list[Any] = []
    while position < len(raw):
        while position < len(raw) and (raw[position].isspace() or raw[position] == ","):
            position += 1
        if position >= len(raw) or raw[position] == "]":
            break
        try:
            value, consumed = decoder.raw_decode(raw[position:])
        except json.JSONDecodeError:
            break
        if not isinstance(value, (dict, list)):
            break
        recovered.append(value)
        position += consumed
    return recovered or None


def _extract_turns(raw: str) -> list[dict[str, Any]] | None:
    values = _extract_array(raw, "turns")
    if values is None:
        return None
    turns: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, dict):
            turns.append(value)
            continue
        if not isinstance(value, list) or len(value) != 4:
            continue
        speaker, act_code, text, claim_ids = value
        act = COMPACT_ACT_CODES.get(str(act_code).upper())
        if not act:
            continue
        turns.append({
            "speaker": f"HOST_{str(speaker).upper()}" if str(speaker).upper() in {"A", "B"} else speaker,
            "dialogue_act": act,
            "text": text,
            "claim_ids": claim_ids if isinstance(claim_ids, list) else [],
        })
    return turns or None


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


# V3 遗留：_grounded_question / _safe_chapter_turns 只被 create_chapter_turns（V3 路径）使用，
# V4 editorial-acts 生产路径不经过；其循环模板正是匿名评审所指“修正模板”的句式来源，保留仅为 V3 兼容。
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
            if len(claims) >= 64:
                return claims
    return claims


def _fallback_episode_plan(claims: list[dict[str, Any]], language: str, act_count: int | None = None) -> dict[str, Any]:
    chapter_count = act_count or max(2, min(6, round(math.sqrt(max(1, len(claims))))))
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
                "lead_host": "HOST_A" if number % 2 else "HOST_B",
                "tension": "检验这一部分最容易被误解或过度推论的地方" if language != "en" else "Test the easiest misunderstanding or overreach in this part",
            }
        )
        if len(chapters) >= chapter_count:
            break
    thesis = claims[0]["text"][:220] if claims else ("资料深度解读" if language != "en" else "A grounded deep dive")
    return {"episode_thesis": thesis, "chapters": chapters, "fallback": True}


async def create_episode_plan(
    claims: list[dict[str, Any]], language: str, focus: str, trace: ContextUsage | None = None, act_count: int | None = None
) -> tuple[dict[str, Any], bool]:
    language_rule = "只使用自然的简体中文" if language != "en" else "Use natural spoken English only"
    target = act_count or max(3, min(6, round(math.sqrt(max(1, len(claims))))))
    prompt_prefix = f"""你是资料型深度播客的总编。{language_rule}。只规划一条能从核心问题逐步走向结论的叙事主线，不写对话，不补充资料外事实。
规划恰好 {target} 个逻辑递进 Act。HOST_A 与 HOST_B 必须按 Act 轮换主导解释、质疑和综合，不能固定成主讲者与采访者。每个 claim 只在真正相关的 Act 使用。第一 Act 必须先用一两轮立起题目与背景（这份材料是什么、核心问题是什么、为什么值得听），再进入细节论证；提出问题但不预告所有答案，最后一 Act 回扣问题且不引入新事实。
输出 JSON：{{"episode_thesis":"","chapters":[{{"title":"","purpose":"","tension":"需要检验的误解或反例","lead_host":"HOST_A|HOST_B","claim_ids":["C1"],"bridge_in":"如何承接上一 Act","bridge_out":"留给下一 Act 的问题"}}]}}。
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
            stage="episode_plan",
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
                    "lead_host": "HOST_B" if str(item.get("lead_host") or "").upper() == "HOST_B" else "HOST_A",
                    "tension": str(item.get("tension") or "")[:260],
                }
            )
            if len(chapters) >= 8:
                break
        thesis = str(parsed.get("episode_thesis") or "").strip()
        if len(chapters) == target and thesis:
            return {"episode_thesis": thesis[:400], "chapters": chapters, "fallback": False}, False
    except Exception:
        pass
    if trace:
        trace.mark_fallback()
    return _fallback_episode_plan(claims, language, target), True


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
        "scene_turns": 4 if tier == "lite" else 18,
        "recent_turns": 4 if tier == "lite" else 6,
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


def _is_question_turn(turn: dict[str, Any]) -> bool:
    return turn.get("dialogue_act") == "question" or str(turn.get("text") or "").rstrip().endswith(("?", "？"))


def _question_count_rule(target: int) -> str:
    minimum = max(1, math.ceil(target * 0.20))
    maximum = max(minimum, math.floor(target * 0.35))
    if maximum == minimum:
        return f"问句必须恰好有 {minimum} 轮"
    return f"问句必须有 {minimum}–{maximum} 轮（占本 Act 的 20%–35%）"


def _only_question_filter_issues(issues: list[str]) -> bool:
    substantive = [issue for issue in issues if not issue.startswith("有效轮次不足")]
    return bool(substantive) and all("问句" in issue for issue in substantive)


def _infer_claim_id(
    text: str,
    claims_by_id: dict[str, dict[str, Any]],
    cards_by_id: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    target = set(tokenize(text))
    if not target:
        return None
    ranked = []
    for claim_id, claim in claims_by_id.items():
        support_text = str(claim["text"])
        if cards_by_id:
            support_text += " " + " ".join(
                str(cards_by_id[evidence_id].get("content") or "")
                for evidence_id in claim.get("evidence_ids") or []
                if evidence_id in cards_by_id
            )
        support = set(tokenize(support_text))
        score = len(target & support) / max(1, min(len(target), len(support)))
        ranked.append((score, claim_id))
    ranked.sort(reverse=True)
    if not ranked or ranked[0][0] < 0.16:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.02:
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
    default_claim_ids: list[str | None] | None = None,
    question_cap: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(raw_turns, list):
        return [], ["模型没有返回 turns 数组"]
    accepted: list[dict[str, Any]] = []
    issues: list[str] = []
    previous = last_speaker
    question_run = 0
    for prior in reversed(existing_turns):
        if not _is_question_turn(prior):
            break
        question_run += 1
    question_count = 0
    max_questions = max(1, math.floor(expected_count * 0.40)) if question_cap is None else max(0, question_cap)
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
        candidate_is_question = act == "question" or text.rstrip().endswith(("?", "？"))
        if candidate_is_question and question_run >= 2:
            issues.append(f"第 {index + 1} 轮造成跨 Act 连续问句过多")
            continue
        if candidate_is_question and question_count >= max_questions:
            issues.append(f"第 {index + 1} 轮使本 Act 问句比例超过 40%")
            continue
        spoken_units = _spoken_unit_count(text, language)
        minimum_ok = len(text) >= 8 if language != "en" else spoken_units >= 4
        maximum_units = 240 if language != "en" else 90
        maximum_ok = spoken_units <= maximum_units
        if not minimum_ok or not maximum_ok:
            direction = "过短" if not minimum_ok else "过长"
            issues.append(f"第 {index + 1} 轮长度不合格（{direction}，口播单位 {spoken_units}）")
            continue
        if not text_matches_language(text, language):
            issues.append(f"第 {index + 1} 轮语言不符合输出要求")
            continue
        if re.search(r"[!！]|[?？]{2,}|\.{3,}|…{2,}", text):
            issues.append(f"第 {index + 1} 轮包含会放大口播情绪的标点")
            continue
        lowered = text.lower()
        if any(stem in lowered for stem in GENERIC_STEMS):
            issues.append(f"第 {index + 1} 轮使用机械模板")
            continue
        claim_id_source = "model" if claim_ids else "none"
        if not claim_ids:
            inferred = _infer_claim_id(text, claims_by_id, cards_by_id)
            if inferred:
                claim_ids = [inferred]
                claim_id_source = "lexical"
        semantic_bridge = act in {"intro", "bridge", "outro"} and spoken_units >= (8 if language == "en" else 20)
        provisional_factual = act in FACTUAL_ACTS or semantic_bridge or bool(NUMBER_PATTERN.search(text))
        if not claim_ids and provisional_factual and default_claim_ids and index < len(default_claim_ids):
            default_claim_id = default_claim_ids[index]
            if default_claim_id in claims_by_id:
                claim_ids = [str(default_claim_id)]
                claim_id_source = "slot"
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
                "claim_id_inferred": claim_id_source in {"lexical", "slot"},
                "claim_id_source": claim_id_source,
                "safe": False,
            }
        )
        previous = speaker
        question_run = question_run + 1 if candidate_is_question else 0
        question_count += int(candidate_is_question)
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
            "act": "Develop the planned act as one continuous exchange: frame the issue, probe common misunderstandings, explain the evidence, draw an implication, and hand off naturally. The first act opens the central question; the last resolves it without new facts.",
        }[scene_kind]
    return {
        "intro": "用核心问题开场，让两位主持人的互补视角自然出现；不要提前罗列所有答案。",
        "chapter": "沿一条推理主线推进；每一轮必须直接回应、修正或承接紧邻的上一轮。",
        "outro": "只用已经讨论过的主张回应开场问题，自然收束，不引入任何新事实。",
        "boundary_repair": "重写本章开头，使其先回应上一段真实对话，再自然进入规划主题。",
        "act": "把当前 Act 写成连续推进的交流：提出局部问题、澄清常见误解、解释证据、形成含义并自然承接。第一 Act 打开核心问题，最后一 Act 只用已讨论事实自然收束核心问题。",
    }[scene_kind]


def _delivery_instruction(language: str) -> str:
    if language == "en":
        return (
            "Keep both hosts in a restrained knowledge-podcast register. Do not use exclamation marks, repeated "
            "punctuation, all-caps emphasis, stage directions, or wording that asks for shouting, anger, or abrupt "
            "emotional changes; questions and challenges must remain calm."
        )
    return (
        "两位主持人始终使用克制、稳定的知识播客表达；禁止感叹号、重复标点、舞台式情绪说明，"
        "也不要用要求喊叫、愤怒或情绪骤变的措辞；提问和质疑都保持平静。"
    )


def _act_output_tokens(duration_budget: dict[str, Any] | None, target: int, language: str) -> int:
    units = float((duration_budget or {}).get("maximum_units") or target * (55 if language == "en" else 110))
    visible = units * (2.0 if language == "en" else 1.5) + target * 24
    return min(10_000, max(6_000, round(visible + 4_500)))


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
    duration_budget: dict[str, Any] | None = None,
) -> SceneDraftResult:
    language_rule = "只输出自然的简体中文口语" if language != "en" else "Use natural spoken English only"
    start_speaker = "HOST_B" if memory.last_speaker == "HOST_A" else "HOST_A"
    memory_json = json.dumps(memory.prompt_payload(profile["recent_turns"]), ensure_ascii=False)
    feedback = "；".join(repair_feedback or [])
    slot_plan = _turn_slot_plan(
        target,
        int((duration_budget or {}).get("minimum_units") or 1),
        language,
        [str(claim["id"]) for claim in claims],
    )
    if duration_budget and language == "en":
        duration_rule = (
            f"This Act should sustain about {duration_budget['target_minutes']:.1f} minutes of natural speech. "
            "Use the sentence density in the slot plan instead of calculating an exact word count."
        )
    elif duration_budget:
        duration_rule = (
            f"本 Act 需要形成约 {duration_budget['target_minutes']:.1f} 分钟的自然口播；"
            "按槽位计划的句数密度直接写，不要计算精确字符数。"
        )
    else:
        duration_rule = ""
    question_rule = _question_count_rule(target)
    prompt_prefix = f"""你是严格资料内的双人深度播客编剧。{language_rule}。两位主持人都能解释、质疑和综合；本 Act 由 {chapter.get('lead_host') or 'HOST_A'} 主导，但另一位必须贡献实质判断，禁止机械采访和孤立事实罗列。
{_scene_instruction(scene_kind, language)}
{_delivery_instruction(language)}
生成恰好 {target} 轮，从 {start_speaker} 开始并严格交替。{question_rule}，不得连续出现超过两个问句；使用 Q act_code 的轮次必须写成自然问句并以问号结尾。长短轮次要有变化，但每一轮都要完成一个实质推进。{duration_rule} {_slot_plan_instruction(slot_plan, language)} 每个 D 槽的 claim_ids 至少填一个允许的 C 编号；S 槽只有在 Q/B/A/I/O 且完全不陈述事实时才允许空数组。围绕本 Act 的“张力”组织论证主线，把前提、机制和含义逐步讲清；张力只用于内部规划，不得照读或转述其措辞。涉及尚未确认的内容时，用一句自然口语限定带过（如“这里原文没明说”“这点还差一点证据”），把不确定体现在论证结构里，不要念成方法论旁白；口播中禁止使用“不能推出、只支持、边界、门槛、范围、回扣、压实、下一层”一类审稿术语。对听者的显性防误读提醒（“别把它读成/夸成/说成 X”“A 不等于 B”“这不意味着…”）每个 Act 至多一处，其余限定直接并入叙述——说“原文给的是 A”，而不是反复敲打“A 不等于 B”。事实、数字、案例、判断必须被所填 claim_ids 直接支持；禁止用“唯一、必然、完全”等绝对措辞放大原主张，也不能从个人行动擅自推演到社会影响。不得使用资料外常识、轶事或类比，不得念出编号，不得重复“所以你的意思是”一类模板句。
只输出一个 JSON 对象，键名为 turns；turns 的每一项必须是四元素数组，依次为 speaker、act_code、text、claim_ids。speaker 只能为 A/B；act_code 只能为 I/F/B/Q/A/X/E/M/C/S/O；claim_ids 只能从下方允许列表逐字复制，不能省略事实轮的编号。不要输出示例、统计、解释或额外字段。
剧集记忆：{memory_json}
当前部分：{chapter.get('title')}；目的：{chapter.get('purpose')}；本 Act 的内部张力（仅用于组织论证主线，不得照读或转述其措辞）：{chapter.get('tension')}；承接：{chapter.get('bridge_in')}；后续钩子：{chapter.get('bridge_out')}。
{'这不是首个 Act：第一轮必须先明确回应剧集记忆中上个钩子的未决关系，再进入新角度；不得直接跳到新类比。' if memory.last_turns else ''}
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
        timeout=420,
        max_tokens=_act_output_tokens(duration_budget, target, language),
        minimum_output_tokens=min(3600, max(700, target * 130)),
        temperature=0.45,
        trace=trace,
        stage="act_draft" if not repair_feedback else "targeted_repair",
    )
    available_claims = {claim["id"]: claim for claim in generated.build.metadata["items"]}
    raw_turns = _extract_turns(generated.content)
    finish_reason = getattr(generated, "finish_reason", None)
    if not isinstance(raw_turns, list):
        reason = f"（结束原因：{finish_reason}）" if finish_reason else ""
        return SceneDraftResult([], [f"模型没有返回可解析的 turns 数组{reason}"], finish_reason)
    validated, issues = validate_scene_turns(
        raw_turns,
        available_claims,
        cards_by_id,
        last_speaker=memory.last_speaker,
        existing_turns=existing_turns,
        language=language,
        expected_count=target,
        scene_kind=scene_kind,
        default_claim_ids=[item["default_claim_id"] for item in slot_plan],
    )
    return SceneDraftResult(validated, issues, finish_reason)


async def _audit_scene(
    turns: list[dict[str, Any]],
    claims_by_id: dict[str, dict[str, Any]],
    memory: EpisodeMemory,
    language: str,
    trace: ContextUsage,
) -> dict[str, Any]:
    used = list(dict.fromkeys(claim_id for turn in turns for claim_id in turn["claim_ids"]))
    transcript = "\n".join(
        f"{index}: {turn['speaker']} [{turn.get('dialogue_act', 'explain')}] {turn['text']} claims={','.join(turn.get('claim_ids') or []) or '-'}"
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


async def _continue_scene(
    *,
    chapter: dict[str, Any],
    claims: list[dict[str, Any]],
    cards_by_id: dict[str, dict[str, Any]],
    memory: EpisodeMemory,
    existing_turns: list[dict[str, Any]],
    partial: list[dict[str, Any]],
    target: int,
    language: str,
    trace: ContextUsage,
    duration_budget: dict[str, Any] | None,
) -> SceneDraftResult:
    missing = target - len(partial)
    actual_units = sum(_spoken_unit_count(turn["text"], language) for turn in partial)
    minimum_units = max(1, round(float((duration_budget or {}).get("minimum_units") or 1) - actual_units))
    maximum_units = max(minimum_units, round(float((duration_budget or {}).get("maximum_units") or minimum_units) - actual_units))
    compact_recent = [
        [turn["speaker"].removeprefix("HOST_"), turn["dialogue_act"], turn["text"], turn.get("claim_ids") or []]
        for turn in (memory.last_turns + partial)[-4:]
    ]
    start_speaker = "B" if partial[-1]["speaker"] == "HOST_A" else "A"
    language_rule = "只输出自然的简体中文口语" if language != "en" else "Use natural spoken English only"
    slot_plan = _turn_slot_plan(missing, minimum_units, language, [str(claim["id"]) for claim in claims])
    remaining_question_cap = max(0, math.floor(target * 0.40) - sum(_is_question_turn(turn) for turn in partial))
    remaining_minutes = minimum_units / (LATIN_WORDS_PER_MINUTE if language == "en" else CJK_CHARS_PER_MINUTE)
    prompt_prefix = f"""你正在补全一段提前结束、结构不完整的资料型双人播客。{language_rule}。不要重写或复述已有轮次，只续写缺失的 {missing} 轮，从 HOST_{start_speaker} 开始严格交替。
{_delivery_instruction(language)}
续写需要补足约 {remaining_minutes:.1f} 分钟自然口播，其中最多 {remaining_question_cap} 轮可以是问句（包括以问号结尾的非 Q 标签轮）。{_slot_plan_instruction(slot_plan, language)} 继续当前推理并完成本 Act 的目的与后续钩子；事实轮必须带受支持的 claim_ids。
只输出一个 JSON 对象，键名为 turns；每一项是 speaker、act_code、text、claim_ids 组成的四元素数组。不要输出短示例、统计或解释。act code 只能使用 I/F/B/Q/A/X/E/M/C/S/O。
当前部分：{chapter.get('title')}；目的：{chapter.get('purpose')}；内部张力（仅用于组织论证主线，不得照读或转述其措辞）：{chapter.get('tension')}；后续钩子：{chapter.get('bridge_out')}。
紧邻的已有对话：{json.dumps(compact_recent, ensure_ascii=False)}
允许使用的主张：
"""
    continuation_budget = {
        "minimum_units": minimum_units,
        "maximum_units": maximum_units,
        "unit": (duration_budget or {}).get("unit"),
    }
    generated = await budgeted_chat(
        lambda budget: _segment_prompt_build(
            budget,
            prefix=prompt_prefix,
            items=claims,
            renderer=lambda claim: f"[{claim['id']}|{','.join(claim['evidence_ids'])}] {claim['text']}",
            group_key=lambda claim: str(claim["source_id"]),
        ),
        json_mode=True,
        timeout=420,
        max_tokens=_act_output_tokens(continuation_budget, missing, language),
        minimum_output_tokens=min(3000, max(700, missing * 130)),
        temperature=0.35,
        trace=trace,
        stage="act_continuation",
    )
    raw_turns = _extract_turns(generated.content)
    finish_reason = getattr(generated, "finish_reason", None)
    if not raw_turns:
        return SceneDraftResult([], ["续写没有返回可解析的 turns"], finish_reason)
    available_claims = {claim["id"]: claim for claim in generated.build.metadata["items"]}
    validated, issues = validate_scene_turns(
        raw_turns,
        available_claims,
        cards_by_id,
        last_speaker=partial[-1]["speaker"],
        existing_turns=existing_turns + partial,
        language=language,
        expected_count=missing,
        scene_kind="act",
        default_claim_ids=[item["default_claim_id"] for item in slot_plan],
        question_cap=remaining_question_cap,
    )
    return SceneDraftResult(validated, issues, finish_reason)


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
    duration_budget: dict[str, Any] | None = None,
    generation_state: EpisodeGenerationState | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    draft: list[dict[str, Any]] = []
    deterministic_issues: list[str] = []
    finish_reason: str | None = None
    continuation_used_here = False
    question_filtered_shortfall = False
    try:
        result = _coerce_scene_draft(await _draft_scene(
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
            duration_budget=duration_budget,
        ))
        draft, deterministic_issues, finish_reason = result.turns, result.issues, result.finish_reason
    except httpx.ConnectError:
        try:
            result = _coerce_scene_draft(await _draft_scene(
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
                duration_budget=duration_budget,
            ))
            draft, deterministic_issues, finish_reason = result.turns, result.issues, result.finish_reason
        except Exception as exc:
            deterministic_issues = [f"场景连接重试失败：{type(exc).__name__}"]
    except Exception as exc:
        deterministic_issues = [f"场景生成失败：{type(exc).__name__}"]
    if (
        scene_kind == "act"
        and not draft
        and finish_reason in {"stop", "length", "max_tokens"}
        and generation_state is not None
        and not generation_state.empty_response_retry_used
    ):
        generation_state.empty_response_retry_used = True
        try:
            result = _coerce_scene_draft(await _draft_scene(
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
                duration_budget=duration_budget,
            ))
            draft, deterministic_issues, finish_reason = result.turns, result.issues, result.finish_reason
        except Exception as exc:
            deterministic_issues = [f"空响应重试失败：{type(exc).__name__}"]
    question_filtered_shortfall = bool(
        draft
        and scene_kind == "act"
        and finish_reason == "stop"
        and len(draft) >= math.ceil(target * 0.85)
        and _only_question_filter_issues(deterministic_issues)
    )
    if (
        draft
        and scene_kind == "act"
        and len(draft) < target
        and (len(draft) < max(1, target - 1) or finish_reason in {"length", "max_tokens"})
        and not question_filtered_shortfall
        and generation_state is not None
        and generation_state.recovery_kind is None
    ):
        generation_state.continuation_used = True
        continuation_used_here = True
        continued = await _continue_scene(
            chapter=chapter,
            claims=claims,
            cards_by_id=cards_by_id,
            memory=memory,
            existing_turns=existing_turns,
            partial=draft,
            target=target,
            language=language,
            trace=trace,
            duration_budget=duration_budget,
        )
        completed_count = len(draft) + len(continued.turns)
        combined_issues = [issue for issue in deterministic_issues if not issue.startswith("有效轮次不足")] + continued.issues
        question_filtered_shortfall = bool(
            continued.finish_reason == "stop"
            and completed_count >= math.ceil(target * 0.85)
            and _only_question_filter_issues(combined_issues)
        )
        minimum_after_continuation = (
            math.ceil(target * 0.85)
            if question_filtered_shortfall
            else target if continued.finish_reason in {"length", "max_tokens"} else max(1, target - 1)
        )
        if completed_count < minimum_after_continuation:
            raise PodcastQualityError(
                f"{chapter.get('title') or scene_kind} 的唯一续写仍不完整",
                {
                    "passed": False,
                    "stage": "act_continuation",
                    "target_turns": target,
                    "accepted_turns": completed_count,
                    "finish_reason": continued.finish_reason,
                    "deterministic_issues": continued.issues,
                },
            )
        draft.extend(continued.turns)
        deterministic_issues = [issue for issue in deterministic_issues if not issue.startswith("有效轮次不足")]
        deterministic_issues.extend(continued.issues)
        finish_reason = continued.finish_reason
    minimum_complete = (
        math.ceil(target * 0.85)
        if question_filtered_shortfall
        else target if scene_kind == "boundary_repair" or finish_reason in {"length", "max_tokens"} else max(1, target - 1)
    )
    if draft and scene_kind in {"act", "boundary_repair"} and len(draft) < minimum_complete:
        raise PodcastQualityError(
            f"{chapter.get('title') or scene_kind} 返回的有效轮次不足",
            {
                "passed": False,
                "stage": scene_kind,
                "target_turns": target,
                "accepted_turns": len(draft),
                "finish_reason": finish_reason,
                "deterministic_issues": deterministic_issues,
            },
        )
    if draft:
        duration = {
            **(duration_budget or {}),
            "estimated_minutes": round(_content_minutes(draft), 3),
            "actual_units": round(sum(_spoken_unit_count(turn["text"], language) for turn in draft)),
        }
        duration["ratio"] = round(duration["estimated_minutes"] / max(0.001, float(duration.get("target_minutes") or 0.001)), 3)
        return draft, {
            "passed": not deterministic_issues,
            "partial": len(draft) < target,
            "deterministic_issues": deterministic_issues,
            "repaired": continuation_used_here,
            "continuation_used": continuation_used_here,
            "finish_reason": finish_reason,
            "duration": duration,
        }
    if scene_kind in {"act", "boundary_repair"}:
        raise PodcastQualityError(
            f"{chapter.get('title') or scene_kind} 未返回可用内容",
            {"passed": False, "stage": scene_kind, "deterministic_issues": deterministic_issues},
        )
    if deterministic_issues:
        feedback = deterministic_issues
        try:
            repair_result = _coerce_scene_draft(await _draft_scene(
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
                duration_budget=duration_budget,
            ))
            repaired, repair_issues = repair_result.turns, repair_result.issues
        except Exception as exc:
            repaired, repair_issues = [], [f"场景修复失败：{type(exc).__name__}"]
        if repaired:
            duration = {
                **(duration_budget or {}),
                "estimated_minutes": round(_content_minutes(repaired), 3),
                "actual_units": round(sum(_spoken_unit_count(turn["text"], language) for turn in repaired)),
            }
            duration["ratio"] = round(duration["estimated_minutes"] / max(0.001, float(duration.get("target_minutes") or 0.001)), 3)
            return repaired, {"passed": not repair_issues, "partial": len(repaired) < target, "deterministic_issues": repair_issues, "repaired": True, "duration": duration}
        report = {
            "passed": False,
            "stage": scene_kind,
            "deterministic_issues": repair_issues or deterministic_issues,
        }
        raise PodcastQualityError(f"{chapter.get('title') or scene_kind} 未通过场景质量检查", report)
    raise PodcastQualityError(f"{chapter.get('title') or scene_kind} 未返回可用内容", {"passed": False, "stage": scene_kind})


def _duration_expansion_plan(
    turns: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    missing_units: int,
    language: str,
) -> list[dict[str, Any]]:
    maximum_units = 105 if language == "en" else 190
    candidates: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(chapters))}
    for turn_index, turn in enumerate(turns):
        if turn.get("dialogue_act") not in FACTUAL_ACTS or not turn.get("claim_ids"):
            continue
        current = round(_spoken_unit_count(str(turn.get("text") or ""), language))
        capacity = maximum_units - current
        if capacity < (10 if language == "en" else 20):
            continue
        chapter_index = next(
            (
                index
                for index, chapter in enumerate(chapters)
                if int(chapter["turn_start"]) <= turn_index <= int(chapter["turn_end"])
            ),
            0,
        )
        candidates.setdefault(chapter_index, []).append(
            {"index": turn_index, "current_units": current, "capacity": capacity, "speaker": turn["speaker"]}
        )
    for values in candidates.values():
        values.sort(key=lambda item: (item["current_units"], item["speaker"], item["index"]))
    ordered: list[dict[str, Any]] = []
    while any(candidates.values()):
        for chapter_index in range(len(chapters)):
            values = candidates.get(chapter_index) or []
            if values:
                ordered.append(values.pop(0))
    target_addition = math.ceil(missing_units * 1.05)
    selected: list[dict[str, Any]] = []
    capacity = 0
    for item in ordered:
        selected.append(item)
        capacity += item["capacity"]
        if capacity >= target_addition:
            break
    if capacity < target_addition:
        return []
    remaining = target_addition
    for position, item in enumerate(selected):
        slots_left = len(selected) - position
        addition = min(item["capacity"], max(1, math.ceil(remaining / slots_left)))
        item["minimum_units"] = item["current_units"] + addition
        item["maximum_units"] = min(maximum_units, item["minimum_units"] + (12 if language == "en" else 24))
        remaining -= addition
    return selected


async def _expand_episode_duration(
    turns: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    claims_by_id: dict[str, dict[str, Any]],
    cards_by_id: dict[str, dict[str, Any]],
    language: str,
    target_minutes: float,
    trace: ContextUsage,
    generation_state: EpisodeGenerationState,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before_minutes = _content_minutes(turns)
    target_budget = _scene_duration_budget(language, target_minutes, len(turns), 0)
    actual_units = round(sum(_spoken_unit_count(turn["text"], language) for turn in turns))
    missing_units = max(0, int(target_budget["minimum_units"]) - actual_units)
    report: dict[str, Any] = {
        "used": False,
        "before_minutes": round(before_minutes, 3),
        "target_minutes": round(target_minutes, 3),
        "missing_units": missing_units,
    }
    if missing_units <= 0:
        return turns, report
    maximum_recovery = MAX_DURATION_EXPANSION_UNITS["en" if language == "en" else "zh-CN"]
    if missing_units > maximum_recovery:
        raise PodcastQualityError(
            "整集口播缺口超过单次通用扩展上限",
            {"passed": False, "stage": "duration_expansion", **report, "maximum_recovery_units": maximum_recovery},
        )
    if generation_state.recovery_kind is not None:
        raise PodcastQualityError(
            "整集口播不足且唯一恢复槽已使用",
            {"passed": False, "stage": "duration_expansion", **report, "recovery_kind": generation_state.recovery_kind},
        )
    plan = _duration_expansion_plan(turns, chapters, missing_units, language)
    if not plan:
        raise PodcastQualityError(
            "没有足够的受支持轮次可用于口播扩展",
            {"passed": False, "stage": "duration_expansion", **report},
        )
    generation_state.duration_expansion_used = True
    items = []
    for item in plan:
        turn = turns[item["index"]]
        claim_ids = [value for value in turn["claim_ids"] if value in claims_by_id]
        items.append(
            {
                **item,
                "speaker": turn["speaker"],
                "dialogue_act": turn["dialogue_act"],
                "text": turn["text"],
                "claim_ids": claim_ids,
                "claims": [claims_by_id[value]["text"] for value in claim_ids],
            }
        )
    language_rule = "Use natural spoken English only." if language == "en" else "只使用自然的简体中文口语。"
    unit = "words" if language == "en" else "中文等价字符"
    prompt_prefix = f"""你是资料型双人播客的精简扩写编辑。{language_rule} 只扩写列出的实质轮次，不改变说话人、dialogue act、claim_ids、结论方向或相邻轮次关系。用对应 claims 补足前提、机制或含义；需要限定时用一句自然口语带过（如“这里原文没明说”），不使用“不能推出、只支持、边界、门槛、回扣、压实”一类审稿术语，也不要新增“别把它读成/夸成”“不等于”一类防误读提醒。不得加入资料外事实、数字、类比、开场白或重复总结。
每项必须达到自己的 minimum_units 且不超过 maximum_units，单位为{unit}。只输出 JSON 对象，键名 replacements；每项是 [原始整数 index, replacement_text]。必须恰好返回全部 index，不输出统计或解释。
待扩写轮次：
"""
    generated = await budgeted_chat(
        lambda budget: _segment_prompt_build(
            budget,
            prefix=prompt_prefix,
            items=items,
            renderer=lambda item: json.dumps(item, ensure_ascii=False, separators=(",", ":")),
        ),
        json_mode=True,
        timeout=420,
        max_tokens=_act_output_tokens({"maximum_units": sum(item["maximum_units"] for item in items)}, len(items), language),
        minimum_output_tokens=min(3600, max(900, len(items) * 140)),
        temperature=0.35,
        trace=trace,
        stage="duration_expansion",
    )
    raw = _extract_array(generated.content, "replacements") or []
    replacements: dict[int, str] = {}
    for value in raw:
        if isinstance(value, list) and len(value) == 2 and str(value[0]).isdigit():
            replacements[int(value[0])] = _normalize_text(str(value[1] or ""))
        elif isinstance(value, dict) and str(value.get("index", "")).isdigit():
            replacements[int(value["index"])] = _normalize_text(str(value.get("text") or ""))
    planned = {item["index"]: item for item in items}
    issues: list[str] = []
    if set(replacements) != set(planned):
        issues.append("扩展结果没有完整返回计划中的 index")
    expanded = [dict(turn) for turn in turns]
    accepted_texts: list[dict[str, Any]] = []
    for index, item in planned.items():
        text = replacements.get(index, "")
        units = _spoken_unit_count(text, language)
        other_turns = [turn for position, turn in enumerate(expanded) if position != index] + accepted_texts
        evidence = _claim_evidence_text(item["claim_ids"], claims_by_id, cards_by_id)
        if not text or not text_matches_language(text, language):
            issues.append(f"第 {index + 1} 轮扩展语言或正文无效")
        elif not item["minimum_units"] <= units <= item["maximum_units"]:
            issues.append(f"第 {index + 1} 轮扩展量不在计划范围")
        elif not _numbers_supported(text, evidence):
            issues.append(f"第 {index + 1} 轮扩展包含资料不支持的数字")
        elif any(stem in text.lower() for stem in GENERIC_STEMS) or _is_duplicate(text, other_turns):
            issues.append(f"第 {index + 1} 轮扩展出现模板或重复")
        else:
            expanded[index]["text"] = text
            accepted_texts.append(expanded[index])
    after_minutes = _content_minutes(expanded)
    if after_minutes < target_minutes:
        issues.append("单次扩展后仍未达到整集口播目标")
    report.update(
        {
            "used": True,
            "selected_turns": sorted(planned),
            "after_minutes": round(after_minutes, 3),
            "finish_reason": getattr(generated, "finish_reason", None),
            "issues": issues,
        }
    )
    if getattr(generated, "finish_reason", None) in {"length", "max_tokens"} or issues:
        raise PodcastQualityError("整集口播扩展未通过本地验证", {"passed": False, "stage": "duration_expansion", **report})
    return expanded, report


def _duration_compression_plan(
    turns: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    excess_units: int,
    language: str,
) -> list[dict[str, Any]]:
    base_floor = 18 if language == "en" else 35
    margin = 12 if language == "en" else 24
    candidates: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(chapters))}
    for turn_index, turn in enumerate(turns):
        if _is_question_turn(turn) or turn.get("dialogue_act") not in FACTUAL_ACTS or not turn.get("claim_ids"):
            continue
        current = round(_spoken_unit_count(str(turn.get("text") or ""), language))
        minimum = max(base_floor, math.ceil(current * 0.45))
        capacity = current - minimum
        if capacity < margin:
            continue
        chapter_index = next(
            (
                index
                for index, chapter in enumerate(chapters)
                if int(chapter["turn_start"]) <= turn_index <= int(chapter["turn_end"])
            ),
            0,
        )
        candidates.setdefault(chapter_index, []).append(
            {"index": turn_index, "current_units": current, "floor_units": minimum, "capacity": capacity}
        )
    for values in candidates.values():
        values.sort(key=lambda item: (-item["capacity"], item["index"]))
    ordered: list[dict[str, Any]] = []
    while any(candidates.values()):
        for chapter_index in range(len(chapters)):
            values = candidates.get(chapter_index) or []
            if values:
                ordered.append(values.pop(0))
    target_reduction = math.ceil(excess_units * 1.05)
    selected: list[dict[str, Any]] = []
    capacity = 0
    for item in ordered:
        selected.append(item)
        capacity += item["capacity"]
        if capacity >= target_reduction:
            break
    if capacity < target_reduction:
        return []
    remaining = target_reduction
    for position, item in enumerate(selected):
        slots_left = len(selected) - position
        reduction = min(item["capacity"], max(1, math.ceil(remaining / slots_left)))
        maximum = item["current_units"] - reduction
        item["maximum_units"] = maximum
        item["minimum_units"] = max(item["floor_units"], maximum - margin)
        item["safe_minimum_units"] = max(base_floor, math.floor(maximum * 0.80))
        remaining -= reduction
    return selected


async def _compress_episode_duration(
    turns: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    claims_by_id: dict[str, dict[str, Any]],
    cards_by_id: dict[str, dict[str, Any]],
    language: str,
    target_minutes: float,
    trace: ContextUsage,
    generation_state: EpisodeGenerationState,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before_minutes = _content_minutes(turns)
    target_budget = _scene_duration_budget(language, target_minutes, len(turns), 0)
    actual_units = round(sum(_spoken_unit_count(turn["text"], language) for turn in turns))
    excess_units = max(0, actual_units - int(target_budget["maximum_units"]))
    report: dict[str, Any] = {
        "used": False,
        "before_minutes": round(before_minutes, 3),
        "target_minutes": round(target_minutes, 3),
        "excess_units": excess_units,
    }
    if before_minutes <= target_minutes * 1.20 or excess_units <= 0:
        return turns, report
    if generation_state.recovery_kind is not None:
        raise PodcastQualityError(
            "整集口播超长且唯一恢复槽已使用",
            {"passed": False, "stage": "duration_compression", **report, "recovery_kind": generation_state.recovery_kind},
        )
    plan = _duration_compression_plan(turns, chapters, excess_units, language)
    if not plan:
        raise PodcastQualityError(
            "没有足够的受支持轮次可用于口播压缩",
            {"passed": False, "stage": "duration_compression", **report},
        )
    generation_state.duration_compression_used = True
    items = []
    for item in plan:
        turn = turns[item["index"]]
        claim_ids = [value for value in turn["claim_ids"] if value in claims_by_id]
        items.append({
            **item,
            "speaker": turn["speaker"],
            "dialogue_act": turn["dialogue_act"],
            "text": turn["text"],
            "claim_ids": claim_ids,
            "claims": [claims_by_id[value]["text"] for value in claim_ids],
        })
    language_rule = "Use natural spoken English only." if language == "en" else "只使用自然的简体中文口语。"
    unit = "words" if language == "en" else "中文等价字符"
    prompt_prefix = f"""你是资料型双人播客的精简编辑。{language_rule} 只压缩列出的实质轮次，不改变说话人、dialogue act、claim_ids、数字、结论方向或相邻轮次关系。删除重复修饰和绕行表达，保留对应 claims 中的前提、机制、限定与关键含义；不得加入资料外事实、类比、审稿术语、开场白或总结。
每项必须达到自己的 minimum_units 且不超过 maximum_units，单位为{unit}。所有输入轮次都不是问句，replacement_text 也不得变成问句或以问号结尾。只输出 JSON 对象，键名 replacements；每项是 [原始整数 index, replacement_text]。必须恰好返回全部 index，不输出统计或解释。
待压缩轮次：
"""
    generated = await budgeted_chat(
        lambda budget: _segment_prompt_build(
            budget,
            prefix=prompt_prefix,
            items=items,
            renderer=lambda item: json.dumps(item, ensure_ascii=False, separators=(",", ":")),
        ),
        json_mode=True,
        timeout=420,
        # Reasoning-capable MAIN models may consume most of a 6k allowance before
        # emitting the compact JSON payload. Compression is a single bounded
        # recovery call, so reserve the full bounded allowance.
        max_tokens=10_000,
        minimum_output_tokens=min(3600, max(700, len(items) * 100)),
        temperature=0.2,
        trace=trace,
        stage="duration_compression",
    )
    raw = _extract_array(generated.content, "replacements") or []
    replacements: dict[int, str] = {}
    for value in raw:
        if isinstance(value, list) and len(value) == 2 and str(value[0]).isdigit():
            replacements[int(value[0])] = _normalize_text(str(value[1] or ""))
        elif isinstance(value, dict) and str(value.get("index", "")).isdigit():
            replacements[int(value["index"])] = _normalize_text(str(value.get("text") or ""))
    planned = {item["index"]: item for item in items}
    issues: list[str] = []
    if set(replacements) != set(planned):
        issues.append("压缩结果没有完整返回计划中的 index")
    compressed = [dict(turn) for turn in turns]
    accepted_texts: list[dict[str, Any]] = []
    unit_results: list[dict[str, int]] = []
    for index, item in planned.items():
        text = replacements.get(index, "")
        units = round(_spoken_unit_count(text, language))
        unit_results.append({
            "index": index,
            "minimum_units": int(item["safe_minimum_units"]),
            "requested_maximum_units": int(item["maximum_units"]),
            "original_units": int(item["current_units"]),
            "actual_units": units,
        })
        other_turns = [turn for position, turn in enumerate(compressed) if position != index] + accepted_texts
        evidence = _claim_evidence_text(item["claim_ids"], claims_by_id, cards_by_id)
        if not text or not text_matches_language(text, language):
            issues.append(f"第 {index + 1} 轮压缩语言或正文无效")
        elif not item["safe_minimum_units"] <= units < item["current_units"]:
            issues.append(f"第 {index + 1} 轮压缩后长度不在安全范围")
        elif _is_question_turn({"dialogue_act": item["dialogue_act"], "text": text}):
            issues.append(f"第 {index + 1} 轮压缩改变了问句属性")
        elif not _numbers_supported(text, evidence):
            issues.append(f"第 {index + 1} 轮压缩包含资料不支持的数字")
        elif any(stem in text.lower() for stem in GENERIC_STEMS) or _is_duplicate(text, other_turns):
            issues.append(f"第 {index + 1} 轮压缩出现模板或重复")
        else:
            compressed[index]["text"] = text
            accepted_texts.append(compressed[index])
    after_minutes = _content_minutes(compressed)
    if not target_minutes * 0.85 <= after_minutes <= target_minutes * 1.20:
        issues.append("单次压缩后整集口播仍不在发布时长范围")
    report.update({
        "used": True,
        "selected_turns": sorted(planned),
        "after_minutes": round(after_minutes, 3),
        "unit_results": unit_results,
        "finish_reason": getattr(generated, "finish_reason", None),
        "issues": issues,
    })
    if getattr(generated, "finish_reason", None) in {"length", "max_tokens"} or issues:
        raise PodcastQualityError("整集口播压缩未通过本地验证", {"passed": False, "stage": "duration_compression", **report})
    return compressed, report


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
    turns: list[dict[str, Any]], chapters: list[dict[str, Any]], thesis: str, language: str, trace: ContextUsage,
    claims_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sampled_indexes: set[int] = set()
    for chapter in chapters:
        start = max(0, int(chapter.get("turn_start") or 0))
        end = min(len(turns) - 1, int(chapter.get("turn_end") if chapter.get("turn_end") is not None else start))
        if end < start:
            continue
        sampled_indexes.update({start, (start + end) // 2, end})
    if not sampled_indexes and turns:
        sampled_indexes.update({0, len(turns) // 2, len(turns) - 1})
    sampled_turns = [(index, turns[index]) for index in sorted(sampled_indexes)]
    transcript = "\n".join(
        f"{index}: {turn.get('speaker', 'Host A')} [{turn.get('dialogue_act', 'explain')}] "
        f"{turn.get('text', '')} claims={','.join(turn.get('claim_ids') or []) or '-'}"
        for index, turn in sampled_turns
    )
    used_claims = list(dict.fromkeys(claim_id for _, turn in sampled_turns for claim_id in turn.get("claim_ids") or []))
    claims = [claims_by_id[value] for value in used_claims if claims_by_id and value in claims_by_id]
    prompt_prefix = f"""你是深度播客主编。全稿已通过时长、引用、说话人平衡、问题密度和重复的逐轮客观检查；下面是每个 Act 的开头、中点和结尾抽样。判断对应事实是否受 claim 支持、论证是否逐步深入、Act 之间是否自然、双方是否都贡献实质内容、是否有套话或过度绝对的结论。额外检查两点：(a) 是否把尚未确认的猜测说成已建立结论、是否把支持性论据过度放大——资料忠实度防线不变；(b) 口播是否把审稿术语（如“不能推出、只支持、边界、门槛、范围、回扣、压实、下一层”）直接说出口——区分确定与不确定应该是一句自然口语限定，而不是方法论旁白；(c) 防误读提醒是否密集重复——“别把它读成/夸成/说成”“不等于/不意味着”一类句式在同一 Act 出现多处，会让节目听起来像持续自我审查；发现任一点即记入 blocking_issues。只输出 JSON：{{"verdict":"pass|fail","scores":{{"grounding":5,"coherence":5,"depth":5,"roles":5,"repetition":5,"completeness":5}},"invalid_boundaries":[1],"blocking_issues":["会阻断发布的具体问题"],"notes":["可选润色建议"]}}。5=优秀、4=可发布、1=严重失败。blocking_issues 只能放会使某项低于 4 分或与 pass 矛盾的发布阻断项；4 分范围内的改善建议必须放 notes，不得放 blocking_issues。只有局限在 Act 边界附近、最多可改 6 轮的问题才放入 invalid_boundaries；需要重写整集时给 fail 但保持该数组为空。语言={language}。核心命题：{thesis}
Act 抽样（保留原始轮次索引）：
{transcript}
抽样所用主张：
"""
    try:
        generated = await budgeted_chat(
            lambda budget: _segment_prompt_build(
                budget,
                prefix=prompt_prefix,
                items=claims,
                renderer=lambda claim: f"[{claim['id']}] {claim['text']}",
                group_key=lambda claim: str(claim["source_id"]),
            ),
            json_mode=True,
            timeout=300,
            max_tokens=structured_output_tokens(2400),
            minimum_output_tokens=450,
            temperature=0.0,
            trace=trace,
            stage="episode_audit",
        )
        parsed = _extract_json(generated.content)
        scores = parsed.get("scores") or {}
        invalid = [int(value) for value in parsed.get("invalid_boundaries") or [] if str(value).isdigit() and 0 < int(value) < len(chapters)]
        verdict = str(parsed.get("verdict") or "").lower()
        names = ("grounding", "coherence", "depth", "roles", "repetition", "completeness")
        raw_scores_valid = isinstance(scores, dict) and all(
            str(scores.get(name, "")).isdigit() and 1 <= int(scores[name]) <= 5 for name in names
        )
        legacy_issues = [str(value)[:240] for value in parsed.get("issues") or []][:8]
        blocking_issues = [str(value)[:240] for value in parsed.get("blocking_issues") or []][:8]
        if "blocking_issues" not in parsed and (verdict == "fail" or not raw_scores_valid or min((int(scores[name]) for name in names), default=0) < 4):
            blocking_issues = legacy_issues
        notes = [str(value)[:240] for value in parsed.get("notes") or []][:8]
        if verdict == "pass" and raw_scores_valid:
            notes = (notes + legacy_issues)[:8]
        if getattr(generated, "finish_reason", None) in {"length", "max_tokens"}:
            return {
                "passed": False,
                "verdict": verdict or "incomplete",
                "scores": {},
                "invalid_boundaries": invalid,
                "issues": ["整集审校输出达到 token 上限，结果不完整"],
            }
        if verdict not in {"pass", "fail"} or not raw_scores_valid:
            return {
                "passed": False,
                "verdict": verdict or "invalid",
                "scores": {},
                "invalid_boundaries": invalid,
                "issues": ["整集审校未返回完整 verdict 与六项分数"],
            }
        normalized = {name: int(scores[name]) for name in names}
        return {
            "passed": verdict == "pass" and not invalid and not blocking_issues and min(normalized.values(), default=0) >= 4,
            "verdict": verdict,
            "scores": normalized,
            "invalid_boundaries": invalid,
            "issues": blocking_issues,
            "notes": notes,
        }
    except Exception as exc:
        detail = str(exc).strip()[:160]
        suffix = f"（{detail}）" if detail else ""
        return {
            "passed": False,
            "scores": {},
            "invalid_boundaries": [],
            "issues": [f"整集审校失败：{type(exc).__name__}{suffix}"],
        }


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
    return cjk_chars / CJK_CHARS_PER_MINUTE + latin_words / LATIN_WORDS_PER_MINUTE + len(turns) * TURN_PAUSE_SECONDS / 60


def _repeated_stem_ratio(turns: list[dict[str, Any]]) -> float:
    stems = [re.sub(r"[“\"].*", "", turn["text"])[:24] for turn in turns if turn["dialogue_act"] == "question"]
    if not stems:
        return 0.0
    repeated = sum(count - 1 for count in Counter(stems).values() if count > 1)
    return repeated / len(stems)


def _count_cliche_hits(text: str, families: dict[str, tuple[str, ...]]) -> dict[str, int]:
    return {family: sum(text.count(phrase) for phrase in phrases) for family, phrases in families.items()}


def _cliche_family_metrics(turns: list[dict[str, Any]], chapter_payloads: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    hard_counts = {family: 0 for family in CLICHE_FAMILIES}
    soft_counts = {family: 0 for family in CLICHE_SOFT_FAMILIES}
    guard_counts = {family: 0 for family in GUARD_FAMILIES}
    for turn in turns:
        text = str(turn.get("text") or "")
        for family, count in _count_cliche_hits(text, CLICHE_FAMILIES).items():
            hard_counts[family] += count
        for family, count in _count_cliche_hits(text, CLICHE_SOFT_FAMILIES).items():
            soft_counts[family] += count
        for family, pattern in GUARD_FAMILIES.items():
            guard_counts[family] += len(pattern.findall(text))
    total = sum(hard_counts.values())
    guard_total = sum(guard_counts.values())
    act_densities: list[dict[str, Any]] = []
    guard_act_densities: list[dict[str, Any]] = []
    for chapter in chapter_payloads or []:
        start = max(0, int(chapter.get("turn_start") or 0))
        end = chapter.get("turn_end")
        segment = turns[start : int(end) + 1 if end is not None else len(turns)]
        if not segment:
            continue
        hits = sum(
            sum(str(turn.get("text") or "").count(phrase) for phrase in phrases)
            for turn in segment
            for phrases in CLICHE_FAMILIES.values()
        )
        act_densities.append({"chapter_id": chapter.get("id"), "density": round(hits / len(segment), 3)})
        guard_hits = sum(
            len(pattern.findall(str(turn.get("text") or "")))
            for turn in segment
            for pattern in GUARD_FAMILIES.values()
        )
        guard_act_densities.append({"chapter_id": chapter.get("id"), "density": round(guard_hits / len(segment), 3)})
    worst_act = max((item["density"] for item in act_densities), default=0.0)
    return {
        "cliche_family_counts": {**hard_counts, **soft_counts},
        "cliche_family_density": round(total / max(1, len(turns)), 3),
        "cliche_max_family_count": max(hard_counts.values(), default=0),
        "cliche_worst_family": max(hard_counts, key=lambda family: hard_counts[family]) if total else None,
        "cliche_act_density": act_densities,
        "cliche_worst_act_density": worst_act,
        "guard_family_counts": guard_counts,
        "guard_density": round(guard_total / max(1, len(turns)), 3),
        "guard_max_family_count": max(guard_counts.values(), default=0),
        "guard_act_density": guard_act_densities,
    }


def _quality_metrics_v3(
    turns: list[dict[str, Any]], citations: list[dict[str, Any]], target_minutes: int, requested_turns: int,
    episode_audit: dict[str, Any], scene_audits: list[dict[str, Any]], selected_source_ids: list[str],
    duration_calibration: dict[str, Any] | None = None,
    chapter_payloads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    a_chars = sum(len(turn["text"]) for turn in turns if turn["speaker"] == "HOST_A")
    b_chars = sum(len(turn["text"]) for turn in turns if turn["speaker"] == "HOST_B")
    total = max(1, a_chars + b_chars)
    estimated = _content_minutes(turns)
    factual = [turn for turn in turns if turn["dialogue_act"] in FACTUAL_ACTS or turn["claim_ids"]]
    source_coverage = len({citation["source_id"] for citation in citations})
    duplicate_pairs = sum(_similar(left["text"], right["text"]) >= 0.84 for index, left in enumerate(turns) for right in turns[index + 1 :])
    question_turns = sum(_is_question_turn(turn) for turn in turns)
    longest_question_run = current_question_run = 0
    for turn in turns:
        if _is_question_turn(turn):
            current_question_run += 1
            longest_question_run = max(longest_question_run, current_question_run)
        else:
            current_question_run = 0
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
        "inferred_claim_turns": sum(bool(turn.get("claim_id_inferred")) for turn in turns),
        "bridge_turns": sum(turn["dialogue_act"] in NONFACTUAL_ACTS and not turn["claim_ids"] for turn in turns),
        "all_factual_turns_cited": all(turn["citation_ids"] for turn in factual),
        "duplicate_pairs": duplicate_pairs,
        "repeated_stem_ratio": round(_repeated_stem_ratio(turns), 3),
        "question_ratio": round(question_turns / max(1, len(turns)), 3),
        "longest_question_run": longest_question_run,
        "source_coverage": source_coverage,
        "selected_sources": len(selected_source_ids),
        "scene_repairs": sum(bool(audit.get("repaired")) for audit in scene_audits),
        "scene_scores": [audit.get("scores") for audit in scene_audits],
        "episode_audit": episode_audit,
        "safe_fallback_turns": 0,
        "duration_calibration": duration_calibration or {},
    }
    report.update(_cliche_family_metrics(turns, chapter_payloads))
    report["deterministic_passed"] = bool(
        0.85 <= report["duration_ratio"] <= 1.20
        and 0.40 <= report["host_a_ratio"] <= 0.60
        and 0.20 <= report["question_ratio"] <= 0.40
        and report["longest_question_run"] <= 2
        and report["uncited_factual_turns"] == 0
        and report["duplicate_pairs"] == 0
        and report["repeated_stem_ratio"] <= 0.10
        and report["cliche_family_density"] <= CLICHE_EPISODE_DENSITY_LIMIT
        and report["cliche_max_family_count"] <= CLICHE_FAMILY_COUNT_LIMIT
        and report["cliche_worst_act_density"] <= CLICHE_ACT_DENSITY_LIMIT
        and report["guard_density"] <= GUARD_EPISODE_DENSITY_LIMIT
        and report["guard_max_family_count"] <= GUARD_FAMILY_COUNT_LIMIT
    )
    report["passed"] = bool(report["deterministic_passed"] and episode_audit.get("passed"))
    return report


def _deterministic_failure_reasons(report: dict[str, Any]) -> list[str]:
    checks = (
        (not 0.85 <= float(report.get("duration_ratio") or 0) <= 1.20, "时长不在目标的 85%–120%"),
        (not 0.40 <= float(report.get("host_a_ratio") or 0) <= 0.60, "主持人篇幅不均衡"),
        (not 0.20 <= float(report.get("question_ratio") or 0) <= 0.40, "问句比例不合格"),
        (int(report.get("longest_question_run") or 0) > 2, "连续问句过多"),
        (int(report.get("uncited_factual_turns") or 0) > 0, "存在未引用的事实轮"),
        (int(report.get("duplicate_pairs") or 0) > 0, "存在重复轮次"),
        (float(report.get("repeated_stem_ratio") or 0) > 0.10, "问句模板重复"),
        (
            float(report.get("cliche_family_density") or 0) > CLICHE_EPISODE_DENSITY_LIMIT
            or int(report.get("cliche_max_family_count") or 0) > CLICHE_FAMILY_COUNT_LIMIT,
            f"审计式套话密度过高（{report.get('cliche_worst_family') or '-'} 出现 {report.get('cliche_max_family_count') or 0} 次，整集 {report.get('cliche_family_density') or 0}/轮）",
        ),
        (
            float(report.get("cliche_worst_act_density") or 0) > CLICHE_ACT_DENSITY_LIMIT,
            f"审计式套话在单个 Act 内过密（最高 {report.get('cliche_worst_act_density') or 0}/轮）",
        ),
        (
            float(report.get("guard_density") or 0) > GUARD_EPISODE_DENSITY_LIMIT
            or int(report.get("guard_max_family_count") or 0) > GUARD_FAMILY_COUNT_LIMIT,
            f"防误读提醒句式过密（整集 {report.get('guard_density') or 0}/轮，{report.get('guard_family_counts') or {}}）",
        ),
    )
    return [message for failed, message in checks if failed]


async def build_podcast_script(
    notebook_id: str,
    payload: dict[str, Any],
    *,
    progress: Callable[[str, float], None] | None = None,
    act_ready: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ids = source_scope(notebook_id, payload.get("source_ids"))
    if not ids:
        raise ValueError("当前范围没有已就绪的文档")
    language, language_selection = resolve_output_language(DB, ids, payload.get("language", "zh-CN"))
    focus = str(payload.get("focus") or "").strip()
    if progress:
        progress("构建全篇证据地图", 0.08)
    requested_hint = int(payload.get("minutes") or 0)
    evidence_per_source = max(20, math.ceil(max(5, requested_hint) * 2 / max(1, len(ids))))
    rows = select_podcast_evidence(notebook_id, ids, focus, evidence_per_source)
    cards, all_citations = build_evidence_cards(rows)
    if len(cards) < 2:
        raise ValueError("资料内容不足，无法生成深度播客")
    if progress:
        progress("提取可引用主张", 0.12)
    context_usage = ContextUsage()
    claims = build_claim_ledger(cards)
    if len(claims) < 2:
        raise ValueError("资料中缺少足够的可验证主张")
    duration_mode = payload.get("duration_mode") or ("fixed" if payload.get("minutes") else "auto")
    requested_minutes = int(payload.get("minutes") or 0) or None
    estimated_chapters = max(3, min(6, round(math.sqrt(max(1, len(claims))))))
    target_minutes = requested_minutes if duration_mode == "fixed" and requested_minutes else estimate_auto_minutes(estimated_chapters, len(cards))
    total_target = target_turn_count(target_minutes)
    profile = podcast_generation_profile()
    act_count = max(2, math.ceil(total_target / profile["scene_turns"]))
    context_usage.request_limit = act_count + 4
    context_usage.total_token_limit = min(45_000, 14_000 + 750 * total_target)
    if progress:
        progress("规划递进式剧集结构", 0.16)
    episode_plan, outline_degraded = await create_episode_plan(claims, language, focus, context_usage, act_count)
    chapters = episode_plan["chapters"]
    chapter_targets = [total_target // len(chapters) for _ in chapters]
    for index in range(total_target % len(chapters)):
        chapter_targets[index] += 1
    cards_by_id = {card["id"]: card for card in cards}
    claims_by_id = {claim["id"]: claim for claim in claims}
    turns: list[dict[str, Any]] = []
    memory = EpisodeMemory(episode_plan["episode_thesis"])
    chapter_payloads: list[dict[str, Any]] = []
    scene_audits: list[dict[str, Any]] = []
    generation_state = EpisodeGenerationState()
    duration_goal = target_minutes * GENERATION_DURATION_TARGET_RATIO
    duration_calibration: dict[str, Any] = {
        "strategy": "slot_budget_with_single_expansion_v3",
        "version": PODCAST_DURATION_CALIBRATION_VERSION,
        "generation_target_ratio": GENERATION_DURATION_TARGET_RATIO,
        "target_estimated_minutes": round(duration_goal, 2),
        "acts": [],
    }
    for chapter_index, chapter in enumerate(chapters):
        if progress:
            progress(f"连贯续写 Act {chapter_index + 1}/{len(chapters)}", 0.20 + 0.34 * chapter_index / max(1, len(chapters)))
        start_index = len(turns)
        chapter_claims = [claims_by_id[value] for value in chapter["claim_ids"] if value in claims_by_id]
        if chapter_index == 0:
            chapter = {**chapter, "bridge_in": "", "purpose": f"{episode_plan['episode_thesis']}；{chapter['purpose']}" if language != "en" else f"Open the central question: {episode_plan['episode_thesis']}. {chapter['purpose']}"}
        if chapter_index == len(chapters) - 1:
            chapter = {**chapter, "bridge_out": "", "purpose": f"{chapter['purpose']}；只用已讨论主张回扣核心问题" if language != "en" else f"{chapter['purpose']}; resolve the central question using only discussed claims"}
        current_minutes = _content_minutes(turns)
        duration_budget = _remaining_scene_duration_budget(
            language,
            duration_goal,
            current_minutes,
            total_target,
            chapter_targets,
            chapter_index,
        )
        try:
            scene_turns, scene_audit = await create_linked_scene(
                scene_kind="act", chapter=chapter, claims=chapter_claims, cards_by_id=cards_by_id, memory=memory,
                existing_turns=turns, target=chapter_targets[chapter_index], language=language, profile=profile, trace=context_usage,
                duration_budget=duration_budget, generation_state=generation_state,
            )
        except PodcastQualityError as exc:
            exc.report.update({
                "chapter_id": chapter.get("id"),
                "chapter_index": chapter_index,
                "completed_acts": list(duration_calibration["acts"]),
                "current_duration_budget": duration_budget,
                "continuation_used": generation_state.continuation_used,
                "context_usage": context_usage.as_dict(),
            })
            raise
        turns.extend(scene_turns)
        if act_ready:
            act_ready({
                "chapter_index": chapter_index,
                "start_index": start_index,
                "language": language,
                "turns": [dict(turn) for turn in scene_turns],
            })
        scene_audits.append(scene_audit)
        duration_calibration["acts"].append({"chapter_id": chapter["id"], **scene_audit.get("duration", duration_budget)})
        _update_memory(memory, scene_turns, chapter, profile["recent_turns"])
        chapter_payloads.append({**chapter, "turn_start": start_index, "turn_end": len(turns) - 1})
    expansion_report: dict[str, Any] = {"used": False}
    current_episode_minutes = _content_minutes(turns)
    release_minimum_minutes = target_minutes * 0.85
    if current_episode_minutes < release_minimum_minutes:
        if progress:
            progress("校准整集口播密度", 0.56)
        try:
            turns, expansion_report = await _expand_episode_duration(
                turns,
                chapter_payloads,
                claims_by_id,
                cards_by_id,
                language,
                duration_goal,
                context_usage,
                generation_state,
            )
        except PodcastQualityError as exc:
            exc.report.update(
                {
                    "completed_acts": list(duration_calibration["acts"]),
                    "recovery_kind": generation_state.recovery_kind,
                    "context_usage": context_usage.as_dict(),
                }
            )
            raise
    elif current_episode_minutes < duration_goal:
        expansion_report = {
            "used": False,
            "skipped": True,
            "reason": "release_duration_gate_already_met",
            "estimated_minutes": round(current_episode_minutes, 3),
            "release_minimum_minutes": round(release_minimum_minutes, 3),
        }
    duration_calibration["expansion"] = expansion_report
    compression_report: dict[str, Any] = {"used": False}
    current_episode_minutes = _content_minutes(turns)
    if current_episode_minutes > target_minutes * 1.20:
        if progress:
            progress("压缩整集口播密度", 0.57)
        try:
            turns, compression_report = await _compress_episode_duration(
                turns,
                chapter_payloads,
                claims_by_id,
                cards_by_id,
                language,
                float(target_minutes),
                context_usage,
                generation_state,
            )
        except PodcastQualityError as exc:
            exc.report.update({
                "completed_acts": list(duration_calibration["acts"]),
                "recovery_kind": generation_state.recovery_kind,
                "context_usage": context_usage.as_dict(),
            })
            raise
    duration_calibration["compression"] = compression_report
    if generation_state.recovery_kind is not None:
        # Expansion/compression is deliberately bounded to one call, but its
        # output must not consume the token allowance reserved for the final
        # publishability audit. The absolute 45k task ceiling still applies.
        _reserve_episode_audit_after_recovery(context_usage)
    provisional_used_evidence = {evidence_id for turn in turns for evidence_id in turn["citation_ids"]}
    provisional_citations = [citation for citation in all_citations if citation["id"] in provisional_used_evidence]
    skipped_audit = {
        "passed": False,
        "skipped": True,
        "scores": {},
        "invalid_boundaries": [],
        "issues": ["客观脚本门禁失败，未调用整集审校"],
    }
    preflight = _quality_metrics_v3(
        turns, provisional_citations, target_minutes, total_target, skipped_audit, scene_audits, ids,
        duration_calibration, chapter_payloads
    )
    if not preflight.get("deterministic_passed", preflight.get("passed", False)):
        preflight["deterministic_failure_reasons"] = _deterministic_failure_reasons(preflight)
        context_usage.stop_reason = "deterministic_quality_gate"
        preflight["context_usage"] = context_usage.as_dict()
        raise PodcastQualityError("整集脚本未达到客观发布门槛", preflight)
    if progress:
        progress("执行整集连贯性审校", 0.58)
    episode_audit = await _audit_episode(turns, chapter_payloads, episode_plan["episode_thesis"], language, context_usage, claims_by_id)
    used_evidence = {evidence_id for turn in turns for evidence_id in turn["citation_ids"]}
    used_citations = [citation for citation in all_citations if citation["id"] in used_evidence]
    remap = {citation["id"]: f"S{index}" for index, citation in enumerate(used_citations, start=1)}
    citations = [{**citation, "id": remap[citation["id"]]} for citation in used_citations]
    for turn in turns:
        turn["citation_ids"] = [remap[value] for value in turn["citation_ids"] if value in remap]
    for index, turn in enumerate(turns, start=1):
        turn["id"] = f"turn_{index}"
        turn["chapter_id"] = next((chapter["id"] for chapter in chapter_payloads if chapter["turn_start"] <= index - 1 <= chapter["turn_end"]), "unknown")
    quality = _quality_metrics_v3(
        turns, citations, target_minutes, total_target, episode_audit, scene_audits, ids,
        duration_calibration, chapter_payloads
    )
    quality["recovery"] = {
        "continuation_used": generation_state.continuation_used,
        "duration_expansion_used": generation_state.duration_expansion_used,
        "duration_compression_used": generation_state.duration_compression_used,
        "empty_response_retry_used": generation_state.empty_response_retry_used,
        "recovery_kind": generation_state.recovery_kind,
        "boundary_repair_allowed": False,
    }
    if not quality["passed"]:
        quality["context_usage"] = context_usage.as_dict()
        raise PodcastQualityError("整集脚本未达到发布门槛", quality)
    if outline_degraded:
        context_usage.mark_fallback()
    script = "\n".join(f"{turn['speaker']}: {turn['text']} {' '.join(f'[{value}]' for value in turn['citation_ids'])}" for turn in turns)
    return {
        "version": PODCAST_ENGINE_VERSION,
        "engine": {
            **profile,
            "strategy": "editorial_acts",
            "version": PODCAST_ENGINE_VERSION,
            "duration_calibration_version": PODCAST_DURATION_CALIBRATION_VERSION,
        },
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
