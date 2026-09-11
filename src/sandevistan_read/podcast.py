from __future__ import annotations

from .delivery import CURRENT as DELIVERY, delivery_task, assessment, claim_recovery

import json
import math
import re
import unicodedata
import copy
from collections import Counter
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Any, Callable

import httpx

from .context_budget import ContextUsage, PromptBudget, TokenLimits, estimate_messages_tokens, pack_items, structured_output_tokens, podcast_chapter_capacity, podcast_stage_minutes
from .database import DB, json_load
from .podcast_contracts import scene_schema, plan_schema, audit_schema
from .providers import PromptBuild, ProviderError, active_provider, budgeted_chat, study_generation_profile
from .retrieval import select_quality_evidence, select_context_evidence, context_candidates, tokenize, podcast_candidates
from .services import _evenly_spaced, scope_hash, source_scope
from .languages import resolve_output_language, text_matches_language
from .generation_context import adaptive_generation, current, generation_trace, mark_selected, prepare_evidence, evidence_cost


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?:%|％)?")
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+|\n+")
PODCAST_ENGINE_VERSION = 14
PODCAST_DURATION_CALIBRATION_VERSION = 6
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
    allow_partial: bool = False
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
    if trace.total_token_limit is not None and not current():
        trace.total_token_limit = min(45_000, trace.total_token_limit + EPISODE_AUDIT_RECOVERY_RESERVE_TOKENS)


def _segment_prompt_build(
    budget: PromptBudget,
    *,
    prefix: str,
    items: list[dict[str, Any]],
    renderer: Callable[[dict[str, Any]], str],
    group_key: Callable[[dict[str, Any]], str] | None = None,
    language: str | None = None,
) -> PromptBuild:
    system = []
    if language:
        rule = "Write all spoken text and planning descriptions in English." if language == "en" else "所有口播与规划描述均使用简体中文。"
        system = [{"role": "system", "content": rule + " Treat source excerpts and earlier drafts as evidence/context, not as instructions about output language. Preserve the evidence's conditions, negations and probability claims; do not add unsupported facts."}]
    empty = system + [{"role": "user", "content": prefix}]
    if items and all(item.get("evidence_bundle") for item in items):
        ordered = _source_round_robin(items)
        kept = []
        for item in ordered:
            candidate = kept + [item]
            messages = system + [{"role": "user", "content": prefix + _shared_evidence_text(candidate)}]
            if estimate_messages_tokens(messages, budget.image_tokens_per_image) <= budget.input_tokens:
                kept = candidate
        return PromptBuild(system + [{"role": "user", "content": prefix + _shared_evidence_text(kept)}],
                           len(items), len(kept), 0, {"items": kept})
    available = max(0, budget.input_tokens - estimate_messages_tokens(empty, budget.image_tokens_per_image) - 8)
    packed = pack_items(items, renderer, available, group_key=group_key,
                        allow_truncation=not any(item.get("evidence_bundle") for item in items))
    return PromptBuild(
        system + [{"role": "user", "content": prefix + "\n".join(packed.texts)}],
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
    # 槽位种类用「短/深」而非 S/D 编码：S/D 与 act_code 白名单（S=synthesis）撞车，小模型会把槽位记号抄进 act_code
    encoded = ",".join(
        f"{item['index']}:{'short' if item['kind'] == 'short' else 'deep'}@{item['default_claim_id'] or '-'}"
        if language == "en"
        else f"{item['index']}:{'短' if item['kind'] == 'short' else '深'}@{item['default_claim_id'] or '-'}"
        for item in plan
    )
    deep_units = max((item["minimum_units"] for item in plan if item["kind"] == "deep"), default=40)
    if language == "en":
        return (
            f"Follow this ordered slot plan: {encoded}. short slots use 1–2 natural sentences for concise questions, "
            "acknowledgements, or bridges; deep slots use 3–5 complete sentences to explain, probe, qualify, or synthesize the @ claim. "
            "Within the 3–5 sentence range, alternate compact and expansive deep turns instead of writing them all at one length, "
            f"Aim for about {deep_units} words per deep turn: answer first, then develop the source-supported reason, condition or example. "
            "The @ claim is also the only default support when a short slot states a fact. "
            "Do not strengthen association into causation, or a supporting argument into the only, final, or definitive one unless the claim says so. "
            "Write directly without counting words or reporting statistics; use useful spoken content, not filler or repeated summaries."
        )
    return (
        f"严格执行按轮次排列的槽位计划：{encoded}。短槽用 1–2 个自然句完成简洁追问、回应或承接；"
        "深槽用 3–5 个完整但紧凑的句子解释、追问、辨析或综合 @ 后的主张；在 3–5 句范围内让紧凑轮与展开轮长短交替，"
        f"深槽以约 {deep_units} 个中文等价字符为篇幅参考：先回应问题，再依据原文展开原因、条件或实例；"
        "短槽一旦陈述事实，也只能使用该槽的 @ 主张作为默认支持。"
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
    state = current()
    if state:
        candidates = podcast_candidates(context_candidates([r for r in state.rows if r["source_id"] in source_ids]))
        groups = _podcast_evidence_groups(candidates)
        if focus.strip():
            ranked = select_quality_evidence(notebook_id, source_ids, limit=limit, focus=focus)
            rank = {r["id"]: i for i, r in enumerate(ranked)}
            groups.sort(key=lambda r: rank.get(r["id"], len(rank)))
            selected, charged = [], set()
            remaining = state.plan.evidence_tokens
            for group in groups:
                fresh = {(r['source_id'], r['id']): r for r in group['evidence_rows']
                         if (r['source_id'], r['id']) not in charged}
                cost = sum(evidence_cost(r) for r in fresh.values())
                if cost <= remaining:
                    selected.append(group)
                    charged.update(fresh)
                    remaining -= cost
        else:
            selected = select_context_evidence(groups, state.plan.evidence_tokens, dependencies=lambda r: r['evidence_rows'])
        result: dict[str, dict[str, Any]] = {}
        for group in selected:
            for row in group["evidence_rows"]:
                result.setdefault(row["id"], {**row, "related_chunk_ids": [], "supporting_only": True})
            result[group["id"]]["related_chunk_ids"] = [r["id"] for r in group["evidence_rows"]]
            result[group["id"]]["supporting_only"] = False
        values = list(result.values())
        mark_selected(values)
        return values
    return podcast_candidates(select_quality_evidence(notebook_id, source_ids, limit=limit, focus=focus))


def _podcast_evidence_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Charge adjacent qualifying context together with its original passage."""
    positions = {(r["source_id"], r.get("ordinal")): r for r in rows if isinstance(r.get("ordinal"), int)}
    continuation = re.compile(r"^(?:however\b|as such\b|unless\b|provided\b|this (?:means|requires)\b|but\b|然而|但是|仅当|前提|因此)", re.I)
    groups = []
    for row in rows:
        pieces = [row]
        ordinal = row.get("ordinal")
        if isinstance(ordinal, int):
            previous = positions.get((row["source_id"], ordinal - 1))
            following = positions.get((row["source_id"], ordinal + 1))
            if previous and (continuation.search(row["content"].lstrip()) or re.match(r"^[a-z]", row["content"].lstrip())):
                pieces.insert(0, previous)
            if following and (continuation.search(following["content"].lstrip()) or re.match(r"^[a-z]", following["content"].lstrip())):
                pieces.append(following)
        groups.append({**row, "content": "\n".join(p["content"] for p in pieces), "evidence_rows": pieces})
    unique = {}
    for group in groups:
        key = tuple(sorted((r['source_id'], r['id']) for r in group['evidence_rows']))
        unique.setdefault(key, group)
    return list(unique.values())


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
            "ordinal": row.get("ordinal", index - 1),
            "related_chunk_ids": row.get("related_chunk_ids", []),
            "supporting_only": row.get("supporting_only", False),
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
        try:
            parsed, consumed = json.JSONDecoder().raw_decode(raw[start:end + 1])
        except json.JSONDecodeError:
            return {}
        # Some compatible services append an extra closing brace. Accept only
        # that unambiguous suffix, never a second object or diagnostic content.
        if raw[start + consumed:end + 1].strip("} \t\r\n"):
            return {}
        return parsed if isinstance(parsed, dict) else {}


def _ordered_records(value: Any) -> list[Any] | None:
    """Accept an array or an unambiguous contiguous numeric-key object."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and value:
        for start in (0, 1):
            keys = [str(i) for i in range(start, start + len(value))]
            if set(value) == set(keys):
                return [value[key] for key in keys]
    return None


def _extract_array(raw: str, key: str) -> list[Any] | None:
    """Recover only complete objects from a possibly truncated JSON list."""
    parsed = _extract_json(raw)
    values = _ordered_records(parsed.get(key))
    if values is not None:
        return values
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


def _spoken_question(text: str) -> bool:
    """Use spoken syntax, never an advisory model label, to detect questions."""
    ending = str(text or "").strip().rstrip('’\'”"')
    if ending.endswith(("?", "？")):
        return True
    if ending.endswith(("。", ".", "!", "！")):
        return False
    last = re.split(r"[。!?！？]\s*", ending)[-1].strip()
    return bool(re.search(r"(?:吗|呢)$", last) or re.match(
        r"^(?:为什么|怎么|如何|是否|能否|难道|(?:so,?\s+)?(?:why|how|when|where|what)\s+(?:is|are|do|does|did|can|could|would|should|will)\b)",
        last, re.I))


def _coerce_dialogue_act(value: Any, text: str, claim_ids: Any) -> str:
    """Map compact act codes; tolerate small-model drift such as slot-plan tokens (短@C1/deep@C2)."""
    raw = str(value or "").split("@", 1)[0].strip()
    mapped = COMPACT_ACT_CODES.get(raw.upper())
    if (mapped or raw.lower()) == "question" and not _spoken_question(text) and re.search(r"[。.!！][’'”\"]?$", str(text).strip()):
        return "explain"
    if mapped:
        return mapped
    if raw.lower() in ALLOWED_DIALOGUE_ACTS:
        return raw.lower()
    if not raw:
        return ""
    # 未知记号（如旧槽位编码 D）按内容形态挽救，后续的确定性门禁仍然照常校验
    if isinstance(claim_ids, list) and claim_ids:
        return "explain"
    if text.rstrip().endswith(("?", "？")):
        return "question"
    return "acknowledgement"


def _extract_turns(raw: str) -> list[dict[str, Any]] | None:
    mapped = _extract_json(raw) if '"chapter_bodies"' in raw else None
    if isinstance(mapped, dict) and "chapter_bodies" in mapped:
        bodies = mapped.get("chapter_bodies")
        if not isinstance(bodies, dict) or not bodies:
            return None
        sections = [("opening", mapped.get("opening")), *bodies.items(), ("closing", mapped.get("closing"))]
        complete = []
        for key, values in sections:
            maximum = 4 if key in {"opening", "closing"} else 32
            if not isinstance(values, list) or not 2 <= len(values) <= maximum:
                return None
            group = _extract_turns(json.dumps({"turns": values}, ensure_ascii=False)) or []
            if len(group) != len(values) or (key != "opening" and _is_question_turn(group[-1])):
                return None
            for i, turn in enumerate(group):
                turn.update(source_chapter_id=key, exchange_id=f"section/{key}", exchange_start=i == 0)
            complete.extend(group)
        return complete
    parsed_core = _extract_json(raw) if '"opening"' in raw else None
    if isinstance(parsed_core, dict) and any(key in parsed_core for key in ("opening", "body", "closing")):
        sections = [parsed_core.get(key) for key in ("opening", "body", "closing")]
        if not all(isinstance(section, list) and 2 <= len(section) <= (32 if index == 1 else 4)
                   for index, section in enumerate(sections)):
            return None
        # These are narrative sections, not independently removable exchanges:
        # an opening question may be answered by the first turn of the body.
        complete = []
        for index, section in enumerate(sections):
            group = _extract_turns(json.dumps({"turns": section}, ensure_ascii=False)) or []
            if len(group) != len(section):
                return None
            for position, turn in enumerate(group):
                turn.update(exchange_id=f"exchange_{index + 1}", exchange_start=position == 0)
            complete.extend(group)
        if not complete or _is_question_turn(complete[-1]):
            return None
        return complete
    if re.search(r'"exchanges"\s*:', raw):
        exchanges = _extract_array(raw, "exchanges") or []
        complete = []
        pending: list[dict[str, Any]] = []
        gap = False
        for index, exchange in enumerate(exchanges):
            if not isinstance(exchange, dict):
                pending = []
                gap = True
                continue
            values = _ordered_records(exchange.get('turns'))
            if values is None:
                pending = []
                gap = True
                continue
            group = _extract_turns(json.dumps({"turns": values}, ensure_ascii=False)) or []
            if len(group) != len(values) or any(
                not isinstance(t.get("text"), str) or not t["text"].strip()
                or t.get("speaker") not in {"HOST_A", "HOST_B"} for t in group
            ):
                pending = []
                gap = True
                continue
            if len(pending) >= 2:
                supports = {cid for t in pending for cid in t.get("claim_ids", [])}
                if (_is_question_turn(pending[-1]) and group and not _is_question_turn(group[0])
                        and pending[-1]["speaker"] != group[0]["speaker"]
                        and (not supports or supports & set(group[0].get("claim_ids", [])))
                        and len(pending) + len(group) <= 8):
                    # A question need not cite a claim. Retain the adjacent
                    # answer without inventing support for either turn.
                    group = pending + group
                    pending = []
                    gap = False
                else:
                    pending = []
                    gap = True
            if gap and group and re.match(r"^(?:并不是|不是的|确实|没错|正是|因此|所以|也就是说|这|no\b|not really\b|yes\b|exactly\b|indeed\b|therefore\b|that\b)", group[0]["text"].lstrip(), re.I):
                continue
            gap = False
            if len(group) == 1:
                # Some providers put each turn in its own exchange object.
                # Join only adjacent alternating turns on the same evidence,
                # or a question and its following statement. Never bridge a
                # malformed object or alter the provider's spoken text.
                previous = pending[0] if pending else None
                if previous and previous.get("speaker") != group[0].get("speaker") and (
                    _is_question_turn(previous)
                    or set(previous.get("claim_ids") or []) & set(group[0].get("claim_ids") or [])
                ) and not _is_question_turn(group[0]):
                    group = pending + group
                    pending = []
                else:
                    pending = group
                    continue
            else:
                pending = []
            if not 2 <= len(group) <= 8:
                pending = []
                gap = True
                continue
            if _is_question_turn(group[-1]):
                pending = group
                continue
            if not re.search(r"[。.!！][’'”\"]?$", str(group[-1].get("text") or "").strip()):
                gap = True
                continue
            for position, turn in enumerate(group):
                turn.update(exchange_id=f"exchange_{index + 1}", exchange_start=position == 0)
            complete.extend(group)
        return complete or None
    values = _extract_array(raw, "turns")
    if values is None:
        values = _ordered_records(_extract_json(raw))
    if values is None:
        return None
    turns: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, dict):
            act = _coerce_dialogue_act(
                value.get("dialogue_act") or value.get("act_code"), str(value.get("text") or ""), value.get("claim_ids")
            )
            if act:
                value = {**value, "dialogue_act": act}
            if str(value.get("speaker")).upper() in {"A", "B"}:
                value = {**value, "speaker": "HOST_" + value["speaker"].upper()}
            turns.append(value)
            continue
        if isinstance(value, list) and len(value) > 4 and all(
            isinstance(identifier, str) and re.fullmatch(r"C\d+(?:[_|]E\d+)?", identifier)
            for identifier in value[3:]
        ):
            # Compact models sometimes spread citation IDs into extra tuple
            # fields. This is unambiguous only for recognized ID syntax.
            value = value[:3] + [value[3:]]
        if not isinstance(value, list) or len(value) not in {3, 4}:
            if DELIVERY.get():
                break
            continue
        speaker, act_code, text = value[:3]
        claim_ids = value[3] if len(value) == 4 else []
        # A common compact variant omits act_code, not claim_ids. Preserve its
        # actual prose instead of turning the reference array into spoken text.
        if len(value) == 3 and isinstance(value[1], str) and isinstance(value[2], list):
            speaker, text, claim_ids = value
            act_code = "question" if text.rstrip().endswith(("?", "？")) else "explain"
        if not isinstance(text, str):
            if DELIVERY.get():
                break
            continue
        act = _coerce_dialogue_act(act_code, str(text or ""), claim_ids)
        if not act:
            continue
        turns.append({
            "speaker": f"HOST_{str(speaker).upper()}" if str(speaker).upper() in {"A", "B"} else speaker,
            "dialogue_act": act,
            "text": text,
            "claim_ids": claim_ids if isinstance(claim_ids, list) else [claim_ids] if isinstance(claim_ids, str) else [],
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
                    "id": f"chapter_{original_index}",
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
    value = re.sub(r"\[[CES]\d+(?:\s*[,|，;]\s*[CES]\d+)*\]", "", value)
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


async def _critic_grounded_pairs(pairs: list[dict[str, Any]], language: str, trace: ContextUsage | None = None, *, strict: bool = False) -> set[int]:
    answers = "\n".join(f"{index}: {pair['answer']}" for index, pair in enumerate(pairs))
    prompt_prefix = f"""你是严格的翻译忠实度审校器。逐项比较原文摘录与回答。回答必须只是摘录的忠实翻译或压缩改述；若新增因果、绝对化结论、实体、数字或摘录没有的判断，就判为不支持。
只输出 JSON：{{"invalid_indexes":[0]}}。语言={language}。
回答：
{answers}
原文摘录：
"""
    indexed_pairs = [{**pair, "index": index} for index, pair in enumerate(pairs)]
    try:
        result = await budgeted_chat(
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
        )
        parsed = _extract_json(result.content)
        if strict and (result.build.truncated_segments or result.build.included_segments != len(pairs)
                       or not isinstance(parsed.get("invalid_indexes"), list)):
            return set(range(len(pairs)))
        values = parsed.get("invalid_indexes") or []
        if strict and any(type(value) is not int or not 0 <= value < len(pairs) for value in values):
            return set(range(len(pairs)))
        return {int(value) for value in values if isinstance(value, int) or str(value).isdigit()}
    except Exception:
        return set(range(len(pairs))) if strict else set()


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
    """Keep source-addressable arguments intact, including their qualifications."""
    claims: list[dict[str, Any]] = []
    by_chunk = {c.get("chunk_id"): c for c in cards if c.get("chunk_id")}
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for card in cards:
        if card.get("supporting_only"):
            continue
        linked = [by_chunk[key] for key in card.get("related_chunk_ids", []) if key in by_chunk]
        if not linked:
            linked = [card]
        original = "\n".join(c["content"] for c in linked)
        identity = (card["source_id"], tuple(sorted(c["id"] for c in linked)))
        if not identity[1] or identity in seen:
            continue
        seen.add(identity)
        claims.append({"id": f"C{len(claims) + 1}", "text": card["content"],
                       "evidence_ids": [c["id"] for c in linked], "source_id": card["source_id"],
                       "filename": card["filename"], "locator": card.get("locator") or {},
                       "qualification": "Read the complete original for its conditions; do not infer unconditional claims.",
                       "original": original, "attribution": card["filename"],
                       "statement_kind": "source_excerpt", "evidence_bundle": True, "preparation_notes": [],
                       "ordinal": card.get("ordinal", len(claims)),
                       "evidence_passages": [{k: c.get(k, "") for k in ("id", "chunk_id", "source_id", "filename", "content")} for c in linked]})
    return claims


def _source_round_robin(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        buckets.setdefault(item["source_id"], []).append(item)
    return [values[i] for i in range(max(map(len, buckets.values()), default=0)) for values in buckets.values() if i < len(values)]


def _shared_evidence_payload(claims: list[dict[str, Any]]) -> dict[str, Any]:
    passages, records = {}, {}
    for claim in claims:
        if claim.get("evidence_passages"):
            for passage in claim["evidence_passages"]:
                passages[passage["id"]] = {k: passage[k] for k in ("source_id", "filename", "content")}
            records[claim["id"]] = {"evidence_ids": claim["evidence_ids"], "reading_hints": claim.get("preparation_notes", [])}
        else:
            records[claim["id"]] = {key: claim.get(key, "") for key in ("text", "filename", "qualification", "statement_kind", "original")}
    return {"passages": passages, "claims": records}


def _shared_evidence_text(claims: list[dict[str, Any]]) -> str:
    return "\nRead each claim's referenced passages together, including all conditions. Shared passages appear once:\n" + json.dumps(_shared_evidence_payload(claims), ensure_ascii=False)


def merge_prepared_claims(claims: list[dict[str, Any]], cards: list[dict[str, Any]],
                          notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Notes guide reading; they never replace the source's remaining argument."""
    result = copy.deepcopy(claims)
    by_chunk = {c.get("chunk_id"): c["id"] for c in cards}
    for note in notes:
        evidence_id = by_chunk.get(note.get("chunk_id"))
        for claim in result:
            if evidence_id not in claim["evidence_ids"]:
                continue
            quote = re.sub(r"\s+", " ", str(note.get("quote") or "")).strip()
            if not quote or quote not in re.sub(r"\s+", " ", claim["original"]):
                continue
            hint = {key: str(note.get(key) or "") for key in ("claim", "qualification", "statement_kind", "quote")}
            hints = claim.setdefault("preparation_notes", [])
            if hint not in hints:
                hints.append(hint)
    return result


def _render_claim_bundle(claim: dict[str, Any]) -> str:
    if claim.get("evidence_bundle"):
        return (f"[{claim['id']}|{claim['filename']}] Source passage (interpret all sentences together):\n"
                + claim["original"] + "\nReading hints, not additional evidence: "
                + json.dumps(claim.get("preparation_notes", []), ensure_ascii=False))
    return f"[{claim['id']}|{claim['filename']}] {claim['text']} Conditions: {claim.get('qualification', '')} Type: {claim.get('statement_kind', 'source_excerpt')}"


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


def _fit_episode_chapters(chapters: list[dict[str, Any]], target: int, language: str) -> list[dict[str, Any]]:
    """Keep usable editorial structure and split long acts locally to fit output capacity."""
    if DELIVERY.get():
        # Reusing a claim for a different mechanism or qualification is legitimate.
        seen = set()
        unique = []
        for chapter in chapters:
            identity = (_review_text(chapter.get("new_information") or chapter.get("purpose", "")), tuple(sorted(chapter["claim_ids"])),
                        *(str(chapter.get(key) or "") for key in ("question", "mechanism", "required_conditions")))
            if identity not in seen:
                seen.add(identity)
                unique.append(chapter)
        if len(unique) <= target:
            return unique
        groups = [[] for _ in range(target)]
        for index, chapter in enumerate(unique):
            groups[index * target // len(unique)].append(chapter)
        result = []
        for index, group in enumerate(groups):
            result.append({**group[0], "id": f"chapter_{index + 1}",
                "claim_ids": list(dict.fromkeys(cid for chapter in group for cid in chapter["claim_ids"])),
                "merged_from": [chapter["id"] for chapter in group],
                "subtopics": [{key: chapter.get(key, "") for key in
                    ("question", "purpose", "mechanism", "required_conditions", "optional_example", "claim_ids")}
                    for chapter in group],
                "bridge_out": group[-1].get("bridge_out", "") if index < target - 1 else ""})
        return result
    if len(chapters) >= target:
        return chapters[:target]
    result = []
    for index, chapter in enumerate(chapters):
        parts = target // len(chapters) + int(index < target % len(chapters))
        for part in range(parts):
            item = {**chapter, "id": f"chapter_{len(result) + 1}"}
            if parts > 1:
                suffix = (f" Part {part + 1}/{parts}: develop the same argument using the actual preceding dialogue; do not repeat the introduction."
                          if language == "en" else f" 第 {part + 1}/{parts} 段：依据真实前文继续展开同一论证，不重复开场。")
                item["purpose"] = chapter["purpose"] + suffix
            result.append(item)
    return result


def _ensure_plan_coverage(plan: dict[str, Any], visible: list[dict[str, Any]],
                          assignments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assign bounded core units locally when an outline omits a source topic."""
    plan = copy.deepcopy(plan)
    chapters = plan["chapters"]
    by_id = {c["id"]: c for c in visible}
    core = _source_round_robin(visible)[:2 * len(chapters)]
    assignments = assignments if isinstance(assignments, dict) else {}
    mapped, repaired = {}, []
    for chapter in chapters:
        chapter["required_unit_ids"] = []
    for unit in core:
        cid = unit["id"]
        proposed = assignments.get(cid)
        existing = next((i for i,c in enumerate(chapters) if cid in c["claim_ids"]), None)
        index = proposed - 1 if type(proposed) is int and 1 <= proposed <= len(chapters) else existing
        if index is None:
            def rank(i: int) -> tuple[float, float, int, int]:
                peers = [by_id[k] for k in chapters[i]["claim_ids"] if k in by_id and by_id[k]["source_id"] == unit["source_id"]]
                distance = abs(float(unit.get("ordinal", 0)) - sum(float(c.get("ordinal", 0)) for c in peers) / len(peers)) if peers else 0
                return (0 if peers else 1, distance, len(chapters[i]["claim_ids"]), i)
            index = min(range(len(chapters)), key=rank)
            repaired.append(cid)
        chapter = chapters[index]
        chapter["required_unit_ids"].append(cid)
        if cid not in chapter["claim_ids"]:
            chapter["claim_ids"].append(cid)
        mapped[cid] = chapter["id"]
    plan["coverage"] = {"provided_unit_ids": list(by_id), "required_unit_ids": [u["id"] for u in core],
                        "assignments": mapped, "locally_assigned_unit_ids": repaired,
                        "unassigned_unit_ids": [], "meaning": "Chapter assignment is not verified explanation coverage."}
    return plan


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
    if language == "en":
        prompt_prefix = f"""Plan a source-grounded two-host knowledge podcast in English. Return exactly {target} connected acts, building one central argument. Do not write dialogue or invent facts.
Alternate the lead host across acts. Use each claim only where relevant. The opening introduces the source and central question before details. Each act reaches a supported conclusion; the final act closes the central question without new facts.
Return JSON with episode_thesis and chapters. Every chapter has title, purpose, tension, lead_host (HOST_A or HOST_B), claim_ids (allowed C IDs), bridge_in, bridge_out. Bridges describe future intentions, not dialogue that already occurred.
Focus: {focus or 'Explain the central argument and its qualifications'}
Available claims:
"""
    if DELIVERY.get():
        prompt_prefix += "\nAlso return answer_path: how the central question reaches its answer. Each chapter includes new_information: the mechanism, source example or qualification it adds. Merge synonymous claims; do not allocate several acts to paraphrases. Preserve each author's distinct view. Do not invent a relation between unrelated sources. Each chapter also returns question, mechanism, required_conditions and optional_example, all grounded in its claim_ids. Keep these descriptions brief. Cover the main explanatory topics across the supplied passages: definitions, how it works, conditions and practical limits; do not crowd independent topics into a final catch-all chapter. The chapter bodies develop the argument; a separate opening and closing will be supplied.\n"
    required_units = [c["id"] for c in _source_round_robin(claims)[:2 * target]]
    if DELIVERY.get():
        prompt_prefix += "\nReturn assignments mapping EVERY core unit ID to its main chapter number (1-based): " + json.dumps(required_units) + ". Each assigned unit's mechanism and essential conditions must be explained there, not merely cited. Reuse in other chapters is background, not a new topic. Cover all topics in a supplied unit, including its definitions and practical limitations.\n"
    try:
        generated = await budgeted_chat(
            lambda budget: _segment_prompt_build(
                budget,
                language=language,
                prefix=prompt_prefix,
                items=claims,
                renderer=_render_claim_bundle,
                group_key=lambda claim: str(claim["source_id"]),
            ),
            json_mode=True,
            **({"response_schema": plan_schema(target, required_units)} if DELIVERY.get() else {}),
            max_tokens=structured_output_tokens(1800),
            minimum_output_tokens=384,
            temperature=0.15,
            trace=trace,
            stage="episode_plan",
        )
        parsed = _extract_json(generated.content)
        available = {claim["id"] for claim in generated.build.metadata["items"]}
        chapters = []
        for original_index, item in enumerate(_extract_array(generated.content, "chapters") or [], start=1):
            if not isinstance(item, dict):
                continue
            claim_ids = [str(value) for value in item.get("claim_ids") or [] if str(value) in available]
            title = str(item.get("title") or "").strip()
            purpose = str(item.get("purpose") or "").strip()
            if not title or not purpose or not claim_ids:
                continue
            chapters.append(
                {
                    "id": f"chapter_{original_index}",
                    "title": title[:120],
                    "purpose": purpose[:300],
                    "claim_ids": claim_ids[:10],
                    "bridge_in": str(item.get("bridge_in") or "")[:220],
                    "bridge_out": str(item.get("bridge_out") or "")[:220],
                    "lead_host": "HOST_B" if str(item.get("lead_host") or "").upper() == "HOST_B" else "HOST_A",
                    "tension": str(item.get("tension") or "")[:260],
                    "new_information": str(item.get("new_information") or purpose)[:400],
                    **{key: str(item.get(key) or "")[:400] for key in ("question", "mechanism", "required_conditions", "optional_example")},
                }
            )
        thesis = str(parsed.get("episode_thesis") or "").strip()
        if chapters and thesis:
            adjusted = len(chapters) != target
            if len(chapters) < target and not DELIVERY.get():
                used = {claim_id for chapter in chapters for claim_id in chapter["claim_ids"]}
                remaining = [claim for claim in generated.build.metadata["items"] if claim["id"] not in used]
                if remaining:
                    # An incomplete outline is not permission to repeat its
                    # opening for the whole episode. Keep it and cover unused
                    # evidence locally, without another planning request.
                    chapters.extend(_fallback_episode_plan(remaining, language, target - len(chapters))["chapters"])
                chapters = [{**chapter, "id": f"chapter_{index + 1}"} for index, chapter in enumerate(chapters)]
            expanded = _fit_episode_chapters(chapters, target, language)
            adjusted = adjusted or len(expanded) != len(chapters)
            expanded[-1]["bridge_out"] = ""
            plan = {"episode_thesis": thesis[:400], "answer_path": str(parsed.get("answer_path") or "")[:600], "chapters": expanded, "fallback": False,
                    "locally_adjusted": adjusted}
            if DELIVERY.get():
                assignments = parsed.get("assignments")
                positions = {original: i for i, chapter in enumerate(expanded, start=1)
                             for original in chapter.get("merged_from", [chapter["id"]])}
                remapped = {cid: positions.get(f"chapter_{number}") for cid, number in assignments.items()} if isinstance(assignments, dict) else {}
                plan = _ensure_plan_coverage(plan, generated.build.metadata["items"], remapped)
            return plan, adjusted
    except Exception:
        pass
    if trace:
        trace.mark_fallback()
    fallback = _fallback_episode_plan(claims, language, target)
    return (_ensure_plan_coverage(fallback, claims) if DELIVERY.get() else fallback), True


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
    return _spoken_question(str(turn.get("text") or ""))


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
    allow_style_degradation: bool = False,
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
            if allow_style_degradation:
                break
            continue
        supplied_speaker = _speaker(source.get("speaker"))
        raw_text = source.get("text")
        if not isinstance(raw_text, str):
            issues.append(f"第 {index + 1} 轮口播不是文本")
            if allow_style_degradation:
                break
            continue
        text = _normalize_text(raw_text)
        if not text:
            issues.append(f"第 {index + 1} 轮为空")
            if allow_style_degradation:
                break
            continue
        act = _coerce_dialogue_act(source.get("dialogue_act"), text, source.get("claim_ids"))
        raw_claim_ids = source.get("claim_ids") or []
        raw_claim_ids = [raw_claim_ids] if isinstance(raw_claim_ids, str) else raw_claim_ids
        claim_ids = list(dict.fromkeys(label for value in raw_claim_ids for label in re.findall(r"(?<![A-Z0-9])C\d+(?!\d)", str(value).upper()) if label in claims_by_id))
        expected_speaker = "HOST_B" if previous == "HOST_A" else "HOST_A"
        if supplied_speaker not in {"HOST_A", "HOST_B"}:
            issues.append(f"第 {index + 1} 轮说话人无效")
            if allow_style_degradation:
                break
            continue
        speaker = supplied_speaker if allow_style_degradation else expected_speaker
        if act not in ALLOWED_DIALOGUE_ACTS and allow_style_degradation:
            act = "explain"
        if act not in ALLOWED_DIALOGUE_ACTS:
            issues.append(f"第 {index + 1} 轮 dialogue_act 无效")
            continue
        if not allow_style_degradation and ((scene_kind == "intro" and act == "outro") or (scene_kind in {"chapter", "boundary_repair"} and act in {"intro", "outro"})):
            issues.append(f"第 {index + 1} 轮 dialogue_act 不适合 {scene_kind}")
            continue
        candidate_is_question = _is_question_turn({"text": text})
        if candidate_is_question and question_run >= 2:
            issues.append(f"第 {index + 1} 轮造成跨 Act 连续问句过多")
            if not allow_style_degradation:
                continue
        if candidate_is_question and question_count >= max_questions:
            issues.append(f"第 {index + 1} 轮使本 Act 问句比例超过 40%")
            if not allow_style_degradation:
                continue
        spoken_units = _spoken_unit_count(text, language)
        minimum_ok = len(text) >= 8 if language != "en" else spoken_units >= 4
        maximum_units = 240 if language != "en" else 90
        maximum_ok = spoken_units <= maximum_units
        if (not minimum_ok or not maximum_ok) and not allow_style_degradation:
            direction = "过短" if not minimum_ok else "过长"
            issues.append(f"第 {index + 1} 轮长度不合格（{direction}，口播单位 {spoken_units}）")
            continue
        if not text_matches_language(text, language) and not allow_style_degradation:
            issues.append(f"第 {index + 1} 轮语言不符合输出要求")
            continue
        if re.search(r"[!！]|[?？]{2,}|\.{3,}|…{2,}", text):
            issues.append(f"第 {index + 1} 轮包含会放大口播情绪的标点")
            if not allow_style_degradation:
                continue
            text = re.sub(r"[!！]+", "。" if language != "en" else ".", text)
            text = re.sub(r"[?？]{2,}", "？" if language != "en" else "?", text)
            text = re.sub(r"\.{3,}|…{2,}", "。" if language != "en" else ".", text)
        lowered = text.lower()
        if any(stem in lowered for stem in GENERIC_STEMS) and not allow_style_degradation:
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
        if not allow_style_degradation and not claim_ids and provisional_factual and default_claim_ids and index < len(default_claim_ids):
            default_claim_id = default_claim_ids[index]
            if default_claim_id in claims_by_id:
                claim_ids = [str(default_claim_id)]
                claim_id_source = "slot"
        factual = provisional_factual or bool(claim_ids)
        if factual and not claim_ids and allow_style_degradation:
            issues.append(f"第 {index + 1} 轮事实引用待核实，保留完整对话")
        if factual and not claim_ids and not allow_style_degradation:
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
        if not allow_style_degradation and claim_ids and not _numbers_supported(text, _claim_evidence_text(claim_ids, claims_by_id, cards_by_id)):
            issues.append(f"第 {index + 1} 轮包含资料不支持的数字")
            continue
        if not allow_style_degradation and _is_duplicate(text, existing_turns + accepted):
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
                "example_kind": source.get("example_kind") if source.get("example_kind") in {"source", "illustrative"} else "none",
                **({"source_chapter_id": source["source_chapter_id"]} if source.get("source_chapter_id") else {}),
                **({"exchange_id": source["exchange_id"], "exchange_start": bool(source.get("exchange_start"))}
                   if source.get("exchange_id") else {}),
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
    return structured_output_tokens(min(10_000, max(6_000, round(visible + 4_500))))


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
    output_boost: float = 1.0,
) -> SceneDraftResult:
    language_rule = "只输出自然的简体中文口语" if language != "en" else "Use natural spoken English only"
    start_speaker = "HOST_B" if memory.last_speaker == "HOST_A" else "HOST_A"
    memory_payload = memory.prompt_payload(profile["recent_turns"])
    if profile.get("chapter_replacement"):
        memory_payload["previously_cited_not_necessarily_explained"] = memory_payload.pop("covered_claim_ids")
    memory_json = json.dumps(memory_payload, ensure_ascii=False)
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
生成恰好 {target} 轮，从 {start_speaker} 开始并严格交替。{question_rule}，不得连续出现超过两个问句；使用 Q act_code 的轮次必须写成自然问句并以问号结尾。长短轮次要有变化，但每一轮都要完成一个实质推进。{duration_rule} {_slot_plan_instruction(slot_plan, language)} 每个深槽的 claim_ids 至少填一个允许的 C 编号；短槽只有在 Q/B/A/I/O 且完全不陈述事实时才允许空数组。围绕本 Act 的“张力”组织论证主线，把前提、机制和含义逐步讲清；张力只用于内部规划，不得照读或转述其措辞。涉及尚未确认的内容时，用一句自然口语限定带过（如“这里原文没明说”“这点还差一点证据”），把不确定体现在论证结构里，不要念成方法论旁白；口播中禁止使用“不能推出、只支持、边界、门槛、范围、回扣、压实、下一层”一类审稿术语。对听者的显性防误读提醒（“别把它读成/夸成/说成 X”“A 不等于 B”“这不意味着…”）每个 Act 至多一处，其余限定直接并入叙述——说“原文给的是 A”，而不是反复敲打“A 不等于 B”。事实、数字、案例、判断必须被所填 claim_ids 直接支持；禁止用“唯一、必然、完全”等绝对措辞放大原主张，也不能从个人行动擅自推演到社会影响。不得使用资料外常识、轶事或类比，不得念出编号，不得重复“所以你的意思是”一类模板句。
只输出一个 JSON 对象，键名为 turns；turns 的每一项必须是四元素数组，依次为 speaker、act_code、text、claim_ids。speaker 只能为 A/B；act_code 只能为 I/F/B/Q/A/X/E/M/C/S/O；claim_ids 只能从下方允许列表逐字复制，不能省略事实轮的编号。不要输出示例、统计、解释或额外字段。
剧集记忆：{memory_json}
当前部分：{chapter.get('title')}；目的：{chapter.get('purpose')}；本 Act 的内部张力（仅用于组织论证主线，不得照读或转述其措辞）：{chapter.get('tension')}；承接：{chapter.get('bridge_in')}；后续钩子：{chapter.get('bridge_out')}。
{'这不是首个 Act：第一轮必须先明确回应剧集记忆中上个钩子的未决关系，再进入新角度；不得直接跳到新类比。' if memory.last_turns else '这是全篇开场，没有任何之前的对话；直接介绍主题，禁止说“我们刚才讨论过”、as we discussed earlier 或使用没有前文的指代。'}
{f'上次草稿问题，必须修复：{feedback}' if feedback else ''}
允许使用的主张：
"""
    exchange_rule = (
        "Write connected exchanges: a question must be directly answered or explicitly clarified in the very next turn before any new question. Both hosts may explain. Keep an answer with its question on the same claim; choose another allowed claim when a slot's suggested claim is irrelevant. End this act with a complete supported statement, not an unanswered question."
        if language == "en" else
        "按完整交流编写：提出问题后，紧接的下一轮必须先直接回答或明确澄清，之后才能提出新问题。双方都可以解释。同一问答使用相关的同一主张；槽位建议的主张不相关时改用允许列表内真正支持回答的主张。每个 Act 以完整且有依据的陈述收束，不留下未回答的问题。"
    )
    if language == "en":
        opening = ("Continue from the ACTUAL recent dialogue. Answer any real unresolved closing question first; planned bridges are only future intentions."
                   if memory.last_turns else "This is the episode opening. Introduce the source and central question. There is no earlier conversation; do not say 'as we discussed' or use missing antecedents.")
        prompt_prefix = f"""Write a source-grounded two-host knowledge podcast in natural spoken English. All spoken text must be English even when evidence is in another language.
{_scene_instruction(scene_kind, language)} {_delivery_instruction(language)}
Generate exactly {target} turns, starting with {start_speaker}, strictly alternating speakers. The lead host for this act is {chapter.get('lead_host') or 'HOST_A'}; both hosts contribute substantive explanations and judgments.
{exchange_rule}
Use no more than {max(1, math.floor(target * .35))} questions in this act, regardless of act code. Most turns should explain rather than ask.
{duration_rule} {_slot_plan_instruction(slot_plan, language)}
Use varied sentence lengths. Explain premises, mechanisms and implications, preserving qualifications and uncertainty. Do not add outside facts, numbers, anecdotes or analogies. Do not read claim IDs or planning language aloud. Avoid repetitive reminders about what cannot be concluded.
Return ONE JSON object with key turns. Each turn MUST be a four-element array: [speaker, act_code, text, claim_ids]. Speaker MUST be A or B. act_code MUST be I/F/B/Q/A/X/E/M/C/S/O. claim_ids MUST be an array of allowed C IDs. Fact-bearing turns need directly supporting claim IDs; purely conversational questions or bridges may use an empty array. Q turns end with a question mark. Do not nest arrays beyond claim_ids, output statistics, or add other fields.
Actual episode memory: {memory_json}
Act title: {chapter.get('title')}; purpose: {chapter.get('purpose')}; organizing tension (never read aloud): {chapter.get('tension')}; planned next topic: {chapter.get('bridge_out')}.
{opening}
Repair feedback: {feedback or 'None'}
Allowed claims:
"""
    else:
        prompt_prefix = exchange_rule + "\n" + prompt_prefix
    if profile.get("allow_partial"):
        exchange_count = max(1, math.ceil(target / 3))
        units = int((duration_budget or {}).get("minimum_units") or target * (55 if language == "en" else 110))
        per_exchange = math.ceil(units / exchange_count)
        if language == "en":
            prompt_prefix = f"""Write {exchange_count} self-contained exchanges for a source-grounded two-host podcast in natural English.
Each exchange has 2–4 alternating turns. Both hosts explain; at most one question per exchange, directly answered by the next turn using the same relevant evidence. Answer the exact mechanism asked about, not just its purpose or a different mechanism. An explanation and its qualification belong together.
Start each exchange by naming its topic so it remains understandable without the previous exchange. Do not start with pronouns or assume earlier dialogue. End each exchange with a complete supported statement. Do not repeat covered explanations, introductions or conclusions. No invented facts, analogies, numbers or spoken citation IDs.
Write about {per_exchange} spoken words per exchange, distributed across the turns. Develop the source's mechanism, example and conditions rather than repeating the claim. If evidence cannot sustain this length, keep the complete shorter exchange.
Act: {chapter.get('title')}; purpose: {chapter.get('purpose')}. Previously discussed: {memory_json}.
Return JSON {{"exchanges":[{{"turns":[["A","X","spoken text",["C1"]],["B","X","spoken text",["C1"]]]}}]}}. The structure is an example, not text to repeat. Use A/B speakers, allowed act codes I/F/B/Q/A/X/E/M/C/S/O, and only actual allowed C IDs supporting each factual turn. Start with {start_speaker}. Keep original conditions, attribution and uncertainty.
Allowed evidence:
"""
        else:
            prompt_prefix = f"""用自然简体中文写 {exchange_count} 个可独立理解的双人播客交流单元。每单元2至4轮，两人交替且都能解释，每单元最多一个问句，紧接下一轮必须回答所问的具体机制，不能拿目的或另一机制代替答案。问答依据同一组相关主张，解释与必要限定一起讲清。
每个单元开头点明具体主题，不用缺少前文的“这、因此、刚才”等指代；结尾为完整且有依据的陈述。不要重复讲过的介绍和结论，不增加外部事实、数字、类比，不念引用编号。
每单元约 {per_exchange} 个汉字的口播，分配到各轮，用原文机制、例证与条件展开，不反复重述同一句主张。证据不足以支撑长度时保留完整短版。
本章：{chapter.get('title')}；目的：{chapter.get('purpose')}。已经讨论过：{memory_json}。
返回JSON {{"exchanges":[{{"turns":[["A","X","口播文本",["C1"]],["B","X","口播文本",["C1"]]]}}]}}。这是格式示意，不要照抄内容。说话人A/B，act_code为I/F/B/Q/A/X/E/M/C/S/O，事实轮只填真正支持内容的允许C编号，从{start_speaker}开始。保留原文的前提、作者及不确定性。
允许证据：
"""
    role = profile.get("complete_role")
    if role:
        if language == "en":
            task = ("This is a COMPLETE short episode, not its first chapter. Use three or more exchanges: introduce the source and central question, explain the answer with evidence, then close that SAME question. The last exchange is the episode conclusion; its last turn uses O. No teasers or promises of later explanation."
                    if role == "core" else
                    "This is OPTIONAL additional depth inserted before an existing conclusion. Explain a new mechanism, example or qualification, not the core explanation again. It must not change the existing conclusion or introduce an unanswered question. Do not say this is the first or final chapter.")
        else:
            task = ("本次要写的是一集完整短播客，不是第一章。至少三个交流单元：介绍资料和核心问题、依据原文解释答案、最后回答同一个核心问题并结束节目。最后单元是整集结论，最后一轮使用O。不要预告稍后再解释，不留下悬念。"
                    if role == "core" else
                    "本次是插入已有结论之前的可选深度展开：提供尚未解释的机制、案例或限定，不复述核心介绍。不能改变已有结论，不留下未答问题，不自称第一章或最后一章。")
        prompt_prefix = task + "\n" + prompt_prefix
    if role:
        if profile.get("chapter_replacement"):
            task = "Develop the current chapter in its narrative position. It replaces a short placeholder that the listener has NOT heard."
            per_exchange = max(30, round(float((duration_budget or {}).get("minimum_units") or 300) / max(2, math.ceil(target / 3))))
        contract = scene_schema(role, [c["id"] for c in claims], profile.get("core_chapter_ids"), max(2, math.ceil(target / 3)))
        shape = ('{"opening":[turn,turn],"chapter_bodies":{' + ','.join(json.dumps(key) + ':[turn,turn]' for key in profile["core_chapter_ids"]) + '},"closing":[turn,turn]}' if profile.get("core_chapter_ids") else
                 '{"opening":[turn,turn],"body":[turn,turn],"closing":[turn,turn]}' if role == "core"
                 else '{"exchanges":[{"turns":[turn,turn]}]}')
        prompt_prefix = f"""Write a source-grounded two-host learning podcast in {'natural English' if language == 'en' else '自然简体中文口语'}.
{task}
Use this JSON shape: {shape}. Each turn is an object with speaker (A/B), act_code (I/F/B/Q/A/X/E/M/C/S/O), text, claim_ids (array of allowed C IDs), example_kind (none/source/illustrative). Each group has 2–4 alternating turns. Start with {start_speaker}. No extra fields.
Keep each question together with its direct answer. End each group with a complete supported statement, not a question. The core closing must answer the opening and end with act_code O.
Aim for {max(2, math.ceil(target / 3)) if role != "core" else 3} groups. Write approximately {per_exchange} {'words' if language == 'en' else 'Chinese characters'} per group, but prefer a complete shorter explanation over padding.
Build one continuous explanation: use the actual prior dialogue to connect ideas; do not restart the source introduction at every group. Both hosts contribute. Preserve qualifications, probability, and attribution to the source author. A hypothesis or philosophical position is not established fact. Prefer source examples. You may use a simple hypothetical analogy explicitly introduced as "打个比方" or "imagine"; mark example_kind illustrative. It illustrates the supported mechanism, not new evidence. Do not invent historical facts, figures or scientific conclusions. Preserve the limits of the analogy. Avoid repeated agreement and paraphrase: the other host probes a mechanism, challenges an inference, or applies it. Do not read IDs aloud.
Chapter explanation contract: {json.dumps({key: chapter.get(key, "") for key in ("question", "mechanism", "required_conditions", "optional_example", "required_unit_ids", "subtopics")}, ensure_ascii=False)}
The contract is a reading guide, not evidence. Verify it against the original, retaining probability and attribution even in a short answer.
Episode question: {memory.thesis}
Part: {chapter.get('title')}; purpose: {chapter.get('purpose')}; new information: {chapter.get('new_information') or chapter.get('purpose')}.
Already explained (do not paraphrase again): {memory_json}
Specific structural errors to fix: {feedback or 'None'}. Return the complete corrected structure; retain valid explanation and conclusion.
Allowed evidence, author and conditions:
"""
    if profile.get("core_chapter_ids"):
        prompt_prefix += "\nCORE MAP CONTRACT: Return opening (2 turns), chapter_bodies (object with exactly these keys, 2 turns each), closing (2 turns). Keys in narrative order: " + json.dumps(profile["core_chapter_ids"]) + ". Each body explains its own chapter question, mechanism and required conditions using its full evidence bundle; Allocate the available output to a complete mechanism and its necessary conditions in each body before optional examples. Do not discard qualifications to meet an arbitrary body length. The opening establishes the question without summarizing the answers. Closing answers that question. Each body ends with a statement, so it remains usable if deeper writing fails.\nChapter purposes and allowed claims: " + json.dumps(profile["core_chapter_plan"], ensure_ascii=False)
    if profile.get("chapter_replacement") and not profile.get("core_chapter_ids"):
        prompt_prefix += "\nThis is the actual body of this chapter, replacing its compact placeholder, not an appendix to a finished explanation. Explain the mechanism step by step, use an example when useful, and keep necessary qualifications. Only the supplied preceding dialogue has already been spoken. No repeated source introduction, no final episode signoff. Complete the current reasoning. Part assignment: " + str(profile.get("part_assignment", ""))
        prompt_prefix += "\nPreserve and develop the mechanisms and conditions in this compact chapter (the listener has NOT heard it): " + json.dumps(profile.get("compact_chapter", []), ensure_ascii=False)
    output_limit = int(profile.get("stage_output_tokens") or round(_act_output_tokens(duration_budget, target, language) * output_boost))
    def build_scene(budget: PromptBudget) -> PromptBuild:
        built = _segment_prompt_build(
            replace(budget, input_tokens=min(budget.input_tokens, output_limit * 3)) if role and not (profile.get("chapter_replacement") or profile.get("core_chapter_ids")) else budget,
            language=language,
            prefix=prompt_prefix,
            items=claims,
            renderer=lambda claim: _render_claim_bundle(claim) if claim.get("evidence_bundle") else f"[{claim['id']}|{','.join(claim['evidence_ids'])}] {claim['text']}\nAuthor/source: {claim.get('filename', '')}; type: {claim.get('statement_kind', 'unspecified')}; conditions: {claim.get('qualification', '')}\n" + _claim_evidence_text([claim["id"]], {claim["id"]: claim}, cards_by_id),
            group_key=lambda claim: str(claim["source_id"]),
        )
        visible = {claim["id"] for claim in built.metadata["items"]}
        required_ids = set(profile.get("preserve_claim_ids") or [])
        if required_ids - visible:
            required = [claim for claim in claims if claim["id"] in required_ids]
            evidence = (_shared_evidence_text(required) if required and all(c.get("evidence_bundle") for c in required)
                        else "\n".join(_render_claim_bundle(c) + _claim_evidence_text([c["id"]], {c["id"]: c}, cards_by_id) for c in required))
            built = PromptBuild([*built.messages[:-1], {"role": "user", "content": prompt_prefix + evidence}],
                                len(claims), len(required), 0, {"items": required, "preserve_evidence": True})
            visible = {claim["id"] for claim in required}
        if role and claims and not visible:
            raise PodcastQualityError("剩余预算无法容纳本章原文，保留已有完整短稿", {"stage": "evidence_budget"})
        def restrict(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: [cid for cid in item if cid in visible] if key in {"claim_ids", "required_unit_ids"}
                        else restrict(item) for key, item in value.items()}
            if isinstance(value, list):
                return [restrict(item) for item in value if not isinstance(item, dict)
                        or not item.get("claim_ids") or visible.intersection(item["claim_ids"])]
            return value
        guides = [profile.get("core_chapter_plan"), {key: chapter.get(key, "") for key in
                  ("question", "mechanism", "required_conditions", "optional_example", "required_unit_ids", "subtopics")}]
        for guide in guides:
            if guide:
                original = json.dumps(guide, ensure_ascii=False)
                filtered = json.dumps(restrict(guide), ensure_ascii=False)
                for message in built.messages:
                    if isinstance(message.get("content"), str):
                        message["content"] = message["content"].replace(original, filtered)
        if role:
            contract.clear()
            contract.update(scene_schema(role, sorted(visible), profile.get("core_chapter_ids"), max(2, math.ceil(target / 3))))
        return built

    generated = await budgeted_chat(
        build_scene,
        json_mode=True,
        **({"response_schema": contract} if role else {}),
        timeout=420,
        max_tokens=output_limit,
        minimum_output_tokens=min(output_limit, 1200) if role else min(3600, max(700, target * 130)),
        temperature=0.45,
        trace=trace,
        stage="act_draft" if not repair_feedback else "targeted_repair",
    )
    available_claims = {claim["id"]: claim for claim in generated.build.metadata["items"]}
    content = generated.content
    ids = profile.get("core_chapter_ids") or []
    if len(ids) == 1:
        parsed = _extract_json(content)
        if isinstance(parsed, dict) and "chapter_bodies" not in parsed and all(isinstance(parsed.get(key), list) and parsed[key] for key in ("opening", "body", "closing")):
            parsed["chapter_bodies"] = {ids[0]: parsed.pop("body")}
            content = json.dumps(parsed, ensure_ascii=False)
    raw_turns = _extract_turns(content)
    finish_reason = getattr(generated, "finish_reason", None)
    if profile.get("core_chapter_ids") and raw_turns and {t.get("source_chapter_id") for t in raw_turns} != {"opening", "closing", *profile["core_chapter_ids"]}:
        return SceneDraftResult([], ["Core map is missing or adds chapter IDs; return every requested section."], finish_reason)
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
        expected_count=max(target, len(raw_turns)) if profile.get("allow_partial") else target,
        scene_kind=scene_kind,
        default_claim_ids=[item["default_claim_id"] for item in slot_plan],
        allow_style_degradation=bool(profile.get("allow_partial")),
    )
    if profile.get("allow_partial") and any(t.get("exchange_id") for t in raw_turns):
        counts = Counter(t.get("exchange_id") for t in raw_turns)
        valid_counts = Counter(t.get("exchange_id") for t in validated)
        validated = [t for t in validated if valid_counts[t.get("exchange_id")] == counts[t.get("exchange_id")]]
        for turn in validated:
            if turn.get("exchange_id"):
                turn["exchange_id"] = f"{chapter['id']}/{turn['exchange_id']}"
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
            language=language,
            prefix=prompt_prefix,
            items=claims,
            renderer=lambda claim: _render_claim_bundle(claim) if claim.get("evidence_bundle") else f"[{claim['id']}|{','.join(claim['evidence_ids'])}] {claim['text']}",
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
    allow_partial = bool(generation_state and generation_state.allow_partial)
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
        if allow_partial and not claim_recovery():
            raise PodcastQualityError("场景连接失败，任务恢复次数已用尽")
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
        and (not allow_partial or claim_recovery())
    ):
        generation_state.empty_response_retry_used = True
        try:
            # 空响应多半是推理耗尽输出预算；续写需要非空草稿，这里只能以翻倍预算整体重试
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
                repair_feedback=deterministic_issues + (["Preserve this valid dialogue: " + json.dumps(draft, ensure_ascii=False)] if draft else []),
                output_boost=2.0,
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
        and not allow_partial
    ):
        generation_state.continuation_used = True
        continuation_used_here = True
        try:
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
        except Exception as exc:
            if not allow_partial:
                raise
            continued = SceneDraftResult([], [f"续写未完成：{type(exc).__name__}: {exc}"], None)
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
        if completed_count < minimum_after_continuation and not allow_partial:
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
    if draft and scene_kind in {"act", "boundary_repair"} and len(draft) < minimum_complete and not allow_partial:
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
            language=language,
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
            language=language,
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
    # Planned bridges are intentions, not statements that have been spoken.
    # Carry only a real unanswered closing question into the next act.
    memory.open_hook = turns[-1]["text"] if turns and _is_question_turn(turns[-1]) else ""
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
    prompt_prefix = f"""你是深度播客主编。全稿已执行时长、引用、说话人平衡、问题密度和重复的逐轮检查，部分指标可能未通过；下面是每个 Act 的开头、中点和结尾抽样。判断对应事实是否受 claim 支持、论证是否逐步深入、Act 之间是否自然、双方是否都贡献实质内容、是否有套话或过度绝对的结论。额外检查两点：(a) 是否把尚未确认的猜测说成已建立结论、是否把支持性论据过度放大——资料忠实度防线不变；(b) 口播是否把审稿术语（如“不能推出、只支持、边界、门槛、范围、回扣、压实、下一层”）直接说出口——区分确定与不确定应该是一句自然口语限定，而不是方法论旁白；(c) 防误读提醒是否密集重复——“别把它读成/夸成/说成”“不等于/不意味着”一类句式在同一 Act 出现多处，会让节目听起来像持续自我审查；发现任一点即记入 blocking_issues。只输出 JSON：{{"verdict":"pass|fail","scores":{{"grounding":5,"coherence":5,"depth":5,"roles":5,"repetition":5,"completeness":5}},"invalid_boundaries":[1],"blocking_issues":["会阻断发布的具体问题"],"notes":["可选润色建议"]}}。5=优秀、4=可发布、1=严重失败。blocking_issues 只能放会使某项低于 4 分或与 pass 矛盾的发布阻断项；4 分范围内的改善建议必须放 notes，不得放 blocking_issues。只有局限在 Act 边界附近、最多可改 6 轮的问题才放入 invalid_boundaries；需要重写整集时给 fail 但保持该数组为空。语言={language}。核心命题：{thesis}
Act 抽样（保留原始轮次索引）：
{transcript}
抽样所用主张：
"""
    prompt_prefix += "\n额外返回 unsupported_turns 数组：仅列出事实无法被主张支持的原始轮次零基索引（可以是 0），不要把风格或连贯性问题放入该数组。没有这类事实问题时返回 []。\n"
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
        unsupported = parsed.get("unsupported_turns") or []
        unsupported = sorted({value for value in unsupported if type(value) is int and value in sampled_indexes}) if isinstance(unsupported, list) else []
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
                "unsupported_turns": unsupported,
                "scores": {},
                "invalid_boundaries": invalid,
                "issues": ["整集审校输出达到 token 上限，结果不完整"],
            }
        if verdict not in {"pass", "fail"} or not raw_scores_valid:
            return {
                "passed": False,
                "verdict": verdict or "invalid",
                "unsupported_turns": unsupported,
                "scores": {},
                "invalid_boundaries": invalid,
                "issues": ["整集审校未返回完整 verdict 与六项分数"],
            }
        normalized = {name: int(scores[name]) for name in names}
        return {
            "passed": verdict == "pass" and not invalid and not blocking_issues and not unsupported and min(normalized.values(), default=0) >= 4,
            "verdict": verdict,
            "unsupported_turns": unsupported,
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


async def _audit_grounded_subset(turns: list[dict[str, Any]], cards_by_id: dict[str, dict[str, Any]], trace: ContextUsage) -> dict[str, Any]:
    """One focused evidence review when an episode score identifies no bad turns."""
    trace.request_limit = max(trace.request_limit or 0, trace.requests + 1)
    if not current():
        trace.total_token_limit = min(45_000, (trace.total_token_limit or 0) + 4000)
    remaining = trace.total_token_limit - trace.accounted_tokens
    if remaining < 800:
        return {"accepted_indexes": [], "issues": ["逐轮事实复核预算不足"]}
    items = [{"index": index, "speaker": turn["speaker"], "text": turn["text"], "evidence": [cards_by_id[label]["content"] for label in turn.get("citation_ids", []) if label in cards_by_id]} for index, turn in enumerate(turns)]
    prefix = """逐轮核对口播与所附原文证据，只检查事实，不评价风格、深度、问句比例、衔接或时长。保持否定词、概率、数值和条件的原意；将概率小夸大为不可能、或把相关性说成确定因果的轮次不能接受。没有事实断言的自然提问和过渡可以接受。只返回明确有证据支持的轮次原始 index；未显示或证据不完整的轮次不要接受。仅输出 JSON：{"accepted_indexes":[0,1],"issues":["被拒绝的索引与事实原因"]}。
逐轮原文与口播：
"""

    def build(budget: PromptBudget) -> PromptBuild:
        available = max(1, remaining - budget.output_tokens - 32)
        return _segment_prompt_build(replace(budget, input_tokens=min(budget.input_tokens, available)), prefix=prefix, items=items, renderer=lambda item: json.dumps(item, ensure_ascii=False))

    try:
        generated = await budgeted_chat(build, json_mode=True, max_tokens=min(structured_output_tokens(800), remaining // 3), minimum_output_tokens=256, trace=trace, stage="grounding_recovery")
        parsed = _extract_json(generated.content)
        shown = generated.build.metadata["items"]
        if generated.build.truncated_segments:
            shown = shown[:-1]
        indexes = {item["index"] for item in shown}
        accepted = parsed.get("accepted_indexes")
        accepted = sorted({index for index in accepted if type(index) is int and index in indexes}) if isinstance(accepted, list) else []
        return {"accepted_indexes": accepted, "issues": parsed.get("issues") or []}
    except (ProviderError, RuntimeError, ValueError) as exc:
        return {"accepted_indexes": [], "issues": [str(exc)]}


async def repair_measured_duration(
    generated: dict[str, Any], actual_seconds: float, trace: ContextUsage,
) -> list[dict[str, Any]]:
    """One evidence-bound edit using measured speech rate; preserve turn identities."""
    state = current()
    if not state or actual_seconds <= 0:
        raise ValueError("实际时长恢复缺少固定资料快照")
    turns = generated["turns"]
    language = generated["language"]
    target_seconds = float(generated["duration"]["target_minutes"]) * 60
    units = sum(_spoken_unit_count(turn["text"], language) for turn in turns)
    change = round(units * abs(target_seconds / actual_seconds - 1))
    planner = _duration_expansion_plan if actual_seconds < target_seconds else _duration_compression_plan
    # Partial scripts can label a factual answer "intro" or a question "frame".
    # Plan by actual question form and attached evidence, not those style labels.
    planning_turns = [{**turn, "dialogue_act": "question" if _is_question_turn(turn) else "explain"}
                      for turn in turns]
    plan = planner(planning_turns, generated["chapters"], change, language)
    if not plan:
        raise ValueError("实际时长偏差超过本轮可安全调整的范围")
    rows = {row["id"]: row for row in state.rows}
    citations = {citation["id"]: citation for citation in generated["citations"]}
    items = []
    for item in plan:
        turn = turns[item["index"]]
        evidence_rows = [rows[citations[label]["chunk_id"]] for label in turn.get("citation_ids", [])
                         if label in citations and citations[label]["chunk_id"] in rows]
        evidence = [row["content"] for row in evidence_rows]
        if not evidence:
            raise ValueError("待调整口播缺少原文证据")
        mark_selected(evidence_rows)
        items.append({**item, "text": turn["text"], "evidence": evidence})
    prefix = (
        ("依据所附原文扩写每段口播，解释原文已有的机制、前提或含义，不要仅替换同义词。" if actual_seconds < target_seconds
         else "依据所附原文压缩每段口播，删去重复修饰和绕行表达。")
        + "保留前提、限定、否定、陈述者及结论方向。"
        "不增加外部事实、数字、类比、重复总结或问句。每项长度在 minimum_units 与 maximum_units 之间，"
        + ("单位为英文单词。" if language == "en" else "单位为中文等价字符。")
        + '只输出 JSON {"replacements":[[0,"修改后的完整口播"]]}，返回每项原 index。\n'
    )
    result = await budgeted_chat(
        lambda budget: _segment_prompt_build(budget, language=language, prefix=prefix, items=items,
            renderer=lambda item: json.dumps(item, ensure_ascii=False)),
        json_mode=True, max_tokens=structured_output_tokens(sum(item["maximum_units"] for item in items) * 2),
        minimum_output_tokens=512, trace=trace, stage="measured_duration_repair",
    )
    if result.build.truncated_segments or result.build.included_segments != len(items):
        raise ValueError("本轮预算无法容纳完整时长修复证据")
    raw = _extract_array(result.content, "replacements") or []
    replacements: dict[int, str] = {}
    for value in raw:
        index, text = (value if isinstance(value, list) and len(value) == 2 else
                       (value.get("index"), value.get("text")) if isinstance(value, dict) else (None, None))
        if (type(index) is int or isinstance(index, str) and index.isdigit()) and isinstance(text, str):
            replacements[int(index)] = _normalize_text(text)
    if not replacements or not set(replacements) <= {item["index"] for item in items}:
        raise ValueError("时长修复没有返回可用的计划轮次")
    revised = [dict(turn) for turn in turns]
    pairs = []
    pair_indexes = []
    for item in items:
        if item["index"] not in replacements:
            continue
        text = replacements[item["index"]]
        length = _spoken_unit_count(text, language)
        length_ok = (item["current_units"] < length <= item["maximum_units"] * 1.3 if actual_seconds < target_seconds
                     else item["safe_minimum_units"] <= length < item["current_units"])
        if (not text_matches_language(text, language)
                or not length_ok
                or not _numbers_supported(text, "\n".join(item["evidence"]))
                or _is_duplicate(text, [turn for index, turn in enumerate(revised) if index != item["index"]])):
            continue
        revised[item["index"]]["text"] = text
        pairs.append({"answer": text, "support_quote": "\n".join(item["evidence"])})
        pair_indexes.append(item["index"])
    predicted_seconds = actual_seconds * sum(_spoken_unit_count(turn["text"], language) for turn in revised) / max(units, 1)
    if not target_seconds * 0.85 <= predicted_seconds <= target_seconds * 1.2:
        raise ValueError("本轮修改按实测语速仍无法达到整集时长范围")
    invalid = await _critic_grounded_pairs(pairs, language, trace, strict=True)
    for index in invalid:
        if 0 <= index < len(pair_indexes):
            original_index = pair_indexes[index]
            revised[original_index] = dict(turns[original_index])
    predicted_seconds = actual_seconds * sum(_spoken_unit_count(turn["text"], language) for turn in revised) / max(units, 1)
    if not target_seconds * 0.85 <= predicted_seconds <= target_seconds * 1.2:
        raise ValueError("事实核验后可用的修改不足以达到整集时长范围")
    return revised


def _complete_core_parts(turns: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the complete closing exchange intact when adding optional depth."""
    if not turns:
        return [], []
    last_id = turns[-1].get("exchange_id")
    split = len(turns) - 2
    if last_id:
        split = next(i for i, turn in enumerate(turns) if turn.get("exchange_id") == last_id)
    return turns[:max(0, split)], turns[max(0, split):]


def _assemble_complete_episode(core: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    body, closing = _complete_core_parts(core)
    turns: list[dict[str, Any]] = []
    chapters = []
    for identifier, title, values, optional in [("core_body", "核心解释 / Core", body, False),
            *[(b["id"], b["title"], b["turns"], True) for b in blocks],
            ("core_closing", "结论 / Conclusion", closing, False)]:
        if not values:
            continue
        start = len(turns)
        turns.extend(copy.deepcopy(values))
        chapters.append({"id": identifier, "title": title, "turn_start": start, "turn_end": len(turns)-1,
                         "optional": optional, "claim_ids": list(dict.fromkeys(c for t in values for c in t.get("claim_ids", [])))})
    return turns, chapters


def _drop_repeated_exchanges(turns: list[dict[str, Any]], prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop only complete repeated groups, retaining new groups from the same draft."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, turn in enumerate(turns):
        groups.setdefault(str(turn.get("exchange_id") or f"legacy_{index}"), []).append(turn)
    kept: list[dict[str, Any]] = []
    history = [_review_text(t["text"]) for t in prior]
    for group in groups.values():
        values = [_review_text(t["text"]) for t in group]
        repeated = len(group) >= 2 and any(history[i:i + len(values)] == values for i in range(len(history) - len(values) + 1))
        if not repeated:
            kept.extend(group)
            history.extend(values)
    return kept


def _assemble_chapter_versions(core: list[dict[str, Any]], plan: dict[str, Any],
                               replacements: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace complete chapter bodies in place; compact and deep prose never stack."""
    turns, chapters = [], []
    for key, title in [("opening", "Opening"), *[(c["id"], c["title"]) for c in plan["chapters"]], ("closing", "Conclusion")]:
        fallback = [t for t in core if t.get("source_chapter_id") == key]
        values = copy.deepcopy(replacements.get(key, fallback))
        start = len(turns)
        for i, turn in enumerate(values):
            turn.update(exchange_id=f"{'optional' if key in replacements else 'core'}/{key}", exchange_start=i == 0)
        turns.extend(values)
        chapters.append({"id": key, "title": title, "turn_start": start, "turn_end": len(turns)-1,
                         "optional": key in replacements, "development": "expanded" if key in replacements else "compact",
                         "claim_ids": list(dict.fromkeys(c for t in values for c in t.get("claim_ids", []))),
                         "fallback_turns": copy.deepcopy(fallback)})
    return turns, chapters


def _restore_reviewed_chapters(turns: list[dict[str, Any]], chapters: list[dict[str, Any]],
                               audit: dict[str, Any], warnings: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    breaks = set(audit.get("breaks") or []) | set(_local_dialogue_breaks(turns))
    if type(audit.get("broken_at")) is int:
        breaks.add(audit["broken_at"])
    duplicates = {v.get("exchange_id") for v in audit.get("duplicates", [])}
    restore = {c["id"] for c in chapters if c.get("development") == "expanded" and
               (any(c["turn_start"] <= i <= c["turn_end"] + 1 for i in breaks if type(i) is int)
                or f"optional/{c['id']}" in duplicates)}
    if not restore:
        return turns, chapters, "draft_only" if breaks else "full"
    result, updated, mapping = [], [], {}
    for chapter in chapters:
        start = len(result)
        replaced = chapter["id"] in restore
        values = copy.deepcopy(chapter["fallback_turns"] if replaced else turns[chapter["turn_start"]:chapter["turn_end"]+1])
        for j, turn in enumerate(values):
            if replaced:
                turn.update(exchange_id=f"core/{chapter['id']}", exchange_start=j == 0)
            else:
                mapping[chapter["turn_start"]+j] = start+j
        result.extend(values)
        updated.append({**chapter, "turn_start": start, "turn_end": len(result)-1,
                        "development": "restored" if replaced else chapter["development"],
                        "optional": False if replaced else chapter["optional"],
                        "claim_ids": list(dict.fromkeys(c for t in values for c in t.get("claim_ids", [])))})
    # New compact text was not part of the final audit. Never inherit its verdict.
    adjacent = {i for i in mapping if i+1 in mapping and mapping[i+1] == mapping[i]+1}
    for key in ("reviewed_indexes", "requested_facts", "unsupported_turns"):
        audit[key] = [mapping[i] for i in audit.get(key, []) if i in mapping]
    for key in ("requested_transitions", "reviewed_transitions"):
        audit[key] = [mapping[i] for i in audit.get(key, []) if type(i) is int and i in adjacent]
    audit["facts"] = [{**f, "index": mapping[f["index"]]} for f in audit.get("facts", []) if f.get("index") in mapping]
    audit["transition_checks"] = [{**f, "index": mapping[f["index"]]} for f in audit.get("transition_checks", []) if f.get("index") in adjacent]
    audit.update(passed=False, restored_chapters=sorted(restore), original_breaks=sorted(breaks), breaks=[], broken_at=None,
                 closure={**audit.get("closure", {}), "verdict": "uncertain", "reason": "Chapter fallback changed the audited context."})
    warnings.append({"code": "chapter_restored", "stage": "script", "message": "部分深入章节未通过衔接检查，已原位恢复短章；恢复后的衔接尚未核实，不增加审校调用。"})
    return result, updated, "draft_only" if _local_dialogue_breaks(result) or any(i in mapping and i-1 in mapping for i in breaks if type(i) is int) else "partial"


async def _generate_chapter_replacement(
    plan: dict[str, Any], claims: list[dict[str, Any]], cards: dict[str, dict[str, Any]],
    language: str, target_minutes: float, profile: dict[str, Any], trace: ContextUsage,
    state: EpisodeGenerationState, check_cancelled: Callable[[], None],
    checkpoint_ready: Callable[[dict[str, Any]], None] | None = None,
    resume: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    resume = resume or {}
    if resume and resume.get("version") != PODCAST_ENGINE_VERSION:
        raise ValueError("Podcast checkpoint version changed; start a new generation.")
    core = copy.deepcopy(resume.get("core_turns") or [])
    replacements = copy.deepcopy(resume.get("chapter_versions") or {})
    drafts = copy.deepcopy(resume.get("chapter_drafts") or {})
    attempted = set(resume.get("attempted_parts") or [])
    warnings, audits = list(resume.get("warnings") or []), list(resume.get("audits") or [])
    output = min(profile["max_output_tokens"], current().plan.output_tokens if current() else 6000)
    if resume.get("recovery_used") and DELIVERY.get():
        claim_recovery()
    if DELIVERY.get():
        DELIVERY.get().stage_output_tokens.update(resume.get("stage_output_tokens") or {})
    by_id = {c["id"]: c for c in claims}
    ids = [c["id"] for c in plan["chapters"]]

    def evidence_ids(turns: list[dict[str, Any]]) -> set[str]:
        return {eid for turn in turns for cid in turn.get("claim_ids", [])
                for eid in (by_id.get(cid, {}).get("evidence_ids") or [cid])}

    def save() -> None:
        if checkpoint_ready:
            checkpoint_ready(copy.deepcopy({"version": PODCAST_ENGINE_VERSION, "episode_plan": plan,
                "claims": claims, "cards": list(cards.values()), "core_turns": core,
                "chapter_versions": replacements, "chapter_drafts": drafts, "attempted_parts": sorted(attempted),
                "warnings": warnings, "audits": audits, "recovery_used": bool(DELIVERY.get() and DELIVERY.get().recoveries),
                "stage_output_tokens": dict(DELIVERY.get().stage_output_tokens) if DELIVERY.get() else {}}))

    if not core:
        selected_ids = list(dict.fromkeys(cid for c in plan["chapters"] for cid in c["claim_ids"]))
        check_cancelled()
        core, audit = await create_linked_scene(scene_kind="act", chapter={"id": "core", "title": plan["episode_thesis"], "purpose": "A concise complete chapter map", "claim_ids": selected_ids},
            claims=[by_id[c] for c in selected_ids if c in by_id], cards_by_id=cards,
            memory=EpisodeMemory(plan["episode_thesis"]), existing_turns=[], target=4+2*len(ids), language=language,
            profile={**profile, "allow_partial": True, "complete_role": "core", "stage_output_tokens": min(6000, output),
                     "core_chapter_ids": ids, "core_chapter_plan": plan["chapters"]}, trace=trace,
            duration_budget=_scene_duration_budget(language, min(3.0, target_minutes*.3), 4+2*len(ids), 0), generation_state=state)
        if {t.get("source_chapter_id") for t in core} != {"opening", "closing", *ids}:
            raise PodcastQualityError("核心稿没有完整对应章节", {"stage": "core_structure"})
        audits.append(audit)
        check_cancelled()
        save()
    boundary_minutes = _content_minutes([t for t in core if t.get("source_chapter_id") in {"opening", "closing"}])
    minutes_per_chapter = max(.3, (target_minutes - boundary_minutes) / max(1, len(ids)))
    parts = max(1, math.ceil(minutes_per_chapter / podcast_stage_minutes(output, language)))
    for chapter in plan["chapters"]:
        key = chapter["id"]
        if key in replacements:
            continue
        existing_parts = drafts.setdefault(key, {})
        for part in range(parts):
            check_cancelled()
            part_id = f"{key}/{part}"
            if str(part) in existing_parts:
                continue
            if part_id in attempted:
                break
            if trace.total_token_limit and trace.total_token_limit - trace.accounted_tokens < trace.episode_audit_reserve_tokens + 1200 + 1024:
                warnings.append({"code": "optional_budget", "stage": "script", "message": "已保留终审预算，未完成章节沿用完整短版。"})
                break
            assembled, positions = _assemble_chapter_versions(core, plan, replacements)
            preceding = assembled[:next(c["turn_start"] for c in positions if c["id"] == key)] + [t for j in range(part) for t in existing_parts.get(str(j), [])]
            memory = EpisodeMemory(plan["episode_thesis"])
            _update_memory(memory, preceding, {"title": "Actual preceding dialogue"}, profile["recent_turns"])
            assignment = "Mechanism and premises" if part == 0 and parts > 1 else "Source example, implications and necessary qualifications" if part == parts-1 and parts > 1 else "Mechanism, useful example and necessary qualifications"
            stage_minutes = minutes_per_chapter / parts
            target = max(4, min(18, math.ceil(stage_minutes*3)))
            attempted.add(part_id)
            compact = [turn for turn in core if turn.get("source_chapter_id") == key]
            preserve_ids = list(dict.fromkeys(cid for turn in compact for cid in turn.get("claim_ids", [])))
            try:
                extra, audit = await create_linked_scene(scene_kind="act", chapter={**chapter, "id": part_id},
                    claims=[by_id[c] for c in chapter["claim_ids"] if c in by_id], cards_by_id=cards,
                    memory=memory, existing_turns=preceding, target=target, language=language,
                    profile={**profile, "allow_partial": True, "complete_role": "expansion", "stage_output_tokens": output,
                             "preserve_claim_ids": preserve_ids,
                             "compact_chapter": [{"text": t["text"], "claim_ids": t.get("claim_ids", [])} for t in compact],
                             "chapter_replacement": True, "part_assignment": f"{part+1}/{parts}: {assignment}"}, trace=trace,
                    duration_budget=_scene_duration_budget(language, stage_minutes, target, 0), generation_state=state)
                extra = _drop_repeated_exchanges(extra, preceding)
                if len(extra) < 2 or _is_question_turn(extra[-1]):
                    raise PodcastQualityError("章节分段没有完整收束", {"stage": "chapter_structure"})
                if parts == 1 and evidence_ids(compact) and not evidence_ids(compact).intersection(evidence_ids(extra)):
                    raise PodcastQualityError("深入版本偏离短章的原文依据，保留完整短章", {"stage": "chapter_evidence_loss"})
                existing_parts[str(part)] = extra
                audits.append(audit)
            except PodcastQualityError as exc:
                warnings.append({"code": "chapter_compact", "stage": "script", "message": f"{chapter['title']} 深入生成未完成，保留对应短章：{exc}"})
            check_cancelled()
            save()
            if str(part) not in existing_parts:
                break
        if all(str(i) in existing_parts for i in range(parts)):
            candidate = [t for i in range(parts) for t in existing_parts[str(i)]]
            compact = [t for t in core if t.get("source_chapter_id") == key]
            if evidence_ids(compact) and not evidence_ids(compact).intersection(evidence_ids(candidate)):
                warnings.append({"code": "chapter_compact", "stage": "script", "message": "深入版本偏离短章的原文依据，保留完整短章。"})
                save()
                continue
            if not evidence_ids(compact) <= evidence_ids(candidate):
                warnings.append({"code": "chapter_coverage_reduced", "stage": "script", "message": "扩写未沿用部分短章引用，已保留结果；内容覆盖仍需核对。"})
            replacements[key] = candidate
            save()
    turns, chapters = _assemble_chapter_versions(core, plan, replacements)
    return turns, chapters, audits, warnings, {"strategy": "chapter_replacement", "version": PODCAST_DURATION_CALIBRATION_VERSION,
        "core_minutes": _content_minutes(core), "accepted_blocks": len(replacements), "attempted_blocks": len(attempted),
        "core_texts": [t["text"] for t in core if t.get("source_chapter_id") in {"opening", "closing"}],
        "expansion": {"used": False}, "compression": {"used": False}}


async def _generate_complete_first(
    plan: dict[str, Any], claims: list[dict[str, Any]], cards: dict[str, dict[str, Any]],
    language: str, target_minutes: float, profile: dict[str, Any], trace: ContextUsage,
    state: EpisodeGenerationState, check_cancelled: Callable[[], None],
    checkpoint_ready: Callable[[dict[str, Any]], None] | None = None,
    resume: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    if plan.get("chapter_replacement"):
        return await _generate_chapter_replacement(plan, claims, cards, language, target_minutes, profile, trace, state, check_cancelled, checkpoint_ready, resume)
    resume = resume or {}
    output = min(6000, (current().plan.core_output_tokens if current() else 0) or profile["max_output_tokens"])
    core_minutes = min(target_minutes, 5.0, max(1.0, output / 1200))
    core = copy.deepcopy(resume.get("core_turns") or [])
    blocks = copy.deepcopy(resume.get("blocks") or [])
    completed = set(resume.get("attempted_blocks") or [])
    warnings = list(resume.get("warnings") or [])
    audits = list(resume.get("audits") or [])
    if resume.get("recovery_used") and DELIVERY.get():
        claim_recovery()
    by_id = {c["id"]: c for c in claims}

    def save() -> None:
        if checkpoint_ready:
            checkpoint_ready(copy.deepcopy({"version": PODCAST_ENGINE_VERSION, "episode_plan": plan,
                "claims": claims, "cards": list(cards.values()), "core_turns": core, "blocks": blocks,
                "attempted_blocks": sorted(completed), "warnings": warnings, "audits": audits,
                "recovery_used": bool(DELIVERY.get() and DELIVERY.get().recoveries)}))

    if not core:
        # One central claim from each planned chapter, source-balanced, instead of an arbitrary prefix.
        selected = []
        for chapter in plan["chapters"]:
            candidate = next((by_id[c] for c in chapter["claim_ids"] if c in by_id), None)
            if candidate and candidate not in selected:
                selected.append(candidate)
        for source in dict.fromkeys(c["source_id"] for c in claims):
            if not any(c["source_id"] == source for c in selected):
                selected.append(next(c for c in claims if c["source_id"] == source))
        selected = selected or claims[:2]
        chapter = {"id": "core", "title": plan["episode_thesis"], "purpose": "Complete episode: opening, explanation, resolved conclusion", "claim_ids": [c["id"] for c in selected]}
        check_cancelled()
        core, audit = await create_linked_scene(scene_kind="act", chapter=chapter, claims=selected,
            cards_by_id=cards, memory=EpisodeMemory(plan["episode_thesis"]), existing_turns=[], target=9,
            language=language, profile={**profile, "allow_partial": True, "complete_role": "core", "stage_output_tokens": output},
            trace=trace, duration_budget=_scene_duration_budget(language, core_minutes, 9, 0), generation_state=state)
        body, closing = _complete_core_parts(core)
        if len(body) < 2 or len(closing) < 2 or _is_question_turn(closing[-1]):
            raise PodcastQualityError("完整短稿缺少可保留的展开或结论", {"stage": "core_structure", "accepted_turns": len(core)})
        audits.append(audit)
        check_cancelled()
        save()

    used = {cid for t in core for cid in t.get("claim_ids", [])}
    used.update(cid for b in blocks for t in b["turns"] for cid in t.get("claim_ids", []))
    for index, chapter in enumerate(plan["chapters"]):
        check_cancelled()
        if chapter["id"] in completed:
            continue
        turns, _ = _assemble_complete_episode(core, blocks)
        if _content_minutes(turns) >= target_minutes * .95:
            break
        # Keep a full review plus one stage's output before accepting more optional work.
        if trace.total_token_limit and trace.total_token_limit - trace.accounted_tokens < trace.episode_audit_reserve_tokens + output * 2:
            warnings.append({"code": "optional_budget", "stage": "script", "message": "展开预算已用尽，保留完整短版及已完成内容。"})
            break
        fresh = [by_id[c] for c in chapter["claim_ids"] if c in by_id and c not in used]
        if not fresh:
            completed.add(chapter["id"])
            save()
            continue
        body, _ = _complete_core_parts(core)
        preceding = body + [t for b in blocks for t in b["turns"]]
        memory = EpisodeMemory(plan["episode_thesis"])
        _update_memory(memory, preceding, chapter, profile["recent_turns"])
        minutes = min(3.0, target_minutes - _content_minutes(turns), max(1.0, output / 1600))
        completed.add(chapter["id"])
        try:
            extra, audit = await create_linked_scene(scene_kind="act", chapter=chapter, claims=fresh,
                cards_by_id=cards, memory=memory, existing_turns=turns, target=6, language=language,
                profile={**profile, "allow_partial": True, "complete_role": "expansion", "stage_output_tokens": output},
                trace=trace, duration_budget=_scene_duration_budget(language, minutes, 6, 0), generation_state=state)
            unique = _drop_repeated_exchanges(extra, turns)
            if len(unique) != len(extra):
                warnings.append({"code": "repeated_exchange_removed", "stage": "script", "message": "已移除原样重复的完整交流，保留同一展开中的新内容。"})
            extra = unique
            text = " ".join(t["text"] for t in extra)
            prior = [" ".join(t["text"] for t in b["turns"]) for b in blocks] + [" ".join(t["text"] for t in core)]
            if len(extra) >= 2 and not _is_question_turn(extra[-1]) and not any(_similar(text, old) > .72 for old in prior):
                # One optional block is one removal unit; never leave half its explanation behind.
                for i, t in enumerate(extra):
                    t.update(exchange_id=f"optional/{chapter['id']}", exchange_start=i == 0)
                blocks.append({"id": chapter["id"], "title": chapter["title"], "turns": extra})
                used.update(cid for t in extra for cid in t.get("claim_ids", []))
                audits.append(audit)
            else:
                warnings.append({"code": "optional_discarded", "stage": "script", "message": "已舍弃重复或未完整收束的可选展开。"})
        except PodcastQualityError as exc:
            warnings.append({"code": "optional_failed", "stage": "script", "message": f"可选展开未完成，基础短稿仍保留：{exc}"})
        check_cancelled()
        save()
    turns, chapters = _assemble_complete_episode(core, blocks)
    return turns, chapters, audits, warnings, {"strategy": "complete_first", "version": PODCAST_DURATION_CALIBRATION_VERSION,
        "core_minutes": _content_minutes(core), "accepted_blocks": len(blocks), "attempted_blocks": len(completed),
        "core_texts": [t["text"] for t in core], "expansion": {"used": False}, "compression": {"used": False}}


@adaptive_generation("podcast")
async def build_podcast_script(
    notebook_id: str,
    payload: dict[str, Any],
    *,
    progress: Callable[[str, float], None] | None = None,
    act_ready: Callable[[dict[str, Any]], None] | None = None,
    allow_partial: bool = False,
    cancel_check: Callable[[], bool] | None = None,
    checkpoint_ready: Callable[[dict[str, Any]], None] | None = None,
    resume_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def check_cancelled() -> None:
        if cancel_check and cancel_check():
            raise RuntimeError("任务已取消")

    check_cancelled()
    ids = source_scope(notebook_id, payload.get("source_ids"))
    if not ids:
        raise ValueError("当前范围没有已就绪的文档")
    language, language_selection = resolve_output_language(DB, ids, payload.get("language", "zh-CN"))
    focus = str(payload.get("focus") or "").strip()
    if progress:
        progress("构建全篇证据地图", 0.08)
    requested_hint = int(payload.get("minutes") or 0)
    evidence_per_source = max(20, math.ceil(max(5, requested_hint) * 2 / max(1, len(ids))))
    resume_checkpoint = resume_checkpoint or {}
    rows = select_podcast_evidence(notebook_id, ids, focus, evidence_per_source)
    cards, all_citations = build_evidence_cards(rows)
    if len(cards) < 2:
        raise ValueError("资料内容不足，无法生成深度播客")
    if progress:
        progress("提取可引用主张", 0.12)
    context_usage = generation_trace()
    if current() and current().plan.preparation_batches and not resume_checkpoint:
        await prepare_evidence(rows, language)
        check_cancelled()
    claims = build_claim_ledger(cards)
    if current() and current().notes:
        claims = merge_prepared_claims(claims, cards, current().notes)
    if resume_checkpoint:
        claims = copy.deepcopy(resume_checkpoint["claims"])
        cards = copy.deepcopy(resume_checkpoint["cards"])
    if len(claims) < 2:
        raise ValueError("资料中缺少足够的可验证主张")
    duration_mode = payload.get("duration_mode") or ("fixed" if payload.get("minutes") else "auto")
    requested_minutes = int(payload.get("minutes") or 0) or None
    estimated_chapters = max(3, min(6, round(math.sqrt(max(1, len(claims))))))
    target_minutes = requested_minutes if duration_mode == "fixed" and requested_minutes else estimate_auto_minutes(estimated_chapters, len(cards))
    total_target = target_turn_count(target_minutes)
    profile = podcast_generation_profile()
    if allow_partial:
        # Output capacity, not just the context window, determines a safe Act size.
        output_capacity = TokenLimits.from_provider(active_provider("main") or {}).max_output_tokens
        profile = {**profile, "scene_turns": min(profile["scene_turns"], max(3, (output_capacity - 768) // 500))}
    act_count = max(2, math.ceil(total_target / profile["scene_turns"]))
    context_usage.request_limit = act_count + 4
    if not current():
        context_usage.total_token_limit = min(45_000, 14_000 + 750 * total_target)
        from .context_budget import high_reasoning
        if allow_partial and high_reasoning(active_provider("main") or {}):
            context_usage.total_token_limit = 45_000
    if allow_partial:
        from .context_budget import reserve_podcast_audit
        from .context_budget import high_reasoning
        audit_provider = active_provider("main") or {}
        reserve_podcast_audit(context_usage, TokenLimits.from_provider(audit_provider), reasoning=high_reasoning(audit_provider))
    if progress:
        progress("规划递进式剧集结构", 0.16)
    if resume_checkpoint:
        episode_plan, outline_degraded = copy.deepcopy(resume_checkpoint["episode_plan"]), False
    else:
        episode_plan, outline_degraded = await create_episode_plan(claims, language, focus, context_usage, podcast_chapter_capacity(min(6000, current().plan.output_tokens if current() else profile["max_output_tokens"]), target_minutes) if allow_partial else act_count)
    if allow_partial:
        episode_plan["chapter_replacement"] = True
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
    generation_state = EpisodeGenerationState(allow_partial=allow_partial)
    warnings: list[dict[str, str]] = []
    if allow_partial:
        profile = {**profile, "allow_partial": True}
        context_usage.request_limit = act_count * 2 + 10
    duration_goal = target_minutes * GENERATION_DURATION_TARGET_RATIO
    duration_calibration: dict[str, Any] = {
        "strategy": "slot_budget_with_single_expansion_v3",
        "version": PODCAST_DURATION_CALIBRATION_VERSION,
        "generation_target_ratio": GENERATION_DURATION_TARGET_RATIO,
        "target_estimated_minutes": round(duration_goal, 2),
        "acts": [],
    }
    if allow_partial:
        turns, chapter_payloads, scene_audits, warnings, duration_calibration = await _generate_complete_first(
            episode_plan, claims, cards_by_id, language, target_minutes, profile, context_usage,
            generation_state, check_cancelled, checkpoint_ready, resume_checkpoint)
    else:
        for chapter_index, chapter in enumerate(chapters):
            check_cancelled()
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
                if not allow_partial:
                    raise
                warnings.append({"code": "act_incomplete", "stage": "script", "message": f"{chapter.get('title') or chapter_index}: {exc}"})
                break
            check_cancelled()
            turns.extend(scene_turns)
            if act_ready and not allow_partial:
                act_ready({
                    "chapter_index": chapter_index,
                    "start_index": start_index,
                    "language": language,
                    "turns": [dict(turn) for turn in scene_turns],
                })
            scene_audits.append(scene_audit)
            if allow_partial and (scene_audit.get("partial") or not scene_audit.get("passed", True)):
                warnings.append({"code": "act_partial", "stage": "script", "message": f"{chapter.get('title') or chapter_index} 保留了有效对话；部分轮次或风格要求未达到。"})
            duration_calibration["acts"].append({"chapter_id": chapter["id"], **scene_audit.get("duration", duration_budget)})
            _update_memory(memory, scene_turns, chapter, profile["recent_turns"])
            chapter_payloads.append({**chapter, "turn_start": start_index, "turn_end": len(turns) - 1})
    if not turns or not any(turn.get("citation_ids") for turn in turns) or len({turn["speaker"] for turn in turns}) < 2:
        raise PodcastQualityError("没有足够的有依据双人对话可交付", {"passed": False, "warnings": warnings, "context_usage": context_usage.as_dict()})
    check_cancelled()
    expansion_report: dict[str, Any] = {"used": False}
    current_episode_minutes = _content_minutes(turns)
    release_minimum_minutes = target_minutes * 0.85
    if not allow_partial and current_episode_minutes < release_minimum_minutes:
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
        except (PodcastQualityError, ProviderError, RuntimeError) as failure:
            exc = failure if isinstance(failure, PodcastQualityError) else PodcastQualityError(str(failure))
            exc.report.update(
                {
                    "completed_acts": list(duration_calibration["acts"]),
                    "recovery_kind": generation_state.recovery_kind,
                    "context_usage": context_usage.as_dict(),
                }
            )
            if not allow_partial:
                raise
            expansion_report = {"used": True, "passed": False, "failure": exc.report}
            warnings.append({"code": "duration_shortfall", "stage": "script", "message": "有限扩写未达到目标时长，保留已有有效对话。"})
    elif current_episode_minutes < duration_goal:
        expansion_report = {
            "used": False,
            "skipped": True,
            "reason": "partial_delivery_no_duration_retry" if allow_partial and current_episode_minutes < release_minimum_minutes else "release_duration_gate_already_met",
            "estimated_minutes": round(current_episode_minutes, 3),
            "release_minimum_minutes": round(release_minimum_minutes, 3),
        }
    duration_calibration["expansion"] = expansion_report
    check_cancelled()
    compression_report: dict[str, Any] = {"used": False}
    current_episode_minutes = _content_minutes(turns)
    if not allow_partial and current_episode_minutes > target_minutes * 1.20:
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
        except (PodcastQualityError, ProviderError, RuntimeError) as failure:
            exc = failure if isinstance(failure, PodcastQualityError) else PodcastQualityError(str(failure))
            exc.report.update({
                "completed_acts": list(duration_calibration["acts"]),
                "recovery_kind": generation_state.recovery_kind,
                "context_usage": context_usage.as_dict(),
            })
            if not allow_partial:
                raise
            compression_report = {"used": True, "passed": False, "failure": exc.report}
            warnings.append({"code": "duration_excess", "stage": "script", "message": "有限压缩未达到目标时长，保留已有有效对话。"})
    duration_calibration["compression"] = compression_report
    if generation_state.recovery_kind is not None:
        # Expansion/compression is deliberately bounded to one call, but its
        # output must not consume the token allowance reserved for the final
        # publishability audit. The absolute 45k task ceiling still applies.
        _reserve_episode_audit_after_recovery(context_usage)
    check_cancelled()
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
        if not allow_partial:
            raise PodcastQualityError("整集脚本未达到客观发布门槛", preflight)
        warnings.extend({"code": "script_quality", "stage": "script", "message": reason} for reason in preflight["deterministic_failure_reasons"])
    if progress:
        progress("执行整集连贯性审校", 0.58)
    audit_claims = {cid: {**claim, "original": _claim_evidence_text([cid], claims_by_id, cards_by_id)} for cid, claim in claims_by_id.items()} if allow_partial else claims_by_id
    episode_audit = await (_audit_product_episode if allow_partial else _audit_episode)(turns, chapter_payloads, episode_plan["episode_thesis"], language, context_usage, audit_claims)
    check_cancelled()
    unsupported = set(episode_audit.get("unsupported_turns") or [])
    low_grounding = 0 < int((episode_audit.get("scores") or {}).get("grounding") or 0) < 4
    delivery_status = "full"
    if allow_partial:
        # Facts remain visible with annotations; never delete individual turns
        # from the middle of an otherwise connected conversation.
        for index in unsupported:
            if 0 <= index < len(turns):
                turns[index]["quality_issues"] = [{"code": "evidence_unconfirmed", "severity": "suspect",
                    "message": fact.get("reason") or "该轮内容的原文支持待核实。", "claim_id": fact.get("claim_id"),
                    "source_quote": fact.get("source_quote")} for fact in episode_audit.get("facts", []) if fact["index"] == index and fact["verdict"] != "supported"] or [{"code": "evidence_unconfirmed", "message": "该轮内容的原文支持待核实。"}]
        if duration_calibration.get("strategy") == "chapter_replacement":
            turns, chapter_payloads, delivery_status = _restore_reviewed_chapters(turns, chapter_payloads, episode_audit, warnings)
        elif any(t.get("exchange_id") for t in turns):
            turns, chapter_payloads, delivery_status = retain_product_exchanges(turns, chapter_payloads, episode_audit, warnings)
        else:
            broken = episode_audit.get("broken_at")
            if type(broken) is int and 0 <= broken < len(turns):
                prefix_turns = turns[:broken]
                while prefix_turns and (_is_question_turn(prefix_turns[-1]) or not re.search(r"[。.!！][’'”\"]?$", prefix_turns[-1]["text"].strip())):
                    prefix_turns.pop()
                if len(prefix_turns) >= 2 and len({t["speaker"] for t in prefix_turns}) == 2:
                    turns = prefix_turns
                    chapter_payloads = [{**c, "turn_end": min(c["turn_end"], len(turns) - 1)} for c in chapter_payloads if c["turn_start"] < len(turns)]
                    delivery_status = "partial"
                    warnings.append({"code": "coherent_short_version", "stage": "script", "message": "发现未解决的衔接问题，仅保留连续且句子完整的短版；自动检查不保证语义完整。"})
                else:
                    delivery_status = "draft_only"
                    warnings.append({"code": "coherence_draft", "stage": "script", "message": "连贯性问题尚未解决，保留脚本草稿，不合成音频。"})
        turns, chapter_payloads, delivery_status = finish_product_script(turns, chapter_payloads, delivery_status, target_minutes, language, warnings)
    if allow_partial:
        _refresh_episode_review(episode_audit, len(turns))
        if episode_audit["status"] != "complete":
            warnings.append({"code": "coherence_unverified", "stage": "script", "message":
                f"连贯性未完整验证：已检查 {episode_audit['checked_transitions']}/{episode_audit['total_transitions']} 处相邻对话；未检查部分不代表通过。"})
    if episode_plan.get("coverage"):
        episode_plan["coverage"]["cited_unit_ids"] = list(dict.fromkeys(cid for t in turns for cid in t.get("claim_ids", [])))
        if current():
            current().audit["planning"] = copy.deepcopy(episode_plan["coverage"])
    narrative_status = "unverified"
    if allow_partial:
        retained_texts = {turn["text"] for turn in turns}
        core_intact = all(text in retained_texts for text in duration_calibration.get("core_texts", []))
        closure = episode_audit.get("closure", {})
        if not core_intact or closure.get("verdict") == "broken":
            delivery_status = "draft_only"
            narrative_status = "incomplete"
            warnings.append({"code": "incomplete_ending", "stage": "script", "message": "基础短稿的完整性未保留，仅保存草稿。"})
        elif closure.get("verdict") == "connected":
            narrative_status = "complete"
        else:
            warnings.append({"code": "ending_unverified", "stage": "script", "message": "已保留短稿结论，语义收束仍待核实。"})
        duration_calibration.pop("core_texts", None)
    coherence_degraded = any(w.get("code") in {"coherent_short_version", "coherence_draft", "incomplete_ending", "ending_short_version"} for w in warnings)
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
        if not allow_partial:
            raise PodcastQualityError("整集脚本未达到发布门槛", quality)
        if not episode_audit.get("passed"):
            warnings.append({"code": "episode_audit", "stage": "script", "message": "整集审校未完成，不表示已判定内容错误；请核对脚本。" if not episode_audit.get("reviewed_indexes") else "部分对话衔接或原文支持需要核对，详见质量报告。"})
    if outline_degraded:
        context_usage.mark_fallback()
    script = "\n".join(f"{turn['speaker']}: {turn['text']} {' '.join(f'[{value}]' for value in turn['citation_ids'])}" for turn in turns)
    unsupported = {f["index"] for f in episode_audit.get("facts", []) if f["verdict"] != "supported"}
    fact_targets = {(i, cid) for i, turn in enumerate(turns) for cid in turn.get("claim_ids", [])}
    facts = [f for f in episode_audit.get("facts", []) if (f["index"], f["claim_id"]) in fact_targets]
    reviewed_facts = {(f["index"], f["claim_id"]) for f in facts}
    fact_review = {"version": 1, "total": len(fact_targets), "reviewed": len(reviewed_facts),
                   "supported": sum(f["verdict"] == "supported" for f in facts),
                   "contradicted": sum(f["verdict"] == "contradicted" for f in facts),
                   "uncertain": sum(f["verdict"] == "uncertain" for f in facts), "checks": facts,
                   "status": "complete" if fact_targets and reviewed_facts == fact_targets and all(f["verdict"] == "supported" for f in facts) else "partial" if facts else "unavailable"}
    cited_sources = {citation["source_id"] for citation in citations}
    source_contributions = [{"source_id": source, "included": source in cited_sources,
                             "reason": "本集已引用" if source in cited_sources else "本集未涵盖"} for source in ids]
    for chapter in chapter_payloads:
        chapter.pop("fallback_turns", None)
    return {
        "fact_review": fact_review, "source_contributions": source_contributions,
        "chapter_development": [{"id": c["id"], "status": c.get("development", "legacy")} for c in chapter_payloads],
        "generation_mode": "expanded" if duration_calibration.get("accepted_blocks") else "complete_short" if allow_partial else "legacy",
        "narrative_status": narrative_status,
        "delivery_status": delivery_status,
        "quality_assessment": assessment(len(turns), len({f["index"] for f in facts}),
            [{"unit": f"turn_{i + 1}", "severity": "suspect", "code": "evidence_unconfirmed", "message": "原文支持待核实。"} for i in unsupported if type(i) is int and 0 <= i < len(turns)]
            + ([{"unit": "episode", "severity": "suspect", "code": "coherence_issue", "message": "发现连贯性问题，已保留短版或草稿；请核对完整性。"}] if coherence_degraded else [])
            + [{"unit": "episode", "code": w["code"], "message": w["message"]} for w in warnings if w.get("code") in {"output_language", "product_duration", "chapter_coverage_reduced"}]
            + ([{"unit": "episode", "code": "local_repair", "message": "已做一次局部衔接修复，修复文本未经独立二次审校。"}] if episode_audit.get("local_repair_applied") else []), method="model_sample"),
        "version": PODCAST_ENGINE_VERSION,
        "engine": {
            **profile,
            "strategy": duration_calibration.get("strategy", "complete_first") if allow_partial else "editorial_acts",
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
        "degraded": outline_degraded or bool(warnings),
        "warnings": warnings,
        "context_usage": context_usage.as_dict(),
        "quality": quality,
        "quality_report": quality,
    }


def retain_product_exchanges(
    turns: list[dict[str, Any]], chapters: list[dict[str, Any]], audit: dict[str, Any],
    warnings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Remove whole broken exchanges and keep independent subsequent material."""
    breaks = set(audit.get("breaks") or []) | set(_local_dialogue_breaks(turns))
    if type(audit.get("broken_at")) is int:
        breaks.add(audit["broken_at"])
    removed: set[str] = set()
    groups: dict[str, list[int]] = {}
    for index, turn in enumerate(turns):
        groups.setdefault(turn.get("exchange_id") or f"legacy_{index}", []).append(index)
    for index in breaks:
        if type(index) is int and 0 <= index < len(turns):
            removed.add(turns[index].get("exchange_id") or f"legacy_{index}")
            if index and _is_question_turn(turns[index - 1]):
                removed.add(turns[index - 1].get("exchange_id") or f"legacy_{index - 1}")
    removed.update(item["exchange_id"] for item in audit.get("duplicates", []) if str(item.get("exchange_id", "")).startswith("optional/"))
    seen = set()
    for key, indexes in groups.items():
        text = _review_text(" ".join(turns[i]["text"] for i in indexes))
        if text in seen:
            removed.add(key)
        seen.add(text)
    if not removed:
        return turns, chapters, "full"
    retained = []
    gap = False
    for key, indexes in groups.items():
        first = turns[indexes[0]]
        dependent = bool(re.match(r"(?:this\b|that\b|therefore\b|so\b|it\b|they\b|as we|这|那|因此|所以|刚才|接着)", first["text"].strip(), re.I))
        if key in removed or (gap and not str(key).startswith("core/") and (not first.get("exchange_start") or dependent)):
            removed.add(key)
            gap = True
            continue
        retained.extend(indexes)
        gap = False
    audit["discarded_exchanges"] = [{"exchange_id": key, "turns": [dict(turns[i]) for i in indexes]}
                                     for key, indexes in groups.items() if key in removed]
    mapping = {old: new for new, old in enumerate(retained)}
    adjacent = {i for i in retained if i + 1 in mapping and mapping[i + 1] == mapping[i] + 1}
    audit["reviewed_indexes"] = [mapping[i] for i in audit.get("reviewed_indexes", []) if i in mapping]
    audit["reviewed_transitions"] = [mapping[i] for i in audit.get("reviewed_transitions", []) if type(i) is int and i in adjacent]
    audit["transition_checks"] = [{**check, "index": mapping[check["index"]]}
                                  for check in audit.get("transition_checks", []) if check.get("index") in adjacent]
    audit["requested_transitions"] = [mapping[i] for i in audit.get("requested_transitions", []) if i in adjacent]
    audit["facts"] = [{**fact, "index": mapping[fact["index"]]} for fact in audit.get("facts", []) if fact.get("index") in mapping]
    audit["unsupported_turns"] = [mapping[i] for i in audit.get("unsupported_turns", []) if i in mapping]
    audit["requested_facts"] = [mapping[i] for i in audit.get("requested_facts", []) if i in mapping]
    audit["original_breaks"] = sorted(breaks)
    audit.update(breaks=[], broken_at=None, passed=False)
    updated = []
    for chapter in chapters:
        indexes = [mapping[i] for i in retained if chapter["turn_start"] <= i <= chapter["turn_end"]]
        if indexes:
            updated.append({**chapter, "turn_start": min(indexes), "turn_end": max(indexes)})
    warnings.append({"code": "coherent_short_version", "stage": "script",
                     "message": f"已移除 {len(removed)} 个不完整、断裂或重复的交流单元；保留可独立理解的后续内容，新衔接未独立复核。"})
    result = [turns[i] for i in retained]
    if len(result) < 2 or len({t['speaker'] for t in result}) < 2:
        # Keep the original as a draft when no playable exchange survives.
        return turns, chapters, "draft_only"
    return result, updated, "partial"


def _review_text(text: str) -> str:
    """Normalize typography, never words, negations, or numbers."""
    normalized = unicodedata.normalize("NFC", text).translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    return " ".join(normalized.split())


def _review_edges(turns: list[dict[str, Any]], chapters: list[dict[str, Any]]) -> list[int]:
    """Prioritize known problems and boundaries before evenly spread questions."""
    questions = [i for i in range(len(turns) - 1) if _is_question_turn(turns[i])]
    spread = _evenly_spaced(questions, min(8, len(questions))) if questions else []
    values = _local_dialogue_breaks(turns) + [0, len(turns) - 2] + [c["turn_start"] - 1 for c in chapters[1:]] + spread + questions
    return list(dict.fromkeys(i for i in values if 0 <= i < len(turns) - 1))


def _product_audit_prompt(budget: PromptBudget, turns: list[dict[str, Any]], chapters: list[dict[str, Any]],
                          prefix: str, claims: dict[str, Any]) -> PromptBuild:
    """Pack whole adjacent turns; gaps are sampling gaps, never dialogue breaks."""
    def build(indexes: set[int]) -> PromptBuild:
        items = [{"index": i, "speaker": turns[i]["speaker"], "text": turns[i]["text"],
                  "claim_ids": turns[i].get("claim_ids", []), "example_kind": turns[i].get("example_kind", "none"),
                  "exchange_id": turns[i].get("exchange_id"), "gap_before": i > 0 and i - 1 not in indexes} for i in sorted(indexes)]
        ids = {c for item in items for c in item["claim_ids"] if c in claims}
        sources = _shared_evidence_payload([claims[c] for c in sorted(ids)])
        edges = _review_edges(turns, chapters)
        requested = [i for i in edges if i in indexes and i + 1 in indexes][:min(12, max(1, (budget.output_tokens - 768) // 400))]
        core = [i for c in chapters if not c.get("optional", False) for i in range(c["turn_start"], c["turn_end"] + 1)]
        risk = re.compile(r"必然|保证|无法|不可能|必须|概率|[0-9]|always|never|must|cannot|probabil|guarantee", re.I)
        by_chapter = [[i for i in range(c["turn_start"], c["turn_end"] + 1) if i in indexes and turns[i].get("claim_ids") and risk.search(turns[i]["text"])] for c in chapters]
        prioritized = [values[j] for j in range(max(map(len, by_chapter), default=0)) for values in by_chapter if j < len(values)]
        factual = [i for i in dict.fromkeys(prioritized + core + sorted(indexes)) if i in indexes and turns[i].get("claim_ids")]
        requested_facts = factual[:min(12, max(1, (budget.output_tokens - 768) // 400))]
        messages = [{"role": "user", "content": prefix + "\nCheck these adjacent transitions (i means i to i+1): " + json.dumps(requested) + "\nCheck these fact-bearing turns first: " + json.dumps(requested_facts) + "\n" + json.dumps({"turns": items, "evidence": sources}, ensure_ascii=False)}]
        return PromptBuild(messages, len(turns), len(items), 0, {"items": items, "requested_transitions": requested, "requested_facts": requested_facts})

    full = build(set(range(len(turns))))
    if estimate_messages_tokens(full.messages, budget.image_tokens_per_image) <= budget.input_tokens:
        return full
    edges = _review_edges(turns, chapters)
    groups = [list(range(c["turn_start"], min(c["turn_end"], len(turns) - 1))) for c in chapters]
    edges += [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)]
    selected: set[int] = set()
    for edge in dict.fromkeys(edges):
        if not 0 <= edge < len(turns) - 1:
            continue
        candidate = selected | {edge, edge + 1}
        trial = build(candidate)
        if estimate_messages_tokens(trial.messages, budget.image_tokens_per_image) <= budget.input_tokens:
            selected = candidate
    return build(selected)


def _refresh_episode_review(audit: dict[str, Any], turn_count: int) -> None:
    """Only retained, unchanged text can contribute to review coverage."""
    repaired = set(audit.get("repaired_indexes") or [])
    reviewed = {i for i in audit.get("reviewed_indexes", []) if type(i) is int and 0 <= i < turn_count and i not in repaired}
    normalized = []
    for value in audit.get("reviewed_transitions", []):
        if type(value) is int:
            normalized.append(value)
        elif isinstance(value, list) and len(value) == 2 and all(type(i) is int for i in value) and value[1] == value[0] + 1:
            normalized.append(value[0])
    edges = sorted({i for i in normalized
                    if type(i) is int and 0 <= i < turn_count - 1 and i in reviewed and i + 1 in reviewed})
    audit.update(reviewed_indexes=sorted(reviewed), reviewed_transitions=edges,
                 total_transitions=max(0, turn_count - 1), checked_transitions=len(edges))
    if "transition_checks" in audit:
        audit["transition_checks"] = [check for check in audit["transition_checks"]
                                      if isinstance(check, dict) and type(check.get("index")) is int and check["index"] in edges]
    if "requested_transitions" in audit:
        audit["requested_transitions"] = [i for i in audit["requested_transitions"]
                                          if type(i) is int and 0 <= i < turn_count - 1]
    audit["status"] = "complete" if turn_count > 1 and len(reviewed) == turn_count and len(edges) == turn_count - 1 else "partial" if reviewed else "unavailable"
    if audit["status"] == "unavailable":
        audit["passed"] = False
        audit.setdefault("reason", "未获得有效的连贯性检查结果")


def _local_dialogue_breaks(turns: list[dict[str, Any]]) -> list[int]:
    """Detect missing opening context and unanswered runs without model calls."""
    broken = []
    if turns and re.search(r"我们刚才|刚才我们|as we (?:just )?discussed|we (?:just|previously) (?:discussed|looked)", turns[0]["text"], re.I):
        broken.append(0)
    start = None
    for i, turn in enumerate(turns):
        sentences = [part.strip() for part in re.split(r"(?<=[。.!！?？])\s*", turn["text"]) if part.strip()]
        has_statement = any(len(part) >= 12 and re.search(r"[。.!！][’'”\"]?$", part) for part in sentences)
        if _is_question_turn(turn) and not has_statement:
            if start is None:
                start = i
            if i - start == 2:
                broken.append(start)
        else:
            start = None
    return sorted(set(broken))


def _validated_content_review(parsed: dict[str, Any], turns: list[dict[str, Any]], visible: set[int],
                              claims: dict[str, Any]) -> dict[str, Any]:
    def anchored(quote: Any, text: str) -> bool:
        return isinstance(quote, str) and 8 <= len(quote.strip()) <= 160 and _review_text(quote) in _review_text(text)

    facts = []
    seen = set()
    for item in parsed.get("facts", []) if isinstance(parsed.get("facts"), list) else []:
        if not isinstance(item, dict):
            continue
        i, cid = item.get("index"), item.get("claim_id")
        if type(i) is not int or i not in visible or not isinstance(cid, str) or cid not in turns[i].get("claim_ids", []):
            continue
        if (i, cid) in seen or item.get("verdict") not in {"supported", "contradicted", "uncertain"}:
            continue
        source = claims.get(cid, {}).get("original", "")
        if not anchored(item.get("script_quote"), turns[i]["text"]) or not anchored(item.get("source_quote"), source):
            continue
        seen.add((i, cid))
        facts.append({**item, "reason": str(item.get("reason") or "")[:300]})
    duplicates = []
    for item in parsed.get("duplicates", []) if isinstance(parsed.get("duplicates"), list) else []:
        if not isinstance(item, dict):
            continue
        i, prior = item.get("index"), item.get("prior_index")
        if type(i) is not int or type(prior) is not int or i not in visible or prior not in visible or prior >= i:
            continue
        key = str(turns[i].get("exchange_id") or "")
        if not key.startswith("optional/") or any(t.get("exchange_id") == key for t in turns[:i]):
            continue
        if anchored(item.get("quote"), turns[i]["text"]) and anchored(item.get("prior_quote"), turns[prior]["text"]):
            duplicates.append({**item, "exchange_id": key})
    return {"facts": facts, "duplicates": duplicates}


async def _audit_product_episode(turns, chapters, thesis, language, trace, claims_by_id=None):
    from .context_budget import reserve_podcast_audit
    from .context_budget import high_reasoning
    audit_provider = active_provider("main") or {}
    reserve_podcast_audit(trace, TokenLimits.from_provider(audit_provider), reasoning=high_reasoning(audit_provider))
    known_breaks = _local_dialogue_breaks(turns)
    prompt = ("检查双人对话的连贯性。只检查实际给出的相邻轮次，抽样间隔不是语义断裂。"
              "同时检查结尾是否回答开场核心问题而不是重新提问或预告；返回closure对象，含opening_quote、closing_quote（逐字8至60字符）、verdict（connected|broken|uncertain）、reason。"
              "必须保留作者归属和原文条件，不把假设变成已证事实、不把概率小变成不可能；仅在facts中记录有原文可核对的判断，不改写。"
              "重点判断前一轮问的具体问题是否被下一轮直接回答。重复问题、重述背景、转谈同主题另一机制都不等于回答；不确定就填uncertain。"
              "例如问温度为何上升，下一轮谈传感器存储数据，属于broken。时长、措辞、主持人比例不影响连贯性判断。"
              "开场不能假装存在更早的对话。每项判断必须引用两轮中的逐字短语，各取8至60字符，不要省略号，理由最多60字符。"
              "按附带的检查索引顺序检查，不要跳过；输出空间不足时只保留完整检查项。"
              "只返回JSON，格式：{\"closure\":{\"opening_quote\":\"开场逐字短语\",\"closing_quote\":\"结尾逐字短语\",\"verdict\":\"connected|broken|uncertain\",\"reason\":\"是否回答核心问题\"},\"unsupported_turns\":[],\"checks\":[{\"index\":0,\"question_quote\":\"前一轮原文短语\",\"answer_quote\":\"后一轮原文短语\",\"verdict\":\"connected|broken|uncertain\",\"reason\":\"具体理由\"}],"
              "\"breaks\":[],\"broken_at\":null,\"repairs\":[]}。breaks列确定断裂的轮次，broken_at填最早断裂轮次或null。"
              "只报告判断，不改写口播，repairs必须为空数组。"
              "系统发现缺少前文或连续三个问题未获回答的索引：" + json.dumps(known_breaks) + "。主题：" + thesis + "\n")
    prompt = prompt[:prompt.index("只返回JSON，格式：")] + (
        "只返回JSON：" + json.dumps(audit_schema(), ensure_ascii=False) + "\n"
        "先做closure，再按requested fact indexes检查facts，最后做checks和duplicates；不得超出请求数量。"
        "facts每项含index、claim_id、script_quote、source_quote、verdict(supported|contradicted|uncertain)、reason。"
        "两个quote均逐字8至60字符，分别来自该轮口播及对应原文。supported要求保留原文所有必要条件及作者归属；"
        "概率不能变保证，多数算力不能变多数节点，作者哲学假说不能变成确定事实。证据不足用uncertain。"
        "标记illustrative的类比必须在口播中表明是假设；假设情境不当作原文事实，但机制映射与推论仍必须有依据，不能因标记而免检。"
        "duplicates只报告整块没有新增机制、例证、限定的可选交流（exchange_id以optional/开始）；结论合理复述不算重复。"
        "index和prior_index分别为可选交流第一轮与前文轮次，quote/prior_quote是两处逐字8至60字符，reason最多60字符。"
        "只作判断，不改写，不用缺失引用的事实或抽样间隔判断断裂。主题：" + thesis + "\n")
    try:
        result = await budgeted_chat(lambda budget: _product_audit_prompt(budget, turns, chapters, prompt, claims_by_id or {}),
            json_mode=True, response_schema=audit_schema(), max_tokens=4096, trace=trace, stage="episode_audit")
        parsed = _extract_json(result.content)
        if result.finish_reason in {"length", "max_tokens"} or not isinstance(parsed, dict):
            raise ValueError("审校回复不完整")
        visible = {item["index"] for item in result.build.metadata.get("items", [])}
        if result.build.truncated_segments:
            visible = set()  # Cannot certify a clipped transcript.
        checks = []
        closure = parsed.get("closure") or {}
        if not (isinstance(closure, dict) and closure.get("verdict") in {"connected", "broken", "uncertain"}
                and isinstance(closure.get("opening_quote"), str) and isinstance(closure.get("closing_quote"), str)
                and min(len(closure["opening_quote"].strip()), len(closure["closing_quote"].strip())) >= 8
                and any(i in visible and _review_text(closure["opening_quote"]) in _review_text(t["text"]) for i, t in enumerate(turns[:2]))
                and any(i in visible and _review_text(closure["closing_quote"]) in _review_text(turns[i]["text"]) for i in range(max(0, len(turns)-2), len(turns)))):
            closure = {"verdict": "uncertain", "reason": "收束检查没有可核对的开场与结尾原文。"}
        raw_checks = parsed.get("checks", parsed.get("transition_checks", parsed.get("reviewed_transitions", [])))
        for check in raw_checks if isinstance(raw_checks, list) else []:
            if not isinstance(check, dict):
                continue
            quote_a, quote_b = check.get("question_quote"), check.get("answer_quote")
            if (not isinstance(quote_a,str) or not isinstance(quote_b,str) or min(len(quote_a.strip()),len(quote_b.strip())) < 8
                    or check.get("verdict") not in {"connected","broken","uncertain"}
                    or not isinstance(check.get("reason"),str) or not check["reason"].strip()):
                continue
            matches = [i for i in sorted(visible) if i + 1 in visible
                       and _review_text(quote_a) in _review_text(turns[i]["text"])
                       and _review_text(quote_b) in _review_text(turns[i+1]["text"])]
            i = check.get("index")
            if type(i) is not int or i not in matches:
                if len(matches) != 1:
                    continue  # Ambiguous quotations cannot relocate a verdict.
                i = matches[0]
            if any(existing["index"] == i for existing in checks):
                continue
            anchored = {**check, "index": i}
            if anchored["verdict"] == "broken" and not _is_question_turn(turns[i]):
                # A resolved statement followed by another topic is not an
                # unanswered question. Keep the suspicion visible, but do not
                # let topic changes delete a complete core or its conclusion.
                anchored["verdict"] = "uncertain"
                anchored["reason"] = "已完成陈述后的主题变化不构成未答问题；衔接待核实。原判断：" + anchored["reason"]
            checks.append(anchored)
            if len(checks) >= max(1, (result.budget.output_tokens - 512) // 160):
                break
        reviewed = sorted({i for check in checks for i in (check["index"],check["index"]+1)})
        # Only anchored adjacent quotations or deterministic local checks may
        # truncate dialogue. Models sometimes number questions instead of turns.
        breaks = set(known_breaks)
        breaks.update(check["index"]+1 for check in checks if check["verdict"] == "broken")
        raw_breaks = parsed.get("breaks", [])
        unanchored = bool(raw_breaks or parsed.get("broken_at") is not None) and not breaks
        # Audits classify existing text only. Suggested prose is never an independently grounded repair.
        applied = False
        repaired_indexes = []
        repair_suspect = False
        broken = min(breaks) if breaks else None
        content_review = _validated_content_review(parsed, turns, visible, claims_by_id or {})
        audit = {"passed": broken is None and not unanchored and bool(checks) and all(c["verdict"] == "connected" for c in checks),
                 "closure": closure, **content_review,
                 "requested_facts": result.build.metadata.get("requested_facts", []),
                 "requested_transitions": result.build.metadata.get("requested_transitions", []), "scores": {}, "invalid_boundaries": [],
                 "issues": ["断裂索引缺少相邻原文依据，未据此截断对话。"] if unanchored else [],
                 "unsupported_turns": sorted({f["index"] for f in content_review["facts"] if f["verdict"] != "supported"}),
                 "broken_at": broken, "breaks": sorted(breaks),
                 "reviewed_indexes": reviewed, "reviewed_transitions": [check["index"] for check in checks], "transition_checks": checks,
                 "repaired_indexes": repaired_indexes, "local_repair_applied": applied, "repair_unreviewed": repair_suspect,
                 "coverage_mode": "full" if len(visible) == len(turns) else "sampled"}
        _refresh_episode_review(audit, len(turns))
        return audit
    except Exception as exc:
        audit = {"passed": False, "scores": {}, "invalid_boundaries": [], "issues": ["连贯性审校未完成"],
                 "reviewed_indexes": [], "reviewed_transitions": [], "coverage_mode": "sampled",
                 "reason": "连贯性审校未完成：" + type(exc).__name__,
                 "broken_at": min(known_breaks) if known_breaks else None}
        _refresh_episode_review(audit, len(turns))
        return audit



_strict_build_podcast_script = build_podcast_script
_product_build_podcast_script = delivery_task(build_podcast_script)

async def build_podcast_script(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if kwargs.get("allow_partial"):
        return await _product_build_podcast_script(*args, **kwargs)
    return await _strict_build_podcast_script(*args, **kwargs)


def finish_product_script(turns: list[dict[str, Any]], chapters: list[dict[str, Any]], status: str,
                          target_minutes: float, language: str, warnings: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Local completion only: retain a continuous closed ending, never fill with new claims."""
    local_breaks = _local_dialogue_breaks(turns)
    if local_breaks and status != "draft_only":
        prefix = turns[:local_breaks[0]]
        while prefix and (_is_question_turn(prefix[-1]) or not re.search(r"[。.!！][’'”\"]?$", prefix[-1]["text"].strip())):
            prefix.pop()
        if len(prefix) >= 2 and len({t["speaker"] for t in prefix}) == 2:
            turns = prefix
            chapters = [{**c, "turn_end": min(c["turn_end"], len(turns)-1)} for c in chapters if c["turn_start"] < len(turns)]
            status = "partial"
            warnings.append({"code":"coherent_short_version","stage":"script","message":"连续提问未得到回答，已保留此前完整的连续短版。"})
        else:
            status = "draft_only"
            warnings.append({"code":"coherence_draft","stage":"script","message":"开场缺少前文或连续提问未获回答，仅保留草稿。"})
    if status != "draft_only" and turns:
        retained = list(turns)
        while retained and (_is_question_turn(retained[-1]) or not re.search(r"[。.!！][’'”\"]?$", retained[-1]["text"].strip())):
            retained.pop()
        if len(retained) != len(turns):
            if len(retained) >= 2 and len({t["speaker"] for t in retained}) == 2:
                turns = retained
                chapters = [{**c, "turn_end": min(c["turn_end"], len(turns) - 1)} for c in chapters if c["turn_start"] < len(turns)]
                status = "partial"
                warnings.append({"code":"ending_short_version","stage":"script","message":"已去掉结尾未回答的问题或残句，保留连续完整的短版。"})
            else:
                status = "draft_only"
                warnings.append({"code":"incomplete_ending","stage":"script","message":"没有可独立交付的完整结尾，仅保留草稿。"})
    estimated = _content_minutes(turns)
    if not .8 * target_minutes <= estimated <= 1.25 * target_minutes:
        if status == "full":
            status = "partial"
        warnings.append({"code":"product_duration","stage":"script","message":f"目标 {target_minutes:g} 分钟，脚本估计 {estimated:.1f} 分钟，超出 80%–125% 参考范围；不为凑时长重写。"})
    # Raw strict metrics remain available, but product warnings use its own range.
    warnings[:] = [w for w in warnings if not (w.get("code") == "script_quality" and "时长" in w.get("message", ""))]
    wrong = [i for i,t in enumerate(turns) if not text_matches_language(t["text"], language)]
    if wrong:
        warnings.append({"code":"output_language","stage":"script","message":f"{len(wrong)} 轮输出语言与请求不同，已保留内容。"})
        for i in wrong:
            turns[i].setdefault("quality_issues", []).append({"code":"output_language","message":"本轮输出语言与请求不同。"})
    return turns, chapters, status
