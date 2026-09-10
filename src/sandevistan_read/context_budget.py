from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar


DEFAULT_CONTEXT_WINDOW_TOKENS = 4096
DEFAULT_IMAGE_TOKENS = 2048
# Enable broad preparation only after paired quality qualification. Explicit
# balanced configuration remains available for isolated evaluation and opt-in.
DEFAULT_CONTEXT_STRATEGY = "conservative"
QUALIFIED_BALANCED_TASKS: frozenset[str] = frozenset()


def context_strategy(provider: dict[str, Any], kind: str) -> str:
    """Explicit opt-in wins; default enablement requires per-feature evidence."""
    configured = (provider.get("config") or {}).get("context_strategy")
    if isinstance(configured, str) and configured in {"balanced", "conservative"}:
        return configured
    if provider.get("id") and kind in {"summary", "chat"}:
        from .context_qualification import qualified
        if qualified(provider, kind):
            return "balanced"
    return "balanced" if kind in QUALIFIED_BALANCED_TASKS else DEFAULT_CONTEXT_STRATEGY


MAX_CONTEXT_WINDOW_TOKENS = 4_194_304
MIN_CONTEXT_WINDOW_TOKENS = 1024
MIN_OUTPUT_WINDOW_TOKENS = 128
SAFETY_RATIO = 0.80
RETRY_SCALES = (1.0, 0.5, 0.25)
# Reactive output-budget escalation ceiling when hidden reasoning exhausts max_tokens;
# only applies to derived output limits (manual/provider-reported caps stay hard).
REASONING_OUTPUT_ESCALATION_CEILING = 16_384
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0


def structured_output_tokens(visible_target: int) -> int:
    """Reserve room for hidden reasoning without tying policy to one provider."""
    visible = max(1, visible_target)
    return visible + max(1024, min(4096, visible))


def positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 0 < number <= MAX_CONTEXT_WINDOW_TOKENS else None


def validate_token_overrides(config: dict[str, Any]) -> None:
    if "context_strategy" in config and config["context_strategy"] not in ("balanced", "conservative"):
        raise ValueError("上下文策略必须是 balanced 或 conservative")
    effort = config.get("reasoning_effort")
    if effort is not None and (not isinstance(effort, str) or effort not in {"low", "medium", "high", "xhigh", "max"}):
        raise ValueError("reasoning_effort 必须是 low、medium、high、xhigh 或 max")
    context_value = config.get("context_window_tokens")
    output_value = config.get("max_output_tokens")
    if context_value is not None:
        context = positive_int(context_value)
        if context is None or context < MIN_CONTEXT_WINDOW_TOKENS:
            raise ValueError(f"上下文窗口必须是 {MIN_CONTEXT_WINDOW_TOKENS} 到 {MAX_CONTEXT_WINDOW_TOKENS} 之间的整数")
    else:
        context = None
    if output_value is not None:
        output = positive_int(output_value)
        if output is None or output < MIN_OUTPUT_WINDOW_TOKENS:
            raise ValueError(f"最大输出必须是 {MIN_OUTPUT_WINDOW_TOKENS} 到 {MAX_CONTEXT_WINDOW_TOKENS} 之间的整数")
        if context is not None and output >= context:
            raise ValueError("最大输出必须小于上下文窗口")


def resolve_temperature(config: dict[str, Any], default: float) -> float:
    value = config.get("temperature")
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Temperature 必须是 0 到 2 之间的数字")
    temperature = float(value)
    if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
        raise ValueError("Temperature 必须是 0 到 2 之间的数字")
    return temperature


def estimate_text_tokens(text: str) -> int:
    """Conservative model-agnostic estimate for mixed CJK and Latin text."""
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 2))


def _content_tokens(content: Any, image_tokens: int) -> int:
    if isinstance(content, str):
        return estimate_text_tokens(content)
    if not isinstance(content, list):
        return estimate_text_tokens(str(content))
    total = 0
    for part in content:
        if not isinstance(part, dict):
            total += estimate_text_tokens(str(part))
        elif part.get("type") in {"image_url", "input_image"} or "image_url" in part:
            total += image_tokens
        else:
            total += estimate_text_tokens(str(part.get("text") or part.get("content") or ""))
    return total


def estimate_messages_tokens(messages: list[dict[str, Any]], image_tokens: int = DEFAULT_IMAGE_TOKENS) -> int:
    total = 16
    for message in messages:
        total += 8 + estimate_text_tokens(str(message.get("role") or ""))
        total += _content_tokens(message.get("content", ""), image_tokens)
        if message.get("images"):
            total += image_tokens * len(message["images"])
    return total


def truncate_text_tokens(text: str, token_budget: int) -> tuple[str, bool]:
    if token_budget <= 0:
        return "", bool(text)
    if estimate_text_tokens(text) <= token_budget:
        return text, False
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_text_tokens(text[:middle]) <= token_budget:
            low = middle
        else:
            high = middle - 1
    clipped = text[:low].rstrip()
    if low < len(text):
        boundary = max(clipped.rfind("。"), clipped.rfind("！"), clipped.rfind("？"), clipped.rfind("\n"), clipped.rfind(". "))
        if boundary >= max(32, len(clipped) // 2):
            clipped = clipped[: boundary + 1].rstrip()
    return clipped, True


@dataclass(frozen=True)
class TokenLimits:
    model_context_tokens: int | None
    effective_context_tokens: int
    max_input_tokens: int | None
    max_output_tokens: int
    context_source: str
    output_source: str
    image_tokens_per_image: int = DEFAULT_IMAGE_TOKENS

    @classmethod
    def from_provider(cls, provider: dict[str, Any]) -> "TokenLimits":
        capabilities = provider.get("capabilities") or {}
        stored = capabilities.get("token_limits") or {}
        config = provider.get("config") or {}
        model_max = positive_int(stored.get("model_context_tokens"))
        manual_context = positive_int(config.get("context_window_tokens"))
        effective = manual_context or positive_int(stored.get("effective_context_tokens")) or DEFAULT_CONTEXT_WINDOW_TOKENS
        if model_max:
            effective = min(effective, model_max)
        manual_output = positive_int(config.get("max_output_tokens"))
        derived_output = max(MIN_OUTPUT_WINDOW_TOKENS, min(4096, effective // 4))
        output = manual_output or positive_int(stored.get("max_output_tokens")) or derived_output
        output = min(output, max(MIN_OUTPUT_WINDOW_TOKENS, effective - 1))
        return cls(
            model_context_tokens=model_max,
            effective_context_tokens=effective,
            max_input_tokens=positive_int(stored.get("max_input_tokens")),
            max_output_tokens=output,
            context_source="manual" if manual_context else str(stored.get("context_source") or "fallback"),
            output_source="manual" if manual_output else str(stored.get("output_source") or "derived"),
            image_tokens_per_image=positive_int(stored.get("image_tokens_per_image")) or DEFAULT_IMAGE_TOKENS,
        )


@dataclass(frozen=True)
class PromptBudget:
    context_tokens: int
    input_tokens: int
    output_tokens: int
    image_tokens_per_image: int
    scale: float


def prompt_budget(limits: TokenLimits, requested_output: int, minimum_output: int, scale: float) -> PromptBudget:
    safe_total = max(1, math.floor(limits.effective_context_tokens * SAFETY_RATIO * scale))
    output_cap = max(minimum_output, math.floor(safe_total * 0.25))
    output = min(requested_output, limits.max_output_tokens, output_cap)
    output = max(1, min(output, max(1, safe_total - 1)))
    input_tokens = max(1, safe_total - output)
    if limits.max_input_tokens:
        input_tokens = min(input_tokens, limits.max_input_tokens)
    return PromptBudget(limits.effective_context_tokens, input_tokens, output, limits.image_tokens_per_image, scale)


CONTEXT_STRATEGY_VERSION = "evidence_v6"
TASK_TOKEN_CEILING = 300_000
PODCAST_TASK_TOKEN_CEILING = TASK_TOKEN_CEILING * 5 // 4


def task_token_ceiling(context_tokens: int, kind: str) -> int:
    """Bound broad synthesis by capacity; small tasks still budget actual demand."""
    return max(TASK_TOKEN_CEILING, math.ceil(context_tokens * 1.25)) if kind in {"summary", "chat"} else PODCAST_TASK_TOKEN_CEILING if kind == "podcast" else TASK_TOKEN_CEILING


def podcast_chapter_capacity(output_tokens: int, minutes: float) -> int:
    """Bound the compact chapter map before any provider call."""
    return min(6, max(1, (output_tokens - 768) // 512), max(1, math.ceil(minutes / 5)))


def podcast_stage_minutes(output_tokens: int, language: str) -> float:
    # Include JSON labels and variable tokenization; this is a planning estimate.
    units_per_minute = 150 * 2 if language == "en" else 300 * 1.5
    return max(.3, min(5.0, (output_tokens - 768) / (units_per_minute + 100)))


@dataclass(frozen=True)
class ContextPlan:
    version: str
    kind: str
    context_tokens: int
    output_tokens: int
    evidence_tokens: int
    batch_evidence_tokens: int
    estimated_segments: int
    preparation_batches: int
    total_token_limit: int
    final_reserve_tokens: int
    output_items: int
    limiting_factor: str
    path: str = "sampled"
    final_evidence_tokens: int = 0
    note_capacity: int = 0
    overview_items: int = 0
    preparation_token_limit: int = 0
    preparation_output_tokens: int = 0
    core_output_tokens: int = 0
    podcast_chapters: int = 0

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def chat_history_budget(budget: PromptBudget) -> int:
    """At most 10% of safe input, bounded by the usable task budget as well.

    Reserving 10% of a million-token window would otherwise reduce evidence
    once the cumulative task ceiling binds. The cap meets that crossing point.
    """
    usable_task_input = max(0, math.floor(task_token_ceiling(budget.context_tokens, "chat") * .75) - budget.output_tokens)
    return min(budget.input_tokens, usable_task_input) // 10


def plan_context(
    limits: TokenLimits, kind: str, *, material_tokens: int | None = None,
    segment_tokens: int = 1000, count: int = 10, minutes: int = 20,
    source_count: int = 1, broad_query: bool = False,
) -> ContextPlan:
    """Plan bounded work from capacity and actual demand, never from a model name."""
    kind = {"flashcards": "flashcard", "podcasts": "podcast"}.get(kind, kind)
    if kind not in {"chat", "summary", "quiz", "flashcard", "podcast"}:
        raise ValueError("Unknown generation kind")
    requested = {"chat": 3600, "summary": 8192, "quiz": 8192,
                 "flashcard": 8192, "podcast": 16_384}[kind]
    budget = prompt_budget(limits, requested, 128, 1.0)
    if kind in {"summary", "chat"}:
        source_count = max(1, source_count)
        desired_items = 6 if source_count == 1 else 6 + 4 * source_count
        output = (min(8192, 1800 + 450 * (source_count - 1)) if broad_query else 1800) if kind == "chat" else max(structured_output_tokens(2200), 512 + desired_items * 350)
        budget = prompt_budget(limits, output, 128, 1.0)
        overhead = 4096 + (chat_history_budget(budget) if kind == "chat" else 0)
        capacity = max(0, budget.input_tokens - overhead)
        demand = material_tokens if material_tokens is not None else capacity
        total = min(task_token_ceiling(limits.effective_context_tokens, kind), max(24_000, math.ceil((min(demand, capacity) * 2 + budget.output_tokens * 4 + overhead) / .75)))
        reserve = math.ceil(total * .25)
        evidence = min(demand, capacity, max(0, total - reserve - overhead - budget.output_tokens))
        direct = material_tokens is not None and evidence >= material_tokens
        reason = "资料已可容纳" if direct else "任务累计预算" if evidence < min(demand, capacity) else "有效上下文与输出预算"
        final_evidence = min(evidence, max(12_000, budget.output_tokens * 6))
        prepare = kind == "summary" and (source_count >= 3 or demand > final_evidence)
        prepare = prepare or (kind == "chat" and broad_query and demand > final_evidence)
        batches = min(4, max(1, math.ceil(evidence / max(1, final_evidence)))) if prepare else 0
        batch_evidence = min(capacity, max(1, math.ceil(evidence / max(1, batches))))
        items = min(desired_items, max(1, (budget.output_tokens - 512) // 350)) if kind == "summary" else 1
        overview = min(6, max(1, items - min(source_count, items - 1))) if kind == "summary" and source_count > 1 else items if kind == "summary" else 0
        return ContextPlan(CONTEXT_STRATEGY_VERSION, kind, limits.effective_context_tokens,
            budget.output_tokens, evidence, batch_evidence if batches else evidence,
            evidence // max(1, segment_tokens), batches, total, reserve, items, reason,
            "prepared" if batches else "direct" if direct and kind == "summary" else "structured" if kind == "summary" else "query",
            final_evidence, batches * max(1, (budget.output_tokens - 512) // 400), overview)
    # Reserve instructions, history, source labels and intermediate reasoning.
    batch = max(0, min(64_000, budget.input_tokens - 2048))
    per_item = {"chat": 450, "summary": 500, "quiz": 900, "flashcard": 500, "podcast": 500}[kind]
    output_items = max(1, min(18 if kind == "podcast" else 12, (budget.output_tokens - 512) // per_item))
    base = {"chat": 24_000, "summary": 24_000, "quiz": 36_000,
            "flashcard": 60_000, "podcast": min(45_000, 14_000 + 2100 * minutes)}[kind]
    # Small documents stay single-pass. Extra preparation is only useful for
    # broad synthesis; questions retain query-focused retrieval instead.
    demand = material_tokens if material_tokens is not None else batch * (4 if kind != "chat" else 1)
    preparation = min(4, math.ceil(demand / max(1, batch))) if demand > batch and kind != "chat" else 0
    evidence = min(demand, batch * max(1, preparation), 120_000 if kind != "chat" else 32_000)
    work = (minutes * 4200 if kind == "podcast" else count * 1400 if kind in {"quiz", "flashcard"} else 12_000)
    total = min(TASK_TOKEN_CEILING, max(base, math.ceil((evidence * 2 + work) / 0.75)))
    if kind == "podcast":
        total = math.floor(total * 1.25)
    reserve = math.ceil(total * 0.25)
    # Requested duration/count can exceed the bounded task budget. Keep room
    # for evidence instead of allowing an aspirational output cost to erase it.
    work = min(work, (total - reserve) // 2)
    evidence = min(evidence, max(0, (total - reserve - work) // 2))
    preparation = min(4, math.ceil(evidence / max(1, batch))) if preparation else 0
    if material_tokens is not None and evidence >= material_tokens:
        reason = "资料已可容纳"
    elif total == task_token_ceiling(limits.effective_context_tokens, kind) or evidence == 120_000:
        reason = "均衡任务总预算"
    elif batch == 64_000 or (kind == "chat" and evidence == 32_000):
        reason = "单阶段均衡预算"
    else:
        reason = "有效上下文与输出预算"
    podcast_prepare = min(2, preparation) if kind == "podcast" else preparation
    preparation_limit = total // 10 if kind == "podcast" else 0
    preparation_output = min(2048, budget.output_tokens) if kind == "podcast" else 0
    if kind == "podcast":
        batch = min(batch, max(0, preparation_limit // max(1, podcast_prepare) - preparation_output - 2048))
    return ContextPlan(CONTEXT_STRATEGY_VERSION, kind, limits.effective_context_tokens,
                       budget.output_tokens, evidence, batch, evidence // max(1, segment_tokens),
                       podcast_prepare, total, reserve, output_items, reason,
                       final_evidence_tokens=evidence,
                       note_capacity=podcast_prepare * max(1, ((preparation_output or budget.output_tokens) - 512) // 400),
                       preparation_token_limit=preparation_limit,
                       preparation_output_tokens=preparation_output,
                       core_output_tokens=min(6000, budget.output_tokens) if kind == "podcast" else 0,
                       podcast_chapters=podcast_chapter_capacity(min(6000, budget.output_tokens), minutes) if kind == "podcast" else 0)


T = TypeVar("T")


@dataclass
class PackedItems:
    items: list[T]
    texts: list[str]
    total: int
    truncated: int = 0


def pack_items(
    items: Iterable[T],
    renderer: Callable[[T], str],
    token_budget: int,
    *,
    group_key: Callable[[T], str] | None = None,
    allow_truncation: bool = True,
) -> PackedItems:
    values = list(items)
    ordered: list[T] = []
    if group_key:
        seen_groups: set[str] = set()
        for item in values:
            group = group_key(item)
            if group not in seen_groups:
                ordered.append(item)
                seen_groups.add(group)
    ordered.extend(item for item in values if item not in ordered)
    selected: list[T] = []
    texts: list[str] = []
    remaining = max(0, token_budget)
    truncated = 0
    for item in ordered:
        rendered = renderer(item)
        cost = estimate_text_tokens(rendered) + 2
        if cost <= remaining:
            selected.append(item)
            texts.append(rendered)
            remaining -= cost
            continue
        if allow_truncation and remaining >= 64 and (not selected or (group_key and group_key(item) not in {group_key(value) for value in selected})):
            clipped, changed = truncate_text_tokens(rendered, remaining - 2)
            if clipped:
                selected.append(item)
                texts.append(clipped)
                truncated += int(changed)
                remaining = 0
        if remaining < 64:
            break
    return PackedItems(selected, texts, len(values), truncated)


@dataclass
class ContextUsage:
    effective_context_tokens: int = 0
    max_output_tokens: int = 0
    context_source: str = "fallback"
    calls: int = 0
    requests: int = 0
    overflow_retries: int = 0
    dropped_segments: int = 0
    truncated_segments: int = 0
    output_limited_calls: int = 0
    budget_clamped_calls: int = 0
    failed_requests: int = 0
    estimated_prompt_tokens: int = 0
    actual_prompt_tokens: int = 0
    actual_completion_tokens: int = 0
    reasoning_tokens: int = 0
    visible_completion_tokens: int = 0
    cached_tokens: int = 0
    accounted_tokens: int = 0
    reserved_tokens: int = 0
    episode_audit_reserve_tokens: int = 0
    temperature_sources: dict[str, int] | None = None
    effective_temperatures: dict[str, int] | None = None
    request_limit: int | None = None
    total_token_limit: int | None = None
    stop_reason: str | None = None
    by_stage: dict[str, dict[str, int]] = field(default_factory=dict)
    fallback_used: bool = False

    def mark_fallback(self) -> None:
        self.fallback_used = True

    @property
    def actual_total_tokens(self) -> int:
        return self.actual_prompt_tokens + self.actual_completion_tokens

    def begin_request(self, *, estimated_tokens: int = 0) -> None:
        if self.request_limit is not None and self.requests >= self.request_limit:
            self.stop_reason = "request_limit"
            raise RuntimeError(f"MAIN 调用达到任务上限（{self.request_limit}）")
        if self.total_token_limit is not None and self.accounted_tokens + max(0, estimated_tokens) > self.total_token_limit:
            self.stop_reason = "token_limit"
            raise RuntimeError(f"MAIN token 达到任务上限（{self.total_token_limit}）")
        self.requests += 1
        self.reserved_tokens = max(0, estimated_tokens)
        self.accounted_tokens += self.reserved_tokens

    def record_failure(self) -> None:
        self.failed_requests += 1
        # No reliable usage on a failed request: retain its conservative charge.
        self.reserved_tokens = 0

    def record(
        self,
        *,
        limits: TokenLimits,
        requested_output: int,
        output_tokens: int,
        estimated_prompt: int,
        actual_prompt: int | None,
        actual_completion: int | None,
        reasoning_tokens: int | None = None,
        cached_tokens: int | None = None,
        temperature: float | None = None,
        temperature_source: str | None = None,
        stage: str = "generation",
        total_segments: int = 0,
        included_segments: int = 0,
        truncated_segments: int = 0,
    ) -> None:
        self.effective_context_tokens = limits.effective_context_tokens
        self.max_output_tokens = limits.max_output_tokens
        self.context_source = limits.context_source
        self.calls += 1
        self.dropped_segments += max(0, total_segments - included_segments)
        self.truncated_segments += truncated_segments
        self.budget_clamped_calls += int(output_tokens < requested_output)
        self.estimated_prompt_tokens += estimated_prompt
        self.actual_prompt_tokens += actual_prompt or 0
        self.actual_completion_tokens += actual_completion or 0
        self.reasoning_tokens += reasoning_tokens or 0
        self.visible_completion_tokens += max(0, (actual_completion or 0) - (reasoning_tokens or 0))
        self.cached_tokens += cached_tokens or 0
        self.accounted_tokens -= self.reserved_tokens
        self.reserved_tokens = 0
        self.accounted_tokens += (actual_prompt if actual_prompt is not None else estimated_prompt)
        self.accounted_tokens += (actual_completion if actual_completion is not None else output_tokens)
        stage_usage = self.by_stage.setdefault(
            stage,
            {"calls": 0, "estimated_prompt_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
        )
        stage_usage["calls"] += 1
        stage_usage["estimated_prompt_tokens"] += estimated_prompt
        stage_usage["prompt_tokens"] += actual_prompt or 0
        stage_usage["completion_tokens"] += actual_completion or 0
        stage_usage["reasoning_tokens"] += reasoning_tokens or 0
        if temperature_source:
            self.temperature_sources = self.temperature_sources or {}
            self.temperature_sources[temperature_source] = self.temperature_sources.get(temperature_source, 0) + 1
        if temperature is not None:
            self.effective_temperatures = self.effective_temperatures or {}
            key = f"{temperature:g}"
            self.effective_temperatures[key] = self.effective_temperatures.get(key, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        adjusted = bool(
            self.overflow_retries or self.dropped_segments or self.truncated_segments or self.output_limited_calls
            or self.budget_clamped_calls or self.failed_requests or self.fallback_used
        )
        return {
            "effective_context_tokens": self.effective_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "context_source": self.context_source,
            "calls": self.calls,
            "requests": self.requests,
            "overflow_retries": self.overflow_retries,
            "dropped_segments": self.dropped_segments,
            "truncated_segments": self.truncated_segments,
            "output_limited_calls": self.output_limited_calls,
            "budget_clamped_calls": self.budget_clamped_calls,
            "failed_requests": self.failed_requests,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "actual_prompt_tokens": self.actual_prompt_tokens or None,
            "actual_completion_tokens": self.actual_completion_tokens or None,
            "reasoning_tokens": self.reasoning_tokens or None,
            "visible_completion_tokens": self.visible_completion_tokens or None,
            "cached_tokens": self.cached_tokens or None,
            "temperature_sources": self.temperature_sources or {},
            "effective_temperatures": self.effective_temperatures or {},
            "episode_audit_reserve_tokens": self.episode_audit_reserve_tokens,
            "request_limit": self.request_limit,
            "total_token_limit": self.total_token_limit,
            "actual_total_tokens": self.actual_total_tokens or None,
            "accounted_total_tokens": self.accounted_tokens or None,
            "by_stage": self.by_stage,
            "stop_reason": self.stop_reason,
            "adjusted": adjusted,
            "fallback_used": self.fallback_used,
        }


CONTEXT_ERROR_PATTERN = re.compile(
    r"context[_ -]length|maximum context|context window|too many tokens|prompt (?:is )?too long|input (?:is )?too long|exceeds? .*context|num_ctx",
    re.I,
)


def is_context_error(status: int | None, code: str, message: str) -> bool:
    if status not in {400, 413, 422}:
        return False
    return code.lower() in {"context_length_exceeded", "context_window_exceeded", "too_many_tokens"} or bool(
        CONTEXT_ERROR_PATTERN.search(message)
    )


def reserve_podcast_audit(trace: ContextUsage, limits: TokenLimits) -> None:
    """Protect one final review inside the existing task and model limits."""
    budget = prompt_budget(limits, 4096, 128, 1.0)
    trace.episode_audit_reserve_tokens = min(
        (trace.total_token_limit or 0) // 4, budget.input_tokens + budget.output_tokens
    )
