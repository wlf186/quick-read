from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar


DEFAULT_CONTEXT_WINDOW_TOKENS = 4096
DEFAULT_IMAGE_TOKENS = 2048
MAX_CONTEXT_WINDOW_TOKENS = 4_194_304
MIN_CONTEXT_WINDOW_TOKENS = 1024
MIN_OUTPUT_WINDOW_TOKENS = 128
SAFETY_RATIO = 0.80
RETRY_SCALES = (1.0, 0.5, 0.25)
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0


def positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 0 < number <= MAX_CONTEXT_WINDOW_TOKENS else None


def validate_token_overrides(config: dict[str, Any]) -> None:
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
        if remaining >= 64 and (not selected or (group_key and group_key(item) not in {group_key(value) for value in selected})):
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
    estimated_prompt_tokens: int = 0
    actual_prompt_tokens: int = 0
    actual_completion_tokens: int = 0
    fallback_used: bool = False

    def mark_fallback(self) -> None:
        self.fallback_used = True

    def record(
        self,
        *,
        limits: TokenLimits,
        requested_output: int,
        output_tokens: int,
        attempts: int,
        estimated_prompt: int,
        actual_prompt: int | None,
        actual_completion: int | None,
        total_segments: int = 0,
        included_segments: int = 0,
        truncated_segments: int = 0,
    ) -> None:
        self.effective_context_tokens = limits.effective_context_tokens
        self.max_output_tokens = limits.max_output_tokens
        self.context_source = limits.context_source
        self.calls += 1
        self.requests += attempts
        self.overflow_retries += max(0, attempts - 1)
        self.dropped_segments += max(0, total_segments - included_segments)
        self.truncated_segments += truncated_segments
        self.output_limited_calls += int(output_tokens < requested_output)
        self.estimated_prompt_tokens += estimated_prompt
        self.actual_prompt_tokens += actual_prompt or 0
        self.actual_completion_tokens += actual_completion or 0

    def as_dict(self) -> dict[str, Any]:
        adjusted = bool(
            self.overflow_retries or self.dropped_segments or self.truncated_segments or self.output_limited_calls or self.fallback_used
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
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "actual_prompt_tokens": self.actual_prompt_tokens or None,
            "actual_completion_tokens": self.actual_completion_tokens or None,
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
