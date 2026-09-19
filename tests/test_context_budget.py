import sqlite3

import pytest

from sandevistan_read import providers
from sandevistan_read.context_budget import (
    ContextUsage,
    TokenLimits,
    approximate_text_tokens,
    estimate_text_tokens,
    pack_items,
    structured_output_tokens,
)
from sandevistan_read.database import Database


def test_token_limits_apply_manual_overrides_and_conservative_defaults() -> None:
    fallback = TokenLimits.from_provider({"capabilities": {}, "config": {}})
    assert fallback.effective_context_tokens == 4096
    assert fallback.max_output_tokens == 1024
    assert fallback.context_source == "fallback"

    manual = TokenLimits.from_provider(
        {
            "capabilities": {"token_limits": {"model_context_tokens": 32768, "effective_context_tokens": 8192}},
            "config": {"context_window_tokens": 16384, "max_output_tokens": 2048},
        }
    )
    assert manual.effective_context_tokens == 16384
    assert manual.max_output_tokens == 2048
    assert manual.context_source == "manual"


def test_structured_output_budget_includes_bounded_reasoning_headroom() -> None:
    assert structured_output_tokens(700) == 1724
    assert structured_output_tokens(1800) == 3600
    assert structured_output_tokens(6000) == 10096


def test_context_usage_enforces_aggregate_budget_and_tracks_stages() -> None:
    trace = ContextUsage(request_limit=2, total_token_limit=500)
    limits = TokenLimits.from_provider({"capabilities": {}, "config": {}})
    trace.begin_request(estimated_tokens=200)
    trace.record(
        limits=limits,
        requested_output=100,
        output_tokens=100,
        estimated_prompt=80,
        actual_prompt=100,
        actual_completion=50,
        stage="outline",
    )
    assert trace.as_dict()["by_stage"]["outline"]["calls"] == 1
    with pytest.raises(RuntimeError, match="token 达到任务上限"):
        trace.begin_request(estimated_tokens=400)
    assert trace.as_dict()["stop_reason"] == "token_limit"

    no_usage = ContextUsage(total_token_limit=250)
    no_usage.begin_request(estimated_tokens=200)
    no_usage.record(
        limits=limits,
        requested_output=100,
        output_tokens=100,
        estimated_prompt=100,
        actual_prompt=None,
        actual_completion=None,
    )
    assert no_usage.as_dict()["accounted_total_tokens"] == 200
    with pytest.raises(RuntimeError, match="token 达到任务上限"):
        no_usage.begin_request(estimated_tokens=100)


def test_approximate_text_tokens_is_calibrated_below_conservative_bound() -> None:
    zh = "播客" * 500  # 1000 CJK chars -> 3000 UTF-8 bytes
    en = "podcast " * 500  # 4000 Latin bytes
    # Calibrated admission estimate: ~bytes/4 for the measured MAIN tokenizer,
    # always below the conservative bytes/2 overflow-safety bound.
    assert approximate_text_tokens(zh) == 750
    assert estimate_text_tokens(zh) == 1500
    assert approximate_text_tokens(en) == 1000
    assert approximate_text_tokens("") == 0


async def test_duration_recovery_extends_audit_reserve_before_recovery_call(monkeypatch) -> None:
    # iter-09 regression: _compress_episode_duration hit
    # RuntimeError("已为最终整集审校保留预算") because the recovery extension
    # (_reserve_episode_audit_after_recovery) was granted only AFTER the
    # recovery call. The admission check in providers.budgeted_chat therefore
    # rejected the bounded recovery call itself. The extension must be granted
    # before the recovery request and must be idempotent.
    from types import SimpleNamespace

    from sandevistan_read import podcast

    line = (
        "比特币系统通过工作量证明机制确保交易记录难以被篡改，"
        "网络中的节点遵循最长链原则对交易顺序达成共识，"
        "每一个新区块都包含前一个区块的哈希值从而形成链条结构。"
    )
    turns = [
        {"speaker": "HOST_A", "dialogue_act": "X", "text": line * 4, "claim_ids": []}
        for _ in range(5)
    ]
    monkeypatch.setattr(
        podcast,
        "_duration_compression_plan",
        lambda _turns, _chapters, _excess, _language: [
            {"index": 0, "safe_minimum_units": 30, "maximum_units": 200, "current_units": 120}
        ],
    )
    captured: dict[str, int] = {}

    async def fake_budgeted_chat(_builder, **_kwargs):
        captured["limit"] = trace.total_token_limit
        replacement = "系统要求参与者提供计算努力作为证明，新区块包含前一区块的哈希值而形成链条。"
        return SimpleNamespace(
            content='{"replacements": [[0, "' + replacement + '"]]}',
            finish_reason="stop",
        )

    monkeypatch.setattr(podcast, "budgeted_chat", fake_budgeted_chat)
    trace = ContextUsage(total_token_limit=43000, request_limit=10)
    trace.accounted_tokens = 41000
    trace.episode_audit_reserve_tokens = 6000
    state = podcast.EpisodeGenerationState()

    compressed, report = await podcast._compress_episode_duration(
        turns, [{"id": "chapter_1"}], {}, {}, "zh-CN", 5.0, trace, state
    )

    assert report["used"] is True
    # Reserve granted BEFORE the recovery call, capped at the 45k absolute ceiling.
    assert captured["limit"] == 45000
    assert trace.total_token_limit == 45000
    # Idempotent: the post-recovery backstop must not extend a second time.
    podcast._reserve_episode_audit_after_recovery(trace, state)
    assert trace.total_token_limit == 45000


async def test_continuation_extends_audit_reserve_before_its_call(monkeypatch) -> None:
    # A3: the grant-before-call contract from A1 must also hold for the bounded
    # act_continuation path; a continuation near the token cap would otherwise
    # be refused by the same admission check that hit iter-09's compression.
    from types import SimpleNamespace

    from sandevistan_read import podcast

    captured: dict[str, int] = {}

    async def fake_budgeted_chat(_builder, **_kwargs):
        captured["limit"] = trace.total_token_limit
        return SimpleNamespace(content='{"turns": []}', finish_reason="stop")

    monkeypatch.setattr(podcast, "budgeted_chat", fake_budgeted_chat)
    trace = ContextUsage(total_token_limit=43000, request_limit=10)
    trace.accounted_tokens = 41000
    trace.episode_audit_reserve_tokens = 6000
    state = podcast.EpisodeGenerationState()
    partial = [
        {"speaker": "HOST_A", "dialogue_act": "explain", "text": "已有轮次的实质内容。", "claim_ids": []}
    ]
    result = await podcast._continue_scene(
        chapter={"title": "t", "purpose": "p", "tension": "x", "bridge_out": "b"},
        claims=[{"id": "C1", "text": "主张", "evidence_ids": ["E1"], "source_id": "s1"}],
        cards_by_id={},
        memory=podcast.EpisodeMemory("thesis"),
        existing_turns=[],
        partial=partial,
        target=4,
        language="zh-CN",
        trace=trace,
        duration_budget={"minimum_units": 300, "maximum_units": 400, "unit": "cjk_equivalent_chars"},
        generation_state=state,
    )
    assert captured["limit"] == 45000
    assert trace.total_token_limit == 45000
    # Grant is idempotent across the later duration-recovery backstop.
    podcast._reserve_episode_audit_after_recovery(trace, state)
    assert trace.total_token_limit == 45000
    assert state.audit_reserve_extended is True


def test_strict_22m_zh_episode_completes_within_45k() -> None:
    # Regression: admission used to reserve the 2x-inflated prompt estimate plus
    # the full padded output budget, aborting evidence-dense 22m zh episodes at
    # ~25k actual of the 45k cap. Admission is now calibrated estimate + minimum
    # output, settled to metered actuals, with a 6000 audit reserve held for
    # every non-audit stage (audit itself exempt).
    trace = ContextUsage(request_limit=8, total_token_limit=45_000, episode_audit_reserve_tokens=6_000)
    limits = TokenLimits.from_provider({"capabilities": {}, "config": {}})
    calls = [
        ("episode_plan", 11_200 + 384, 10_900, 800),
        ("act_draft", 4_800 + 2_080, 4_700, 1_500),
        ("act_draft", 4_800 + 2_080, 4_700, 1_500),
        ("act_draft", 4_800 + 2_080, 4_700, 1_500),
        ("act_draft", 4_800 + 2_080, 4_700, 1_500),
        ("episode_audit", 5_200 + 450, 5_155, 207),
    ]
    for stage, admission, actual_prompt, actual_completion in calls:
        if stage != "episode_audit":
            assert trace.accounted_tokens + admission <= trace.total_token_limit - trace.episode_audit_reserve_tokens
        trace.begin_request(estimated_tokens=admission)
        trace.record(
            limits=limits,
            requested_output=4_800,
            output_tokens=4_800,
            estimated_prompt=admission,
            actual_prompt=actual_prompt,
            actual_completion=actual_completion,
            stage=stage,
        )
    assert trace.requests == 6
    assert trace.accounted_tokens == 11_700 + 4 * 6_200 + 5_362
    assert trace.accounted_tokens <= 45_000


def test_soft_cap_aborts_at_next_admission_after_bounded_overshoot() -> None:
    # A call already in flight always finishes and settles to actuals, even when
    # that carries the total past the cap; the overshoot is bounded by that one
    # call, and the next admission stops the task.
    trace = ContextUsage(total_token_limit=20_000)
    limits = TokenLimits.from_provider({"capabilities": {}, "config": {}})
    trace.begin_request(estimated_tokens=5_000)
    trace.record(
        limits=limits,
        requested_output=4_000,
        output_tokens=4_000,
        estimated_prompt=5_000,
        actual_prompt=18_000,
        actual_completion=3_000,
    )
    assert trace.accounted_tokens == 21_000
    with pytest.raises(RuntimeError, match="token 达到任务上限"):
        trace.begin_request(estimated_tokens=100)
    assert trace.as_dict()["stop_reason"] == "token_limit"
    assert trace.accounted_tokens == 21_000


def test_usage_less_response_charges_measured_content_before_full_reserve() -> None:
    limits = TokenLimits.from_provider({"capabilities": {}, "config": {}})

    measured = ContextUsage(total_token_limit=50_000)
    measured.begin_request(estimated_tokens=6_000)
    measured.record(
        limits=limits,
        requested_output=11_000,
        output_tokens=11_000,
        estimated_prompt=5_000,
        actual_prompt=None,
        actual_completion=None,
        fallback_prompt=4_800,
        fallback_completion=1_200,
    )
    assert measured.accounted_tokens == 6_000

    empty = ContextUsage(total_token_limit=50_000)
    empty.begin_request(estimated_tokens=16_000)
    empty.record(
        limits=limits,
        requested_output=11_000,
        output_tokens=11_000,
        estimated_prompt=5_000,
        actual_prompt=None,
        actual_completion=None,
        fallback_prompt=4_800,
        fallback_completion=None,  # empty content: no measurable completion, keep the conservative charge
    )
    assert empty.accounted_tokens == 15_800


def test_packing_preserves_source_coverage_within_budget() -> None:
    items = [
        {"source": "a", "text": "甲" * 120},
        {"source": "a", "text": "乙" * 120},
        {"source": "b", "text": "丙" * 120},
    ]
    packed = pack_items(items, lambda item: item["text"], 400, group_key=lambda item: item["source"])
    assert {item["source"] for item in packed.items} == {"a", "b"}
    assert sum(estimate_text_tokens(text) + 2 for text in packed.texts) <= 400


@pytest.mark.asyncio
async def test_budgeted_chat_retries_context_overflow_at_smaller_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = {
        "kind": "openai",
        "model": "chat",
        "base_url": "https://example.com",
        "api_key": "",
        "capabilities": {"token_limits": {"effective_context_tokens": 8192, "max_output_tokens": 2048}},
        "config": {},
    }
    monkeypatch.setattr(providers, "active_provider", lambda role: provider)
    calls = 0
    scales: list[float] = []

    async def fake_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise providers.ContextOverflowError("maximum context length", status=400, code="context_length_exceeded")
        return providers.ChatCompletion("ok", prompt_tokens=100, completion_tokens=8)

    monkeypatch.setattr(providers, "_chat_once", fake_once)
    trace = ContextUsage()

    def build(budget):
        scales.append(budget.scale)
        return providers.PromptBuild([{"role": "user", "content": "short"}], 10, int(10 * budget.scale))

    result = await providers.budgeted_chat(build, max_tokens=1000, trace=trace)
    assert result.content == "ok"
    assert scales == [1.0, 0.5, 0.25]
    assert trace.as_dict()["overflow_retries"] == 2
    assert trace.as_dict()["adjusted"] is True


def test_message_metadata_migration_preserves_legacy_rows(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_versions (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
            INSERT INTO schema_versions VALUES (1, 'old');
            INSERT INTO schema_versions VALUES (2, 'old');
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations_json TEXT NOT NULL DEFAULT '[]',
                scope_hash TEXT,
                state TEXT NOT NULL DEFAULT 'complete',
                created_at TEXT NOT NULL
            );
            INSERT INTO messages VALUES ('m1','c1','assistant','legacy','[]',NULL,'complete','old');
            """
        )
    database = Database(path)
    database._migrate_v3()
    row = database.fetchone("SELECT content,metadata_json FROM messages WHERE id='m1'")
    assert row == {"content": "legacy", "metadata_json": "{}"}
    assert database.fetchone("SELECT MAX(version) AS version FROM schema_versions")["version"] == 3
