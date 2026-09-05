import json
from types import SimpleNamespace

import pytest

from sandevistan_read import jobs, podcast
from sandevistan_read.context_budget import PromptBudget
from sandevistan_read.schemas import PodcastRequest


def test_duration_modes_are_backward_compatible() -> None:
    legacy = PodcastRequest(minutes=20)
    assert legacy.minutes == 20
    assert legacy.duration_mode == "fixed"
    assert PodcastRequest().language == "zh-CN"
    assert podcast.estimate_auto_minutes(4, 20) == 18
    assert podcast.estimate_auto_minutes(7, 20) == 21
    assert podcast.target_turn_count(5) == 18
    assert podcast.target_turn_count(20) == 56


def test_question_rule_translates_ratio_into_actionable_counts() -> None:
    assert podcast._question_count_rule(9) == "问句必须有 2–3 轮（占本 Act 的 20%–35%）"
    assert podcast._question_count_rule(4) == "问句必须恰好有 1 轮"


def test_remaining_duration_budget_carries_short_act_debt_forward() -> None:
    targets = [22, 22, 23, 23]
    first = podcast._remaining_scene_duration_budget("zh-CN", 23.75, 0, 90, targets, 0)
    second_on_target = podcast._remaining_scene_duration_budget("zh-CN", 23.75, first["target_minutes"], 90, targets, 1)
    second_after_short_act = podcast._remaining_scene_duration_budget("zh-CN", 23.75, first["target_minutes"] - 1.5, 90, targets, 1)
    assert first["unit"] == "cjk_equivalent_chars"
    assert second_after_short_act["minimum_units"] > second_on_target["minimum_units"]
    assert second_after_short_act["carry_in_minutes"] > 0
    nominal = 23.75 * targets[1] / 90
    assert second_after_short_act["target_minutes"] <= round(nominal * 1.20, 3)


def test_turn_slot_plan_covers_act_floor_with_varied_turns() -> None:
    for language, minimum in (("zh-CN", 1695), ("en", 945)):
        plan = podcast._turn_slot_plan(18, minimum, language)
        assert len(plan) == 18
        assert sum(item["minimum_units"] for item in plan) >= minimum
        assert {item["kind"] for item in plan} == {"short", "deep"}
        assert 5 <= sum(item["kind"] == "short" for item in plan) <= 8


def test_scene_validation_accepts_substantive_chinese_turn_below_spoken_cap() -> None:
    text = "这是一段用来说明资料中核心主张的完整口播。" * 11 + "补充说明"
    assert 220 < podcast._spoken_unit_count(text, "zh-CN") <= 240
    accepted, issues = podcast.validate_scene_turns(
        [{"speaker": "HOST_A", "text": text, "dialogue_act": "explain", "claim_ids": ["C1"]}],
        {"C1": {"id": "C1", "text": text, "evidence_ids": ["E1"]}},
        {"E1": {"id": "E1", "content": text}},
        last_speaker=None,
        existing_turns=[],
        language="zh-CN",
        expected_count=1,
        scene_kind="act",
    )
    assert len(accepted) == 1
    assert not any("长度不合格" in issue for issue in issues)


def test_scene_validation_rejects_punctuation_that_drives_emotional_tts() -> None:
    text = "这个结论确实值得继续讨论！"
    accepted, issues = podcast.validate_scene_turns(
        [{"speaker": "HOST_A", "text": text, "dialogue_act": "explain", "claim_ids": ["C1"]}],
        {"C1": {"id": "C1", "text": text, "evidence_ids": ["E1"]}},
        {"E1": {"id": "E1", "content": text}},
        last_speaker=None,
        existing_turns=[],
        language="zh-CN",
        expected_count=1,
        scene_kind="act",
    )
    assert accepted == []
    assert any("放大口播情绪" in issue for issue in issues)


def test_duration_budget_and_audio_gate_are_language_aware() -> None:
    english = podcast._scene_duration_budget("en", 5, 18, 0)
    chinese = podcast._scene_duration_budget("zh-CN", 5, 18, 0)
    assert english["unit"] == "words" and chinese["unit"] == "cjk_equivalent_chars"
    assert chinese["minimum_units"] > english["minimum_units"]
    assert podcast._content_minutes([{
        "text": "知" * 225, "speaker": "HOST_A", "dialogue_act": "explain", "claim_ids": ["C1"],
    }]) == pytest.approx(1 + podcast.TURN_PAUSE_SECONDS / 60)
    assert jobs._actual_duration_check(20, 20 * 60)["passed"] is True
    assert jobs._actual_duration_check(20, 16 * 60)["passed"] is False


def test_safe_turns_are_short_traceable_dialogue() -> None:
    cards = [
        {"id": "E1", "content": "第一条证据解释了系统如何建立顺序。它还说明参与者为什么需要验证记录。"},
        {"id": "E2", "content": "第二条证据描述了工作量证明的作用。这个机制让修改历史记录需要重新计算。"},
    ]
    turns = podcast._safe_chapter_turns(cards, 6, "zh-CN")
    assert len(turns) == 6
    assert all(turn["citation_ids"][0] in {"E1", "E2"} for turn in turns)
    assert all(len(turn["text"]) < 180 for turn in turns)


def test_claim_ledger_and_fallback_plan_preserve_source_order() -> None:
    cards = [
        {"id": "E1", "source_id": "s1", "filename": "one.md", "locator": {"section": "第一章"}, "content": "第一条主张说明系统先建立公开顺序。第二条主张解释参与者随后验证记录。"},
        {"id": "E2", "source_id": "s2", "filename": "two.md", "locator": {"section": "第二章"}, "content": "第三条主张说明冲突记录必须被排除。第四条主张说明最终结果可以复核。"},
    ]
    claims = podcast.build_claim_ledger(cards)
    plan = podcast._fallback_episode_plan(claims, "zh-CN")
    planned_ids = [claim_id for chapter in plan["chapters"] for claim_id in chapter["claim_ids"]]
    assert planned_ids == [claim["id"] for claim in claims]
    assert len(planned_ids) == len(set(planned_ids))


def test_scene_validation_allows_uncited_bridge_but_rejects_uncited_fact() -> None:
    claims = {"C1": {"id": "C1", "text": "系统通过公开记录建立可验证的交易顺序。", "evidence_ids": ["E1"]}}
    cards = {"E1": {"id": "E1", "content": "系统通过公开记录建立可验证的交易顺序。"}}
    valid, issues = podcast.validate_scene_turns(
        [
            {"speaker": "HOST_A", "dialogue_act": "bridge", "text": "我们先把刚才的结论接起来。", "claim_ids": []},
            {"speaker": "HOST_B", "dialogue_act": "explain", "text": "资料说明公开记录让交易顺序可以被验证。", "claim_ids": ["C1"]},
        ],
        claims,
        cards,
        last_speaker=None,
        existing_turns=[],
        language="zh-CN",
        expected_count=2,
    )
    assert not issues and valid[0]["citation_ids"] == [] and valid[1]["citation_ids"] == ["E1"]

    invalid, issues = podcast.validate_scene_turns(
        [
            {"speaker": "HOST_A", "dialogue_act": "explain", "text": "火星基地已经在2025年完成部署。", "claim_ids": []},
            {"speaker": "HOST_B", "dialogue_act": "question", "text": "那验证者下一步需要检查什么？", "claim_ids": []},
        ],
        claims,
        cards,
        last_speaker=None,
        existing_turns=[],
        language="zh-CN",
        expected_count=2,
    )
    assert len(invalid) == 1
    assert invalid[0]["dialogue_act"] == "question"
    assert any("没有 claim_id" in issue for issue in issues)


def test_scene_validation_rejects_third_question_across_act_boundary() -> None:
    existing = [
        {"speaker": "HOST_A", "dialogue_act": "question", "text": "第一个问题？"},
        {"speaker": "HOST_B", "dialogue_act": "question", "text": "第二个问题？"},
    ]
    valid, issues = podcast.validate_scene_turns(
        [
            {"speaker": "HOST_A", "dialogue_act": "acknowledgement", "text": "标签不是 question 但文本仍是第三个问题？", "claim_ids": []},
            {"speaker": "HOST_B", "dialogue_act": "acknowledgement", "text": "先回应已经提出的问题。", "claim_ids": []},
        ],
        {},
        {},
        last_speaker="HOST_B",
        existing_turns=existing,
        language="zh-CN",
        expected_count=2,
    )
    assert len(valid) == 1
    assert valid[0]["dialogue_act"] == "acknowledgement"
    assert any("连续问句过多" in issue for issue in issues)


def test_scene_validation_caps_semantic_questions_at_forty_percent() -> None:
    raw = [
        {"speaker": "HOST_A", "dialogue_act": "question", "text": "我们先问清楚第一个问题究竟是什么？", "claim_ids": []},
        {"speaker": "HOST_B", "dialogue_act": "acknowledgement", "text": "先回应第一个问题。", "claim_ids": []},
        {"speaker": "HOST_A", "dialogue_act": "acknowledgement", "text": "这一轮标签是回应但文本仍在提问？", "claim_ids": []},
        {"speaker": "HOST_B", "dialogue_act": "acknowledgement", "text": "再回应第二个问题。", "claim_ids": []},
        {"speaker": "HOST_A", "dialogue_act": "question", "text": "第三个问题会超过比例吗？", "claim_ids": []},
    ]
    valid, issues = podcast.validate_scene_turns(
        raw, {}, {}, last_speaker=None, existing_turns=[], language="zh-CN", expected_count=5
    )
    assert len(valid) == 4
    assert sum(podcast._is_question_turn(turn) for turn in valid) == 2
    assert any("问句比例超过 40%" in issue for issue in issues)


def test_scene_validation_can_infer_claim_from_supporting_evidence() -> None:
    claims = {
        "C1": {"id": "C1", "text": "公开排序", "evidence_ids": ["E1"]},
        "C2": {"id": "C2", "text": "发行规则", "evidence_ids": ["E2"]},
    }
    cards = {
        "E1": {"id": "E1", "content": "所有参与者都可以沿公开记录核对交易先后关系并排除冲突记录。"},
        "E2": {"id": "E2", "content": "新增单位的发行遵循另一套资料内规则。"},
    }
    valid, issues = podcast.validate_scene_turns(
        [{
            "speaker": "HOST_A", "dialogue_act": "explain",
            "text": "参与者能够根据公开记录核对交易先后，并据此排除冲突。", "claim_ids": [],
        }],
        claims,
        cards,
        last_speaker=None,
        existing_turns=[],
        language="zh-CN",
        expected_count=1,
    )
    assert not issues
    assert valid[0]["claim_ids"] == ["C1"]
    assert valid[0]["claim_id_inferred"] is True


def test_scene_validation_uses_only_declared_slot_claim_as_last_resort() -> None:
    claims = {
        "C1": {"id": "C1", "text": "第一项资料结论", "evidence_ids": ["E1"]},
        "C2": {"id": "C2", "text": "第二项资料结论", "evidence_ids": ["E2"]},
    }
    cards = {
        "E1": {"id": "E1", "content": "第一项资料结论"},
        "E2": {"id": "E2", "content": "第二项资料结论"},
    }
    valid, issues = podcast.validate_scene_turns(
        [{"speaker": "HOST_A", "dialogue_act": "synthesis", "text": "这一步形成了需要继续检验的阶段性判断。", "claim_ids": []}],
        claims,
        cards,
        last_speaker=None,
        existing_turns=[],
        language="zh-CN",
        expected_count=1,
        default_claim_ids=["C2"],
    )
    assert not issues
    assert valid[0]["claim_ids"] == ["C2"]
    assert valid[0]["claim_id_source"] == "slot"


def test_scene_validation_uses_language_specific_maximums() -> None:
    claims = {"C1": {"id": "C1", "text": "word " * 100, "evidence_ids": ["E1"]}}
    cards = {"E1": {"id": "E1", "content": "word " * 100}}
    valid, issues = podcast.validate_scene_turns(
        [{"speaker": "HOST_A", "dialogue_act": "explain", "text": "word " * 91, "claim_ids": ["C1"]}],
        claims,
        cards,
        last_speaker=None,
        existing_turns=[],
        language="en",
        expected_count=1,
    )
    assert not valid
    assert "长度不合格" in issues[0]


def test_extract_turns_recovers_complete_objects_from_truncated_json() -> None:
    raw = '{"turns":[{"speaker":"HOST_A","text":"完整一轮","claim_ids":[]},{"speaker":"HOST_B","text":"未完成'
    assert podcast._extract_turns(raw) == [{"speaker": "HOST_A", "text": "完整一轮", "claim_ids": []}]


def test_extract_turns_accepts_compact_and_truncated_tuples() -> None:
    raw = '{"turns":[["A","X","完整解释",["C1"]],["B","Q","继续追问",[]],["A","S","未完成"'
    assert podcast._extract_turns(raw) == [
        {"speaker": "HOST_A", "dialogue_act": "explain", "text": "完整解释", "claim_ids": ["C1"]},
        {"speaker": "HOST_B", "dialogue_act": "question", "text": "继续追问", "claim_ids": []},
    ]
    assert podcast._act_output_tokens({"maximum_units": 4000}, 20, "zh-CN") == 10000


@pytest.mark.asyncio
async def test_chapter_generation_rejects_duplicates_and_unsupported_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_budgeted_chat(builder, **kwargs):
        build = builder(PromptBudget(4096, 2600, 900, 2048, 1.0))
        return SimpleNamespace(content='{"pairs":[]}', build=build)

    monkeypatch.setattr(podcast, "budgeted_chat", fake_budgeted_chat)
    cards = {"E1": {"id": "E1", "content": "系统必须解决双重支付问题，并以工作量证明建立公开的交易顺序。"}}
    chapter = {"title": "双重支付", "purpose": "解释问题", "evidence_ids": ["E1"]}
    turns, degraded = await podcast.create_chapter_turns(chapter, cards, 6, "zh-CN", "节目开篇")
    normalized = [podcast._normalize_text(turn["text"]) for turn in turns]
    assert len(normalized) == len(set(normalized))
    assert all(turn["citation_ids"] == ["E1"] for turn in turns)
    assert degraded is True


@pytest.mark.asyncio
async def test_linked_scene_repairs_a_bad_draft_without_per_scene_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"draft": 0, "audit": 0}

    async def fake_budgeted_chat(builder, **kwargs):
        build = builder(PromptBudget(8192, 6000, 1200, 2048, 1.0))
        prompt = build.messages[0]["content"]
        if "待审场景" in prompt:
            calls["audit"] += 1
            content = json.dumps({"invalid_indexes": [], "scores": {"grounding": 5, "continuity": 5, "roles": 5, "repetition": 5}, "issues": []}, ensure_ascii=False)
        else:
            calls["draft"] += 1
            if "上次草稿问题" not in prompt:
                turns = [
                    {"speaker": "HOST_A", "dialogue_act": "question", "text": "这条材料明确说明了什么？", "claim_ids": ["C1"]},
                    {"speaker": "HOST_B", "dialogue_act": "explain", "text": "资料给出的直接线索是：公开记录。", "claim_ids": ["C1"]},
                ]
            else:
                turns = [
                    {"speaker": "HOST_A", "dialogue_act": "explain", "text": "资料说明系统通过公开记录建立可验证的交易顺序。", "claim_ids": ["C1"]},
                    {"speaker": "HOST_B", "dialogue_act": "question", "text": "顺着这个结论，验证者接下来需要核对什么？", "claim_ids": []},
                ]
            content = json.dumps({"turns": turns}, ensure_ascii=False)
        return SimpleNamespace(content=content, build=build)

    monkeypatch.setattr(podcast, "budgeted_chat", fake_budgeted_chat)
    claim = {"id": "C1", "text": "系统通过公开记录建立可验证的交易顺序。", "evidence_ids": ["E1"], "source_id": "s1", "filename": "source.md"}
    turns, audit = await podcast.create_linked_scene(
        scene_kind="chapter",
        chapter={"title": "公开顺序", "purpose": "解释验证机制", "bridge_in": "承接问题", "bridge_out": "继续讨论验证"},
        claims=[claim],
        cards_by_id={"E1": {"id": "E1", "content": claim["text"]}},
        memory=podcast.EpisodeMemory("系统如何建立可信顺序"),
        existing_turns=[],
        target=2,
        language="zh-CN",
        profile={"tier": "lite", "scene_turns": 4, "recent_turns": 4},
        trace=podcast.ContextUsage(),
    )
    assert len(turns) == 2
    assert audit["passed"] is True and audit["repaired"] is True
    assert calls == {"draft": 2, "audit": 0}


@pytest.mark.asyncio
async def test_editorial_act_retries_only_connection_setup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_draft(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise podcast.httpx.ConnectError("connection unavailable")
        return ([
            {"speaker": "HOST_A", "text": "完整推进。", "dialogue_act": "explain", "claim_ids": ["C1"], "citation_ids": ["E1"], "safe": False},
            {"speaker": "HOST_B", "text": "继续追问。", "dialogue_act": "question", "claim_ids": [], "citation_ids": [], "safe": False},
        ], [])

    monkeypatch.setattr(podcast, "_draft_scene", fake_draft)
    turns, audit = await podcast.create_linked_scene(
        scene_kind="act",
        chapter={"title": "Act"},
        claims=[],
        cards_by_id={},
        memory=podcast.EpisodeMemory("thesis"),
        existing_turns=[],
        target=2,
        language="zh-CN",
        profile={"recent_turns": 4},
        trace=podcast.ContextUsage(),
    )
    assert len(turns) == 2
    assert audit["passed"] is True
    assert calls == 2


@pytest.mark.asyncio
async def test_editorial_act_retries_one_zero_turn_provider_response(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_draft(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return podcast.SceneDraftResult([], ["模型没有返回可解析的 turns 数组（结束原因：stop）"], "stop")
        return podcast.SceneDraftResult([
            {"speaker": "HOST_A", "text": "完整解释受支持的前提。", "dialogue_act": "explain", "claim_ids": ["C1"], "citation_ids": ["E1"], "safe": False},
            {"speaker": "HOST_B", "text": "继续检验受支持的限制。", "dialogue_act": "challenge", "claim_ids": ["C1"], "citation_ids": ["E1"], "safe": False},
        ], [], "stop")

    monkeypatch.setattr(podcast, "_draft_scene", fake_draft)
    state = podcast.EpisodeGenerationState()
    turns, audit = await podcast.create_linked_scene(
        scene_kind="act", chapter={"title": "Act"}, claims=[], cards_by_id={},
        memory=podcast.EpisodeMemory("thesis"), existing_turns=[], target=2, language="zh-CN",
        profile={"recent_turns": 4}, trace=podcast.ContextUsage(), generation_state=state,
    )
    assert len(turns) == 2 and audit["passed"] is True
    assert calls == 2
    assert state.empty_response_retry_used is True


@pytest.mark.asyncio
async def test_editorial_act_rejects_partial_recovered_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def partial_draft(**kwargs):
        return ([
            {"speaker": "HOST_A", "text": "只恢复出一轮有效内容。", "dialogue_act": "explain", "claim_ids": ["C1"], "citation_ids": ["E1"], "safe": False},
        ], ["有效轮次不足：1/3"])

    monkeypatch.setattr(podcast, "_draft_scene", partial_draft)
    with pytest.raises(podcast.PodcastQualityError) as captured:
        await podcast.create_linked_scene(
            scene_kind="act",
            chapter={"title": "Act"},
            claims=[],
            cards_by_id={},
            memory=podcast.EpisodeMemory("thesis"),
            existing_turns=[],
            target=4,
            language="zh-CN",
            profile={"recent_turns": 4},
            trace=podcast.ContextUsage(),
        )
    assert captured.value.report["target_turns"] == 4
    assert captured.value.report["accepted_turns"] == 1


@pytest.mark.asyncio
async def test_editorial_act_keeps_target_minus_one_safe_turns_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    async def nearly_complete(**kwargs):
        return podcast.SceneDraftResult([
            {"speaker": "HOST_A", "text": "第一轮建立受支持的前提。", "dialogue_act": "explain", "claim_ids": ["C1"], "citation_ids": ["E1"], "safe": False},
            {"speaker": "HOST_B", "text": "第二轮继续检验这个前提。", "dialogue_act": "challenge", "claim_ids": ["C1"], "citation_ids": ["E1"], "safe": False},
            {"speaker": "HOST_A", "text": "第三轮给出受支持的综合。", "dialogue_act": "synthesis", "claim_ids": ["C1"], "citation_ids": ["E1"], "safe": False},
        ], ["第 4 轮包含事实但没有 claim_id"], "stop")

    monkeypatch.setattr(podcast, "_draft_scene", nearly_complete)
    turns, audit = await podcast.create_linked_scene(
        scene_kind="act",
        chapter={"title": "Act"},
        claims=[],
        cards_by_id={},
        memory=podcast.EpisodeMemory("thesis"),
        existing_turns=[],
        target=4,
        language="zh-CN",
        profile={"recent_turns": 4},
        trace=podcast.ContextUsage(),
        generation_state=podcast.EpisodeGenerationState(),
    )
    assert len(turns) == 3
    assert audit["partial"] is True
    assert audit["deterministic_issues"] == ["第 4 轮包含事实但没有 claim_id"]


@pytest.mark.asyncio
async def test_editorial_act_uses_only_one_length_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    continuation_calls = 0

    async def partial_draft(**kwargs):
        return podcast.SceneDraftResult([
            {"speaker": "HOST_A", "text": "第一轮已经建立了一个受资料支持的完整前提。", "dialogue_act": "explain", "claim_ids": ["C1"], "citation_ids": ["E1"], "safe": False},
        ], ["有效轮次不足：1/1"], "length")

    async def continuation(**kwargs):
        nonlocal continuation_calls
        continuation_calls += 1
        return podcast.SceneDraftResult([
            {"speaker": "HOST_B", "text": "第二轮顺着前提继续说明限制与含义。", "dialogue_act": "synthesis", "claim_ids": ["C1"], "citation_ids": ["E1"], "safe": False},
        ], [], "stop")

    monkeypatch.setattr(podcast, "_draft_scene", partial_draft)
    monkeypatch.setattr(podcast, "_continue_scene", continuation)
    state = podcast.EpisodeGenerationState()
    kwargs = dict(
        scene_kind="act",
        chapter={"title": "Act"},
        claims=[],
        cards_by_id={},
        memory=podcast.EpisodeMemory("thesis"),
        existing_turns=[],
        target=2,
        language="zh-CN",
        profile={"recent_turns": 4},
        trace=podcast.ContextUsage(),
        generation_state=state,
    )
    turns, audit = await podcast.create_linked_scene(**kwargs)
    assert len(turns) == 2 and audit["continuation_used"] is True
    assert state.continuation_used is True and continuation_calls == 1
    with pytest.raises(podcast.PodcastQualityError):
        await podcast.create_linked_scene(**kwargs)
    assert continuation_calls == 1


@pytest.mark.asyncio
async def test_editorial_act_continues_severely_short_stop_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def short_stop(**kwargs):
        return podcast.SceneDraftResult([
            {"speaker": "HOST_A", "text": "先建立一个受支持的前提。", "dialogue_act": "explain", "claim_ids": ["C1"], "citation_ids": ["E1"], "safe": False},
        ], ["有效轮次不足：1/2"], "stop")

    async def continuation(**kwargs):
        return podcast.SceneDraftResult([
            {"speaker": "HOST_B", "text": "再检验前提的限制。", "dialogue_act": "challenge", "claim_ids": ["C1"], "citation_ids": ["E1"], "safe": False},
        ], ["第 2 轮长度不合格"], "stop")

    monkeypatch.setattr(podcast, "_draft_scene", short_stop)
    monkeypatch.setattr(podcast, "_continue_scene", continuation)
    state = podcast.EpisodeGenerationState()
    turns, audit = await podcast.create_linked_scene(
        scene_kind="act", chapter={"title": "Act"}, claims=[], cards_by_id={},
        memory=podcast.EpisodeMemory("thesis"), existing_turns=[], target=3, language="zh-CN",
        profile={"recent_turns": 4}, trace=podcast.ContextUsage(), generation_state=state,
    )
    assert len(turns) == 2
    assert audit["continuation_used"] is True
    assert audit["partial"] is True
    assert state.recovery_kind == "length_continuation"


@pytest.mark.asyncio
async def test_question_filter_shortfall_does_not_spend_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    turns = [
        {
            "speaker": "HOST_A" if index % 2 == 0 else "HOST_B",
            "text": f"第 {index + 1} 轮保留的实质内容。",
            "dialogue_act": "explain",
            "claim_ids": ["C1"],
            "citation_ids": ["E1"],
            "safe": False,
        }
        for index in range(15)
    ]

    async def filtered_draft(**kwargs):
        return podcast.SceneDraftResult(
            turns,
            ["第 16 轮使本 Act 问句比例超过 40%", "有效轮次不足：15/16"],
            "stop",
        )

    async def unexpected_continuation(**kwargs):
        raise AssertionError("不应为本地问句过滤花费续写请求")

    monkeypatch.setattr(podcast, "_draft_scene", filtered_draft)
    monkeypatch.setattr(podcast, "_continue_scene", unexpected_continuation)
    result, audit = await podcast.create_linked_scene(
        scene_kind="act",
        chapter={"title": "Act"},
        claims=[],
        cards_by_id={},
        memory=podcast.EpisodeMemory("thesis"),
        existing_turns=[],
        target=17,
        language="zh-CN",
        profile={"recent_turns": 4},
        trace=podcast.ContextUsage(),
        generation_state=podcast.EpisodeGenerationState(),
    )
    assert len(result) == 15
    assert audit["partial"] is True
    assert audit["continuation_used"] is False


@pytest.mark.asyncio
async def test_duration_expansion_uses_grounded_replacements_once(monkeypatch: pytest.MonkeyPatch) -> None:
    claims = {
        "C1": {"id": "C1", "text": "公开记录让参与者验证交易顺序。", "evidence_ids": ["E1"]},
        "C2": {"id": "C2", "text": "参与者依据同一记录排除冲突交易。", "evidence_ids": ["E2"]},
    }
    cards = {
        "E1": {"id": "E1", "content": claims["C1"]["text"]},
        "E2": {"id": "E2", "content": claims["C2"]["text"]},
    }
    turns = [
        {"speaker": "HOST_A", "text": "公开记录支持验证。", "dialogue_act": "explain", "claim_ids": ["C1"], "citation_ids": ["E1"]},
        {"speaker": "HOST_B", "text": "冲突交易需要排除。", "dialogue_act": "challenge", "claim_ids": ["C2"], "citation_ids": ["E2"]},
    ]
    chapters = [{"turn_start": 0, "turn_end": 1}]

    async def fake_chat(builder, **kwargs):
        build = builder(PromptBudget(16_000, 12_000, 4000, 2048, 1.0))
        replacements = []
        for item in build.metadata["items"]:
            stem = "公开记录验证交易顺序机制参与者复核冲突结果"
            text = (stem * 10)[: item["minimum_units"]]
            replacements.append([item["index"], text])
        return SimpleNamespace(content=json.dumps({"replacements": replacements}, ensure_ascii=False), build=build, finish_reason="stop")

    monkeypatch.setattr(podcast, "budgeted_chat", fake_chat)
    state = podcast.EpisodeGenerationState()
    expanded, report = await podcast._expand_episode_duration(
        turns, chapters, claims, cards, "zh-CN", 0.30, podcast.ContextUsage(), state
    )
    assert report["used"] is True
    assert report["after_minutes"] >= 0.30
    assert state.recovery_kind == "duration_expansion"
    assert expanded != turns

    with pytest.raises(podcast.PodcastQualityError, match="唯一恢复槽"):
        await podcast._expand_episode_duration(
            turns, chapters, claims, cards, "zh-CN", 0.30, podcast.ContextUsage(), state
        )


@pytest.mark.asyncio
async def test_duration_compression_preserves_questions_and_uses_recovery_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_texts = [
        "公开记录让参与者验证交易顺序与一致结果",
        "时间戳结构把历史条目按可复核顺序连接",
        "参与者依据共同记录识别并排除冲突交易",
        "验证过程让不同参与者检查相同历史状态",
    ]
    claims = {
        f"C{index}": {"id": f"C{index}", "text": text, "evidence_ids": [f"E{index}"]}
        for index, text in enumerate(claim_texts, start=1)
    }
    cards = {
        f"E{index}": {"id": f"E{index}", "content": text}
        for index, text in enumerate(claim_texts, start=1)
    }
    turns = []
    for index, text in enumerate(claim_texts, start=1):
        turns.append({
            "speaker": "HOST_A" if index % 2 else "HOST_B",
            "text": (text * 20)[:200],
            "dialogue_act": "explain",
            "claim_ids": [f"C{index}"],
            "citation_ids": [f"E{index}"],
        })
    question = {
        "speaker": "HOST_A", "text": "那么共同记录究竟怎样帮助参与者复核？",
        "dialogue_act": "question", "claim_ids": [], "citation_ids": [],
    }
    turns.insert(2, question)
    chapters = [{"turn_start": 0, "turn_end": 2}, {"turn_start": 3, "turn_end": 4}]

    async def fake_chat(builder, **kwargs):
        assert kwargs["max_tokens"] == 10_000
        build = builder(PromptBudget(16_000, 12_000, 4000, 2048, 1.0))
        replacements = []
        for item in build.metadata["items"]:
            source = claims[item["claim_ids"][0]]["text"]
            text = (source * 20)[: item["safe_minimum_units"]]
            replacements.append([item["index"], text])
        return SimpleNamespace(
            content=json.dumps({"replacements": replacements}, ensure_ascii=False),
            build=build,
            finish_reason="stop",
        )

    monkeypatch.setattr(podcast, "budgeted_chat", fake_chat)
    state = podcast.EpisodeGenerationState()
    compressed, report = await podcast._compress_episode_duration(
        turns, chapters, claims, cards, "zh-CN", 2.0, podcast.ContextUsage(), state
    )
    assert report["used"] is True
    assert report["after_minutes"] <= 2.4
    assert any(item["actual_units"] < item["requested_maximum_units"] for item in report["unit_results"])
    assert all(item["minimum_units"] >= item["requested_maximum_units"] * 0.80 - 1 for item in report["unit_results"])
    assert compressed[2] == question
    assert state.recovery_kind == "duration_compression"

    with pytest.raises(podcast.PodcastQualityError, match="唯一恢复槽"):
        await podcast._compress_episode_duration(
            turns, chapters, claims, cards, "zh-CN", 2.0, podcast.ContextUsage(), state
        )


def test_recovery_reserves_final_episode_audit_without_exceeding_hard_cap() -> None:
    trace = podcast.ContextUsage(total_token_limit=27_500)
    podcast._reserve_episode_audit_after_recovery(trace)
    assert trace.total_token_limit == 35_500

    capped = podcast.ContextUsage(total_token_limit=42_000)
    podcast._reserve_episode_audit_after_recovery(capped)
    assert capped.total_token_limit == 45_000


@pytest.mark.asyncio
async def test_build_podcast_script_emits_v4_editorial_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    cards = [
        {"id": "E1", "source_id": "s1", "filename": "source.md", "locator": {"section": "一"}, "content": "公开记录支持验证交易顺序。"},
        {"id": "E2", "source_id": "s1", "filename": "source.md", "locator": {"section": "二"}, "content": "参与者可以复核记录中的先后关系。"},
    ]
    citations = [
        {"id": card["id"], "source_id": "s1", "chunk_id": f"chunk-{index}", "filename": "source.md", "locator": card["locator"], "quote": card["content"]}
        for index, card in enumerate(cards, start=1)
    ]
    claims = [
        {"id": "C1", "text": cards[0]["content"], "evidence_ids": ["E1"], "source_id": "s1", "filename": "source.md"},
        {"id": "C2", "text": cards[1]["content"], "evidence_ids": ["E2"], "source_id": "s1", "filename": "source.md"},
    ]
    plan = {
        "episode_thesis": "公开记录如何支持复核",
        "chapters": [
            {"id": "chapter_1", "title": "建立顺序", "purpose": "解释顺序", "claim_ids": ["C1"], "bridge_in": "", "bridge_out": "转向复核"},
            {"id": "chapter_2", "title": "完成复核", "purpose": "解释复核", "claim_ids": ["C2"], "bridge_in": "承接顺序", "bridge_out": ""},
        ],
        "fallback": False,
    }
    monkeypatch.setattr(podcast, "source_scope", lambda *args: ["s1"])
    monkeypatch.setattr(podcast, "resolve_podcast_language", lambda *args: "zh-CN")
    monkeypatch.setattr(podcast, "select_podcast_evidence", lambda *args: [{}])
    monkeypatch.setattr(podcast, "build_evidence_cards", lambda rows: (cards, citations))
    monkeypatch.setattr(podcast, "build_claim_ledger", lambda values: claims)
    monkeypatch.setattr(podcast, "create_episode_plan", lambda *args: _async_value((plan, False)))
    monkeypatch.setattr(podcast, "podcast_generation_profile", lambda: {"tier": "lite", "model": "test-main", "scene_turns": 4, "recent_turns": 4})
    monkeypatch.setattr(podcast, "scope_hash", lambda source_ids: "test-scope")

    counter = {"value": 0}

    async def linked_scene(**kwargs):
        result = []
        speaker = "HOST_B" if kwargs["memory"].last_speaker == "HOST_A" else "HOST_A"
        available = kwargs["claims"] or claims
        for _ in range(kwargs["target"]):
            counter["value"] += 1
            claim = available[(counter["value"] - 1) % len(available)]
            result.append({
                "speaker": speaker,
                "text": f"第{counter['value']}轮继续解释资料中可验证的关系与当前问题。",
                "dialogue_act": "explain",
                "claim_ids": [claim["id"]],
                "citation_ids": claim["evidence_ids"],
                "safe": False,
            })
            speaker = "HOST_B" if speaker == "HOST_A" else "HOST_A"
        return result, {"passed": True, "scores": {"grounding": 5}, "repaired": False}

    monkeypatch.setattr(podcast, "create_linked_scene", linked_scene)
    monkeypatch.setattr(
        podcast,
        "_expand_episode_duration",
        lambda turns, *args, **kwargs: _async_value((turns, {"used": False, "test_bypass": True})),
    )
    monkeypatch.setattr(podcast, "_audit_episode", lambda *args: _async_value({"passed": True, "scores": {"coherence": 5}, "invalid_boundaries": [], "issues": []}))
    monkeypatch.setattr(podcast, "_quality_metrics_v3", lambda *args: {"passed": True, "estimated_minutes": 5.0})
    ready_acts: list[dict] = []
    result = await podcast.build_podcast_script(
        "n1", {"source_ids": ["s1"], "minutes": 5, "duration_mode": "fixed"},
        act_ready=ready_acts.append,
    )
    assert result["version"] == 4
    assert result["engine"]["strategy"] == "editorial_acts"
    assert result["quality_report"]["passed"] is True
    assert result["chapters"][0]["turn_start"] < result["chapters"][1]["turn_start"]
    assert {citation["id"] for citation in result["citations"]} == {"S1", "S2"}
    assert [item["start_index"] for item in ready_acts] == [0, 9]
    assert all(item["language"] == "zh-CN" and item["turns"] for item in ready_acts)


def test_podcast_overlap_requires_distinct_remote_main_and_audio_hosts() -> None:
    assert jobs._podcast_overlap_safe(
        {"base_url": "https://main.example.com"}, {"base_url": "http://audio.lan:20810"},
    )
    assert not jobs._podcast_overlap_safe(
        {"base_url": "http://127.0.0.1:11434"}, {"base_url": "http://audio.lan:20810"},
    )
    assert not jobs._podcast_overlap_safe(
        {"base_url": "https://shared.example.com:11434/api"}, {"base_url": "https://shared.example.com:20810/audio"},
    )


@pytest.mark.asyncio
async def test_objective_gate_skips_episode_audit_for_short_script(monkeypatch: pytest.MonkeyPatch) -> None:
    cards = [
        {"id": "E1", "source_id": "s1", "filename": "source.md", "locator": {}, "content": "公开记录支持验证交易顺序。"},
        {"id": "E2", "source_id": "s1", "filename": "source.md", "locator": {}, "content": "参与者可以复核记录。"},
    ]
    citations = [
        {"id": card["id"], "source_id": "s1", "chunk_id": card["id"], "filename": "source.md", "locator": {}, "quote": card["content"]}
        for card in cards
    ]
    claims = [
        {"id": f"C{index}", "text": card["content"], "evidence_ids": [card["id"]], "source_id": "s1", "filename": "source.md"}
        for index, card in enumerate(cards, start=1)
    ]
    plan = {
        "episode_thesis": "公开记录如何支持复核",
        "chapters": [
            {"id": "chapter_1", "title": "顺序", "purpose": "解释顺序", "claim_ids": ["C1"], "bridge_in": "", "bridge_out": "复核"},
            {"id": "chapter_2", "title": "复核", "purpose": "解释复核", "claim_ids": ["C2"], "bridge_in": "顺序", "bridge_out": ""},
        ],
    }
    monkeypatch.setattr(podcast, "source_scope", lambda *args: ["s1"])
    monkeypatch.setattr(podcast, "resolve_output_language", lambda *args: ("zh-CN", {"source": "test"}))
    monkeypatch.setattr(podcast, "select_podcast_evidence", lambda *args: [{}])
    monkeypatch.setattr(podcast, "build_evidence_cards", lambda rows: (cards, citations))
    monkeypatch.setattr(podcast, "build_claim_ledger", lambda values: claims)
    monkeypatch.setattr(podcast, "create_episode_plan", lambda *args: _async_value((plan, False)))
    monkeypatch.setattr(podcast, "podcast_generation_profile", lambda: {"tier": "lite", "scene_turns": 24, "recent_turns": 4})

    async def short_scene(**kwargs):
        speaker = "HOST_B" if kwargs["memory"].last_speaker == "HOST_A" else "HOST_A"
        claim = kwargs["claims"][0]
        turns = []
        for _ in range(kwargs["target"]):
            turns.append({
                "speaker": speaker,
                "text": "简短解释资料。",
                "dialogue_act": "explain",
                "claim_ids": [claim["id"]],
                "citation_ids": claim["evidence_ids"],
                "safe": False,
            })
            speaker = "HOST_B" if speaker == "HOST_A" else "HOST_A"
        return turns, {"passed": True, "repaired": False, "duration": kwargs["duration_budget"]}

    audit_calls = 0

    async def forbidden_audit(*args):
        nonlocal audit_calls
        audit_calls += 1
        return {"passed": True}

    monkeypatch.setattr(podcast, "create_linked_scene", short_scene)
    monkeypatch.setattr(
        podcast,
        "_expand_episode_duration",
        lambda turns, *args, **kwargs: _async_value((turns, {"used": False, "test_bypass": True})),
    )
    monkeypatch.setattr(podcast, "_audit_episode", forbidden_audit)
    with pytest.raises(podcast.PodcastQualityError) as captured:
        await podcast.build_podcast_script("n1", {"source_ids": ["s1"], "minutes": 5, "duration_mode": "fixed"})
    assert audit_calls == 0
    assert captured.value.report["episode_audit"]["skipped"] is True
    assert captured.value.report["context_usage"]["stop_reason"] == "deterministic_quality_gate"


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_episode_audit_rejects_fail_verdict_even_with_four_point_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_chat(builder, **kwargs):
        build = builder(PromptBudget(8192, 6000, 1500, 2048, 1.0))
        content = json.dumps({
            "verdict": "fail",
            "scores": {"grounding": 5, "coherence": 4, "depth": 4, "roles": 5, "repetition": 5, "completeness": 5},
            "invalid_boundaries": [],
            "issues": ["A minor bridge could be more explicit."],
        })
        return SimpleNamespace(content=content, build=build)

    monkeypatch.setattr(podcast, "budgeted_chat", fake_chat)
    turns = [
        {"speaker": "HOST_A", "text": "First grounded point."},
        {"speaker": "HOST_B", "text": "Second grounded point."},
    ]
    chapters = [
        {"title": "One", "turn_start": 0, "turn_end": 0},
        {"title": "Two", "turn_start": 1, "turn_end": 1},
    ]
    audit = await podcast._audit_episode(turns, chapters, "Thesis", "en", podcast.ContextUsage())
    assert audit["passed"] is False
    assert audit["issues"]


@pytest.mark.asyncio
async def test_episode_audit_keeps_publishable_notes_non_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_chat(builder, **kwargs):
        build = builder(PromptBudget(8192, 6000, 1500, 2048, 1.0))
        return SimpleNamespace(content=json.dumps({
            "verdict": "pass",
            "scores": {name: 4 for name in ("grounding", "coherence", "depth", "roles", "repetition", "completeness")},
            "invalid_boundaries": [],
            "blocking_issues": [],
            "notes": ["One transition could be polished."],
        }), build=build, finish_reason="stop")

    monkeypatch.setattr(podcast, "budgeted_chat", fake_chat)
    audit = await podcast._audit_episode(
        [{"speaker": "HOST_A", "text": "Grounded point.", "claim_ids": []}],
        [{"turn_start": 0, "turn_end": 0}],
        "Thesis",
        "en",
        podcast.ContextUsage(),
    )
    assert audit["passed"] is True
    assert audit["issues"] == []
    assert audit["notes"] == ["One transition could be polished."]


@pytest.mark.asyncio
async def test_episode_audit_rejects_incomplete_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_chat(builder, **kwargs):
        build = builder(PromptBudget(8192, 6000, 1500, 2048, 1.0))
        return SimpleNamespace(content="{}", build=build, finish_reason="length")

    monkeypatch.setattr(podcast, "budgeted_chat", fake_chat)
    audit = await podcast._audit_episode(
        [{"speaker": "HOST_A", "text": "Grounded point."}],
        [{"title": "One", "turn_start": 0, "turn_end": 0}],
        "Thesis",
        "en",
        podcast.ContextUsage(),
    )
    assert audit["passed"] is False
    assert audit["verdict"] == "incomplete"
    assert "token 上限" in audit["issues"][0]


@pytest.mark.asyncio
async def test_episode_audit_samples_each_act_instead_of_resending_full_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    async def fake_chat(builder, **kwargs):
        build = builder(PromptBudget(8192, 6000, 1500, 2048, 1.0))
        captured["prompt"] = build.messages[0]["content"]
        captured["max_tokens"] = kwargs["max_tokens"]
        return SimpleNamespace(content=json.dumps({
            "verdict": "pass",
            "scores": {name: 5 for name in ("grounding", "coherence", "depth", "roles", "repetition", "completeness")},
            "invalid_boundaries": [],
            "issues": [],
        }), build=build, finish_reason="stop")

    monkeypatch.setattr(podcast, "budgeted_chat", fake_chat)
    turns = [
        {"speaker": "HOST_A" if index % 2 == 0 else "HOST_B", "text": f"unique-{index}", "claim_ids": []}
        for index in range(10)
    ]
    audit = await podcast._audit_episode(
        turns,
        [{"turn_start": 0, "turn_end": 4}, {"turn_start": 5, "turn_end": 9}],
        "Thesis",
        "en",
        podcast.ContextUsage(),
    )
    assert audit["passed"] is True
    assert captured["max_tokens"] == podcast.structured_output_tokens(2400)
    for index in (0, 2, 4, 5, 7, 9):
        assert f"unique-{index}" in captured["prompt"]
    for index in (1, 3, 6, 8):
        assert f"unique-{index}" not in captured["prompt"]


def test_v3_quality_gate_rejects_underlength_episode() -> None:
    turns = [
        {"speaker": "HOST_A", "text": "简短开场。", "dialogue_act": "intro", "claim_ids": [], "citation_ids": []},
        {"speaker": "HOST_B", "text": "简短回应。", "dialogue_act": "outro", "claim_ids": [], "citation_ids": []},
    ]
    report = podcast._quality_metrics_v3(
        turns,
        [],
        5,
        30,
        {"passed": True, "scores": {"grounding": 5, "coherence": 5, "roles": 5, "repetition": 5, "completeness": 5}},
        [],
        ["s1"],
    )
    assert report["duration_ratio"] < 0.85
    assert report["passed"] is False


@pytest.mark.asyncio
async def test_quality_failure_stops_before_tts_and_persists_report(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = tmp_path / "job-work"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(jobs, "PATHS", SimpleNamespace(job_work=work, artifacts=artifacts, root=tmp_path))
    monkeypatch.setattr(jobs, "active_provider", lambda role: {"name": "TTS", "model": "tts", "config": {}, "capabilities": {}})
    monkeypatch.setattr(jobs, "audio_provider_readiness", lambda provider: (True, "ready"))
    monkeypatch.setattr(jobs, "register_resource", lambda *args: None)
    monkeypatch.setattr(jobs, "DB", SimpleNamespace(fetchone=lambda *args: {"cancel_requested": 0}, execute=lambda *args: None))

    async def fail_script(*args, **kwargs):
        raise podcast.PodcastQualityError("跨章不连贯", {"passed": False, "stage": "episode"})

    synthesized: list[str] = []

    async def fake_synthesize(*args, **kwargs):
        synthesized.append("called")

    monkeypatch.setattr(jobs, "build_podcast_script", fail_script)
    monkeypatch.setattr(jobs, "synthesize", fake_synthesize)
    with pytest.raises(RuntimeError, match="未通过质量门槛"):
        await jobs._podcast("n1", {"source_ids": ["s1"], "minutes": 5}, "job_quality")
    manifest = json.loads((work / "job_quality" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 4
    assert manifest["quality_failure"]["stage"] == "episode"
    assert not synthesized


@pytest.mark.asyncio
async def test_episode_audit_reserves_reasoning_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    async def fake_budgeted_chat(builder, **kwargs):
        captured.update(kwargs)
        build = builder(PromptBudget(8192, 6000, 1200, 2048, 1.0))
        content = json.dumps({
            "verdict": "pass",
            "scores": {"grounding": 5, "coherence": 5, "depth": 5, "roles": 5, "repetition": 5, "completeness": 5},
            "invalid_boundaries": [],
            "blocking_issues": [],
        }, ensure_ascii=False)
        return SimpleNamespace(content=content, build=build)

    monkeypatch.setattr(podcast, "budgeted_chat", fake_budgeted_chat)
    turns = _gate_ready_turns(12)
    chapters = [{"id": "chapter_1", "turn_start": 0, "turn_end": 11}]
    result = await podcast._audit_episode(turns, chapters, "命题", "zh-CN", podcast.ContextUsage())
    assert result["passed"] is True
    # 推理模型的隐藏 reasoning 会占用 max_tokens：审计必须像其他结构化调用一样预留余量
    assert captured["max_tokens"] >= 2400 + 1024


def _gate_ready_turns(count: int = 30) -> list[dict]:
    # 三个槽位取互质周期（6/5/7），保证 30 轮内没有两轮同时命中两个槽位，
    # 使 _similar 的 SequenceMatcher 比率稳定在 0.84 重复阈值以下
    topics = ["信任机制", "层次结构", "网络协议", "证明过程", "历史背景", "符号系统"]
    angles = ["从成本角度剖析", "用具体案例说明", "对照替代方案比较", "拆开前提逐层验证", "换成反方立场审视"]
    endings = ["并给出对应的实例", "再补充一个反例", "同时交代适用条件", "由此引出后续讨论", "并指出常见误读", "最后落到实际影响", "顺手厘清相邻概念"]
    turns: list[dict] = []
    for index in range(count):
        is_question = index % 5 == 2
        text = (
            f"主持人就{topics[index % 6]}的第{index}点{angles[index % 5]}，{endings[index % 7]}，"
            f"并进一步交代这一点的适用条件与常见误解（材料第{index}段）"
            + ("？" if is_question else "。")
        )
        turns.append({
            "speaker": "HOST_A" if index % 2 == 0 else "HOST_B",
            "text": text,
            "dialogue_act": "question" if is_question else "explain",
            "claim_ids": [] if is_question else ["C1"],
            "citation_ids": [] if is_question else ["S1"],
        })
    return turns


def _gate_chapters(count: int = 30, acts: int = 3) -> list[dict]:
    size = count // acts
    return [
        {"id": f"chapter_{index + 1}", "turn_start": index * size, "turn_end": (index + 1) * size - 1}
        for index in range(acts)
    ]


def _run_quality_gate(turns: list[dict], chapters: list[dict] | None = None) -> dict:
    return podcast._quality_metrics_v3(
        turns,
        [{"source_id": "s1", "id": "S1"}],
        6,
        30,
        {"passed": True, "scores": {"grounding": 5, "coherence": 5, "depth": 5, "roles": 5, "repetition": 5, "completeness": 5}},
        [],
        ["s1"],
        None,
        chapters,
    )


def test_cliche_gate_passes_clean_episode() -> None:
    report = _run_quality_gate(_gate_ready_turns(), _gate_chapters())
    assert report["deterministic_passed"] is True
    assert report["cliche_family_density"] == 0
    assert report["cliche_family_counts"] == {"audit_negation": 0, "recap_meta": 0, "next_layer": 0, "boundary_meta": 0}
    assert len(report["cliche_act_density"]) == 3


def test_cliche_gate_rejects_dense_episode() -> None:
    turns = _gate_ready_turns()
    for index in (0, 5, 10, 15, 20, 25):
        turns[index]["text"] += "这里先把第一主张压实，再回扣开篇问题。"
    report = _run_quality_gate(turns, _gate_chapters())
    assert report["cliche_family_counts"]["recap_meta"] == 12
    assert report["deterministic_passed"] is False
    reasons = podcast._deterministic_failure_reasons(report)
    assert any("套话" in reason and "recap_meta" in reason for reason in reasons)


def test_cliche_gate_family_count_boundary() -> None:
    turns = _gate_ready_turns()
    for index in (0, 5, 10):
        turns[index]["text"] += "这一点只支持到这里。"
    report = _run_quality_gate(turns, _gate_chapters())
    # 3 次命中：密度 0.10、单族 3，恰好都在阈值上，应通过
    assert report["cliche_family_density"] == 0.1
    assert report["cliche_max_family_count"] == 3
    assert report["deterministic_passed"] is True
    turns[15]["text"] += "这里不能推出更多结论。"
    report = _run_quality_gate(turns, _gate_chapters())
    assert report["cliche_max_family_count"] == 4
    assert report["deterministic_passed"] is False


def test_cliche_gate_rejects_single_act_cluster() -> None:
    turns = _gate_ready_turns()
    # 全部命中集中在第一个 Act：整集密度与单族计数都在阈值内，但单 Act 密度 0.3 超限
    for index in (0, 1, 2):
        turns[index]["text"] += "下一层问题自然冒出来。"
    report = _run_quality_gate(turns, _gate_chapters())
    assert report["cliche_family_density"] == 0.1
    assert report["cliche_act_density"][0]["density"] == 0.3
    assert report["deterministic_passed"] is False
    assert any("单个 Act" in reason for reason in podcast._deterministic_failure_reasons(report))


def test_cliche_gate_ignores_soft_boundary_content_words() -> None:
    turns = _gate_ready_turns()
    soft_additions = [
        "这个门槛随攻击条件移动。",
        "机制边界在材料里交代得很清楚。",
        "理解的门槛并不等于失效。",
        "演示流畅不代表跨过了那道门槛。",
        "它给出的边界是双值性留下的入口。",
    ]
    for addition, index in zip(soft_additions, (0, 3, 6, 9, 12)):
        turns[index]["text"] += addition
    report = _run_quality_gate(turns, _gate_chapters())
    # 软族只统计、不参与判定：正当内容词不应导致失败
    assert report["cliche_family_counts"]["boundary_meta"] == 5
    assert report["cliche_family_density"] == 0
    assert report["deterministic_passed"] is True


def test_guard_gate_passes_clean_episode() -> None:
    report = _run_quality_gate(_gate_ready_turns(), _gate_chapters())
    assert report["deterministic_passed"] is True
    assert report["guard_family_counts"] == {"guard_disclaimer": 0, "neq_disclaimer": 0}
    assert report["guard_density"] == 0
    assert len(report["guard_act_density"]) == 3


def test_guard_gate_rejects_dense_disclaimers() -> None:
    turns = _gate_ready_turns()
    for index in (0, 5, 10, 15, 20, 25):
        turns[index]["text"] += "这里别把它读成绝对结论。"
    report = _run_quality_gate(turns, _gate_chapters())
    # 6 次命中：整集密度 0.2 超过 0.18 上限
    assert report["guard_family_counts"]["guard_disclaimer"] == 6
    assert report["guard_density"] == 0.2
    assert report["deterministic_passed"] is False
    assert any("防误读" in reason for reason in podcast._deterministic_failure_reasons(report))


def test_guard_gate_density_boundary() -> None:
    turns = _gate_ready_turns()
    for index in (0, 5, 10, 15, 20):
        turns[index]["text"] += "风险不等于消失。"
    report = _run_quality_gate(turns, _gate_chapters())
    # 5/30 = 0.167，低于 0.18 且单族 5 ≤ 10，应通过
    assert report["guard_density"] == 0.167
    assert report["deterministic_passed"] is True
    turns[25]["text"] += "速度快不意味着安全。"
    report = _run_quality_gate(turns, _gate_chapters())
    assert report["guard_density"] == 0.2
    assert report["deterministic_passed"] is False


def test_guard_regex_ignores_legitimate_reading_verbs() -> None:
    turns = _gate_ready_turns()
    # 正当用法：没有“别”字告诫语境的“读成/当成”不计入
    turns[0]["text"] += "先把这个读成工程上的顺手设计。"
    turns[1]["text"] += "可以把它当成一次架构演示来听。"
    report = _run_quality_gate(turns, _gate_chapters())
    assert report["guard_family_counts"]["guard_disclaimer"] == 0
    assert report["deterministic_passed"] is True
    turns[2]["text"] += "先别急着把它当成定论。"
    report = _run_quality_gate(turns, _gate_chapters())
    assert report["guard_family_counts"]["guard_disclaimer"] == 1


def test_guard_act_density_is_report_only() -> None:
    turns = _gate_ready_turns()
    # 命中集中在首个 Act：整集密度 0.1 与单族 3 都在限内，Act 密度 0.3 只记录不判定
    for index in (0, 1, 2):
        turns[index]["text"] += "这不意味着机制失效。"
    report = _run_quality_gate(turns, _gate_chapters())
    assert report["guard_act_density"][0]["density"] == 0.3
    assert report["guard_density"] == 0.1
    assert report["deterministic_passed"] is True



def test_cliche_metrics_english_turns_no_false_positive() -> None:
    turns = [
        {"speaker": "HOST_A", "text": "The threshold here is a real boundary condition for the mechanism.", "dialogue_act": "explain"},
        {"speaker": "HOST_B", "text": "Right, and we cannot push the claim beyond what the text supports.", "dialogue_act": "challenge"},
    ]
    metrics = podcast._cliche_family_metrics(turns)
    assert metrics["cliche_family_density"] == 0
    assert metrics["cliche_worst_family"] is None
