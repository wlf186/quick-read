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
    assert 30 <= podcast.target_turn_count(5) < podcast.target_turn_count(20) <= 200


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


def test_extract_turns_recovers_complete_objects_from_truncated_json() -> None:
    raw = '{"turns":[{"speaker":"HOST_A","text":"完整一轮","claim_ids":[]},{"speaker":"HOST_B","text":"未完成'
    assert podcast._extract_turns(raw) == [{"speaker": "HOST_A", "text": "完整一轮", "claim_ids": []}]


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
async def test_build_podcast_script_emits_v3_linked_payload(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(podcast, "_audit_episode", lambda *args: _async_value({"passed": True, "scores": {"coherence": 5}, "invalid_boundaries": [], "issues": []}))
    monkeypatch.setattr(podcast, "_quality_metrics_v3", lambda *args: {"passed": True, "estimated_minutes": 5.0})
    result = await podcast.build_podcast_script("n1", {"source_ids": ["s1"], "minutes": 5, "duration_mode": "fixed"})
    assert result["version"] == 3
    assert result["engine"]["strategy"] == "linked_scenes"
    assert result["quality_report"]["passed"] is True
    assert result["chapters"][0]["turn_start"] < result["chapters"][1]["turn_start"]
    assert {citation["id"] for citation in result["citations"]} == {"S1", "S2"}


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_episode_audit_treats_four_point_minor_notes_as_publishable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_chat(builder, **kwargs):
        build = builder(PromptBudget(8192, 6000, 1500, 2048, 1.0))
        content = json.dumps({
            "verdict": "fail",
            "scores": {"grounding": 5, "coherence": 4, "roles": 5, "repetition": 5, "completeness": 5},
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
    assert audit["passed"] is True
    assert audit["issues"]


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
    assert manifest["version"] == 3
    assert manifest["quality_failure"]["stage"] == "episode"
    assert not synthesized
