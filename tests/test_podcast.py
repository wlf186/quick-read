import json

import pytest

from sandevistan_read import podcast
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


@pytest.mark.asyncio
async def test_chapter_generation_rejects_duplicates_and_unsupported_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    generated = {
        "turns": [
            {"speaker": "A", "text": "资料提出了双重支付问题。", "citation_ids": ["E1"]},
            {"speaker": "B", "text": "攻击者拥有 51% 算力就绝对无法追上。", "citation_ids": ["E1"]},
            {"speaker": "A", "text": "资料提出了双重支付问题。", "citation_ids": ["E1"]},
        ]
    }

    async def fake_chat(messages, **kwargs):
        if "事实审校器" in messages[0]["content"]:
            return json.dumps({"invalid_indexes": []})
        return json.dumps(generated, ensure_ascii=False)

    monkeypatch.setattr(podcast, "chat", fake_chat)
    cards = {"E1": {"id": "E1", "content": "系统必须解决双重支付问题，并以工作量证明建立公开的交易顺序。"}}
    chapter = {"title": "双重支付", "purpose": "解释问题", "evidence_ids": ["E1"]}
    turns, degraded = await podcast.create_chapter_turns(chapter, cards, 6, "zh-CN", "节目开篇")
    normalized = [podcast._normalize_text(turn["text"]) for turn in turns]
    assert len(normalized) == len(set(normalized))
    assert all(turn["citation_ids"] == ["E1"] for turn in turns)
    assert degraded is True
