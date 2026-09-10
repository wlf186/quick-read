"""Narrative replacement, source reuse and rollback must not depend on live models."""
import copy
import json

import pytest

from sandevistan_read import podcast
from sandevistan_read.context_budget import ContextUsage, TokenLimits, plan_context, podcast_chapter_capacity, podcast_stage_minutes
from sandevistan_read.delivery import CURRENT, DeliveryBudget


def pair(label, chapter):
    return [{"speaker": "HOST_A" if i == 0 else "HOST_B", "dialogue_act": "explain", "text": f"{label} {i} completes a supported argument.",
             "claim_ids": ["C1"], "citation_ids": ["E1"], "source_chapter_id": chapter,
             "exchange_id": f"core/{chapter}", "exchange_start": i == 0} for i in range(2)]


def episode():
    plan = {"episode_thesis": "How does it work?", "chapter_replacement": True,
            "chapters": [{"id": key, "title": key, "purpose": key, "claim_ids": ["C1"]} for key in ["one", "two"]]}
    core = [t for key in ["opening", "one", "two", "closing"] for t in pair(key, key)]
    return plan, core


def test_replacement_is_in_place_and_same_claim_can_be_developed_twice():
    plan, core = episode()
    turns, chapters = podcast._assemble_chapter_versions(core, plan, {"one": pair("deep mechanism", "one")})
    assert [c["id"] for c in chapters] == ["opening", "one", "two", "closing"]
    assert turns[2]["text"].startswith("deep mechanism")
    assert not any(t["text"].startswith("one ") for t in turns)
    assert turns[4:] == core[4:]
    token = CURRENT.set(DeliveryBudget())
    try:
        assert len(podcast._fit_episode_chapters(plan["chapters"]*2, 10, "en")) == 2
    finally:
        CURRENT.reset(token)


def test_core_map_keeps_opening_question_and_rejects_incomplete_map():
    plan, core = episode()
    raw = {"opening": core[:2], "chapter_bodies": {"one": core[2:4], "two": core[4:6]}, "closing": core[6:]}
    raw["opening"][-1]["text"] = "How does this mechanism work?"
    raw["opening"][-1]["dialogue_act"] = "question"
    parsed = podcast._extract_turns(json.dumps(raw))
    assert len(parsed) == 8
    assert parsed[1]["text"].endswith("?")
    raw["closing"] = []
    assert podcast._extract_turns(json.dumps(raw)) is None


@pytest.mark.asyncio
async def test_failed_chapter_keeps_short_body_and_resume_never_redrafts(monkeypatch):
    plan, core = episode()
    calls, snapshots = [], []
    async def scene(**kwargs):
        key = kwargs['chapter']['id'];calls.append(key)
        if key == 'core':return copy.deepcopy(core), {'passed': True}
        if key.startswith('one/'):
            raise podcast.PodcastQualityError('one chapter failed')
        assert not any(t['text'].startswith('closing ') for t in kwargs['existing_turns'])
        assert kwargs['claims'][0]['id'] == 'C1'  # Already cited in core and first chapter.
        return pair('deeper second mechanism', 'two'), {'passed': True}
    monkeypatch.setattr(podcast, 'create_linked_scene', scene)
    args = (plan, [{'id':'C1','source_id':'s','text':'Original mechanism','evidence_ids':['E1']}], {}, 'en', 8,
            {'max_output_tokens':4096,'recent_turns':6}, ContextUsage(total_token_limit=100000),
            podcast.EpisodeGenerationState(allow_partial=True), lambda:None)
    first = await podcast._generate_complete_first(*args, checkpoint_ready=snapshots.append)
    assert calls == ['core','one/0','two/0']
    assert first[0][2:4] == core[2:4]
    assert first[0][4]['text'].startswith('deeper second')
    assert first[4]['accepted_blocks'] == 1
    again = await podcast._generate_complete_first(*args, resume=snapshots[-1])
    assert len(calls) == 3 and again[0] == first[0]
    with pytest.raises(ValueError, match='version'):
        await podcast._generate_complete_first(*args, resume={**snapshots[-1], 'version':8})


def test_audit_rollback_restores_whole_chapter_and_invalidates_changed_coverage():
    plan, core = episode()
    turns, chapters = podcast._assemble_chapter_versions(core, plan, {'one':pair('deep mechanism','one')+pair('extra condition','one')})
    audit = {'breaks':[3], 'facts':[{'index':2,'claim_id':'C1','verdict':'supported'}, {'index':6,'claim_id':'C1','verdict':'supported'}],
             'reviewed_indexes':list(range(10)), 'reviewed_transitions':list(range(9)), 'closure':{'verdict':'connected'}}
    result, updated, status = podcast._restore_reviewed_chapters(turns, chapters, audit, [])
    assert status == 'partial'
    assert result[2:4] == core[2:4]
    assert len(result) == 8 and updated[1]['development'] == 'restored'
    assert [f['index'] for f in audit['facts']] == [4]
    assert 2 not in audit['reviewed_indexes'] and 3 not in audit['reviewed_indexes']
    assert audit['closure']['verdict'] == 'uncertain'


@pytest.mark.parametrize('minutes', [5,10,14,20,22,25,30])
@pytest.mark.parametrize('window,output', [(4096,1024),(30720,4096),(204800,16384),(1000000,384000)])
def test_capacity_matches_planner_and_budget_does_not_compound(minutes,window,output):
    limits = TokenLimits.from_provider({'config':{'context_window_tokens':window,'max_output_tokens':output}})
    p = plan_context(limits,'podcast',material_tokens=1000000,minutes=minutes)
    assert p.podcast_chapters == podcast_chapter_capacity(p.core_output_tokens, minutes)
    assert p.total_token_limit <= 375000
    assert 1 <= p.podcast_chapters <= 6
    assert .3 <= podcast_stage_minutes(p.output_tokens,'en') <= 5
    assert .3 <= podcast_stage_minutes(p.output_tokens,'zh-CN') <= 5


def test_illustration_metadata_survives_without_creating_fake_citations():
    raw = pair('Imagine a simplified mechanism','one')
    raw[0]['example_kind']='illustrative'
    for t in raw:t['claim_ids']=[]
    result, _ = podcast.validate_scene_turns(raw, {}, {}, last_speaker=None, existing_turns=[], language='en', expected_count=2, allow_style_degradation=True)
    assert result[0]['example_kind']=='illustrative'
    assert result[0]['citation_ids']==[]


@pytest.mark.asyncio
async def test_cancelled_multibatch_chapter_resumes_saved_part_without_publishing_it(monkeypatch):
    plan, core = episode()
    calls, snapshots = [], []
    async def scene(**kwargs):
        key = kwargs['chapter']['id']; calls.append(key)
        return (copy.deepcopy(core) if key == 'core' else pair('Detailed '+key, key)), {'passed':True}
    monkeypatch.setattr(podcast, 'create_linked_scene', scene)
    def cancelled():
        if snapshots and snapshots[-1].get('chapter_drafts',{}).get('one',{}).get('0'):
            raise RuntimeError('cancelled')
    args = (plan, [{'id':'C1','source_id':'s','text':'Original','evidence_ids':['E1']}], {}, 'en', 20,
            {'max_output_tokens':4096,'recent_turns':6}, ContextUsage(total_token_limit=100000),
            podcast.EpisodeGenerationState(allow_partial=True))
    with pytest.raises(RuntimeError,match='cancelled'):
        await podcast._generate_complete_first(*args, cancelled, checkpoint_ready=snapshots.append)
    assert snapshots[-1]['chapter_versions']=={}
    assert calls==['core','one/0']
    result = await podcast._generate_complete_first(*args, lambda:None, resume=snapshots[-1])
    assert calls==['core','one/0','one/1','two/0','two/1']
    assert result[4]['accepted_blocks']==2


@pytest.mark.asyncio
async def test_reserved_audit_budget_keeps_all_compact_chapters(monkeypatch):
    plan, core = episode()
    calls=[]
    async def scene(**kwargs):
        calls.append(kwargs['chapter']['id'])
        return copy.deepcopy(core), {'passed':True}
    monkeypatch.setattr(podcast,'create_linked_scene',scene)
    trace=ContextUsage(total_token_limit=1000);trace.episode_audit_reserve_tokens=1000
    result=await podcast._generate_complete_first(plan,[{'id':'C1','source_id':'s'}],{},'zh-CN',30,
        {'max_output_tokens':4096,'recent_turns':6},trace,podcast.EpisodeGenerationState(allow_partial=True),lambda:None)
    assert calls==['core']
    assert result[4]['accepted_blocks']==0
    assert len(result[0])==len(core)
    assert any(w['code']=='optional_budget' for w in result[3])
