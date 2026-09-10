"""Argument retention under budget, preparation and chapter fallback."""
import copy
from types import SimpleNamespace

import pytest

from sandevistan_read import podcast
from sandevistan_read.context_budget import ContextUsage, TokenLimits, prompt_budget, pack_items
from sandevistan_read.generation_context import CURRENT, GenerationContext


def card(key, text, **extra):
    return {'id':key, 'chunk_id':'chunk_'+key, 'source_id':'source', 'filename':'synthetic.md',
            'content':text, 'locator':{}, **extra}


@pytest.mark.parametrize('text', [
    'Records are shared. Participants compare histories. This works only if honest processing power exceeds attackers. The risk decreases but never becomes impossible.',
    '所有参与者都能查看记录。节点对历史顺序进行比较。只有诚实算力占多数时才有此结论。攻击概率降低，并不是绝不可能。',
])
def test_third_sentence_conditions_survive_ledger_and_preparation(text):
    cards=[card('E1',text)]
    original=podcast.build_claim_ledger(cards)
    note={'chunk_id':'chunk_E1','claim':'Records are shared.', 'quote':text.split('.')[0], 'qualification':'', 'statement_kind':'assertion'}
    merged=podcast.merge_prepared_claims(original,cards,[note,note,{**note,'quote':'invented quotation'}])
    assert len(merged)==1 and merged[0]['original']==text and merged[0]['text']==text
    assert len(merged[0]['preparation_notes'])==1
    assert original[0]['preparation_notes']==[]
    assert text in podcast._render_claim_bundle(merged[0])


def test_adjacent_qualification_keeps_its_own_citation_and_never_crosses_sources():
    rows=[{'id':'a','source_id':'s','ordinal':0,'content':'The simplified verifier accepts the chain.'},
          {'id':'b','source_id':'s','ordinal':1,'content':'However, it depends on honest nodes and cannot independently validate every transaction.'},
          {'id':'c','source_id':'other','ordinal':2,'content':'However, this is a different author.'}]
    groups=podcast._podcast_evidence_groups(rows)
    assert [r['id'] for r in groups[0]['evidence_rows']]==['a','b']
    assert len(groups)==2
    assert [r['id'] for r in groups[1]['evidence_rows']]==['c']
    cards=[card('E1',rows[0]['content'],related_chunk_ids=['chunk_E1','chunk_E2']),card('E2',rows[1]['content'])]
    claim=podcast.build_claim_ledger(cards)[0]
    assert claim['evidence_ids']==['E1','E2'] and rows[1]['content'] in claim['original']


def test_atomic_packing_never_sends_a_claim_without_its_late_condition():
    claims=podcast.build_claim_ledger([card('E1','Reason. '*2000+'Only under the final condition.'),card('E2','A short complete conditional explanation.')])
    budget=prompt_budget(TokenLimits.from_provider({'config':{'context_window_tokens':4096,'max_output_tokens':1024}}),1024,128,1)
    build=podcast._segment_prompt_build(budget,prefix='Explain the source.\n',items=claims,renderer=podcast._render_claim_bundle,group_key=lambda c:c['source_id'])
    text=build.messages[-1]['content']
    assert 'Reason.' not in text and 'short complete conditional' in text
    assert build.truncated_segments==0 and [c['id'] for c in build.metadata['items']]==['C2']
    # Other generation tasks retain their existing optional clipping behavior.
    assert pack_items(['word '*5000],lambda x:x,100).truncated==1


def test_breadth_selection_keeps_short_definitions_and_multiple_sources(monkeypatch):
    rows=[{'id':f'{s}-{i}','source_id':s,'ordinal':i*10,'content':f'Only if condition {i} holds.', 'locator':{'page':i*10+1}}
          for s in ['one','two'] for i in range(3)]
    state=GenerationContext({},SimpleNamespace(evidence_tokens=600),rows,[],ContextUsage())
    token=CURRENT.set(state)
    try:
        selected=podcast.select_podcast_evidence('notebook',['one','two'],'')
        assert {r['source_id'] for r in selected}=={'one','two'}
        assert all(r['id'] in state.selected for r in selected)
        assert any(r['ordinal']>0 for r in selected)
    finally:
        CURRENT.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize('language',['en','zh-CN'])
async def test_core_receives_all_chapter_conditions_and_failed_expansion_keeps_them(monkeypatch,language):
    from sandevistan_read.delivery import CURRENT as DELIVERY, DeliveryBudget
    claims=[{'id':f'C{i}','source_id':'s','text':f'Condition {i}','evidence_ids':['E1']} for i in range(1,4)]
    chapter={'id':'one','title':'One mechanism','purpose':'Explain it','claim_ids':['C1','C2','C3'],
             'question':'How?', 'mechanism':'Shared records', 'required_conditions':'Condition 3'}
    plan={'episode_thesis':'How?', 'chapters':[chapter], 'chapter_replacement':True}
    core=[{'speaker':'HOST_A' if i%2==0 else 'HOST_B','text':f'Complete condition 3 explanation {i}.',
           'dialogue_act':'explain','source_chapter_id':key,'claim_ids':['C3'],'citation_ids':['E1']}
          for key in ['opening','one','closing'] for i in range(2)]
    calls=[]
    async def scene(**kwargs):
        calls.append(kwargs)
        if kwargs['chapter']['id']=='core':return copy.deepcopy(core),{'passed':True}
        raise podcast.PodcastQualityError('upstream unavailable')
    monkeypatch.setattr(podcast,'create_linked_scene',scene)
    token=DELIVERY.set(DeliveryBudget())
    try:
        result=await podcast._generate_chapter_replacement(plan,claims,{},language,5,{'max_output_tokens':4096,'recent_turns':6},
            ContextUsage(total_token_limit=100000),podcast.EpisodeGenerationState(allow_partial=True),lambda:None)
    finally:
        DELIVERY.reset(token)
    assert [c['id'] for c in calls[0]['claims']]==['C1','C2','C3']
    assert calls[0]['profile']['core_chapter_plan'][0]['required_conditions']=='Condition 3'
    assert [(t['text'], t['claim_ids']) for t in result[0]]==[(t['text'], t['claim_ids']) for t in core]
    assert len(calls)==2


def test_focus_preserves_relevance_and_adds_adjacent_conditions(monkeypatch):
    rows=[{'id':str(i),'source_id':'s','ordinal':i,'content':text} for i,text in enumerate([
        'A broad introduction to a different topic.', 'A particular mechanism is described here.',
        'However, that mechanism requires a specific assumption.'])]
    state=GenerationContext({},SimpleNamespace(evidence_tokens=1000),rows,[],ContextUsage())
    seen=[]
    def ranked(*args,**kwargs):
        seen.append(kwargs['focus'])
        return [rows[1]]
    monkeypatch.setattr(podcast,'select_quality_evidence',ranked)
    token=CURRENT.set(state)
    try:
        result=podcast.select_podcast_evidence('n',['s'],'particular mechanism')
    finally:
        CURRENT.reset(token)
    assert seen==['particular mechanism']
    assert result[0]['id']=='1' and result[0]['related_chunk_ids']==['1','2']


@pytest.mark.asyncio
async def test_v9_checkpoint_is_rejected_before_any_scene_call(monkeypatch):
    async def scene(**kwargs):
        pytest.fail('A stale checkpoint must not call MAIN')
    monkeypatch.setattr(podcast,'create_linked_scene',scene)
    with pytest.raises(ValueError,match='version'):
        await podcast._generate_chapter_replacement({},[],{},'en',5,{},ContextUsage(),
            podcast.EpisodeGenerationState(allow_partial=True),lambda:None,resume={'version':9})
