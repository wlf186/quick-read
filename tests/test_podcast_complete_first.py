"""Whole-episode fallback, bounded preparation and recoverable expansion contracts."""
import copy
import json
import pytest
from sandevistan_read import podcast, providers
from sandevistan_read.context_budget import ContextUsage, TokenLimits, plan_context, prompt_budget
from sandevistan_read.delivery import CURRENT as DELIVERY, DeliveryBudget
from sandevistan_read.generation_context import CURRENT, GenerationContext, prepare_evidence


def dialogue(prefix, claim, pairs=3):
    return [{"speaker": "HOST_A" if i % 2 == 0 else "HOST_B", "text": f"{prefix} complete supported explanation {i}.",
             "claim_ids": [claim], "citation_ids": ['E1'], "dialogue_act": "explain",
             "exchange_id": f"{prefix}/{i//2}", "exchange_start": i % 2 == 0} for i in range(pairs*2)]


@pytest.mark.asyncio
async def test_optional_failure_preserves_core_ending_and_resume_skips_calls(monkeypatch):
    claims = [{'id': f'C{i}', 'source_id': 's', 'text': f'claim {i}', 'evidence_ids': ['E1']} for i in range(1,5)]
    plan = {'episode_thesis': 'A central question', 'chapters': [{'id': 'one', 'title': 'first', 'claim_ids': ['C1','C2']}, {'id': 'two', 'title': 'second', 'claim_ids': ['C3','C4']}]}
    calls, snapshots = [], []
    async def scene(**kw):
        calls.append(kw['chapter']['id'])
        if calls[-1] == 'core':return dialogue('core','C1'), {'passed':True}
        if calls[-1] == 'one':raise podcast.PodcastQualityError('invalid optional JSON')
        return dialogue('additional different example','C4',1), {'passed':True}
    monkeypatch.setattr(podcast, 'create_linked_scene', scene)
    profile={'max_output_tokens':4096, 'recent_turns':6}
    args=(plan,claims,{},'en',20,profile,ContextUsage(total_token_limit=100000),podcast.EpisodeGenerationState(allow_partial=True),lambda:None)
    token=DELIVERY.set(DeliveryBudget())
    try:
        turns,chapters,_,warnings,report=await podcast._generate_complete_first(*args,checkpoint_ready=snapshots.append)
        assert calls == ['core','one','two']
        assert turns[-2:] == dialogue('core','C1')[-2:]
        assert chapters[-1]['id']=='core_closing' and report['accepted_blocks']==1
        assert any(w['code']=='optional_failed' for w in warnings)
        assert snapshots[0]['blocks']==[] and snapshots[0]['core_turns']
        again=await podcast._generate_complete_first(*args,resume=snapshots[-1])
        assert calls == ['core','one','two'] and again[0]==turns
    finally:DELIVERY.reset(token)


@pytest.mark.asyncio
async def test_cancellation_does_not_start_optional_work(monkeypatch):
    calls=[]
    async def scene(**kw):calls.append(1);return dialogue('core','C1'), {'passed':True}
    monkeypatch.setattr(podcast,'create_linked_scene',scene)
    def cancelled():
        if calls:raise RuntimeError('任务已取消')
    with pytest.raises(RuntimeError,match='取消'):
        await podcast._generate_complete_first({'episode_thesis':'x','chapters':[{'id':'c','claim_ids':['C1']}]},
            [{'id':'C1','source_id':'s'}],{},'en',20,{'max_output_tokens':4096,'recent_turns':6},ContextUsage(),
            podcast.EpisodeGenerationState(allow_partial=True),cancelled)
    assert len(calls)==1


def test_product_outline_never_duplicates_a_chapter_to_fill_time():
    token=DELIVERY.set(DeliveryBudget())
    try:
        chapters=[{'id':'c','claim_ids':['C1'],'purpose':'one'}]
        assert podcast._fit_episode_chapters(chapters,10,'en')==chapters
    finally:DELIVERY.reset(token)


@pytest.mark.parametrize('window,output',[(4096,1024),(30720,4096),(204800,16384),(1000000,384000),(2000000,500000)])
def test_podcast_planning_protects_core_and_caps_preparation(window,output):
    p=plan_context(TokenLimits.from_provider({'config':{'context_window_tokens':window,'max_output_tokens':output}}),'podcast',material_tokens=2000000,minutes=30)
    assert p.preparation_batches<=2
    assert p.preparation_token_limit<=p.total_token_limit*.1
    assert p.preparation_output_tokens<=2048
    assert p.core_output_tokens<=min(6000,output)
    assert p.batch_evidence_tokens>=0


@pytest.mark.asyncio
async def test_zero_valid_podcast_notes_stops_after_first_batch(monkeypatch):
    provider={'config':{'context_window_tokens':30720,'max_output_tokens':4096}}
    limits=TokenLimits.from_provider(provider)
    rows=[{'id':str(i),'source_id':'s','content':'A substantive qualified original statement. '*100,'ordinal':i} for i in range(10)]
    state=GenerationContext(provider,plan_context(limits,'podcast',material_tokens=200000),rows,[],ContextUsage(total_token_limit=300000))
    calls=[]
    async def chat(builder,**kw):
        calls.append(kw)
        budget=prompt_budget(limits,kw['max_tokens'],128,1)
        return providers.BudgetedCompletion(json.dumps({'notes':[{'chunk_id':'unknown','claim':'wrong','quote':'invented'}]}),builder(budget),budget)
    monkeypatch.setattr(providers,'budgeted_chat',chat)
    token=CURRENT.set(state)
    try:
        assert await prepare_evidence(rows,'en')==[]
        assert len(calls)==1 and calls[0]['max_tokens']<=2048
        assert state.preparation['stop_reason']=='no_accepted_notes'
        assert state.preparation['rejected']['invalid_source_or_shape']==1
    finally:CURRENT.reset(token)


def test_index_filter_preserves_formula_explanation():
    from sandevistan_read.retrieval import podcast_candidates
    rows=[{'id':'index','content':'\n'.join('term; variants; another; ' for _ in range(10))},
          {'id':'math','content':'Assume p > q. The probability q/p decreases with confirmations. This is a conditional calculation.'},
          {'id':'corrupt','content':'broken '+ '\ufffd'*30}]
    assert [r['id'] for r in podcast_candidates(rows)]==['math']


@pytest.mark.asyncio
async def test_closure_contract_includes_schema_and_requires_exact_anchors(monkeypatch):
    from sandevistan_read.providers import BudgetedCompletion
    turns=dialogue('A complete source argument','C1',2)
    provider={'config':{'context_window_tokens':30720,'max_output_tokens':4096}}
    monkeypatch.setattr(podcast,'active_provider',lambda role:provider)
    wrong=False
    async def chat(builder,**kwargs):
        budget=prompt_budget(TokenLimits.from_provider(provider),4096,128,1)
        built=builder(budget)
        assert '\\"closure\\"' not in built.messages[0]['content']
        assert "closure" in kwargs["response_schema"]["required"]
        data={'closure':{'opening_quote':turns[0]['text'], 'closing_quote':'Invented absent conclusion.' if wrong else turns[-1]['text'], 'verdict':'connected','reason':'The conclusion answers the opening.'},'checks':[]}
        return BudgetedCompletion(json.dumps(data),built,budget,'stop')
    monkeypatch.setattr(podcast,'budgeted_chat',chat)
    a=await podcast._audit_product_episode(turns,[{'turn_start':0,'turn_end':3}],'thesis','en',ContextUsage(),{})
    assert a['closure']['verdict']=='connected'
    wrong=True
    a=await podcast._audit_product_episode(turns,[{'turn_start':0,'turn_end':3}],'thesis','en',ContextUsage(),{})
    assert a['closure']['verdict']=='uncertain'


@pytest.mark.asyncio
async def test_podcast_note_prefix_recovery_still_requires_visible_id_and_quote(monkeypatch):
    provider={'config':{'context_window_tokens':30720,'max_output_tokens':4096}}
    limits=TokenLimits.from_provider(provider)
    quote='An original statement with a precisely preserved qualification.'
    rows=[{'id':'chunk_abc','source_id':'s','content':quote,'ordinal':0}]
    state=GenerationContext(provider,plan_context(limits,'podcast',material_tokens=200000),rows,[],ContextUsage(total_token_limit=300000))
    async def chat(builder,**kw):
        budget=prompt_budget(limits,kw['max_tokens'],128,1)
        notes=[{'chunk_id':'abc','claim':'valid claim','quote':quote},
               {'chunk_id':'abc','claim':'invented claim','quote':'A completely invented quotation without support.'},
               {'chunk_id':'invisible','claim':'unknown source','quote':quote},
               {'chunk_id':['abc'],'claim':'invalid shape','quote':quote}]
        return providers.BudgetedCompletion(json.dumps({'notes':notes}),builder(budget),budget)
    monkeypatch.setattr(providers,'budgeted_chat',chat)
    token=CURRENT.set(state)
    try:
        notes=await prepare_evidence(rows,'en')
        assert len(notes)==1 and notes[0]['chunk_id']=='chunk_abc'
        assert state.preparation['rejected']=={'quote_not_found_or_short':1,'invalid_source_or_shape':2}
    finally:CURRENT.reset(token)
