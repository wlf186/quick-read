"""Regression coverage for bounded reasoning and complete evidence delivery."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sandevistan_read import podcast, providers
from sandevistan_read.context_budget import ContextUsage, PromptBudget, TokenLimits, plan_context
from sandevistan_read.delivery import CURRENT as DELIVERY, DeliveryBudget
from sandevistan_read.generation_context import CURRENT, GenerationContext, prepare_evidence


def profile(output=100000):
    return {'id': 'fixture', 'role': 'main', 'kind': 'openai', 'model': 'synthetic',
            'config': {'context_window_tokens': 256000, 'max_output_tokens': output, 'reasoning_effort': 'high'}}


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [3, 4])
async def test_outline_merges_last_mechanism_and_remaps_assignments(monkeypatch, count):
    claims = [{'id': f'C{i}', 'source_id': 'source', 'ordinal': i} for i in range(1, count+1)]
    chapters = [{'title': f'Topic {i}', 'purpose': f'Purpose {i}', 'mechanism': f'Mechanism {i}',
                 'required_conditions': f'Condition {i}', 'claim_ids': [f'C{i}']} for i in range(1,count+1)]
    async def reply(*args, **kwargs):
        return SimpleNamespace(content=json.dumps({'episode_thesis': 'Explain the mechanism', 'chapters': chapters,
                               'assignments': {'C1': count, 'C2': count}}), build=SimpleNamespace(metadata={'items':claims}))
    monkeypatch.setattr(podcast, 'budgeted_chat', reply)
    token = DELIVERY.set(DeliveryBudget())
    try:
        plan, adjusted = await podcast.create_episode_plan(claims, 'en', '', act_count=1)
    finally:
        DELIVERY.reset(token)
    assert adjusted and plan['locally_adjusted']
    assert len(plan['chapters']) == 1
    chapter = plan['chapters'][0]
    assert set(chapter['claim_ids']) == {c['id'] for c in claims}
    assert chapter['subtopics'][-1]['mechanism'] == f'Mechanism {count}'
    assert chapter['subtopics'][-1]['required_conditions'] == f'Condition {count}'
    assert chapter['bridge_out'] == ''
    assert plan['coverage']['assignments']['C1'] == chapter['id']


@pytest.mark.asyncio
async def test_reasoning_recovery_is_reused_without_growing_visible_work(monkeypatch):
    calls, targets = [], []
    async def respond(provider, messages, **kwargs):
        calls.append(kwargs['max_tokens'])
        return providers.ChatCompletion('' if len(calls)==1 else '{"notes":[]}', prompt_tokens=10,
            completion_tokens=8192 if len(calls)==1 else 9000,
            reasoning_tokens=8192 if len(calls)==1 else 8500,
            finish_reason='length' if len(calls)==1 else 'stop')
    monkeypatch.setattr(providers, '_chat_once', respond)
    state = DeliveryBudget(provider=profile())
    token = DELIVERY.set(state)
    def build(budget):
        targets.append(budget.output_tokens)
        return providers.PromptBuild([{'role':'user','content':'Extract a small fixed set of notes.'}])
    try:
        for _ in range(2):
            await providers.budgeted_chat(build, json_mode=True, max_tokens=4400, stage='context_prepare')
    finally:
        DELIVERY.reset(token)
    assert calls == [8192, 16384, 16384]
    assert targets == [4400, 4400]
    assert state.recoveries == 1


@pytest.mark.asyncio
async def test_small_manual_cap_does_not_consume_unused_recovery(monkeypatch):
    calls = []
    async def respond(provider, messages, **kwargs):
        calls.append(kwargs['max_tokens'])
        return providers.ChatCompletion('', completion_tokens=512, reasoning_tokens=512, finish_reason='length')
    monkeypatch.setattr(providers, '_chat_once', respond)
    state = DeliveryBudget(provider=profile(512)); token = DELIVERY.set(state)
    try:
        await providers.budgeted_chat(lambda b: providers.PromptBuild([{'role':'user','content':'Small task.'}]),
                                      json_mode=True, max_tokens=4400)
    finally:
        DELIVERY.reset(token)
    assert calls == [512] and state.recoveries == 0


@pytest.mark.asyncio
async def test_final_audit_uses_available_output_without_a_third_clamp(monkeypatch):
    calls = []
    async def respond(provider, messages, **kwargs):
        calls.append(kwargs['max_tokens'])
        return providers.ChatCompletion('{}', prompt_tokens=20, completion_tokens=100, finish_reason='stop')
    monkeypatch.setattr(providers, '_chat_once', respond)
    state = DeliveryBudget(provider=profile()); token = DELIVERY.set(state)
    trace = ContextUsage(total_token_limit=14000, accounted_tokens=4000, episode_audit_reserve_tokens=9000)
    try:
        await providers.budgeted_chat(lambda b: providers.PromptBuild([{'role':'user','content':'Audit these turns.'}]),
            json_mode=True, max_tokens=4096, trace=trace, stage='episode_audit')
    finally:
        DELIVERY.reset(token)
    assert calls == [8192] and state.audits == 1 and state.recoveries == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('recovery_batch', [1, 2, 3, 4])
async def test_four_preparation_batches_survive_one_recovery(monkeypatch, recovery_batch):
    rows = [{'id':f'c{i}', 'source_id':'source', 'ordinal':i,
             'content':f'Original region {i} contains a distinct claim and its necessary limiting conditions.'} for i in range(4)]
    plan = replace(plan_context(TokenLimits.from_provider(profile()), 'summary'), preparation_batches=4, batch_evidence_tokens=150)
    trace = ContextUsage(request_limit=7, total_token_limit=300000)
    state = GenerationContext(profile(), plan, rows, [], trace)
    delivery = DeliveryBudget(provider=profile()); calls=[]
    async def respond(builder, **kwargs):
        calls.append(kwargs['stage']); trace.requests += 1
        if len(calls) == recovery_batch:
            trace.requests += 1; delivery.recoveries = 1
        built = builder(PromptBudget(256000, 100000, 4400, 2048, 1))
        notes = [{'chunk_id':r['id'], 'claim':'A supported claim', 'quote':r['content']} for r in built.metadata['chunks']]
        return SimpleNamespace(content=json.dumps({'notes':notes}), build=built)
    monkeypatch.setattr(providers, 'budgeted_chat', respond)
    dt = DELIVERY.set(delivery); ct = CURRENT.set(state)
    try:
        notes = await prepare_evidence(rows, 'en')
    finally:
        CURRENT.reset(ct); DELIVERY.reset(dt)
    assert len(calls) == 4 and trace.requests == 5
    assert {n['chunk_id'] for n in notes} == {r['id'] for r in rows}
    assert state.preparation['attempted_batches'] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize('chapters', [['only'], ['one', 'two']])
async def test_single_chapter_legacy_body_keeps_original_text(monkeypatch, chapters):
    raw = {key:[{'speaker':'A' if i==0 else 'B','act_code':'X','text':f'{key} explanation {i}.','claim_ids':['C1']}
                for i in range(2)] for key in ['opening','body','closing']}
    claims = [{'id':cid,'source_id':'source','text':'A mechanism with a condition.','evidence_ids':['E1']} for cid in ['C1','C2']]
    monkeypatch.setattr(podcast,'_segment_prompt_build',lambda *args,**kwargs: providers.PromptBuild(
        [{'role':'user','content':kwargs['prefix']}],metadata={'items':claims[:1]}))
    prompts=[]
    async def reply(builder, **kwargs):
        built=builder(PromptBudget(256000,100000,6000,2048,1));prompts.append(built.messages[0]['content'])
        assert kwargs['response_schema']['properties']['opening']['items']['properties']['claim_ids']['items']['enum']==['C1']
        return SimpleNamespace(content=json.dumps(raw), finish_reason='stop', build=built)
    monkeypatch.setattr(podcast,'budgeted_chat',reply)
    monkeypatch.setattr(podcast,'validate_scene_turns',lambda turns,*args,**kwargs:(turns,[]))
    result=await podcast._draft_scene(scene_kind='act',chapter={'id':'core','title':'Test','purpose':'Explain'},claims=claims,
        cards_by_id={'E1':{'id':'E1','content':'A mechanism with a condition.'}},memory=podcast.EpisodeMemory('How?'),existing_turns=[],
        target=6,language='en',profile={'recent_turns':4,'allow_partial':True,'complete_role':'core','core_chapter_ids':chapters,
        'core_chapter_plan':[{'id':c,'claim_ids':['C1']} for c in chapters],'stage_output_tokens':6000},trace=ContextUsage())
    assert len(prompts)==1 and 'OVERRIDES THE EXAMPLE' not in prompts[0]
    if len(chapters)==1:
        assert [t['text'] for t in result.turns] == [t['text'] for group in raw.values() for t in group]
        assert {t['source_chapter_id'] for t in result.turns} == {'opening','only','closing'}
    else:
        assert not result.turns and result.issues


def test_same_topic_with_distinct_conditions_is_not_deduplicated():
    chapters = [{'id':f'c{i}', 'purpose':'Same topic', 'claim_ids':['C1'],
                 'mechanism':'Same mechanism', 'required_conditions':f'Condition {i}'} for i in range(2)]
    token=DELIVERY.set(DeliveryBudget())
    try:
        result=podcast._fit_episode_chapters(chapters,1,'en')
    finally:
        DELIVERY.reset(token)
    assert [c['required_conditions'] for c in result[0]['subtopics']] == ['Condition 0','Condition 1']


@pytest.mark.asyncio
@pytest.mark.parametrize('size,possible', [(10000,True),(30000,False)])
async def test_required_evidence_borrows_output_locally_or_skips_call(monkeypatch,size,possible):
    calls=[]
    async def respond(provider,messages,**kwargs):
        calls.append(kwargs)
        return providers.ChatCompletion('{}',prompt_tokens=5000,completion_tokens=100,finish_reason='stop')
    monkeypatch.setattr(providers,'_chat_once',respond)
    state=DeliveryBudget(provider=profile());token=DELIVERY.set(state)
    trace=ContextUsage(total_token_limit=14000,accounted_tokens=4000,episode_audit_reserve_tokens=1)
    async def invoke():
        return await providers.budgeted_chat(lambda b:providers.PromptBuild([{'role':'user','content':'x'*size}],
             metadata={'preserve_evidence':True}),json_mode=True,max_tokens=6000,trace=trace,stage='act_draft')
    try:
        if possible:
            await invoke()
        else:
            with pytest.raises(providers.ContextOverflowError):await invoke()
    finally:
        DELIVERY.reset(token)
    assert len(calls)==int(possible) and state.recoveries==0
    if possible:
        assert 4096 <= calls[0]['max_tokens'] < 8192
        assert calls[0]['timeout'] > 180


@pytest.mark.asyncio
async def test_expansion_cannot_replace_core_with_less_evidence(monkeypatch):
    plan={'episode_thesis':'Explain the whole mechanism','chapter_replacement':True,
          'chapters':[{'id':'one','title':'Mechanism','purpose':'Explain','claim_ids':['C1','C2']}]}
    def pair(section,claim):
        return [{'speaker':'HOST_A' if i==0 else 'HOST_B','text':f'{section} supported explanation {i}.',
                 'claim_ids':[claim],'source_chapter_id':section} for i in range(2)]
    core=pair('opening','C1')+pair('one','C2')+pair('closing','C2');calls=[]
    async def scene(**kwargs):
        calls.append(kwargs)
        return (core if kwargs['chapter']['id']=='core' else pair('one','C1')), {'passed':True}
    monkeypatch.setattr(podcast,'create_linked_scene',scene)
    result=await podcast._generate_chapter_replacement(plan,
        [{'id':'C1','evidence_ids':['E1']},{'id':'C2','evidence_ids':['E2']}],{},'en',5,
        {'max_output_tokens':6000,'recent_turns':4},ContextUsage(total_token_limit=100000),
        podcast.EpisodeGenerationState(allow_partial=True),lambda:None)
    assert len(calls)==2 and calls[1]['profile']['preserve_claim_ids']==['C2']
    assert [t['text'] for t in result[0]]==[t['text'] for t in core]
    assert result[4]['accepted_blocks']==0
    assert any('原文依据' in w['message'] for w in result[3])


@pytest.mark.asyncio
async def test_evidence_reallocation_respects_provider_input_cap(monkeypatch):
    p=profile();p['capabilities']={'token_limits':{'max_input_tokens':2000}}
    async def forbidden(*args,**kwargs):
        pytest.fail('An oversized input must never reach the provider')
    monkeypatch.setattr(providers,'_chat_once',forbidden)
    state=DeliveryBudget(provider=p);token=DELIVERY.set(state)
    try:
        with pytest.raises(providers.ContextOverflowError):
            await providers.budgeted_chat(lambda b:providers.PromptBuild([{'role':'user','content':'x'*10000}],
                metadata={'preserve_evidence':True}),json_mode=True,max_tokens=6000)
    finally:
        DELIVERY.reset(token)
    assert state.recoveries==0


@pytest.mark.parametrize('wrapped', [True,False])
def test_summary_format_normalization_keeps_claims_and_citations(wrapped):
    from sandevistan_read.services import summary_point_values
    point={'text':'中文要点，保留原文条件。','citations':['[S1]'],'qualification':'只在此前提下成立'}
    payload={'points':[{'__error__':'Ignore this diagnostic','points':[point]}] if wrapped else [point]}
    values,normalized=summary_point_values(payload)
    assert normalized and len(values)==1
    assert values[0]['claim']==point['text'] and values[0]['citations']==['[S1]']
    assert values[0]['qualification']==point['qualification']
    assert '__error__' not in values[0]


def test_summary_does_not_reinterpret_diagnostics_or_arbitrary_nested_data():
    from sandevistan_read.services import summary_point_values
    assert summary_point_values({'__error__':'Diagnostic only'}) == ([],False)
    values,_=summary_point_values({'points':[{'claim':{'text':'not a string'}},
        {'points':[{'points':[{'claim':'too deeply nested'}]}]}, {'text':'Valid','citations':['UNKNOWN']}]})
    assert len(values)==1 and values[0]['claim']=='Valid'
    assert values[0]['citations']==['UNKNOWN']  # Actual visible-label filtering remains in the parser.


def test_complete_six_turn_chapter_is_preserved_with_extra_closing_brace():
    def group(n):
        return [{'speaker':'A' if i%2==0 else 'B','act_code':'X','text':f'Complete explanation {i}.','claim_ids':['C1']} for i in range(n)]
    payload={'opening':group(2),'chapter_bodies':{'chapter_1':group(6)},'closing':group(2)}
    turns=podcast._extract_turns(json.dumps(payload)+' }')
    assert len(turns)==10
    assert [t['text'] for t in turns[2:8]]==[t['text'] for t in payload['chapter_bodies']['chapter_1']]
    assert podcast._extract_json('{"a":1} {"b":2}')=={}
    assert podcast._extract_json('{"a":[')=={}


@pytest.mark.asyncio
async def test_partial_citation_loss_is_a_warning_not_a_generation_gate(monkeypatch):
    plan={'episode_thesis':'Explain','chapter_replacement':True,'chapters':[{'id':'one','title':'Mechanism','purpose':'Explain','claim_ids':['C1','C2']}]}
    def pair(section,claims):
        return [{'speaker':'HOST_A' if i==0 else 'HOST_B','text':f'{section} complete explanation {i}.',
                 'claim_ids':claims,'source_chapter_id':section} for i in range(2)]
    core=pair('opening',['C1'])+pair('one',['C1','C2'])+pair('closing',['C1'])
    async def scene(**kwargs):return (core if kwargs['chapter']['id']=='core' else pair('expanded',['C1'])),{}
    monkeypatch.setattr(podcast,'create_linked_scene',scene)
    result=await podcast._generate_chapter_replacement(plan,[{'id':'C1','evidence_ids':['E1']},{'id':'C2','evidence_ids':['E2']}],{},
        'en',5,{'max_output_tokens':6000,'recent_turns':4},ContextUsage(total_token_limit=100000),podcast.EpisodeGenerationState(allow_partial=True),lambda:None)
    assert result[4]['accepted_blocks']==1
    assert any(w['code']=='chapter_coverage_reduced' for w in result[3])
