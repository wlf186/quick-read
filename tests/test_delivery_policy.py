from __future__ import annotations
import json
import pytest
from sandevistan_read import delivery, providers, podcast, study_sessions
from sandevistan_read.context_budget import ContextUsage, PromptBudget

@pytest.mark.asyncio
async def test_recovery_shared_across_calls_and_audit_never_retries(monkeypatch):
    attempts = []
    async def once(*args, **kwargs):
        attempts.append(1)
        raise providers.ProviderError('unavailable', status=503)
    monkeypatch.setattr(providers, '_chat_once', once)
    model = {'kind':'ollama','model':'fixture','config':{'context_window_tokens':30720}}
    @delivery.delivery_task
    async def run():
        for stage in ('generation', 'generation', 'aggregate_audit', 'aggregate_audit'):
            with pytest.raises((providers.ProviderError, RuntimeError)):
                await providers.budgeted_chat(lambda budget: providers.PromptBuild([{'role':'user','content':'test'}]), stage=stage, provider_override=model)
        assert delivery.CURRENT.get().recoveries == 1
        assert delivery.CURRENT.get().audits == 1
    await run()
    assert len(attempts) == 4  # generation+one retry, second generation, audit
    assert delivery.CURRENT.get() is None

def test_quiz_rating_cannot_leak_answer():
    item = {'type':'quiz','payload':{'items':[{'question':'Q','answer_index':0,'quality_issues':[{'message':'answer is A'}]}], 'quality_assessment':{'level':'needs_review','issues':[{'message':'answer is A'}]}}, 'citations':[]}
    public = study_sessions.public_artifact(item)
    assert 'answer is A' not in json.dumps(public)
    assert item['payload']['items'][0]['quality_issues']  # Stored result preserved.

@pytest.mark.asyncio
@pytest.mark.parametrize('response', ['{}', '{"reviewed_indexes":[0,1],"broken_at":1,"unsupported_turns":[999],"repairs":[]}', 'invalid'])
async def test_coherence_review_discloses_only_visible_turns(monkeypatch, response):
    turns = [{'speaker':'HOST_A','text':'Where is the explanation?','claim_ids':[]}, {'speaker':'HOST_B','text':'An unrelated answer.','claim_ids':[]}]
    monkeypatch.setattr(podcast, 'active_provider', lambda role: {'config':{'context_window_tokens':30720,'max_output_tokens':4096}})
    async def chat(builder, **kwargs):
        budget = PromptBudget(30720, 10000, 4096, 2048, 1)
        return providers.BudgetedCompletion(response, builder(budget), budget)
    monkeypatch.setattr(podcast, 'budgeted_chat', chat)
    result = await podcast._audit_product_episode(turns, [{'turn_start':0,'turn_end':1}], 'Topic', 'en', ContextUsage(), {})
    assert 999 not in result.get('unsupported_turns', [])
    if '"broken_at":1' in response:
        assert result['broken_at'] == 1
    else:
        assert not result['reviewed_indexes']

@pytest.mark.parametrize('language', ['en', 'zh-CN', 'auto'])
@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard', 'mixed'])
@pytest.mark.parametrize('count', [1, 10, 30, 50])
def test_legal_configuration_keeps_delivery_budget_independent(language, difficulty, count):
    from sandevistan_read.context_budget import plan_context, TokenLimits
    from sandevistan_read.schemas import QuizRequest, FlashcardRequest
    request = (QuizRequest if count <= 30 else FlashcardRequest)(count=count, language=language, difficulty=difficulty)
    for window, output in [(30720,4096),(256000,100000),(1000000,384000)]:
        plan = plan_context(TokenLimits.from_provider({'config':{'context_window_tokens':window,'max_output_tokens':output}}), 'quiz' if count <= 30 else 'flashcard', count=request.count)
        assert plan.output_tokens <= output and plan.total_token_limit <= 300000

@pytest.mark.asyncio
async def test_conservative_delivery_pins_main_across_calls(monkeypatch):
    selected = {'kind':'ollama','model':'first','config':{'context_window_tokens':30720},'api_key':''}
    monkeypatch.setitem(globals(), 'active_provider', lambda role: selected)
    seen = []
    async def once(provider, *args, **kwargs):
        seen.append(provider['model'])
        selected['model'] = 'replacement'
        return providers.ChatCompletion('usable')
    monkeypatch.setattr(providers, '_chat_once', once)
    @delivery.delivery_task
    async def run():
        for _ in range(2):
            await providers.budgeted_chat(lambda budget: providers.PromptBuild([{'role':'user','content':'test'}]))
    await run()
    assert seen == ['first', 'first']


def test_summary_readback_and_list_preserve_rating_without_answer_text(tmp_path, monkeypatch):
    from sandevistan_read import app
    from sandevistan_read.database import Database, json_dump
    db = Database(tmp_path / 'delivery.db')
    db.initialize()
    monkeypatch.setattr(app, 'DB', db)
    db.execute("INSERT INTO notebooks(id,title,created_at,updated_at) VALUES('n','Fixture','now','now')")
    db.execute("INSERT INTO summaries VALUES('summary','n','scope','Preserved content','[]','now')")
    quality = delivery.assessment(2, 1, [{'unit':'point_2','code':'evidence_unconfirmed','severity':'suspect','message':'Verify the claim.'}])
    db.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", ('artifact','n','summary','Summary','[]','en','ready',json_dump({'content':'Preserved content','quality_assessment':quality,'delivery_status':'full'}),'[]',None,'now','now'))
    assert app.latest_summary('n')['quality_assessment'] == quality
    listed = app.artifacts('n', view='summary')[0]['payload']
    assert listed['quality_assessment']['level'] == 'needs_review'
    assert listed['quality_assessment']['issues'] == []
    assert listed['delivery_status'] == 'full'


def test_compact_podcast_recovers_question_and_explicit_composite_reference():
    raw = json.dumps({'turns':[['A','Q','What does this mechanism do?'],['B','F','It converts light energy into chemical energy.', '[C1|E1]']]})
    turns = podcast._extract_turns(raw)
    assert len(turns) == 2 and turns[0]['claim_ids'] == []
    accepted, _ = podcast.validate_scene_turns(turns, {'C1':{'text':'Light energy becomes chemical energy.','evidence_ids':['E1']}}, {'E1':{'content':'Light energy becomes chemical energy.'}}, last_speaker=None, existing_turns=[], language='en', expected_count=2, allow_style_degradation=True)
    assert len(accepted) == 2
    assert accepted[1]['claim_ids'] == ['C1'] and accepted[1]['citation_ids'] == ['E1']


def test_unknown_composite_reference_never_attaches_an_unrelated_citation():
    turns = podcast._extract_turns('{"turns":[["A","F","This is an unverified statement.","C999|E1"]]}')
    accepted, _ = podcast.validate_scene_turns(turns, {'C1':{'text':'Actual original.','evidence_ids':['E1']}}, {'E1':{'content':'Actual original.'}}, last_speaker=None, existing_turns=[], language='en', expected_count=1, allow_style_degradation=True)
    assert accepted[0]['citation_ids'] == []

@pytest.mark.asyncio
async def test_known_missing_opening_is_not_cleared_by_blanket_audit_pass(monkeypatch):
    monkeypatch.setattr(podcast, 'active_provider', lambda role: {'config':{'context_window_tokens':30720,'max_output_tokens':4096}})
    async def chat(builder, **kwargs):
        budget=PromptBudget(30720,10000,4096,2048,1)
        return providers.BudgetedCompletion('{"reviewed_indexes":[0,1],"broken_at":null,"breaks":[],"repairs":[]}',builder(budget),budget)
    monkeypatch.setattr(podcast,'budgeted_chat',chat)
    result=await podcast._audit_product_episode([{'speaker':'HOST_A','text':'我们刚才讨论了这个问题。'},{'speaker':'HOST_B','text':'这是后续结论。'}],[{'turn_start':0,'turn_end':1}],'Topic','zh-CN',ContextUsage(),{})
    assert result['broken_at'] == 0 and not result['passed']


def test_hanging_final_question_keeps_connected_short_script_without_model_call():
    turns = [{'speaker':'HOST_A','text':'The mechanism records order.','dialogue_act':'explain'}, {'speaker':'HOST_B','text':'That resolves the question.','dialogue_act':'summary'}, {'speaker':'HOST_A','text':'What about another subject?','dialogue_act':'question'}]
    warnings=[]
    kept, chapters, status=podcast.finish_product_script(turns,[{'turn_start':0,'turn_end':2}], 'full', 5, 'en', warnings)
    assert kept == turns[:2] and chapters[0]['turn_end'] == 1 and status == 'partial'
    assert {w['code'] for w in warnings} >= {'ending_short_version','product_duration'}


def test_duration_deviation_is_not_a_coherence_failure():
    warnings=[]
    turns=[{'speaker':'HOST_A','text':'The mechanism orders records.','dialogue_act':'explain'},{'speaker':'HOST_B','text':'That is the complete conclusion.','dialogue_act':'summary'}]
    _,_,status=podcast.finish_product_script(turns,[{'turn_start':0,'turn_end':1}],'full',30,'en',warnings)
    assert status == 'partial'
    assert [w['code'] for w in warnings] == ['product_duration']

@pytest.mark.parametrize('window,output', [(1024,128),(30720,4096),(256000,100000),(1000000,384000)])
@pytest.mark.parametrize('minutes', [5,10,20,30])
def test_podcast_audit_reserve_stays_inside_existing_limits(window, output, minutes):
    from sandevistan_read.context_budget import TokenLimits, reserve_podcast_audit, prompt_budget
    trace=ContextUsage(total_token_limit=min(45000,14000+750*podcast.target_turn_count(minutes)))
    limits=TokenLimits.from_provider({'config':{'context_window_tokens':window,'max_output_tokens':output}})
    reserve_podcast_audit(trace,limits)
    safe=prompt_budget(limits,4096,128,1)
    assert 0 < trace.episode_audit_reserve_tokens <= trace.total_token_limit//4
    assert trace.episode_audit_reserve_tokens <= safe.input_tokens+safe.output_tokens
    assert trace.accounted_tokens == 0

@pytest.mark.asyncio
async def test_generation_cannot_spend_reserved_review_request(monkeypatch):
    from sandevistan_read.context_budget import TokenLimits, reserve_podcast_audit
    model={'kind':'ollama','model':'fixture','config':{'context_window_tokens':30720,'max_output_tokens':4096}}
    trace=ContextUsage(total_token_limit=45000,request_limit=2,requests=1,accounted_tokens=30000)
    reserve_podcast_audit(trace,TokenLimits.from_provider(model))
    seen=[]
    async def once(provider,messages,**kwargs):
        seen.append(kwargs['max_tokens'])
        return providers.ChatCompletion('{}',prompt_tokens=100,completion_tokens=20)
    monkeypatch.setattr(providers,'_chat_once',once)
    @delivery.delivery_task
    async def run():
        build=lambda budget: providers.PromptBuild([{'role':'user','content':'Review this.'}])
        with pytest.raises(RuntimeError,match='连贯性审校'):
            await providers.budgeted_chat(build,trace=trace,provider_override=model)
        await providers.budgeted_chat(build,trace=trace,stage='episode_audit',provider_override=model)
    await run()
    assert len(seen)==1 and trace.requests==2
    assert trace.accounted_tokens==30120


def test_audit_packing_keeps_adjacent_turns_and_explicit_sampling_gaps():
    from sandevistan_read.context_budget import estimate_messages_tokens
    turns=[{'speaker':'HOST_A' if i%2==0 else 'HOST_B','text':f'Sentence {i}. '+ 'Detailed content. '*20,'claim_ids':[]} for i in range(24)]
    chapters=[{'turn_start':i,'turn_end':i+5} for i in range(0,24,6)]
    budget=PromptBudget(30720,1800,4096,2048,1)
    built=podcast._product_audit_prompt(budget,turns,chapters,'Check continuity. ',{})
    assert estimate_messages_tokens(built.messages)<=budget.input_tokens
    visible={t['index'] for t in built.metadata['items']}
    assert {0,1,22,23} <= visible and len(visible)<24
    assert all(i-1 in visible or i+1 in visible for i in visible)
    assert built.truncated_segments==0
    for t in built.metadata['items']:
        assert t['text']==turns[t['index']]['text']
        assert t['gap_before']==(t['index']>0 and t['index']-1 not in visible)


def test_review_coverage_excludes_trimmed_and_repaired_text():
    audit={'passed':True,'reviewed_indexes':list(range(10)), 'reviewed_transitions':list(range(9)), 'repaired_indexes':[2]}
    podcast._refresh_episode_review(audit,5)
    assert audit['reviewed_indexes']==[0,1,3,4]
    assert audit['reviewed_transitions']==[0,3]
    assert audit['checked_transitions']==2 and audit['total_transitions']==4
    assert audit['status']=='partial'

@pytest.mark.asyncio
@pytest.mark.parametrize('finish', [None,'length'])
async def test_complete_review_requires_explicit_complete_contract(monkeypatch,finish):
    model={'config':{'context_window_tokens':30720,'max_output_tokens':4096}}
    monkeypatch.setattr(podcast,'active_provider',lambda role:model)
    async def chat(builder,**kwargs):
        budget=PromptBudget(30720,10000,4096,2048,1)
        result={'reviewed_indexes':[0,1,True,99],'reviewed_transitions':[0,99], 'breaks':[], 'broken_at':None,'repairs':[],'unsupported_turns':[], 'transition_checks':[{'index':0,'question_quote':'What is recorded?','answer_quote':'The transaction order is recorded.','verdict':'connected','reason':'Directly identifies what is recorded.'}]}
        return providers.BudgetedCompletion(json.dumps(result),builder(budget),budget,finish)
    monkeypatch.setattr(podcast,'budgeted_chat',chat)
    turns=[{'speaker':'HOST_A','text':'What is recorded?'},{'speaker':'HOST_B','text':'The transaction order is recorded.'}]
    result=await podcast._audit_product_episode(turns,[{'turn_start':0,'turn_end':1}],'Records','en',ContextUsage(),{})
    assert result['status']==('unavailable' if finish else 'complete')
    assert result['checked_transitions']==(0 if finish else 1)


def test_review_accepts_explicit_adjacent_pairs_but_not_gaps_or_booleans():
    audit={'reviewed_indexes':[0,1,2,3],'reviewed_transitions':[[0,1],[1,3],[True,2],[2,3],False,99]}
    podcast._refresh_episode_review(audit,4)
    assert audit['reviewed_transitions']==[0,2]
    assert audit['status']=='partial'

@pytest.mark.asyncio
async def test_strict_audit_retains_original_network_recovery(monkeypatch):
    model={'kind':'ollama','model':'fixture','config':{'context_window_tokens':30720}}
    attempts=[]
    async def once(*args,**kwargs):
        attempts.append(1)
        if len(attempts)==1:
            raise providers.ProviderError('temporary',status=503)
        return providers.ChatCompletion('{}')
    monkeypatch.setattr(providers,'_chat_once',once)
    await providers.budgeted_chat(lambda budget:providers.PromptBuild([{'role':'user','content':'Audit.'}]),provider_override=model,stage='episode_audit')
    assert len(attempts)==2


def test_openapi_retains_episode_review_metadata():
    from sandevistan_read.api_docs import ARTIFACT_SCHEMA
    payload=ARTIFACT_SCHEMA['properties']['payload']['properties']
    audit=payload['quality_report']['properties']['episode_audit']['properties']
    assert audit['status']['enum']==['complete','partial','unavailable']
    assert {'performance','provider','quality','quality_report','quality_assessment'} <= payload.keys()

@pytest.mark.asyncio
async def test_missing_pinned_main_never_falls_back(monkeypatch):
    monkeypatch.setitem(globals(),'provider_by_id',lambda provider_id:None)
    @delivery.delivery_task
    async def run(payload):
        pytest.fail('Missing pinned provider must not start generation')
    with pytest.raises(RuntimeError,match='MAIN Provider'):
        await run({'provider_ids':{'main':'deleted'}})

@pytest.mark.asyncio
async def test_coherence_break_needs_no_second_audit_and_quotes_must_be_real(monkeypatch):
    model={'config':{'context_window_tokens':30720,'max_output_tokens':4096}}
    monkeypatch.setattr(podcast,'active_provider',lambda role:model)
    calls=[]
    async def chat(builder,**kwargs):
        calls.append(1)
        budget=PromptBudget(30720,10000,4096,2048,1)
        response={'reviewed_indexes':[0,1,2],'reviewed_transitions':[0,1],'breaks':[],'broken_at':None,'repairs':[],'unsupported_turns':[],
                  'transition_checks':[{'index':0,'question_quote':'Why did the temperature rise?','answer_quote':'The sensor stores its readings in memory.','verdict':'broken','reason':'Data storage does not explain the temperature change.'},
                                       {'index':1,'question_quote':'Invented quotation','answer_quote':'Another invented quotation','verdict':'connected','reason':'A fabricated check cannot count.'}]}
        return providers.BudgetedCompletion(json.dumps(response),builder(budget),budget)
    monkeypatch.setattr(podcast,'budgeted_chat',chat)
    turns=[{'speaker':'HOST_A','text':'Why did the temperature rise?'},{'speaker':'HOST_B','text':'The sensor stores its readings in memory.'},{'speaker':'HOST_A','text':'That describes data storage.'}]
    result=await podcast._audit_product_episode(turns,[{'turn_start':0,'turn_end':2}],'An experiment','en',ContextUsage(),{})
    assert result['broken_at']==1 and not result['passed']
    assert result['reviewed_transitions']==[0] and len(calls)==1


def test_three_unanswered_questions_preserve_only_complete_prefix():
    turns=[{'speaker':'HOST_A','text':'The topic is a distributed record.'}, {'speaker':'HOST_B','text':'The record orders transactions.'}]
    turns += [{'speaker':'HOST_A' if i%2==0 else 'HOST_B','text':q} for i,q in enumerate(['How does it work?','What mechanism is used?','How is it verified?'])]
    assert podcast._local_dialogue_breaks(turns)==[2]
    kept,chapters,status=podcast.finish_product_script(turns,[{'turn_start':0,'turn_end':4}],'full',5,'en',[])
    assert len(kept)==2 and status=='partial' and chapters[0]['turn_end']==1


def test_single_clarification_question_does_not_trigger_local_break():
    turns=[{'speaker':'HOST_A','text':'How is it checked?'},{'speaker':'HOST_B','text':'Do you mean transaction order?'},{'speaker':'HOST_A','text':'Yes. Nodes check the order.'}]
    assert podcast._local_dialogue_breaks(turns)==[]


def test_answer_followed_by_question_interrupts_unanswered_run():
    turns=[{'speaker':'HOST_A','text':'How is the record checked?'},
           {'speaker':'HOST_B','text':'Nodes check the transaction order. Do you mean the order?'},
           {'speaker':'HOST_A','text':'Which order is checked?'}]
    assert podcast._local_dialogue_breaks(turns)==[]
