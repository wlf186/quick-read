"""Learning Podcast contracts and source-anchored quality diagnostics."""
import json

import httpx
import pytest

from sandevistan_read import audio_quality, podcast, providers
from sandevistan_read.podcast_contracts import scene_schema


def test_word_speakers_override_a_cross_host_segment_and_detect_real_collapse():
    turns=[{'speaker':'HOST_A','text':'First answer','start_seconds':0,'end_seconds':1},
           {'speaker':'HOST_B','text':'Second answer','start_seconds':1,'end_seconds':2}]
    words=[{'start':.1,'end':.8,'speaker':'x','text':'First answer'},
           {'start':1.1,'end':1.8,'speaker':'y','text':'Second answer'}]
    result={'segments':[{'start':0,'end':2,'speaker':'x','words':words}]}
    report=audio_quality.assess_transcription(turns,result,'en')
    assert report['passed'] and report['speaker_alignment']==1
    assert report['speaker_method']=='word_speaker' and report['speaker_coverage']==1
    words[1]['speaker']='x'
    bad=audio_quality.assess_transcription(turns,result,'en')
    assert not bad['passed'] and bad['speaker_alignment']==.5 and bad['speaker_issues']


def test_segment_fallback_uses_overlap_and_missing_labels_are_not_full_coverage():
    turns=[{'speaker':'HOST_A','start_seconds':0,'end_seconds':1}, {'speaker':'HOST_B','start_seconds':1,'end_seconds':3}]
    report=audio_quality._speaker_assessment(turns,[{'start':0,'end':3,'speaker':'x'}])
    assert report['speaker_alignment']==.6667 and report['speaker_method']=='segment_overlap'
    report=audio_quality._speaker_assessment(turns,[{'start':0,'end':1,'speaker':'x'},{'start':1,'end':3}])
    assert report['speaker_coverage']==.3333


def test_core_contract_requires_three_sections_and_preserves_turn_content():
    pair=[{'speaker':'A','act_code':'X','text':'The author proposes a conditional argument.','claim_ids':['C1']},
          {'speaker':'B','act_code':'O','text':'Its conclusion depends on that assumption.','claim_ids':['C1']}]
    raw={key:pair for key in ['opening','body','closing']}
    turns=podcast._extract_turns(json.dumps(raw))
    assert len(turns)==6 and turns[-1]['speaker']=='HOST_B'
    assert turns[-1]['text']==pair[-1]['text']
    del raw['closing']
    assert podcast._extract_turns(json.dumps(raw)) is None
    schema=scene_schema('core',['C1'])
    assert schema['required']==['opening','body','closing']


def test_fact_verdicts_require_both_original_and_script_anchors():
    text='Any majority of nodes can ensure the longest chain.'
    original='The majority of CPU power must be controlled by honest nodes.'
    turns=[{'text':text,'claim_ids':['C1']}]
    valid={'index':0,'claim_id':'C1','script_quote':text,'source_quote':original,'verdict':'contradicted','reason':'CPU power is not node count.'}
    parsed={'facts':[valid,{**valid,'claim_id':'C2'},{**valid,'index':1}]}
    report=podcast._validated_content_review(parsed,turns,{0},{'C1':{'original':original}})
    assert report['facts']==[valid]
    for field in ['script_quote','source_quote']:
        report=podcast._validated_content_review({'facts':[{**valid,field:'Invented absent evidence.'}]},turns,{0},{'C1':{'original':original}})
        assert report['facts']==[]


def test_semantic_duplicate_removes_only_whole_optional_block_and_remaps_facts():
    def turn(text,group,speaker,start=False):
        return {'text':text,'speaker':speaker,'exchange_id':group,'exchange_start':start,'claim_ids':['C1'],'citation_ids':['S1'],'dialogue_act':'explain'}
    turns=[turn('The main explanation is conditional.','core/exchange_1','HOST_A',True),
           turn('The condition is majority CPU power.','core/exchange_1','HOST_B'),
           turn('Again the explanation has a condition.','optional/one','HOST_A',True),
           turn('That condition is the majority of CPU power.','optional/one','HOST_B'),
           turn('Therefore the original assumption matters.','core/exchange_3','HOST_A',True),
           turn('This completes the conditional explanation.','core/exchange_3','HOST_B')]
    parsed={'duplicates':[{'index':2,'prior_index':0,'quote':turns[2]['text'],'prior_quote':turns[0]['text'],'reason':'No new content.'},
                          {'index':4,'prior_index':0,'quote':turns[4]['text'],'prior_quote':turns[0]['text'],'reason':'Repeated conclusion.'}]}
    audit=podcast._validated_content_review(parsed,turns,set(range(6)),{})
    assert len(audit['duplicates'])==1
    audit['facts']=[{'index':5,'claim_id':'C1','verdict':'supported'}]
    kept,_,_=podcast.retain_product_exchanges(turns,[{'turn_start':0,'turn_end':5}],audit,[])
    assert kept==turns[:2]+turns[4:] and audit['facts'][0]['index']==3


@pytest.mark.asyncio
async def test_schema_is_sent_to_native_ollama_without_changing_plain_json(monkeypatch):
    seen=[]
    real=httpx.AsyncClient
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'message':{'content':'{}'},'done_reason':'stop'})
    monkeypatch.setattr(providers.httpx,'AsyncClient',lambda **kw:real(transport=httpx.MockTransport(handler),**kw))
    provider={'kind':'ollama','model':'test','api_key':'','base_url':'http://local','config':{}}
    schema=scene_schema('core',['C1'])
    await providers._chat_once(provider,[],json_mode=True,response_schema=schema,timeout=1,max_tokens=100,temperature=.15)
    await providers._chat_once(provider,[],json_mode=True,timeout=1,max_tokens=100,temperature=.15)
    assert seen[0]['format']==schema and seen[1]['format']=='json'


@pytest.mark.asyncio
async def test_resolved_topic_shift_cannot_delete_core_closing(monkeypatch):
    from sandevistan_read.context_budget import ContextUsage, TokenLimits, prompt_budget
    turns=[{'speaker':'HOST_A','text':'The chance of success decreases as the chain grows.','claim_ids':[]},
           {'speaker':'HOST_B','text':'The conclusion is that this design works without a trusted party.','claim_ids':[]}]
    provider={'config':{'context_window_tokens':30720,'max_output_tokens':4096}}
    monkeypatch.setattr(podcast,'active_provider',lambda role:provider)
    async def chat(builder,**kwargs):
        budget=prompt_budget(TokenLimits.from_provider(provider),4096,128,1)
        data={'checks':[{'index':0,'question_quote':turns[0]['text'],'answer_quote':turns[1]['text'],
                         'verdict':'broken','reason':'The conclusion changes the topic.'}]}
        return providers.BudgetedCompletion(json.dumps(data),builder(budget),budget,'stop')
    monkeypatch.setattr(podcast,'budgeted_chat',chat)
    audit=await podcast._audit_product_episode(turns,[{'turn_start':0,'turn_end':1}], 'main question','en',ContextUsage(),{})
    assert audit['breaks']==[] and audit['transition_checks'][0]['verdict']=='uncertain'
    turns[0]['text']='How does the algorithm prevent double spending?'
    audit=await podcast._audit_product_episode(turns,[{'turn_start':0,'turn_end':1}], 'main question','en',ContextUsage(),{})
    assert audit['breaks']==[1]


def test_contained_repeated_exchange_does_not_discard_the_new_half():
    old=[{'text':'The security assumption is explicit.','exchange_id':'prior'},
         {'text':'Honest CPU power must exceed attacker power.','exchange_id':'prior'}]
    repeated=[{**t,'exchange_id':'new/one'} for t in old]
    fresh=[{'text':'Now the source describes transaction fees.','exchange_id':'new/two'},
           {'text':'Fees also incentivize miners.','exchange_id':'new/two'}]
    assert podcast._drop_repeated_exchanges(repeated+fresh,old)==fresh
    # Shared questions alone never justify deleting an answer or half an exchange.
    changed=[repeated[0],{**repeated[1],'text':'A distinct qualified answer.'}]
    assert podcast._drop_repeated_exchanges(changed,old)==changed


def test_core_preserves_opening_question_answered_in_the_body():
    def turn(speaker,text,act='X'):
        return {'speaker':speaker,'act_code':act,'text':text,'claim_ids':['C1']}
    raw={'opening':[turn('A','We discuss a conditional system.','I'),turn('B','What condition makes it work?','Q')],
         'body':[turn('A','The source requires a specific assumption.'),turn('B','The result depends on that assumption.')],
         'closing':[turn('A','That answers the opening question.'),turn('B','The conditional conclusion is now complete.','O')]}
    turns=podcast._extract_turns(json.dumps(raw))
    assert len(turns)==6 and turns[1]['text']==raw['opening'][1]['text']
    assert turns[0]['exchange_id']=='exchange_1' and turns[-1]['exchange_id']=='exchange_3'
    raw['closing'][-1]=turn('B','An unresolved final question?','Q')
    assert podcast._extract_turns(json.dumps(raw)) is None
