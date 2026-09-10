"""Shared-source budgets, mandatory chapter coverage and intact question recovery."""
import json
from types import SimpleNamespace

import pytest

from sandevistan_read import podcast
from sandevistan_read.context_budget import ContextUsage, TokenLimits, prompt_budget, estimate_messages_tokens
from sandevistan_read.generation_context import CURRENT, GenerationContext, evidence_cost
from sandevistan_read.retrieval import select_context_evidence


def rows():
    return [{'id':str(i),'source_id':'one','ordinal':i,'content':text} for i,text in enumerate([
        'The mechanism is based on shared work.',
        'However, it requires an explicit assumption.',
        'A separate privacy limitation is also important.'])]


def test_unique_cost_covers_every_passage_at_exact_budget():
    data=rows();groups=podcast._podcast_evidence_groups(data)
    assert len(groups)==2
    budget=sum(evidence_cost(r) for r in data)
    selected=select_context_evidence(groups,budget,dependencies=lambda r:r['evidence_rows'])
    assert {r['id'] for g in selected for r in g['evidence_rows']}=={'0','1','2'}
    state=GenerationContext({},SimpleNamespace(evidence_tokens=budget),data,[],ContextUsage())
    token=CURRENT.set(state)
    try:
        result=podcast.select_podcast_evidence('n',['one'],'')
    finally:
        CURRENT.reset(token)
    assert len(result)==3
    assert sum(not r['supporting_only'] for r in result)==2


def test_partial_overlap_is_charged_once_but_keeps_distinct_groups():
    data=rows()
    groups=[{**data[0],'evidence_rows':data[:2]}, {**data[1],'evidence_rows':data[1:]}]
    selected=select_context_evidence(groups,sum(evidence_cost(r) for r in data),dependencies=lambda r:r['evidence_rows'])
    assert len(selected)==2


def bundles():
    cards=[{'id':f'E{i}','chunk_id':f'chunk{i}','source_id':'one','filename':'test.txt','locator':{},
            'content':f'Distinct complete original passage {i} with its required condition.', 'related_chunk_ids':linked}
           for i,linked in [(0,['chunk0','chunk1']),(1,['chunk1','chunk2']),(2,[])]]
    return podcast.build_claim_ledger(cards)[:2]


def test_prompt_sends_shared_original_once_and_counts_complete_request():
    claims=bundles();budget=prompt_budget(TokenLimits.from_provider({'config':{'context_window_tokens':4096,'max_output_tokens':1024}}),1024,128,1)
    build=podcast._segment_prompt_build(budget,prefix='Explain the complete conditions.',items=claims,renderer=podcast._render_claim_bundle)
    text=build.messages[-1]['content']
    assert text.count('Distinct complete original passage 1')==1
    assert all(cid in text for cid in ['C1','C2','E0','E1','E2'])
    assert len(build.metadata['items'])==2 and build.truncated_segments==0
    assert estimate_messages_tokens(build.messages)<=budget.input_tokens


def test_core_mapping_preserves_assignment_and_fills_same_source_nearest_chapter():
    claims=[{'id':f'C{i}','source_id':'s','ordinal':i} for i in range(1,5)]
    plan={'chapters':[{'id':'one','claim_ids':['C1']},{'id':'two','claim_ids':['C4']}]}
    result=podcast._ensure_plan_coverage(plan,claims,{'C1':1,'C4':2,'C2':999})
    assert result['coverage']['assignments']=={'C1':'one','C2':'one','C3':'two','C4':'two'}
    assert result['coverage']['locally_assigned_unit_ids']==['C2','C3']
    assert result['chapters'][1]['required_unit_ids']==['C3','C4']
    assert plan['chapters'][0]['claim_ids']==['C1']


def test_core_capacity_round_robins_sources_without_claiming_complete_knowledge():
    claims=[{'id':f'{s}{i}','source_id':s,'ordinal':i} for s in ['a','b','c'] for i in range(10)]
    result=podcast._ensure_plan_coverage({'chapters':[{'id':'one','claim_ids':['a0']},{'id':'two','claim_ids':['b0']}]},claims)
    assert result['coverage']['required_unit_ids']==['a0','b0','c0','a1']
    assert len(result['coverage']['provided_unit_ids'])==30
    assert 'not verified' in result['coverage']['meaning']


def turn(speaker,act,text,cids=None):
    return {'speaker':speaker,'act_code':act,'text':text,'claim_ids':cids if cids is not None else ['C1']}


@pytest.mark.parametrize('question,answer', [('Is this imperfect?','Indeed, this has a specific limitation.'),('这种办法有什么限制？','确实不完美，存在一个明确的限制。')])
def test_adjacent_groups_recover_question_and_answer_without_rewriting(question,answer):
    first=[turn('A','I','A source-backed mechanism is introduced.'),turn('B','Q',question,[])]
    second=[turn('A','X',answer),turn('B','X','The stated condition explains this limitation.')]
    raw={'exchanges':[{'turns':first},{'turns':second}]}
    result=podcast._extract_turns(json.dumps(raw))
    assert [t['text'] for t in result]==[t['text'] for t in first+second]
    assert len({t['exchange_id'] for t in result})==1


def test_bad_object_cannot_bridge_question_to_dependent_answer():
    first=[turn('A','I','An introduced mechanism.'),turn('B','Q','What condition does it require?',[])]
    orphan=[turn('A','X','Indeed, that is the condition.'),turn('B','X','The mechanism has this limitation.')]
    independent=[turn('A','X','Another independent source argument.'),turn('B','X','Its own condition is stated here.')]
    result=podcast._extract_turns(json.dumps({'exchanges':[{'turns':first}, {'broken':True},{'turns':orphan},{'turns':independent}]}))
    assert [t['text'] for t in result]==[t['text'] for t in independent]


def test_pending_recovery_is_bounded_to_eight_turns():
    groups=[]
    for i in range(5):groups.append({'turns':[turn('A','X',f'Explanation {i}.'),turn('B','Q',f'Follow-up {i}?',[])]})
    assert podcast._extract_turns(json.dumps({'exchanges':groups})) is None


def test_audit_shares_originals_and_prioritizes_risky_assertions_across_chapters():
    claims=bundles();by_id={c['id']:c for c in claims}
    turns=[{'speaker':'HOST_A' if i%2==0 else 'HOST_B','text':text,'claim_ids':['C1','C2']} for i,text in enumerate([
        'An ordinary opening statement.', 'The next ordinary statement.', 'It must guarantee success.', 'A different conclusion.', 'The probability is 0.01.', 'Another complete conclusion.'])]
    chapters=[{'turn_start':0,'turn_end':3},{'turn_start':4,'turn_end':5}]
    budget=prompt_budget(TokenLimits.from_provider({'config':{'context_window_tokens':30720,'max_output_tokens':4096}}),4096,128,1)
    build=podcast._product_audit_prompt(budget,turns,chapters,'Check against the evidence.',by_id)
    assert build.metadata['requested_facts'][:2]==[2,4]
    assert build.messages[0]['content'].count('Distinct complete original passage 1')==1
