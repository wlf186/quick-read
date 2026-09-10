import json
import math
from types import SimpleNamespace

import pytest

from sandevistan_read import podcast
from sandevistan_read.context_budget import ContextUsage, TokenLimits, plan_context
from sandevistan_read.generation_context import CURRENT, GenerationContext, context_material, evidence_cost


def test_preview_and_execution_share_metadata_inclusive_podcast_budget(monkeypatch):
    from sandevistan_read import app
    from sandevistan_read.schemas import ContextPreviewRequest
    rows = [{'id': str(i), 'source_id': 's', 'ordinal': i,
             'filename': 'A descriptive source filename.pdf', 'locator_json': json.dumps({'page': i+1}),
             'content': text} for i, text in enumerate([
                 'The mechanism uses shared records. ' * 30,
                 'However, it depends on the stated condition. ' * 30,
                 'Only if the participants cooperate.'])]
    monkeypatch.setattr(app, '_require_notebook', lambda _: None)
    monkeypatch.setattr(app, 'DB', SimpleNamespace(fetchall=lambda sql, args: rows if 'JOIN sources' in sql else [{'id':'s','selected':1}]))
    config = {'context_window_tokens':30720,'max_output_tokens':4096,'context_strategy':'balanced'}
    preview = app.context_preview(ContextPreviewRequest(notebook_id='n',model='fixture',config=config))
    eligible, costs = context_material('podcast', rows)
    assert len(eligible)==3 and costs==[evidence_cost(r) for r in rows]
    plan = plan_context(TokenLimits.from_provider({'config':config}), 'podcast', material_tokens=sum(costs), segment_tokens=math.ceil(sum(costs)/len(costs)), minutes=20)
    assert next(p for p in preview['plans'] if p['kind']=='podcast') == plan.as_dict()
    token = CURRENT.set(GenerationContext({}, plan, eligible, [], ContextUsage()))
    try:
        selected = podcast.select_podcast_evidence('n',['s'],'')
    finally:
        CURRENT.reset(token)
    assert {r['id'] for r in selected} == {'0','1','2'}


def dialogue():
    return [
        {'speaker':'A','act_code':'I','text':'系统允许合并和拆分支付金额，下面讨论它的具体机制。','claim_ids':[]},
        {'speaker':'B','act_code':'Q','text':'难道每个最小金额都需要单独进行一笔交易吗？','claim_ids':[]},
        {'speaker':'A','act_code':'X','text':'并不是的，交易可以包含多个输入和多个输出。','claim_ids':['C1']},
        {'speaker':'B','act_code':'X','text':'剩余金额可以通过输出作为找零返回。','claim_ids':['C1']},
    ]


def test_uncited_question_and_answer_survive_parse_validation_and_assembly():
    turns=dialogue()
    parsed=podcast._extract_turns(json.dumps({'exchanges':[{'turns':turns[:2]},{'turns':turns[2:]}]}))
    assert [t['text'] for t in parsed] == [t['text'] for t in turns]
    validated,issues=podcast.validate_scene_turns(parsed,{'C1':{'evidence_ids':['E1'],'text':'Input and output.'}},
        {'E1':{'content':'Input and output.'}},last_speaker=None,existing_turns=[],language='zh-CN',expected_count=4,allow_style_degradation=True)
    assert any('待核实' in issue for issue in issues)
    assert validated[0]['claim_ids']==[]
    result,_=podcast._assemble_chapter_versions([],{'chapters':[{'id':'one','title':'one'}]},{'one':validated})
    assert [t['text'] for t in result] == [t['text'] for t in turns]


@pytest.mark.parametrize('bad', [{'broken':True}, {'turns':[{'speaker':'A','text':None}]}])
def test_invalid_group_never_bridges_uncited_question_and_negative_answer(bad):
    turns=dialogue()
    assert podcast._extract_turns(json.dumps({'exchanges':[{'turns':turns[:2]},bad,{'turns':turns[2:]}]})) is None

@pytest.mark.asyncio
@pytest.mark.parametrize('failure_status', [None, 429, 500])
async def test_diagnostic_runner_is_bounded_and_stops_on_service_failure(tmp_path, monkeypatch, failure_status):
    import importlib.util
    import sys
    from pathlib import Path
    from sandevistan_read import providers
    scripts=Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('podcast_diagnostic_eval',scripts/'evaluate_podcast_diagnostics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cases=[{'id':str(i),'title':'A mechanism','rows':[{'id':'one','source_id':'s','filename':'synthetic.txt','content':'A source statement with an explicit condition.'}]} for i in range(3)]
    path=tmp_path/'cases.json';path.write_text(json.dumps(cases))
    calls=[]
    async def chat(provider,messages,**kwargs):
        calls.append(kwargs)
        assert provider['model']=='fixture' and kwargs['max_tokens']==4096
        if failure_status:
            error=RuntimeError('synthetic provider error');error.status=failure_status;raise error
        turns=[{'speaker':'A','text':'这里先介绍资料中提供的具体机制。','claim_ids':['C1']},
               {'speaker':'B','text':'这项机制依赖资料中给出的明确条件。','claim_ids':['C1']}]
        value={'turns':turns} if 'turns' in kwargs['response_schema']['properties'] else {'exchanges':[{'turns':turns}]}
        return providers.ChatCompletion(json.dumps(value),100,100,'stop')
    monkeypatch.setattr(providers,'_chat_once',chat)
    before=podcast.budgeted_chat
    code=await module.run(SimpleNamespace(cases=path,output=tmp_path/'output',main_url='http://localhost:9',model='fixture'))
    assert len(calls)==({None:12,429:1,500:2}[failure_status])
    assert code==(0 if failure_status is None else 2)
    assert podcast.budgeted_chat is before

@pytest.mark.parametrize('text', [
    '用户保留区块头，并用路径确认该交易已经被网络接受。',
    'The reader keeps headers and verifies the transaction inclusion path.',
    'How does it work? The reader checks its inclusion in the chain.',
    '原文说的是“有条件地可靠”。',
])
def test_question_label_cannot_turn_a_complete_statement_into_a_question(text):
    turn={'speaker':'A','act_code':'Q','text':text,'claim_ids':['C1']}
    parsed=podcast._extract_turns(json.dumps({'turns':[turn]}))[0]
    assert parsed['text']==text and parsed['dialogue_act']=='explain'
    assert not podcast._is_question_turn({**parsed,'dialogue_act':'question'})


@pytest.mark.parametrize('text', ['为什么需要区块头？','How does it work?', '“How does it work?”', '如何验证交易', 'How does it work'])
def test_question_shape_survives_incorrect_or_missing_labels(text):
    assert podcast._is_question_turn({'text':text,'dialogue_act':'explain'})


def test_mislabeled_statement_preserves_complete_chapter_through_finalization():
    raw=[{'speaker':'A' if i%2==0 else 'B','act_code':act,'text':text,'claim_ids':['C1']} for i,(act,text) in enumerate([
        ('I','用户如何验证资料中描述的交易？'),
        ('Q','用户可以通过区块头和路径确认交易已被网络接受。'),
        ('F','这种方法是否有可靠性方面的前提？'),
        ('A','是的，它依赖诚实节点控制网络，受到攻击时可能被骗。'),
        ('F','多输入交易又如何处理支付金额的合并？'),
        ('O','多个较小的输入可以组合成支付，并通过输出返回找零。'),
    ])]
    parsed=podcast._extract_turns(json.dumps({'exchanges':[{'turns':raw[i:i+2]} for i in range(0,6,2)]}))
    valid,_=podcast.validate_scene_turns(parsed,{'C1':{'text':'Source mechanism.','evidence_ids':['E1']}},{'E1':{'content':'Source mechanism.'}},
        last_speaker=None,existing_turns=[],language='zh-CN',expected_count=6,allow_style_degradation=True)
    turns,chapters=podcast._assemble_chapter_versions([],{'chapters':[{'id':'one','title':'One'}]},{'one':valid})
    final,_,status=podcast.finish_product_script(turns,chapters,'full',5,'zh-CN',[])
    assert status!='draft_only'
    assert [t['text'] for t in final]==[t['text'] for t in raw]
