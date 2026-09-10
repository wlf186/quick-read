#!/usr/bin/env python3
"""Bounded, isolated short-chapter protocol comparison; no production DB writes.

Cases contain id, title and complete source rows. Review criteria are separate and
must never be passed as case fields. Outputs contain source text: keep in runtime.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import sys
from types import SimpleNamespace

from evaluate_context_strategy import EvaluationFailurePolicy
from sandevistan_read import podcast, providers
from sandevistan_read.context_budget import ContextUsage, TokenLimits, estimate_messages_tokens, prompt_budget
from sandevistan_read.podcast_contracts import array, record, string


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def material(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    cards = [{'id': f'E{i}', 'chunk_id': row['id'], 'source_id': row['source_id'],
              'filename': row['filename'], 'content': row['content'],
              'locator': row.get('locator') or json.loads(row.get('locator_json') or '{}')}
             for i, row in enumerate(rows, 1)]
    return cards, podcast.build_claim_ledger(cards)


def flat_turns(raw: str) -> list[dict]:
    """Wire-level view before recovery/validation, only for complete JSON."""
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(value, dict):
        return []
    if isinstance(value.get('turns'), list):
        return value['turns']
    return [t for group in value.get('exchanges', []) if isinstance(group, dict)
            for t in group.get('turns', []) if isinstance(t, dict)]


def turn_diff(before: list[dict], after: list[dict], reason: str) -> list[dict]:
    """Match in order, retaining duplicate occurrences separately."""
    cursor = 0
    changes = []
    for i, turn in enumerate(before):
        retained = cursor < len(after) and turn.get('text') == after[cursor].get('text')
        changes.append({'index': i, 'text': turn.get('text'), 'retained': retained,
                        'reason': 'retained' if retained else reason})
        cursor += int(retained)
    return changes


async def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.read_text())
    if len(cases) != 3 or any(set(case) != {'id','title','rows'} for case in cases):
        raise ValueError('Exactly three cases containing only id, title, rows are required')
    if len({case['id'] for case in cases}) != 3:
        raise ValueError('Case IDs must be unique')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'cases.json', cases)
    policy = EvaluationFailurePolicy(args.output/'failure-state.json')
    provider = {'kind':'ollama','base_url':args.main_url,'model':args.model,'api_key':'',
                'config':{'context_window_tokens':30720,'max_output_tokens':4096}}
    budget = prompt_budget(TokenLimits.from_provider(provider),4096,128,1)
    save(args.output/'manifest.json', {'model':args.model,'main_url':args.main_url,'context':30720,
         'output':4096,'think':False,'temperature':.45,'maximum_calls':12,'retries':0,
         'cases_sha256':hashlib.sha256(args.cases.read_bytes()).hexdigest(),
         'protocols':['current','simple'],'repeats':2,'minutes':2.5,
         'qualification':'All three cases in both repeats: complete necessary conditions, no clear factual errors, coherence and naturalness >=4/5. Offline editorial review only.'})
    results=[]
    original=podcast.budgeted_chat
    try:
        for repeat in (1,2):
            for case in cases:
                for variant in (('current','simple') if repeat==1 else ('simple','current')):
                    if policy.state['stop_reason']:
                        break
                    cards,claims=material(case['rows']); by_id={c['id']:c for c in claims}; card_map={c['id']:c for c in cards}
                    event={'id':f"{case['id']}-{variant}-{repeat}",'case':case['id'],'variant':variant,'repeat':repeat}
                    request_dir=args.output/event['id'];request_dir.mkdir()
                    async def captured(builder, **options):
                        build=builder(budget)
                        # Both protocols must receive every original in the case.
                        assert len(build.metadata['items'])==len(claims), 'Case does not fit; do not compare different evidence'
                        assert estimate_messages_tokens(build.messages)<=budget.input_tokens
                        save(request_dir/'request.json',{'messages':build.messages,'schema':options.get('response_schema'),
                             'estimated_input_tokens':estimate_messages_tokens(build.messages),'input_budget':budget.input_tokens})
                        policy.before_request()
                        started=time.monotonic()
                        try:
                            response=await providers._chat_once(provider,build.messages,json_mode=True,
                                response_schema=options.get('response_schema'),timeout=420,max_tokens=4096,temperature=.45)
                        except Exception as exc:
                            policy.request(False,getattr(exc,'status',None))
                            event.update(error=type(exc).__name__,error_message=str(exc),http_status=getattr(exc,'status',None))
                            raise
                        else:
                            policy.request(True)
                            event['response']=asdict(response)
                            return SimpleNamespace(content=response.content,finish_reason=response.finish_reason,build=build)
                        finally:
                            event['seconds']=round(time.monotonic()-started,3)
                    podcast.budgeted_chat=captured
                    print('START',event['id'],flush=True)
                    try:
                        if variant=='current':
                            result=await podcast._draft_scene(scene_kind='act',
                                chapter={'id':case['id'],'title':case['title'],'purpose':case['title'],
                                         'required_unit_ids':list(by_id)},claims=claims,cards_by_id=card_map,
                                memory=podcast.EpisodeMemory(case['title']),existing_turns=[],target=8,language='zh-CN',
                                profile={'recent_turns':6,'allow_partial':True,'complete_role':'expansion',
                                         'chapter_replacement':True,'stage_output_tokens':4096},trace=ContextUsage(),
                                duration_budget=podcast._scene_duration_budget('zh-CN',2.5,8,0))
                            retained,issues=result.turns,result.issues
                        else:
                            schema=record({'turns':array(record({'speaker':{'type':'string','enum':['A','B']},
                                'text':string(),'claim_ids':array({'type':'string','enum':list(by_id)},6)}),16,2)})
                            prefix=(f'用自然简体中文写一段双人播客，主题：{case["title"]}。约2至3分钟，8至12轮左右。'
                                '连续讲清原文的机制和必要条件，保留概率与作者归属，不增加资料外事实或数字。'
                                '两人都参与解释，提问后直接回答，不反复附和或总结。引用编号不念出口。'
                                '只返回JSON turns数组，每轮含speaker(A/B)、text、claim_ids。事实填实际支持的C编号，纯提问可为空。\n')
                            response=await captured(lambda b:podcast._segment_prompt_build(b,prefix=prefix,items=claims,
                                renderer=podcast._render_claim_bundle,language='zh-CN'),response_schema=schema)
                            parsed=podcast._extract_turns(response.content) or []
                            retained,issues=podcast.validate_scene_turns(parsed,by_id,card_map,last_speaker=None,
                                existing_turns=[],language='zh-CN',expected_count=max(8,len(parsed)),allow_style_degradation=True)
                        raw=event.get('response',{}).get('content','')
                        parsed=podcast._extract_turns(raw) or []
                        assembled,_=podcast._assemble_chapter_versions([],{'chapters':[{'id':case['id'],'title':case['title']}]},{case['id']:retained})
                        event.update(status='completed' if assembled else 'failed',issues=issues,
                            raw_turns=flat_turns(raw),parsed_turns=parsed,validated_turns=retained,final_turns=assembled,
                            loss={'parse':turn_diff(flat_turns(raw),parsed,'parser structure/recovery rejection'),
                                  'validation':turn_diff(parsed,retained,'validation/group rejection; see issues'),
                                  'assembly':turn_diff(retained,assembled,'chapter assembly rejection')})
                    except Exception as exc:
                        event.update(status='failed',error=type(exc).__name__,error_message=str(exc))
                    policy.task({'status':'ready' if event['status']=='completed' else 'failed',
                                 'result':{'delivery_status':'partial','narrative_status':'unverified'} if event['status']=='completed' else {}})
                    save(request_dir/'result.json',event);results.append(event)
                    save(args.output/'results.json',results)
                    print('END',event['id'],event['status'],flush=True)
    finally:
        podcast.budgeted_chat=original
    save(args.output/'summary.json',{'calls':policy.state['requests'],'completed':sum(r['status']=='completed' for r in results),
         'stop_reason':policy.state['stop_reason'],'quality':'awaiting_offline_review'})
    return 0 if len(results)==12 else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--main-url',required=True)
    parser.add_argument('--model',required=True)
    sys.exit(asyncio.run(run(parser.parse_args())))
