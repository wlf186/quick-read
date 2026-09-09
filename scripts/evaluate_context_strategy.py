#!/usr/bin/env python3
"""Explicit paired evaluations using frozen sources and isolated runtime data.

Preparation parses local samples without MAIN/VLM. Runs call the selected MAIN;
--audio additionally calls AUDIO. Results and credentials never enter Git.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time


ROOT = Path(os.environ.get("QUICK_READ_EVAL_REPOSITORY", Path(__file__).resolve().parents[1])).resolve()


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def initialize_instance(path: Path, source: Path) -> None:
    path.mkdir(parents=True)
    shutil.copytree(source / "src", path / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (path / "runtime").mkdir()
    (path / "runtime/models").symlink_to(ROOT / "runtime/models", target_is_directory=True)
    (path / ".tools").symlink_to(ROOT / ".tools", target_is_directory=True)
    (path / "frontend").mkdir()
    (path / "frontend/dist").symlink_to(ROOT / "frontend/dist", target_is_directory=True)
    (path / "runtime/config.toml").write_text('[server]\nport=20839\n', encoding="utf-8")
    save(path / "source-hashes.json", {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path / "src").rglob("*.py")})


async def worker(args) -> int:
    from sandevistan_read.database import DB, json_dump, utc_now
    from sandevistan_read import providers, services, study, podcast
    from sandevistan_read.security import VAULT
    DB.initialize()
    DB.seed("http://100.80.59.126:11434", "gemma4:e4b", args.audio_url)
    DB.execute("UPDATE provider_profiles SET active=0,selected=0 WHERE role='vlm'")
    DB.execute("UPDATE provider_role_settings SET enabled=0 WHERE role='vlm'")
    notebook = "context-eval"
    if args.prepare:
        now = utc_now()
        DB.execute("INSERT OR IGNORE INTO notebooks(id,title,created_at,updated_at) VALUES(?,?,?,?)", (notebook,"Context evaluation",now,now))
        for name in ("bitcoin", "geb", "strange-loop"):
            supplied=dict(value.split('=',1) for value in args.sample)
            sample = Path(supplied[name]).resolve() if name in supplied else next(p for p in (ROOT / ".experiment/samples" / name / "source").iterdir() if p.suffix.lower() in {".pdf", ".epub"})
            checksum = hashlib.sha256(sample.read_bytes()).hexdigest()
            existing=DB.fetchone("SELECT state FROM sources WHERE id=?",(name,))
            if existing and existing['state']=='ready':continue
            blob=args.output/'instance/runtime/data/blobs'/f'{name}{sample.suffix}';blob.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(sample,blob)
            DB.execute("INSERT OR REPLACE INTO sources(id,notebook_id,revision_id,filename,media_type,size_bytes,sha256,blob_path,state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (name,notebook,checksum,sample.name,"application/pdf" if sample.suffix=='.pdf' else "application/epub+zip",sample.stat().st_size,checksum,str(blob.relative_to(args.output/'instance')),"queued",now,now))
            print("Preparing",name,flush=True)
            await services.ingest_source(name,image_policy={"mode":"off","processors":[]})
            print("Prepared",name,flush=True)
        save(args.output / "fixture-manifest.json",DB.fetchall("SELECT id,revision_id,filename,page_count FROM sources WHERE notebook_id=?",(notebook,)))
        return 0
    pinned = json.loads(os.environ.pop("QUICK_READ_EVAL_PROVIDER"))
    if pinned['kind'] == 'ollama' and args.main_concurrency != 1:
        raise ValueError('Ollama evaluation must remain serial')
    pinned['config']={**pinned.get('config',{}),'context_strategy':args.strategy}
    if args.context_window:
        pinned['config']['context_window_tokens'] = args.context_window
    DB.execute("UPDATE provider_profiles SET kind=?,base_url=?,model=?,config_json=?,capabilities_json=?,secret_enc=? WHERE role='main'",
               (pinned["kind"],pinned["base_url"],pinned["model"],json_dump(pinned["config"]),json_dump(pinned.get("capabilities",{})),VAULT.encrypt(pinned.get("api_key", ""))))
    original = providers._chat_once
    consecutive_failures = 0
    async def recorded(provider,messages,**kwargs):
        nonlocal consecutive_failures
        if consecutive_failures >= 3:
            raise RuntimeError("Evaluation stopped after three consecutive provider failures")
        assert provider["base_url"].rstrip('/')==pinned["base_url"].rstrip('/') and provider["model"]==pinned["model"], "MAIN changed during evaluation"
        started=time.monotonic()
        event={"at":time.time(),"model":provider["model"],"options":kwargs,"estimated_input_chars":sum(len(str(m.get('content',''))) for m in messages)}
        key=hashlib.sha256((pinned['base_url']+pinned['model']).encode()).hexdigest()[:16]
        lock = None
        try:
            while lock is None:
                for slot in range(args.main_concurrency):
                    suffix = '' if slot == 0 else f'-{slot}'
                    attempt = sqlite3.connect(ROOT / 'runtime/evals' / f'.context-main-{key}{suffix}.sqlite', timeout=0)
                    try:
                        attempt.execute('BEGIN IMMEDIATE')
                    except sqlite3.OperationalError as exc:
                        attempt.close()
                        if 'locked' not in str(exc):
                            raise
                    else:
                        lock = attempt
                        event['concurrency_slot'] = slot
                        break
                if lock is None:
                    await asyncio.sleep(.5)
            event["queue_seconds"]=round(time.monotonic()-started,3)
            inference_started=time.monotonic()
            result=await original(provider,messages,**kwargs)
            event["inference_seconds"]=round(time.monotonic()-inference_started,3)
            consecutive_failures = 0
            event["response"]=vars(result)
            return result
        except Exception as exc:
            consecutive_failures += 1
            event["error"]=type(exc).__name__
            event["error_code"]=getattr(exc,"code",None)
            event["http_status"]=getattr(exc,"status",None)
            raise
        finally:
            if lock is not None:
                lock.close()
            event["seconds"]=round(time.monotonic()-started,2)
            with (args.output / "main-calls.jsonl").open("a",encoding="utf-8") as handle:
                handle.write(json.dumps(event,ensure_ascii=False)+"\n")
    providers._chat_once=recorded
    if args.selection == 'conservative':
        # Diagnostic only: old local selection with the candidate's prompt,
        # output target and cumulative budget. This avoids confusing the old
        # 24K task-limit rejection with evidence-selection quality.
        from sandevistan_read.generation_context import CURRENT, mark_selected
        original_selection = services.select_quality_evidence
        def conservative_selection(*values, **options):
            token = CURRENT.set(None)
            try:
                selected = original_selection(*values, **options)
            finally:
                CURRENT.reset(token)
            mark_selected(selected)
            return selected
        services.select_quality_evidence = conservative_selection
    if args.audit_mode == 'external':
        async def external_audit(points, *unused):
            return points
        services._audit_summary_points = external_audit
    if args.script_file:
        from sandevistan_read import jobs
        fixed = json.loads((args.output / 'fixed-script.json').read_text())
        fixed = fixed.get('generated', fixed.get('payload', fixed))
        async def fixed_script(*unused, **kwargs):
            result = copy.deepcopy(fixed)
            result['script_reused_for_evaluation'] = True
            return result
        jobs.build_podcast_script = fixed_script
    if args.audio:
        DB.execute("UPDATE provider_profiles SET model='qwen3-tts-0.6b' WHERE role='audio'")
        DB.execute("UPDATE provider_profiles SET config_json=? WHERE role='audio'",(json_dump({"auto_select":False,"compute_device":"gpu","allow_device_fallback":True,"host_a":"Vivian","host_b":"Dylan","asr_model":"qwen3-asr-0.6b","asr_auto_select":False,"asr_compute_device":"gpu","asr_allow_device_fallback":True,"podcast_sequence_tts":True}),))
        await providers.probe_audio_provider(providers.active_provider("audio")["id"],apply_defaults=True)
    result_path=args.output / "results.json"
    results=json.loads(result_path.read_text()) if result_path.exists() else []
    for corpus in args.corpus:
        ids=[corpus] if corpus!='multi' else ['bitcoin','geb','strange-loop']
        for repeat in range(1,args.repeats+1):
            for kind in args.kind:
                identifier=f"{corpus}-{kind}-{repeat}-{args.language}-{'audio' if args.audio else 'text'}"
                if any(item['id']==identifier for item in results):continue
                event={"id":identifier,"kind":kind,"corpus":corpus,"repeat":repeat,"model":pinned['model'],"status":"running", "script_reused":bool(args.script_file), 'strategy':args.strategy,'audit_mode':args.audit_mode}
                save(args.output / "current.json",event)
                started=time.monotonic()
                try:
                    if kind=='summary':
                        result=await services.make_summary(notebook,ids,args.language)
                    elif kind in {'quiz','flashcard'}:
                        result=await study.generate_study_artifact(notebook,kind,args.count or (10 if kind=='quiz' else 20),ids,args.language,args.difficulty)
                    elif kind=='chat':
                        from sandevistan_read.app import ask
                        from sandevistan_read.schemas import ChatRequest
                        topic='工作量证明与双重支付' if corpus=='bitcoin' else '形式系统、自指与思维' if corpus=='geb' else '自指、层级与系统中的信任'
                        questions=[f'依据选中资料，解释{topic}的核心论点。','这个论点具体依赖哪些条件？','请举出资料中一个具体例子，并解释它的作用。','上面的解释有哪些限制或反例？','对照资料前部与后部，哪些内容补充或修正了前面的论点？','最后归纳这些区别，保留必要限定并给出原文依据。']
                        turns=[];conversation=None
                        for question in questions:
                            answer=await ask(notebook,ChatRequest(question=question,source_ids=ids,language=args.language,conversation_id=conversation))
                            conversation=answer['conversation_id'];turns.append(answer)
                        result={'turns':turns,'degraded':any(t.get('degraded') for t in turns)}
                    elif args.audio:
                        from sandevistan_read.jobs import enqueue,_podcast
                        payload={'source_ids':ids,'language':args.language,'minutes':args.minutes or (5 if corpus=='bitcoin' else 30),'duration_mode':args.duration_mode}
                        job=enqueue('podcast',notebook,payload)
                        audio_lock=sqlite3.connect(ROOT/'runtime/evals/.context-audio.sqlite',timeout=0)
                        try:
                            while True:
                                try:audio_lock.execute('BEGIN IMMEDIATE');break
                                except sqlite3.OperationalError as exc:
                                    if 'locked' not in str(exc):raise
                                    await asyncio.sleep(.5)
                            result=await _podcast(notebook,payload,job['id'])
                            artifact=DB.fetchone('SELECT status,payload_json,media_path FROM artifacts WHERE id=?',(result['id'],))
                            if not artifact:raise RuntimeError('Audio artifact was not persisted')
                            result.update(status=artifact['status'],payload=json.loads(artifact['payload_json']),media_path=artifact['media_path'])
                        finally:
                            audio_lock.close()
                    else:
                        result=await podcast.build_podcast_script(notebook,{'source_ids':ids,'language':args.language,'minutes':args.minutes or (5 if corpus=='bitcoin' else 30 if corpus=='geb' else 20),'duration_mode':args.duration_mode},allow_partial=True)
                        event['script_only']=True
                    event['result']=result
                    payload=result.get('payload') or result
                    event['delivery_status']=payload.get('delivery_status') or ('script_only' if event.get('script_only') else 'full')
                    event['full_target_completed']=event['delivery_status']=='full' and not event.get('script_only')
                    event['status']='degraded' if payload.get('degraded') or payload.get('warnings') or result.get('status')=='partial' else 'passed'
                except Exception as exc:
                    event.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                event['seconds']=round(time.monotonic()-started,2)
                results.append(event);save(result_path,results)
                save(args.output / (identifier+'.json'),event)
                print(identifier,event['status'],event['seconds'],flush=True)
                if consecutive_failures >= 3:
                    save(args.output / 'stopped.json', {'reason':'three_consecutive_provider_failures'})
                    return 2
    return int(any(item['status']=='failed' for item in results))


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--sample',action='append',default=[],metavar='NAME=PATH',help='Preparation sample overrides: bitcoin, geb, strange-loop')
    parser.add_argument('--provider-id')
    parser.add_argument('--main-concurrency',type=int,choices=[1,2],default=1,help='At most two independent remote MAIN calls; default serial. Ollama remains serial.')
    parser.add_argument('--selection',choices=['native','conservative'],default='native',help='Diagnostic only: legacy selection with the candidate budget and prompts; requires external audit.')
    parser.add_argument('--strategy',choices=['balanced','conservative'],default='balanced')
    parser.add_argument('--context-window',type=int,help='Application-side input capacity for same-model comparisons; cannot exceed the configured window.')
    parser.add_argument('--audit-mode',choices=['native','external'],default='native',help='External is an ablation only; results require independent review and cannot qualify native behavior.')
    parser.add_argument('--corpus',nargs='+',choices=['bitcoin','geb','multi'],default=['bitcoin','geb','multi'])
    parser.add_argument('--kind',nargs='+',choices=['summary','chat','quiz','flashcard','podcast'],default=['summary','chat','quiz','flashcard','podcast'])
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--language',choices=['zh-CN','en','auto'],default='zh-CN')
    parser.add_argument('--audio',action='store_true')
    parser.add_argument('--count',type=int,help='Quiz or Flashcard item count, validated against the request schema.')
    parser.add_argument('--difficulty',choices=['easy','medium','hard','mixed'],default='mixed')
    parser.add_argument('--duration-mode',choices=['auto','fixed'],default='fixed')
    parser.add_argument('--minutes',type=int,choices=[5,10,20,30])
    parser.add_argument('--script-file',type=Path,help='Replay a frozen Podcast script through the complete AUDIO job; requires --audio.')
    parser.add_argument('--reference',type=Path,help='Frozen source-backed quality rubric, hashed into the run identity.')
    parser.add_argument('--audio-url',default='http://127.0.0.1:20810')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();args.output=args.output.resolve()
    if args.script_file and (not args.audio or args.kind != ['podcast']):
        parser.error('--script-file requires --audio --kind podcast')
    if args.count is not None and (not set(args.kind) <= {'quiz','flashcard'} or not 1 <= args.count <= (30 if 'quiz' in args.kind else 50)):
        parser.error('--count requires only quiz/flashcard and a legal item count')
    if args.selection != 'native' and (args.audit_mode != 'external' or args.strategy != 'balanced' or args.kind != ['summary']):
        parser.error('--selection conservative requires --strategy balanced --audit-mode external --kind summary')
    if args.worker:return asyncio.run(worker(args))
    pinned = None
    if not args.prepare:
        if args.provider_id:
            from sandevistan_read.providers import provider_by_id
            pinned=provider_by_id(args.provider_id)
            if not pinned or pinned.get('role')!='main':raise ValueError('A configured MAIN provider is required')
        else:
            pinned={'kind':'ollama','base_url':'http://100.80.59.126:11434','model':'gemma4:e4b','api_key':'','config':{'context_window_tokens':30720,'max_output_tokens':4096},'capabilities':{}}
        if args.context_window is not None:
            from sandevistan_read.context_budget import TokenLimits
            if not 1024 <= args.context_window <= TokenLimits.from_provider(pinned).effective_context_tokens:
                raise ValueError('Comparison window must be between 1024 and the configured effective window')
    if args.output.exists() and not args.resume:raise ValueError('Use a new output directory, or --resume for the same frozen run')
    identity={'source_root':str(args.source_root.resolve()),'provider_id':args.provider_id,'corpus':args.corpus,'kind':args.kind,'repeats':args.repeats,'language':args.language,'audio':args.audio}
    identity['delivery_policy']='rated_v2' if (args.source_root/'src/sandevistan_read/delivery.py').exists() else 'legacy'
    identity['duration_mode']=args.duration_mode
    identity.update(count=args.count,difficulty=args.difficulty,minutes=args.minutes)
    identity.update(strategy=args.strategy,context_window=args.context_window,audit_mode=args.audit_mode)
    if args.main_concurrency != 1:
        identity['main_concurrency'] = args.main_concurrency
    if args.selection != 'native':
        identity['selection'] = args.selection
    if args.script_file:
        identity['script_hash']=hashlib.sha256(args.script_file.read_bytes()).hexdigest()
    if args.reference:
        identity['reference_hash']=hashlib.sha256(args.reference.read_bytes()).hexdigest()
    if pinned:
        public_settings={key:pinned.get(key) for key in ('kind','base_url','model','config','capabilities')}
        from sandevistan_read.context_qualification import fingerprint
        effective_provider = copy.deepcopy(pinned)
        if args.context_window is not None:
            effective_provider['config'] = {**effective_provider.get('config', {}), 'context_window_tokens': args.context_window}
        identity['provider_fingerprint']=fingerprint(effective_provider)
        identity['provider_settings_hash']=hashlib.sha256(json.dumps(public_settings,sort_keys=True).encode()).hexdigest()
        identity['audio_url']=args.audio_url
        if args.fixture:
            manifest=args.fixture.resolve()/'fixture-manifest.json'
            identity['fixture_manifest_hash']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if args.resume and not args.prepare:
        if json.loads((args.output/'manifest.json').read_text())!=identity:raise ValueError('Resume identity differs from frozen run')
    if not args.output.exists():
        args.output.mkdir(parents=True)
        initialize_instance(args.output/'instance',args.source_root.resolve())
        if not args.prepare:
            if not args.fixture:raise ValueError('--fixture is required')
            source_db=args.fixture.resolve()/'instance/runtime/data/sandevistan-read.db'
            target=args.output/'instance/runtime/data/sandevistan-read.db';target.parent.mkdir(parents=True,exist_ok=True)
            with sqlite3.connect(f'file:{source_db}?mode=ro',uri=True) as source,sqlite3.connect(target) as dest:source.backup(dest)
            (target.parent/'blobs').symlink_to(source_db.parent/'blobs',target_is_directory=True)
        save(args.output/'manifest.json',identity)
        if args.script_file:
            shutil.copy2(args.script_file,args.output/'fixed-script.json')
        if args.reference:
            shutil.copy2(args.reference,args.output/'quality-reference.json')
        shutil.copy2(Path(__file__),args.output/'evaluator.py')
    instance=args.output/'instance'
    environment={**os.environ,'SANDEVISTAN_PROJECT_ROOT':str(instance),'PYTHONPATH':str(instance/'src'),'QUICK_READ_EVAL_REPOSITORY':str(ROOT),'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
    # The parent environment's proxy is not reachable here. Direct connections
    # affect this explicit evaluation child only, never production settings.
    for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):environment.pop(name,None)
    if not args.prepare:
        environment['QUICK_READ_EVAL_PROVIDER']=json.dumps(pinned)
    script=Path(__file__).resolve() if args.prepare else args.output/'evaluator.py'
    command=[sys.executable,str(script),*sys.argv[1:],'--worker']
    try:
        return subprocess.call(command,env=environment)
    finally:
        # The child may need credentials during evaluation, but finished
        # snapshots and failure reports do not. Never touch the production DB.
        database = instance / 'runtime/data/sandevistan-read.db'
        if not args.prepare and database.exists():
            with sqlite3.connect(database) as isolated:
                isolated.execute('PRAGMA secure_delete=ON')
                isolated.execute("UPDATE provider_profiles SET secret_enc='' WHERE role='main'")
                isolated.commit()
                isolated.execute('PRAGMA wal_checkpoint(TRUNCATE)')


if __name__=='__main__':raise SystemExit(main())
