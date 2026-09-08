#!/usr/bin/env python3
"""Explicit HTTP upload-to-index evaluation using a frozen, isolated application."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


def save(path: Path, data: object) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


async def worker(args: argparse.Namespace) -> int:
    from sandevistan_read.database import DB, json_dump, json_load, utc_now
    from sandevistan_read import jobs, providers, retrieval
    from sandevistan_read.app import api
    DB.initialize()
    DB.seed(args.main_url, args.model, 'http://127.0.0.1:9')
    DB.execute("UPDATE provider_profiles SET base_url=?,model=?,config_json=? WHERE role='main'", (
        args.main_url, args.model, json_dump({'context_window_tokens': 30720, 'max_output_tokens': 4096})))
    DB.execute("UPDATE provider_profiles SET base_url=?,model=?,capabilities_json=? WHERE role='vlm'", (
        args.vlm_url, args.vlm_model, json_dump({'vision': True})))
    reports = []
    original = providers._chat_once

    async def measured(provider, messages, **kwargs):
        started = time.perf_counter()
        event = {'role': provider.get('role'), 'model': provider['model']}
        try:
            result = await original(provider, messages, **kwargs)
            event['usage'] = {key: value for key, value in vars(result).items() if 'token' in key}
            return result
        except Exception as exc:
            event['error'] = type(exc).__name__
            raise
        finally:
            event['seconds'] = round(time.perf_counter() - started, 3)
            with (args.output / 'provider-calls.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + '\n')
    providers._chat_once = measured
    layouts = ['single', 'batch'] if args.layout == 'both' else [args.layout]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url='http://localhost') as client:
        for repeat in range(1, args.repeats + 1):
            for layout in layouts:
                groups = [[path] for path in args.sample] if layout == 'single' else [args.sample]
                for group in groups:
                    notebook = (await client.post('/notebooks', json={'title': f'Import evaluation {layout} {repeat}'})).json()['id']
                    uploaded_at = time.perf_counter()
                    with ExitStack() as stack:
                        response = await client.post(f'/notebooks/{notebook}/sources', files=[
                            ('files', (path.name, stack.enter_context(path.open('rb')), 'application/octet-stream')) for path in group])
                    if response.status_code != 200:
                        reports.append({'layout': layout, 'repeat': repeat, 'files': [path.name for path in group],
                                        'status': 'upload_failed', 'http_status': response.status_code, 'detail': response.json()})
                        save(args.output / 'results.json', reports)
                        continue
                    for uploaded in response.json():
                        job = DB.fetchone('SELECT * FROM jobs WHERE id=?', (uploaded['job']['id'],))
                        source_id = uploaded['source_id']
                        started = time.perf_counter()
                        report = {'layout': layout, 'repeat': repeat, 'source_id': source_id,
                                  'queue_seconds': round(started - uploaded_at, 3), 'notebook_id': notebook}
                        DB.execute("UPDATE jobs SET state='running',started_at=? WHERE id=?", (utc_now(), job['id']))
                        try:
                            result = await jobs.execute(job)
                            report['processing_seconds'] = round(time.perf_counter() - started, 3)
                            report['upload_to_ready_seconds'] = round(time.perf_counter() - uploaded_at, 3)
                            source = DB.fetchone('SELECT * FROM sources WHERE id=?', (source_id,))
                            rows = DB.fetchall('SELECT * FROM chunks WHERE source_id=? ORDER BY ordinal', (source_id,))
                            metadata = json_load(source['metadata_json'], {})
                            report.update(filename=source['filename'], status=source['state'], chunks=len(rows),
                                          metadata=metadata, searchable=bool(rows), quality_candidates=sum(retrieval.is_quality_chunk(row) for row in rows))
                            report['locators'] = [json_load(row['locator_json'], {}) for row in rows]
                            save(args.output / f'{source_id}-text.json', [{'text': row['content'], 'locator': json_load(row['locator_json'], {})} for row in rows])
                            if args.question:
                                answer = await client.post(f'/notebooks/{notebook}/chat', json={
                                    'question': args.question, 'source_ids': [source_id], 'language': 'zh-CN'})
                                report['question_status'] = answer.status_code
                                report['answer'] = answer.json()
                            DB.execute("UPDATE jobs SET state='complete',finished_at=?,result_json=? WHERE id=?", (utc_now(), json_dump(result), job['id']))
                        except Exception as exc:
                            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                            DB.execute("UPDATE jobs SET state='failed',finished_at=?,error=? WHERE id=?", (utc_now(), str(exc), job['id']))
                        report.setdefault('processing_seconds', round(time.perf_counter() - started, 3))
                        report.setdefault('upload_to_ready_seconds', round(time.perf_counter() - uploaded_at, 3))
                        reports.append(report)
                        save(args.output / 'results.json', reports)
                        print(report.get('filename', source_id), layout, repeat, report['status'], report['processing_seconds'], flush=True)
    return int(any(item['status'] != 'ready' or not item.get('searchable') for item in reports))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--main-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--vlm-url', required=True)
    parser.add_argument('--vlm-model', required=True)
    parser.add_argument('--layout', choices=['single', 'batch', 'both'], default='both')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--question', help='Optional source-grounded question; calls MAIN after indexing.')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.sample = [path.resolve(strict=True) for path in args.sample]
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.worker:
        if Path(os.environ.get('SANDEVISTAN_PROJECT_ROOT', '')).resolve() != args.output / 'instance':
            parser.error('Worker requires its isolated application root')
        return asyncio.run(worker(args))
    from evaluate_context_strategy import initialize_instance
    args.output.mkdir(parents=True, exist_ok=False)
    instance = args.output / 'instance'
    initialize_instance(instance, args.source_root.resolve())
    save(args.output / 'manifest.json', {'samples': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.sample},
        'source_root': str(args.source_root.resolve()), 'main_url': args.main_url, 'model': args.model,
        'vlm_url': args.vlm_url, 'vlm_model': args.vlm_model, 'layout': args.layout, 'repeats': args.repeats,
        'image_policy': {'mode': 'process', 'processors': ['vlm', 'main', 'ocr']}})
    frozen = args.output / 'evaluator.py'
    shutil.copy2(__file__, frozen)
    environment = {**os.environ, 'SANDEVISTAN_PROJECT_ROOT': str(instance), 'PYTHONPATH': str(instance / 'src'),
                   'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        environment.pop(name, None)
    return subprocess.call([sys.executable, str(frozen), *sys.argv[1:], '--worker'], env=environment)


if __name__ == '__main__':
    raise SystemExit(main())
