from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from sandevistan_read import app as app_module, providers, usage, review, delivery
from sandevistan_read.database import Database, json_dump
from sandevistan_read.context_budget import study_batch_size


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = Database(tmp_path / 'experience.sqlite')
    database.initialize()
    database.execute("INSERT INTO notebooks(id,title,created_at,updated_at) VALUES('n','Example','now','now')")
    monkeypatch.setattr(app_module, 'DB', database)
    from sandevistan_read import services
    monkeypatch.setattr(services, 'DB', database)
    monkeypatch.setitem(app_module.api.dependency_overrides, app_module.require_access, lambda: None)
    return database


def provider():
    return {'id': 'main', 'role': 'main', 'kind': 'openai', 'model': 'fixture',
            'base_url': 'https://example.invalid', 'api_key': 'secret-not-stored',
            'config': {'context_window_tokens': 8192, 'max_output_tokens': 1024}}


def mock_http(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(providers.httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


@pytest.mark.asyncio
async def test_metering_includes_compatibility_retries_without_double_counting(db, monkeypatch):
    def handler(request):
        payload = json.loads(request.content)
        if 'thinking' in payload:
            return httpx.Response(400, json={'error': {'message': 'unknown thinking parameter'}})
        return httpx.Response(200, json={'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 50, 'completion_tokens': 30,
                      'completion_tokens_details': {'reasoning_tokens': 20}, 'prompt_tokens_details': {'cached_tokens': 10}}})
    mock_http(monkeypatch, handler)
    model = provider(); model['config']['thinking'] = 'disabled'
    with usage.running(db, 'n', 'chat') as run:
        await providers._chat_once(model, [{'role':'user','content':'private prompt'}], json_mode=False, timeout=1, max_tokens=128, temperature=.1)
    result = usage.summarize(db, run_id=run.id)
    assert result['calls'] == 2 and result['unknown_calls'] == 1
    assert result['input_tokens'] + result['output_tokens'] == 80
    assert result['reasoning_tokens'] == 20 and result['cached_tokens'] == 10
    assert not result['complete']
    stored = json.dumps(db.fetchall('SELECT * FROM provider_calls'))
    assert 'private prompt' not in stored and 'secret-not-stored' not in stored
    assert 'thinking' in result['requests'][0]['requested_controls']
    assert 'thinking' not in result['requests'][1]['requested_controls']


@pytest.mark.asyncio
async def test_unknown_timeout_keeps_reservation_and_prevents_next_request(db, monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout('interrupted')
    mock_http(monkeypatch, handler)
    with usage.running(db, 'n', 'review', token_limit=200) as run:
        for i in range(2):
            with pytest.raises(httpx.ReadTimeout if i == 0 else RuntimeError):
                await providers._chat_once(provider(), [{'role':'user','content':'test'}], json_mode=False, timeout=1, max_tokens=128, temperature=.1)
    assert len(calls) == 1
    result = usage.summarize(db, run_id=run.id)
    assert result['unknown_calls'] == 1 and result['accounted_tokens'] >= 128


@pytest.mark.asyncio
async def test_zero_usage_is_known_and_separate_runs_are_not_duplicated(db, monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={'choices':[{'message':{'content':'ok'}}], 'usage':{'prompt_tokens':0,'completion_tokens':0}}))
    for _ in range(2):
        with usage.running(db, 'n', 'chat', target_id='same'):
            await providers._chat_once(provider(), [], json_mode=False, timeout=1, max_tokens=128, temperature=.1)
    result = usage.summarize(db, target_id='same')
    assert result['calls'] == 2 and result['complete'] and result['input_tokens'] == 0
    assert usage.summarize(db, target_id='historical')['recorded'] is False
    assert usage.summarize(db, notebook_id='n')['calls'] == 2


def seed_source(db):
    text = 'This original passage provides the complete evidence and its necessary qualifications.'
    db.execute("""INSERT INTO sources(id,notebook_id,revision_id,filename,media_type,size_bytes,sha256,blob_path,state,created_at,updated_at)
                  VALUES('s','n','rev','Example.txt','text/plain',100,'hash','unused','ready','now','now')""")
    db.execute("INSERT INTO chunks VALUES('chunk','s','rev',0,?,'{}',NULL,'hash','now')", (text,))
    return [{'id':'S1','source_id':'s','chunk_id':'chunk','filename':'Example.txt','locator':{},'quote':text}]


def seed_artifact(db, kind='summary'):
    citations = seed_source(db)
    payload = {'points':[{'claim':'The result is supported.', 'citations':['S1']}],
               'items':[{'question':'Which option?', 'options':['secret answer','wrong'], 'answer_index':0, 'explanation':'secret answer', 'citations':['S1']}]}
    db.execute("INSERT INTO artifacts VALUES('a','n',?,'Example','[\"s\"]','en','ready',?,?,NULL,'now','now')", (kind, json_dump(payload), json_dump(citations)))


def test_task_preview_is_local_and_respects_actual_study_batch_rules(db, monkeypatch):
    seed_source(db)
    monkeypatch.setattr(app_module, 'active_provider', lambda role: provider())
    client = TestClient(app_module.api)
    response = client.post('/notebooks/n/task-preview', json={'kind':'quiz','source_ids':['s'],'count':10})
    assert response.status_code == 200
    data = response.json()
    assert data['strategy'] == 'conservative'
    assert data['calls_range'][0] == 10 // study_batch_size('quiz',10,'lite',1024) + 1
    assert data['source_count'] == 1
    assert db.fetchall('SELECT * FROM provider_calls') == []
    assert client.post('/notebooks/n/task-preview', json={'kind':'quiz','token_limit':1}).status_code == 422


def test_review_is_independent_preserves_original_and_redacts_quiz(db, monkeypatch):
    seed_artifact(db, 'quiz')
    before = db.fetchone("SELECT payload_json FROM artifacts WHERE id='a'")
    monkeypatch.setattr(app_module, 'active_provider', lambda role: provider())
    async def fake_review(units, chunks):
        assert 'secret answer' in units[0]['claim']
        return delivery.assessment(1, 1, [{'message':'secret answer', 'code':'suspect'}], method='model_sample')
    monkeypatch.setattr(review, 'review_units', fake_review)
    client = TestClient(app_module.api)
    result = client.post('/reviews', json={'target_type':'artifact','target_id':'a'})
    assert result.status_code == 200
    assert 'secret answer' not in result.text
    assert 'secret answer' not in client.get('/reviews?target_type=artifact&target_id=a').text
    assert db.fetchone("SELECT payload_json FROM artifacts WHERE id='a'") == before
    assert len(db.fetchall('SELECT * FROM review_reports')) == 1
    assert len(db.fetchall('SELECT * FROM generation_runs')) == 1
    db.execute("DELETE FROM chunks WHERE id='chunk'")
    assert client.post('/reviews', json={'target_type':'artifact','target_id':'a'}).status_code == 409
    assert len(db.fetchall('SELECT * FROM generation_runs')) == 1


def test_review_reason_distinguishes_failed_check_from_wrong_content():
    token = delivery.CURRENT.set(delivery.DeliveryBudget())
    try:
        delivery.audit_reason(RuntimeError('用量上限'))
        result = delivery.assessment(6, method='model_sample')
        assert result['review_status'] == 'unavailable' and result['reason_code'] == 'budget_exhausted'
        assert result['issues'] == [] and result['reviewed_units'] == 0
        assert delivery.assessment(6, method='local_excerpt')['review_status'] == 'not_applicable'
    finally:
        delivery.CURRENT.reset(token)


def test_restart_preserves_unknown_attempts(db):
    with usage.running(db, 'n', 'summary') as run:
        db.execute("INSERT INTO provider_calls(id,run_id,model,role,stage,state,estimated_input_tokens,output_limit,accounted_tokens,created_at) VALUES('call',?,'model','main','summary','pending',100,200,300,'now')", (run.id,))
    db.execute("UPDATE generation_runs SET state='running' WHERE id=?", (run.id,))
    db.reset_running_jobs()
    assert db.fetchone("SELECT state FROM provider_calls WHERE id='call'")['state'] == 'unknown'
    assert usage.summarize(db, run_id=run.id)['accounted_tokens'] == 300


@pytest.mark.asyncio
async def test_resumed_job_keeps_previous_budget_and_review_is_separate(db, monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'choices':[{'message':{'content':'ok'}}]})
    mock_http(monkeypatch, handler)
    with usage.running(db, 'n', 'review', job_id='job', token_limit=200):
        await providers._chat_once(provider(), [], json_mode=False, timeout=1, max_tokens=128, temperature=.1)
    with usage.running(db, 'n', 'review', job_id='job', target_id='result', token_limit=200):
        with pytest.raises(RuntimeError, match='用量上限'):
            await providers._chat_once(provider(), [], json_mode=False, timeout=1, max_tokens=128, temperature=.1)
    assert len(calls) == 1
    assert usage.summarize(db, target_id='result')['calls'] == 1
    assert usage.summarize(db, target_id='result', include_reviews=False)['calls'] == 0


@pytest.mark.asyncio
async def test_concurrent_requests_reserve_before_sending(db, monkeypatch):
    import asyncio
    requests = []
    async def handler(request):
        requests.append(request)
        await asyncio.sleep(.01)
        return httpx.Response(200, json={'choices':[{'message':{'content':'ok'}}]})
    mock_http(monkeypatch, handler)
    with usage.running(db, 'n', 'review', token_limit=200):
        results = await asyncio.gather(*(providers._chat_once(provider(), [], json_mode=False, timeout=1, max_tokens=128, temperature=.1) for _ in range(2)), return_exceptions=True)
    assert len(requests) == 1
    assert sum(isinstance(result, RuntimeError) for result in results) == 1


@pytest.mark.asyncio
async def test_media_attempts_stay_separate_from_token_totals(db):
    @usage.metered_media('tts')
    async def synthesize(provider, text):
        raise RuntimeError('connection lost')
    with usage.running(db, 'n', 'podcast') as run:
        with pytest.raises(RuntimeError):
            await synthesize(provider(), 'sample')
    result = usage.summarize(db, run_id=run.id)
    assert result['calls'] == 0 and result['input_tokens'] == 0
    assert result['media'][0]['submitted_chars'] == 6
    assert result['media'][0]['state'] == 'unknown'
