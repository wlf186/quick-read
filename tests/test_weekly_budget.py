from datetime import UTC, datetime
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from test_usage_experience import db, provider, mock_http
from sandevistan_read import app as app_module, providers, usage, weekly_budget as weekly


def configure(db, tokens=None, calls=None, mode='block', overrides=None, zone='UTC'):
    return weekly.save_settings(db, {'timezone':zone, 'global_limit':{'tokens':tokens,'calls':calls,'mode':mode}, 'providers':overrides or {}})


async def call(model=None):
    return await providers._chat_once(model or provider(), [], json_mode=False, timeout=1, max_tokens=128, temperature=.1)


@pytest.mark.asyncio
async def test_weekly_concurrent_admission_and_deleted_notebook(db, monkeypatch):
    requests=[]
    async def handler(request):
        requests.append(request)
        await asyncio.sleep(.01)
        return httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}], 'usage':{'prompt_tokens':10,'completion_tokens':5}})
    mock_http(monkeypatch,handler)
    configure(db,calls=1)
    with usage.running(db,'n','summary'):
        results=await asyncio.gather(call(),call(),return_exceptions=True)
    assert sum(isinstance(item,weekly.QuotaExceeded) for item in results)==1
    assert len(requests)==1
    db.execute("DELETE FROM notebooks WHERE id='n'")
    assert not db.fetchall('SELECT * FROM provider_calls')
    result=weekly.overview(db)['global_usage']
    assert result['known_tokens']==15 and result['calls']==1
    db.initialize()
    assert weekly.overview(db)['global_usage']['calls']==1


@pytest.mark.asyncio
async def test_soft_global_hard_provider_and_unknown_reservation(db,monkeypatch):
    mock_http(monkeypatch,lambda request:httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}]}))
    configure(db,tokens=1,mode='warn',overrides={'main':{'tokens':200,'calls':None,'mode':'block'}})
    with usage.running(db,'n','review'):
        await call()
        with pytest.raises(weekly.QuotaExceeded) as exc:
            await call()
        assert exc.value.detail['scope']=='main'
        await call({**provider(),'id':'another'})
    result=weekly.overview(db)['global_usage']
    assert result['calls']==2 and result['unknown_calls']==2 and result['unconfirmed_tokens']>=256
    configure(db,tokens=None,calls=None)
    with usage.running(db,'n','review'):
        await call()
    assert weekly.overview(db)['global_usage']['calls']==3


@pytest.mark.asyncio
async def test_task_limit_rejection_rolls_back_weekly_reservation(db,monkeypatch):
    mock_http(monkeypatch,lambda request:pytest.fail('must not send'))
    with usage.running(db,'n','review',token_limit=1):
        with pytest.raises(RuntimeError):
            await call()
    assert weekly.overview(db)['global_usage']['calls']==0


@pytest.mark.asyncio
async def test_cross_week_response_settles_original_period_and_retry_counts(db,monkeypatch):
    sunday=datetime(2026,9,27,23,59,59,tzinfo=UTC)
    monday=datetime(2026,9,28,0,0,1,tzinfo=UTC)
    monkeypatch.setattr(usage,'utc_now',lambda:sunday.isoformat())
    async def handler(request):
        monkeypatch.setattr(usage,'utc_now',lambda:monday.isoformat())
        return httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}], 'usage':{'prompt_tokens':0,'completion_tokens':0}})
    mock_http(monkeypatch,handler)
    configure(db,calls=1)
    with usage.running(db,'n','review'):
        await call()
        await call()
    assert weekly.overview(db,sunday)['global_usage']['calls']==1
    result=weekly.overview(db,monday)['global_usage']
    assert result['calls']==1 and result['unknown_calls']==0 and result['known_tokens']==0


def test_timezone_dst_and_settings_do_not_reset_history(db):
    start,end=weekly.period('America/New_York',datetime(2026,3,8,18,tzinfo=UTC))
    assert (datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds()==167*3600
    initial=weekly.settings(db)
    assert initial['global_limit']=={'tokens':1000000,'calls':500,'mode':'warn'}
    weekly.save_settings(db,{**initial,'timezone':'Asia/Shanghai'},initialize_only=True)
    weekly.save_settings(db,{**initial,'timezone':'UTC'},initialize_only=True)
    assert weekly.settings(db)['timezone']=='Asia/Shanghai'
    assert weekly.period('Asia/Shanghai',datetime(2026,9,27,16,tzinfo=UTC))[0]=='2026-09-27T16:00:00+00:00'


@pytest.mark.asyncio
async def test_connection_test_metering_and_structured_quota_response(db,monkeypatch):
    mock_http(monkeypatch,lambda request:httpx.Response(200,json={'choices':[{'message':{'content':'OK'}}], 'usage':{'prompt_tokens':2,'completion_tokens':1}}))
    token=usage.ACCOUNTING_DB.set(db)
    try:
        await providers._deep_verify(provider())
    finally:
        usage.ACCOUNTING_DB.reset(token)
    assert weekly.overview(db)['global_usage']['known_tokens']==3
    assert not db.fetchall('SELECT * FROM generation_runs')
    client=TestClient(app_module.api)
    assert client.get('/usage/weekly').status_code==200
    assert client.put('/settings/usage-budget',json={'timezone':'unknown'}).status_code==422
    assert client.put('/settings/usage-budget',json={'global_limit':{'calls':True}}).status_code==422
    configure(db,calls=1)
    # Inspection routes must propagate a quota denial instead of retrying it.
    async def inspect(candidate,mode):
        await providers._deep_verify(candidate)
    monkeypatch.setattr(app_module,'inspect_provider',inspect)
    response=client.post('/providers/inspect',json={'role':'main','kind':'openai','base_url':'https://example.invalid','model':'fixture','mode':'deep'})
    assert response.status_code==409
    assert response.json()['detail']['code']=='quota_exhausted'
    assert weekly.overview(db)['global_usage']['calls']==1


@pytest.mark.asyncio
async def test_cancelled_call_and_restart_retain_weekly_reservation(db,monkeypatch):
    def handler(request):
        raise asyncio.CancelledError()
    mock_http(monkeypatch,handler)
    with pytest.raises(asyncio.CancelledError):
        with usage.running(db,'n','chat'):
            await call()
    before=weekly.overview(db)['global_usage']
    db.reset_running_jobs()
    after=weekly.overview(db)['global_usage']
    assert after==before and after['unknown_calls']==1 and after['occupied_tokens']>=128


def test_v7_backfill_is_idempotent(db):
    with usage.running(db,'n','summary') as run:
        db.execute("INSERT INTO provider_calls(id,run_id,model,role,stage,state,estimated_input_tokens,output_limit,accounted_tokens,created_at) VALUES('old',?,'fixture','main','summary','unknown',10,20,30,?)",(run.id,usage.utc_now()))
    db._migrate_v8();db._migrate_v8()
    assert weekly.overview(db)['global_usage']['occupied_tokens']==30
    assert len(db.fetchall('SELECT * FROM usage_events'))==1


@pytest.mark.asyncio
async def test_compatibility_retry_hits_call_limit_without_sending(db,monkeypatch):
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(400,json={'error':{'message':'unknown thinking parameter'}})
    mock_http(monkeypatch,handler)
    configure(db,calls=1)
    model=provider();model['config']['thinking']='disabled'
    with usage.running(db,'n','summary') as run:
        with pytest.raises(weekly.QuotaExceeded):
            await call(model)
    assert len(requests)==1
    assert run.quota_denial['code']=='quota_exhausted'
    assert weekly.overview(db)['global_usage']['calls']==1


def test_quota_reason_contains_reset_time_and_scope():
    from sandevistan_read import delivery
    state=delivery.CURRENT.set(delivery.DeliveryBudget())
    try:
        delivery.audit_reason(weekly.QuotaExceeded('main','tokens','2026-10-05T00:00:00+00:00'))
        report=delivery.assessment(6,method='model_sample')
        assert report['reason_code']=='weekly_quota_exhausted'
        assert '2026-10-05' in report['reason']
        assert report['quota_denial']['scope']=='main'
    finally:
        delivery.CURRENT.reset(state)


def test_overview_detects_instance_work_and_pending_connection_checks(db):
    assert not weekly.overview(db)['has_active_work']
    with usage.running(db,'n','summary'):
        assert weekly.overview(db)['has_active_work']
    with db.transaction() as conn:
        weekly.reserve(conn,call_id='pending',provider=provider(),estimated=1,output=128,created_at=usage.utc_now())
    assert weekly.overview(db)['has_active_work']
    db.reset_running_jobs()
    assert not weekly.overview(db)['has_active_work']
