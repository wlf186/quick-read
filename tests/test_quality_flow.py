from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from sandevistan_read import app, quality, jobs, providers, services, study, podcast, observability, usage
from sandevistan_read.database import Database, json_dump, json_load
from sandevistan_read.schemas import QualityAction


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = Database(tmp_path / 'quality.sqlite')
    db.initialize()
    for module in (app, quality, jobs, providers, services, study, podcast, observability):
        monkeypatch.setattr(module, 'DB', db)
    model = {'id':'main', 'role':'main', 'kind':'openai', 'base_url':'https://example.invalid', 'model':'fixture', 'api_key':'never-persist-this', 'config':{'context_window_tokens':8192, 'max_output_tokens':2048}}
    for module in (app, jobs, providers, services, study, podcast):
        if hasattr(module, 'active_provider'):
            monkeypatch.setattr(module, 'active_provider', lambda role: model if role == 'main' else None)
        if hasattr(module, 'provider_by_id'):
            monkeypatch.setattr(module, 'provider_by_id', lambda identity: model if identity == 'main' else None)
    db.execute("INSERT INTO notebooks(id,title,created_at,updated_at) VALUES('n','Test','now','now')")
    db.execute("INSERT INTO sources(id,notebook_id,revision_id,filename,media_type,size_bytes,sha256,blob_path,state,created_at,updated_at) VALUES('s','n','r','test.txt','text/plain',12,'hash','unused','ready','now','now')")
    db.execute("INSERT INTO chunks VALUES('chunk','s','r',0,'A source passage about learning.','{}',NULL,'hash','now')")
    monkeypatch.setitem(app.api.dependency_overrides, app.require_access, lambda: None)
    async def candidate(run):
        if run['kind'] == 'quiz':
            return {'items':[{'id':'q1','question':'Question','options':['secret answer','b','c','d'],'answer_index':0,'explanation':'secret answer','citations':[]}], 'citations':[]}, None
        return {'content':'First draft','scope_hash':'scope','citations':[]}, None
    monkeypatch.setattr(quality, 'first_candidate', candidate)
    return db, TestClient(app.api)


def score(total, blocking=False):
    # Use valid rubric dimensions for the supplied total.
    left=total; parts={}
    for name, maximum in quality.DIMENSIONS.items():
        parts[name]=min(left,maximum);left-=parts[name]
    return quality.validate_score({'scores':parts,'blocking':blocking,'suggestions':['Make it clearer']}, {'mode':'full'})


def enqueue(level='low', kind='summary'):
    job=jobs.enqueue(kind,'n',{'quality_level':level,'source_ids':['s']})
    return job, json_load(job['payload_json'])['quality_run_id']


@pytest.mark.asyncio
@pytest.mark.parametrize('level,target,retries', [('low',60,1),('medium',75,1),('high',85,2),('extreme',92,3)])
async def test_policy_visible_first_draft_best_version_and_bound(env,monkeypatch,level,target,retries):
    db,_=env
    job,identity=enqueue(level)
    calls=[]
    async def step(run,candidate,improve=False):
        # The first draft is persisted and readable before scoring starts.
        assert quality.public(identity)['target_id']
        assert db.fetchone('SELECT * FROM artifacts WHERE id=?',(candidate['target_id'],))
        calls.append(improve)
        return {**candidate['content'],'content':'Revised'} if improve else score(50 if len(calls)==1 else 40)
    monkeypatch.setattr(quality,'model_step',step)
    result=await jobs.execute(job)
    state=quality.public(identity)
    assert state['score']==50 and state['target_score']==target
    assert state['attempts']==retries and state['stop_reason']=='attempt_limit'
    assert calls.count(True)==retries and calls.count(False)==retries+1
    assert len(state['versions'])==retries+1
    assert db.fetchone('SELECT payload_json FROM artifacts WHERE id=?',(result['id'],))
    assert 'never-persist-this' not in db.fetchone('SELECT state_json FROM quality_runs')['state_json']


@pytest.mark.asyncio
async def test_target_stops_and_failures_never_fabricate_score(env,monkeypatch):
    job,identity=enqueue()
    async def step(*args,**kwargs):return score(80)
    monkeypatch.setattr(quality,'model_step',step)
    await jobs.execute(job)
    assert quality.public(identity)['attempts']==0
    assert quality.public(identity)['met_target']
    job,identity=enqueue()
    async def fail(*args,**kwargs):raise ValueError('bad model JSON')
    monkeypatch.setattr(quality,'model_step',fail)
    await jobs.execute(job)
    state=quality.public(identity)
    assert state['score'] is None and state['target_id']
    assert state['stop_reason']=='score_unavailable' and state['attempts']==0


@pytest.mark.asyncio
async def test_lower_adopts_without_calls_and_does_not_rewrite_old_versions(env,monkeypatch):
    db,client=env
    job,identity=enqueue('high')
    async def step(run,candidate,improve=False):return {**candidate['content'],'content':'new'} if improve else score(70)
    monkeypatch.setattr(quality,'model_step',step)
    await jobs.execute(job)
    state=quality.public(identity)
    before=db.fetchone('SELECT content_json FROM quality_versions WHERE id=?',(state['version_id'],))
    body={'base_version':state['version_id'],'action':'lower','quality_level':'low','attempts':5,'request_id':'same-click'}
    result=client.post(f'/quality-runs/{identity}/actions',json=body)
    assert result.status_code==200 and result.json()['met_target']
    assert len(db.fetchall('SELECT * FROM jobs'))==1
    assert client.post(f'/quality-runs/{identity}/actions',json=body).json()==result.json()
    assert before==db.fetchone('SELECT content_json FROM quality_versions WHERE id=?',(state['version_id'],))


@pytest.mark.asyncio
async def test_manual_retry_is_exact_addition_and_grade_failure_regrades(env,monkeypatch):
    db,client=env
    job,identity=enqueue()
    async def fail(*args,**kwargs):raise ValueError('bad JSON')
    monkeypatch.setattr(quality,'model_step',fail)
    await jobs.execute(job)
    first=quality.public(identity)
    response=client.post(f'/quality-runs/{identity}/actions',json={'base_version':first['version_id'],'action':'retry','attempts':2,'request_id':'append'})
    assert response.status_code==200
    calls=[]
    async def step(run,candidate,improve=False):
        calls.append(improve)
        return {**candidate['content'],'content':'improved'} if improve else score(30)
    monkeypatch.setattr(quality,'model_step',step)
    await jobs.execute(db.fetchone('SELECT * FROM jobs WHERE id=?',(response.json()['job_id'],)))
    assert calls==[False,True,False,True,False]
    assert quality.public(identity)['attempts']==2


@pytest.mark.asyncio
async def test_keep_during_grade_prevents_late_replacement(env,monkeypatch):
    _,client=env
    job,identity=enqueue('extreme')
    async def step(run,candidate,improve=False):
        response=client.post(f'/quality-runs/{identity}/actions',json={'base_version':candidate['id'],'action':'keep','request_id':'keep'})
        assert response.status_code==200
        return score(10)
    monkeypatch.setattr(quality,'model_step',step)
    await jobs.execute(job)
    state=quality.public(identity)
    assert state['stop_reason']=='kept' and state['attempts']==0 and state['score'] is None


@pytest.mark.asyncio
async def test_source_revision_change_and_unknown_inflight_preserve_content(env,monkeypatch):
    db,_=env
    job,identity=enqueue()
    async def step(*args,**kwargs):
        db.execute("UPDATE sources SET revision_id='changed' WHERE id='s'")
        return score(99)
    monkeypatch.setattr(quality,'model_step',step)
    await jobs.execute(job)
    assert quality.public(identity)['stop_reason']=='sources_changed'
    run=quality.load(identity);run['data'].update(phase='scoring',in_flight=True);quality.save(run)
    await jobs.execute(job)
    assert quality.public(identity)['stop_reason']=='interrupted'


@pytest.mark.asyncio
async def test_async_chat_first_version_and_quiz_public_redaction(env,monkeypatch):
    db,client=env
    response=client.post('/notebooks/n/chat-runs',json={'question':'总结资料中的3个核心要点','source_ids':['s']})
    assert response.status_code==202
    identity=response.json()['run_id']
    async def step(*args,**kwargs):return score(80)
    monkeypatch.setattr(quality,'model_step',step)
    await jobs.execute(db.fetchone('SELECT * FROM jobs WHERE id=?',(response.json()['id'],)))
    messages=client.get(f"/conversations/{response.json()['conversation_id']}/messages").json()
    assert messages[-1]['content']=='First draft' and messages[-1]['metadata']['quality_control']['score']==80
    job,identity=enqueue(kind='quiz')
    await jobs.execute(job)
    state=client.get(f'/quality-runs/{identity}').json()
    assert 'secret answer' not in json.dumps(state)
    artifact=client.get('/artifacts/'+state['target_id']).json()
    assert 'answer_index' not in artifact['payload']['items'][0]
    assert 'citations' not in artifact['payload']
    assert 'secret answer' not in json.dumps(artifact['payload'].get('quality_control'))


def test_invalid_rubric_and_structures():
    for value in (-1,41,True,1.5):
        with pytest.raises(ValueError):
            quality.validate_score({'scores':{'fidelity':value,'requirements':30,'structure':20,'clarity':10},'blocking':False,'suggestions':[]},{})
    with pytest.raises(ValueError):quality.accept_rewrite('quiz',{}, {'items':[{'question':'x'}]})


@pytest.mark.asyncio
async def test_real_chat_pipeline_uses_one_grade_call_and_records_it(env,monkeypatch):
    import httpx
    db,client=env
    # Restore the actual candidate adapter; every provider request is intercepted.
    from importlib.util import spec_from_file_location, module_from_spec
    spec=spec_from_file_location('sandevistan_read._quality_adapter',quality.__file__)
    module=module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'DB',db)
    monkeypatch.setattr(quality,'first_candidate',module.first_candidate)
    chunk=db.fetchone("SELECT c.*,s.filename FROM chunks c JOIN sources s ON s.id=c.source_id")
    chunk['locator']={}
    monkeypatch.setattr(services,'retrieve',lambda *args,**kwargs:[chunk])
    requests=[]
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        text='A source passage about learning. [S1]' if len(requests)==1 else json_dump({'scores':{'fidelity':35,'requirements':25,'structure':15,'clarity':9},'blocking':False,'suggestions':[]})
        return httpx.Response(200,json={'choices':[{'message':{'content':text},'finish_reason':'stop'}],'usage':{'prompt_tokens':100,'completion_tokens':50}})
    original=httpx.AsyncClient
    monkeypatch.setattr(providers.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    response=client.post('/notebooks/n/chat-runs',json={'question':'生成双人播客脚本','source_ids':['s']}).json()
    result=await jobs.execute(db.fetchone('SELECT * FROM jobs WHERE id=?',(response['id'],)))
    assert len(requests)==2
    assert result['quality_control']['score']==84
    assert result['usage']['calls']==2 and result['usage']['accounted_tokens']==300
    assert 'quality_audit' in result['usage']['stages']
    assert '双人播客脚本' in requests[0]['messages'][-1]['content']


@pytest.mark.asyncio
async def test_manual_continuation_cannot_reset_token_limit(env,monkeypatch):
    import httpx
    db,client=env
    job=jobs.enqueue('summary','n',{'quality_level':'low','source_ids':['s'],'token_limit':1024})
    identity=json_load(job['payload_json'])['quality_run_id']
    async def grade(run,candidate,improve=False):
        if improve:raise RuntimeError('已达到本次自定义用量上限')
        return score(30)
    monkeypatch.setattr(quality,'model_step',grade)
    await jobs.execute(job)
    run_id=db.fetchone('SELECT id FROM generation_runs')['id']
    db.execute("INSERT INTO provider_calls(id,run_id,provider_id,model,role,stage,state,estimated_input_tokens,output_limit,accounted_tokens,controls_json,created_at) VALUES('used',?,'main','fixture','main','quality_audit','complete',800,200,1000,'{}','now')",(run_id,))
    state=quality.public(identity)
    response=client.post(f'/quality-runs/{identity}/actions',json={'base_version':state['version_id'],'action':'retry','attempts':1,'request_id':'retry-cap'}).json()
    sent=[]
    original=httpx.AsyncClient
    monkeypatch.setattr(providers.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(lambda request:sent.append(request)),**kwargs))
    async def improve(run,candidate,improve=False):
        await providers._chat_once({**run['data']['provider'],'api_key':''},[{'role':'user','content':'test'}],json_mode=False,timeout=1,max_tokens=128,temperature=.1)
        raise AssertionError('Budget must reject before sending')
    monkeypatch.setattr(quality,'model_step',improve)
    result=await jobs.execute(db.fetchone('SELECT * FROM jobs WHERE id=?',(response['job_id'],)))
    assert not sent
    assert result['quality_control']['stop_reason']=='budget_exhausted'
    assert result['usage']['accounted_tokens']==1000


@pytest.mark.asyncio
async def test_active_flashcard_session_keeps_original_version(env,monkeypatch):
    db,client=env
    from sandevistan_read import study_sessions
    monkeypatch.setattr(study_sessions,'DB',db)
    async def candidate(run):
        return {'items':[{'id':'c1','front':'Question','back':'Original answer','citations':[]}],'citations':[]},None
    monkeypatch.setattr(quality,'first_candidate',candidate)
    sessions=[]
    async def step(run,candidate,improve=False):
        if improve:
            return {**candidate['content'],'items':[{'id':'c1','front':'Question','back':'Improved answer','citations':[]}]}
        if not sessions:
            response=client.post(f"/artifacts/{candidate['target_id']}/study-sessions",json={})
            assert response.status_code==200
            sessions.append(response.json())
            return score(40)
        return score(85)
    monkeypatch.setattr(quality,'model_step',step)
    job,identity=enqueue(kind='flashcard')
    await jobs.execute(job)
    current=client.get('/study-sessions/'+sessions[0]['id']).json()
    assert current['items'][0]['back']=='Original answer'
    assert current['artifact_id']!=quality.public(identity)['target_id']
    listing=client.get('/notebooks/n/artifacts?view=summary').json()
    assert len(listing)==1 and listing[0]['id']==quality.public(identity)['target_id']


@pytest.mark.asyncio
async def test_podcast_renders_selected_script_once_and_reuses_unchanged_best(env,monkeypatch):
    db,client=env
    async def candidate(run):
        return {'version':4,'turns':[{'id':'t1','speaker':'HOST_A','text':'First'},{'id':'t2','speaker':'HOST_B','text':'Second'}],'script':'First, Second','source_ids':['s'],'language':'en','citations':[]},None
    monkeypatch.setattr(quality,'first_candidate',candidate)
    async def step(run,candidate,improve=False):
        return {**candidate['content'],'script':'Attempt'} if improve else score(70)
    monkeypatch.setattr(quality,'model_step',step)
    renders=[]
    async def render(notebook_id,payload,job_id):
        renders.append(payload['_quality_script']['script'])
        identity='rendered-'+job_id
        db.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(identity,'n','podcast','Audio','["s"]','en','ready',json_dump(payload['_quality_script']),'[]',None,'now','now'))
        return {'id':identity}
    monkeypatch.setattr(jobs,'_podcast',render)
    job,identity=enqueue('high','podcast')
    await jobs.execute(job)
    state=quality.public(identity)
    assert renders==['First, Second']
    response=client.post(f'/quality-runs/{identity}/actions',json={'base_version':state['version_id'],'action':'retry','attempts':1,'request_id':'audio-repeat'}).json()
    await jobs.execute(db.fetchone('SELECT * FROM jobs WHERE id=?',(response['job_id'],)))
    assert renders==['First, Second']
    assert len(client.get('/notebooks/n/artifacts?view=summary').json())==1
