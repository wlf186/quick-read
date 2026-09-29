"""Offline contract tests for configuration-independent reuse and learning flows."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sandevistan_read import app as api_module, jobs, observability, podcast, providers, study_sessions
from sandevistan_read.database import Database, json_dump, json_load
from sandevistan_read.schemas import SummaryRequest


@pytest.fixture
def db(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.sqlite")
    db.initialize()
    db.execute("INSERT INTO notebooks(id,title,description,created_at,updated_at) VALUES('n','Notebook','','now','now')")
    db.execute("INSERT INTO sources(id,notebook_id,revision_id,filename,media_type,size_bytes,sha256,blob_path,state,selected,created_at,updated_at) VALUES('s','n','r','source.txt','text/plain',1,'h','blobs/source.txt','ready',1,'now','now')")
    for module in (api_module, jobs, observability, study_sessions):
        monkeypatch.setattr(module, 'DB', db)
    monkeypatch.setattr(api_module.CONFIG.security, 'access_key', '')
    monkeypatch.setattr(jobs, 'active_provider', lambda role: None)
    return db


def artifact(db, kind='podcast', payload=None, identity='a'):
    payload = payload or {'version': 2, 'turns': [{'id': 't1', 'speaker': 'HOST_A', 'text': 'Evidence'}, {'id': 't2', 'speaker': 'HOST_B', 'text': 'Limits'}]}
    db.execute("INSERT INTO artifacts(id,notebook_id,type,title,scope_json,language,status,payload_json,citations_json,created_at,updated_at) VALUES(?,'n',?,'Result','[\"s\"]','en','ready',?,'[]','now','now')", (identity, kind, json_dump(payload)))
    return db.fetchone('SELECT * FROM artifacts WHERE id=?', (identity,))


def audio_ready(monkeypatch):
    provider = {'id': 'audio', 'name': 'Audio', 'kind': 'sandevistan_audio', 'base_url': 'http://audio.invalid', 'model': 'tts', 'config': {'host_a': 'A', 'host_b': 'B'}, 'capabilities': {}, 'api_key': 'private-test-key'}
    monkeypatch.setattr(api_module, 'active_provider', lambda role: provider)
    monkeypatch.setattr(providers, 'audio_provider_readiness', lambda p: (True, ''))
    return provider


def test_atomic_selection_and_empty_scope(db):
    client = TestClient(api_module.api)
    assert client.put('/notebooks/n/source-selection', json={'source_ids': ['s', 'missing']}).status_code == 409
    assert db.fetchone("SELECT selected FROM sources WHERE id='s'")['selected'] == 1
    assert client.put('/notebooks/n/source-selection', json={'source_ids': []}).status_code == 200
    assert db.fetchone("SELECT selected FROM sources WHERE id='s'")['selected'] == 0
    db.execute("UPDATE sources SET state='queued' WHERE id='s'")
    assert client.put('/notebooks/n/source-selection', json={'source_ids': ['s']}).status_code == 409


def test_original_file_revision_and_path_boundary(db, tmp_path, monkeypatch):
    blobs = tmp_path / 'blobs'; blobs.mkdir(); (blobs / 'source.txt').write_text('original')
    monkeypatch.setattr(api_module, 'PATHS', SimpleNamespace(root=tmp_path, blobs=blobs))
    client = TestClient(api_module.api)
    assert client.get('/sources/s/file?revision=old').status_code == 409
    response = client.get('/sources/s/file?revision=r')
    assert response.text == 'original' and 'attachment' in response.headers['content-disposition']
    db.execute("UPDATE sources SET blob_path='../outside.txt' WHERE id='s'")
    assert client.get('/sources/s/file?revision=r').status_code == 404


def test_exports_keep_requested_version_and_require_access(db, monkeypatch):
    artifact(db, 'summary', {'content': 'Original result', 'quality_control': {'score': 60}}, 'old')
    artifact(db, 'summary', {'content': 'New result'}, 'new')
    client = TestClient(api_module.api)
    response = client.get('/artifacts/old/summary.md')
    assert 'Original result' in response.text and 'New result' not in response.text
    assert '不代表事实正确率' in response.text
    monkeypatch.setattr(api_module.CONFIG.security, 'access_key', 'test-key')
    assert client.get('/artifacts/old/summary.md').status_code == 401
    assert client.get('/sources/s/file?revision=r').status_code == 401


def test_summary_defaults_and_bounds():
    value = SummaryRequest()
    assert (value.length, value.focus, value.quality_level) == ('standard', '', 'low')
    with pytest.raises(ValueError):
        SummaryRequest(focus='x' * 1001)


def test_overview_read_only_pause_restore_preserves_schedule(db):
    artifact(db, 'flashcard', {'items': [{'id': 'f', 'front': 'Question', 'back': 'Answer'}]})
    before = db.fetchall('SELECT * FROM study_sessions')
    assert study_sessions.study_overview()['groups'][0]['new'] == 1
    assert db.fetchall('SELECT * FROM study_sessions') == before
    assert db.fetchall('SELECT * FROM flashcard_states') == []
    session = study_sessions.create_session('a', 'due')
    review = study_sessions.review_flashcard(session['id'], 'f', 'good')
    schedule = db.fetchone("SELECT fsrs_json,due_at,last_rating FROM flashcard_states")
    study_sessions.suspend_flashcard('a', 'f')
    assert study_sessions.study_overview()['groups'][0]['paused'][0]['id'] == 'f'
    restored = study_sessions.restore_flashcard('a', 'f', session['id'])
    assert restored['session']['items'][0]['id'] == 'f'
    assert db.fetchone("SELECT fsrs_json,due_at,last_rating FROM flashcard_states") == schedule
    assert schedule['due_at'] == review['due_at']


def test_audio_preview_pins_without_credentials_and_deduplicates(db, monkeypatch):
    original = artifact(db)
    provider = audio_ready(monkeypatch)
    client = TestClient(api_module.api)
    preview = client.get('/artifacts/a/audio-renders').json()
    body = {k: preview[k] for k in ('script_hash', 'provider_hash')} | {'request_id': 'one'}
    result = client.post('/artifacts/a/audio-renders', json=body)
    assert result.status_code == 202
    job_id = result.json()['job_id']
    job = db.fetchone('SELECT * FROM jobs WHERE id=?', (job_id,))
    assert job['kind'] == 'podcast_audio'
    assert 'private-test-key' not in job['payload_json']
    assert json_load(job['payload_json'], {})['_quality_audio_provider']['config']['host_a'] == 'A'
    assert client.post('/artifacts/a/audio-renders', json=body).json()['job_id'] == job_id
    assert client.post('/artifacts/a/audio-renders', json=body | {'request_id': 'two'}).json()['job_id'] == job_id
    provider['config']['host_a'] = 'changed'
    assert client.post('/artifacts/a/audio-renders', json=body | {'request_id': 'three'}).status_code == 409
    assert db.fetchone("SELECT * FROM artifacts WHERE id='a'") == original
    assert db.fetchall('SELECT * FROM quality_runs') == []


@pytest.mark.asyncio
async def test_audio_execution_bypasses_main_and_attaches_separate_version(db, monkeypatch):
    original = artifact(db); audio_ready(monkeypatch)
    client = TestClient(api_module.api)
    preview = client.get('/artifacts/a/audio-renders').json()
    result = client.post('/artifacts/a/audio-renders', json={k: preview[k] for k in ('script_hash', 'provider_hash')} | {'request_id': 'one'}).json()
    async def render(notebook_id, payload, job_id):
        assert payload['_audio_only'] and payload['_quality_script']['turns'][0]['text'] == 'Evidence'
        artifact(db, identity='child')
        return {'id': 'child'}
    monkeypatch.setattr(jobs, '_synthesize_podcast', render)
    monkeypatch.setattr(jobs, 'active_provider', lambda role: pytest.fail('Execution must not resolve MAIN'))
    await jobs.execute(db.fetchone('SELECT * FROM jobs WHERE id=?', (result['job_id'],)))
    assert db.fetchone("SELECT artifact_id FROM podcast_audio_renders WHERE id=?", (result['id'],))['artifact_id'] == 'child'
    assert db.fetchone("SELECT * FROM artifacts WHERE id='a'") == original
    assert [row['id'] for row in client.get('/notebooks/n/artifacts').json()] == ['a']


def test_audio_restart_only_stops_started_requests(db):
    queued = jobs.enqueue('podcast_audio', 'n', {})
    running = jobs.enqueue('podcast_audio', 'n', {})
    db.execute("UPDATE jobs SET state='running' WHERE id=?", (running['id'],))
    db.reset_running_jobs()
    assert db.fetchone('SELECT state FROM jobs WHERE id=?', (queued['id'],))['state'] == 'queued'
    assert db.fetchone('SELECT state FROM jobs WHERE id=?', (running['id'],))['state'] == 'failed'


def test_legacy_script_requires_explicit_labels(db):
    row = artifact(db, payload={'script': 'A: Hello\nB: World'})
    assert len(podcast.reusable_audio_script(row)['turns']) == 2
    row['payload_json'] = json_dump({'script': 'Unlabelled prose'})
    with pytest.raises(ValueError, match='A/B'):
        podcast.reusable_audio_script(row)


def test_v10_migration_backup_is_idempotent(db):
    backup = db.path.with_name(db.path.name + '.pre-v10.bak')
    saved = backup.read_bytes()
    db.initialize()
    assert backup.read_bytes() == saved
    assert len(db.fetchall('SELECT * FROM schema_versions WHERE version=10')) == 1


@pytest.mark.asyncio
async def test_real_audio_pipeline_reuses_parts_and_never_calls_main(db, monkeypatch, tmp_path):
    import copy
    import wave
    original = artifact(db)
    provider = audio_ready(monkeypatch)
    monkeypatch.setattr(jobs, 'PATHS', SimpleNamespace(root=tmp_path, job_work=tmp_path/'work', artifacts=tmp_path/'artifacts'))
    monkeypatch.setattr(jobs.CONFIG, 'tools', SimpleNamespace(ffmpeg_path=''))
    monkeypatch.setattr(jobs, 'register_resource', lambda *args: None)
    monkeypatch.setattr(jobs, 'audio_provider_readiness', lambda p: (True, ''))
    monkeypatch.setattr(jobs, 'provider_by_id', lambda identity: provider)
    monkeypatch.setattr(jobs, 'active_provider', lambda role: pytest.fail('Bound audio must not resolve an active provider'))
    script = podcast.reusable_audio_script(original)
    snapshot = {k: v for k, v in provider.items() if k != 'api_key'}
    calls = []
    async def synthesize(text, voice, destination, **kwargs):
        assert kwargs['provider']['model'] == 'tts'
        calls.append(text)
        if text == 'Limits' and calls.count('Limits') == 1:
            raise RuntimeError('interrupted')
        with wave.open(str(destination), 'wb') as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(24000)
            output.writeframes(b'\x00\x00' * 2400)
    async def transcribe(*args, **kwargs):
        return {}
    monkeypatch.setattr(jobs, 'synthesize', synthesize)
    monkeypatch.setattr(jobs, 'transcribe_audio', transcribe)
    monkeypatch.setattr(jobs, 'assess_transcription', lambda *args: {'passed': False, 'error_rate': .8})
    async def no_main(*args, **kwargs):
        pytest.fail('Audio-only job called MAIN')
    monkeypatch.setattr(jobs, 'build_podcast_script', no_main)
    payload = {'_audio_only': True, '_quality_script': script, '_quality_audio_provider': snapshot, 'provider_ids': {'audio': 'audio'}, 'source_ids': ['s']}
    # Enqueue without consulting providers; the API pins the snapshot before this.
    monkeypatch.setattr(jobs, 'active_provider', lambda role: provider if role == 'audio' else pytest.fail('MAIN requested'))
    first = jobs.enqueue('podcast_audio', 'n', copy.deepcopy(payload))
    with pytest.raises(RuntimeError, match='interrupted'):
        await jobs._synthesize_podcast('n', payload, first['id'])
    second = jobs.enqueue('podcast_audio', 'n', copy.deepcopy(payload))
    result = await jobs._synthesize_podcast('n', payload | {'audio_retry_job_id': first['id']}, second['id'])
    assert calls == ['Evidence', 'Limits', 'Limits']
    assert result['id'] != original['id']
    assert db.fetchone("SELECT * FROM artifacts WHERE id='a'") == original
    rendered = json_load(db.fetchone('SELECT payload_json FROM artifacts WHERE id=?', (result['id'],))['payload_json'], {})
    assert rendered['audio_quality']['passed'] is False
    assert rendered['turns'][0]['text'] == 'Evidence'


def test_chat_records_exact_scope_on_both_messages(db, monkeypatch):
    from sandevistan_read import quality
    monkeypatch.setattr(api_module, 'source_scope', lambda notebook, ids: ['s'])
    monkeypatch.setattr(api_module, 'enqueue', lambda *args: {'id': 'job', 'payload_json': json_dump({'quality_run_id': 'quality'})})
    monkeypatch.setattr(quality, 'public', lambda identity: {'run_id': identity, 'phase': 'queued'})
    response = TestClient(api_module.api).post('/notebooks/n/chat-runs', json={'question': 'Question', 'source_ids': ['s']})
    assert response.status_code == 202
    rows = db.fetchall('SELECT metadata_json,citations_json FROM messages')
    assert len(rows) == 2
    for row in rows:
        assert json_load(row['metadata_json'], {})['source_scope'] == [{'id': 's', 'revision_id': 'r', 'filename': 'source.txt'}]
        assert json_load(row['citations_json'], []) == []


def test_legacy_review_overview_does_not_write_but_sessions_migrate(db):
    from datetime import UTC, datetime
    artifact(db, 'flashcard', {'items': [{'id': 'f', 'front': 'Question', 'back': 'Answer'}]})
    db.execute("INSERT INTO flashcard_reviews(id,artifact_id,card_id,rating,created_at) VALUES('review','a','f','easy',?)", (datetime.now(UTC).isoformat(),))
    group = study_sessions.study_overview()['groups'][0]
    assert group['new'] == 0 and group['due'] == 0 and group['next_due']
    assert db.fetchall('SELECT * FROM flashcard_states') == []
    session = study_sessions.create_session('a', 'due')
    assert session['items'] == []
    assert db.fetchone("SELECT last_rating FROM flashcard_states WHERE card_id='f'")['last_rating'] == 'easy'


@pytest.mark.asyncio
async def test_old_health_request_cannot_repopulate_invalidated_cache(monkeypatch):
    import asyncio
    monkeypatch.setattr(api_module, '_STATUS_CACHE', {'at': 0., 'value': None, 'revision': 0})
    monkeypatch.setattr(api_module, '_STATUS_LOCK', asyncio.Lock())
    entered, release = asyncio.Event(), asyncio.Event()
    async def old_health(role):
        entered.set(); await release.wait()
        return {'ok': False, 'message': 'old'}
    monkeypatch.setattr(api_module, 'health', old_health)
    task = asyncio.create_task(api_module.status())
    await entered.wait()
    api_module.invalidate_status_cache()
    release.set(); await task
    assert api_module._STATUS_CACHE['value'] is None
    async def new_health(role):
        return {'ok': True, 'message': 'current'}
    monkeypatch.setattr(api_module, 'health', new_health)
    assert (await api_module.status())['providers']['main']['message'] == 'current'
