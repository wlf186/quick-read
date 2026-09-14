from __future__ import annotations
import io
import json
import zipfile
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from sandevistan_read import podcast


def _fixture_db(tmp_path, monkeypatch, *, media_path=None, payload=None, artifact_type='podcast'):
    from sandevistan_read import app
    from sandevistan_read.database import Database, json_dump
    db = Database(tmp_path / 'download.db')
    db.initialize()
    monkeypatch.setattr(app, 'DB', db)
    db.execute("INSERT INTO notebooks(id,title,created_at,updated_at) VALUES('n','Fixture','now','now')")
    payload = payload if payload is not None else {
        'delivery_status': 'full',
        'duration': {'target_minutes': 5, 'actual_seconds': 280.0},
        'chapters': [{'title': 'Intro', 'turn_start': 0, 'turn_end': 1}],
        'turns': [
            {'speaker': 'HOST_A', 'text': '今天我们讨论记录排序。', 'dialogue_act': 'question', 'claim_ids': [], 'citation_ids': ['E1'], 'start_seconds': 0.0, 'end_seconds': 4.0},
            {'speaker': 'HOST_B', 'text': '节点会检查交易顺序。', 'dialogue_act': 'explain', 'claim_ids': ['C1'], 'citation_ids': ['E1'], 'start_seconds': 4.0, 'end_seconds': 9.5},
        ],
        'warnings': [],
    }
    citations = [{'id': 'E1', 'filename': '报告.pdf', 'quote': '节点检查交易顺序。', 'locator': {'page': 3}}]
    db.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
               ('artifact_p1', 'n', artifact_type, '双人音频解读', '[]', 'zh-CN',
                'partial' if media_path is None else 'ready', json_dump(payload), json_dump(citations), media_path, 'now', 'now'))
    return db


def _fixture_paths(tmp_path, monkeypatch, *, with_audio=True):
    from sandevistan_read import app
    artifacts = tmp_path / 'artifacts'
    media = None
    if with_audio:
        media_dir = artifacts / 'podcast_p1'
        media_dir.mkdir(parents=True)
        media = media_dir / 'podcast.m4a'
        media.write_bytes(b'fake-audio')
    monkeypatch.setattr(app, 'PATHS', SimpleNamespace(root=tmp_path, artifacts=artifacts))
    return media


def _zip_names(response) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        return set(archive.namelist())


def test_submit_accepts_unready_audio(tmp_path, monkeypatch):
    from sandevistan_read import app
    from sandevistan_read.schemas import PodcastRequest
    _fixture_db(tmp_path, monkeypatch)
    enqueued = []
    monkeypatch.setattr(app, 'enqueue', lambda kind, notebook_id, payload: enqueued.append((kind, notebook_id, payload)) or {'id': 'job_x'})
    result = app.podcast('n', PodcastRequest(language='zh-CN', focus=''))
    assert enqueued and enqueued[0][0] == 'podcast' and result == {'id': 'job_x'}


def test_script_markdown_minimal_format():
    md = podcast.script_markdown({'turns': [
        {'speaker': 'HOST_A', 'text': ' 什么是排序？ ', 'citation_ids': ['E1'], 'start_seconds': 0.0},
        {'speaker': 'HOST_B', 'text': '排序决定先后。', 'claim_ids': ['C1']},
        {'speaker': 'HOST_C', 'text': '补充一点。'},
    ]})
    assert md == '1. A：什么是排序？\n\n2. B：排序决定先后。\n\n3. HOST_C：补充一点。'


def test_script_markdown_legacy_fallback():
    assert podcast.script_markdown({'script': '旧版脚本文本'}) == '旧版脚本文本'
    assert podcast.script_markdown({}) == ''
    assert podcast.script_markdown({'turns': []}) == ''


def test_download_all_with_audio_produces_zip(tmp_path, monkeypatch):
    from sandevistan_read import app
    _fixture_db(tmp_path, monkeypatch, media_path='artifacts/podcast_p1/podcast.m4a')
    _fixture_paths(tmp_path, monkeypatch)
    response = app.download_artifact('artifact_p1')
    assert response.media_type == 'application/zip'
    assert 'attachment' in response.headers['content-disposition']
    assert _zip_names(response) == {'podcast.m4a', 'script.md', 'script.json'}


def test_download_all_script_only_returns_script_zip(tmp_path, monkeypatch):
    from sandevistan_read import app
    _fixture_db(tmp_path, monkeypatch, media_path=None)
    _fixture_paths(tmp_path, monkeypatch, with_audio=False)
    response = app.download_artifact('artifact_p1')
    assert response.media_type == 'application/zip'
    assert _zip_names(response) == {'script.md', 'script.json'}


def test_download_audio_missing_404(tmp_path, monkeypatch):
    from sandevistan_read import app
    _fixture_db(tmp_path, monkeypatch, media_path=None)
    _fixture_paths(tmp_path, monkeypatch, with_audio=False)
    with pytest.raises(HTTPException) as exc:
        app.download_artifact('artifact_p1', part='audio')
    assert exc.value.status_code == 404


def test_download_non_podcast_404(tmp_path, monkeypatch):
    from sandevistan_read import app
    _fixture_db(tmp_path, monkeypatch, artifact_type='summary')
    _fixture_paths(tmp_path, monkeypatch, with_audio=False)
    with pytest.raises(HTTPException) as exc:
        app.download_artifact('artifact_p1')
    assert exc.value.status_code == 404


def test_download_rejects_path_escape(tmp_path, monkeypatch):
    from sandevistan_read import app
    _fixture_db(tmp_path, monkeypatch, media_path='../escape.m4a')
    _fixture_paths(tmp_path, monkeypatch, with_audio=False)
    (tmp_path / 'escape.m4a').write_bytes(b'escape')
    with pytest.raises(HTTPException) as exc:
        app.download_artifact('artifact_p1', part='audio')
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        app.artifact_media('artifact_p1')
    assert exc.value.status_code == 404


def test_download_script_formats_and_json_structure(tmp_path, monkeypatch):
    from sandevistan_read import app
    _fixture_db(tmp_path, monkeypatch, media_path=None)
    _fixture_paths(tmp_path, monkeypatch, with_audio=False)
    markdown = app.download_artifact('artifact_p1', part='script', script_format='markdown')
    assert markdown.media_type.startswith('text/markdown')
    assert markdown.body.decode() == '1. A：今天我们讨论记录排序。\n\n2. B：节点会检查交易顺序。'
    raw = app.download_artifact('artifact_p1', part='script', script_format='json')
    assert raw.media_type == 'application/json'
    data = json.loads(raw.body.decode())
    assert data['delivery_status'] == 'full' and data['language'] == 'zh-CN'
    assert [turn['index'] for turn in data['turns']] == [1, 2]
    assert data['turns'][0]['speaker'] == 'HOST_A' and 'citation_ids' in data['turns'][0]
    assert data['citations'][0]['id'] == 'E1' and data['chapters'][0]['title'] == 'Intro'
    zipped = app.download_artifact('artifact_p1', part='script', script_format='both')
    assert _zip_names(zipped) == {'script.md', 'script.json'}
