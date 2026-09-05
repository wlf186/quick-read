import importlib
import json
import sqlite3
from urllib.parse import parse_qs
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from sandevistan_read.database import Database, json_dump, utc_now
from sandevistan_read import providers
from sandevistan_read.schemas import PodcastRequest, ProviderCreate, ProviderUpdate


def mock_client(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(providers.httpx, "AsyncClient", factory)


def candidate(**overrides):
    value = {
        "name": "Local",
        "role": "main",
        "kind": "ollama",
        "base_url": "http://localhost:11434/api",
        "model": "qwen:latest",
        "api_key": "",
        "config": {},
    }
    value.update(overrides)
    return value


def audio_capabilities() -> dict:
    return {
        "models": [{
            "id": "qwen3-tts-1.7b",
            "name": "Qwen3 TTS 1.7B",
            "installed": True,
            "devices": [{"id": "gpu", "available": True}, {"id": "cpu", "available": True}],
        }],
        "asr": {
            "default_model": "qwen3-asr-0.6b",
            "recommended": {"model": "qwen3-asr-0.6b", "compute_device": "gpu"},
            "models": [{
                "id": "qwen3-asr-0.6b",
                "name": "Qwen3 ASR 0.6B",
                "installed": True,
                "devices": [
                    {"id": "gpu", "available": True, "default": True},
                    {"id": "cpu", "available": True, "default": False},
                ],
            }],
            "diarization": "CAM++",
            "languages": ["Chinese", "English"],
            "aligner_languages": ["Chinese", "English"],
            "timestamp_precisions": ["segment", "word_or_character"],
        }
    }


def test_audio_capability_uses_only_qualified_service_default_and_normalizes_sequence(monkeypatch) -> None:
    monkeypatch.setattr(
        providers,
        "PODCAST_TTS_QUALIFIED_TARGETS",
        {"qwen3-tts-0.6b": {
            "checkpoints": providers.PODCAST_TTS_CANDIDATE_REVISIONS["qwen3-tts-0.6b"],
            "devices": ["gpu"],
        }},
    )
    payload = {
        "tts": {
            "default_model": "qwen3-tts-0.6b",
            "model_capabilities": [{
                "id": "qwen3-tts-0.6b", "installed": True, "default": True,
                "compute_devices": [{"id": "gpu", "available": True}],
                "checkpoints": [
                    {"variant": variant, "revision": revision}
                    for variant, revision in providers.PODCAST_TTS_CANDIDATE_REVISIONS["qwen3-tts-0.6b"].items()
                ],
            }],
            "sequence_jobs": {
                "supported": True, "contract_version": 1, "endpoint": "/api/v1/tts/sequence-jobs",
                "voice_modes": ["preset", "voiceprint"], "artifact_mode": "per_item", "format": "wav",
                "max_items": 100, "max_total_chars": 5000,
            },
        }
    }
    result = providers._normalized_audio_capabilities(payload)
    assert result["recommended"] == {
        "model": "qwen3-tts-0.6b", "compute_device": "gpu", "reason": "service_default",
    }
    assert result["sequence_jobs"]["supported"] is True
    payload["tts"]["model_capabilities"][0]["checkpoints"][0]["revision"] = "changed"
    assert providers._normalized_audio_capabilities(payload)["recommended"] is None


def test_qualified_gpu_default_does_not_authorize_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        providers,
        "PODCAST_TTS_QUALIFIED_TARGETS",
        {"qwen3-tts-0.6b": {
            "checkpoints": providers.PODCAST_TTS_CANDIDATE_REVISIONS["qwen3-tts-0.6b"],
            "devices": ["gpu"],
        }},
    )
    checkpoints = [
        {"variant": variant, "revision": revision}
        for variant, revision in providers.PODCAST_TTS_CANDIDATE_REVISIONS["qwen3-tts-0.6b"].items()
    ]
    payload = {"tts": {
        "default_model": "qwen3-tts-0.6b",
        "model_capabilities": [
            {
                "id": "qwen3-tts-0.6b", "installed": True, "default": True,
                "compute_devices": [{"id": "cpu", "available": True}], "checkpoints": checkpoints,
            },
            {
                "id": "qwen3-tts-1.7b", "installed": True,
                "compute_devices": [{"id": "cpu", "available": True}], "checkpoints": [],
            },
        ],
    }}
    assert providers._normalized_audio_capabilities(payload)["recommended"] == {
        "model": "qwen3-tts-1.7b", "compute_device": "cpu", "reason": "installed_fallback",
    }


def test_unqualified_service_default_does_not_replace_higher_quality_model() -> None:
    payload = {
        "tts": {
            "default_model": "qwen3-tts-0.6b",
            "model_capabilities": [
                {
                    "id": "qwen3-tts-0.6b", "name": "Qwen3 TTS 0.6B", "installed": True,
                    "default": True, "compute_devices": [{"id": "gpu", "available": True}],
                    "checkpoints": [],
                },
                {
                    "id": "qwen3-tts-1.7b", "name": "Qwen3 TTS 1.7B", "installed": True,
                    "compute_devices": [{"id": "cpu", "available": True}],
                    "checkpoints": [],
                },
            ],
        }
    }

    assert providers._normalized_audio_capabilities(payload)["recommended"] == {
        "model": "qwen3-tts-1.7b", "compute_device": "cpu", "reason": "installed_fallback",
    }


def test_tts_auto_selection_preserves_user_authored_preset_instructions() -> None:
    normalized = {
        "models": [
            {"id": "fast", "installed": True, "devices": [{"id": "gpu", "available": True}], "controls": {}},
            {
                "id": "expressive-1.7b", "name": "Expressive 1.7B", "installed": True,
                "devices": [{"id": "gpu", "available": True}],
                "controls": {"instruction_voice_modes": ["preset"]},
            },
        ]
    }
    result = providers._instruction_safe_tts_recommendation(
        normalized, {"host_a_instruct": "我自己的稳定表达规范"},
        {"model": "fast", "compute_device": "gpu", "reason": "service_default"},
    )
    assert result == {
        "model": "expressive-1.7b", "compute_device": "gpu", "reason": "preserve_custom_instructions",
    }


def test_provider_role_kind_and_base_url_are_normalized() -> None:
    with pytest.raises(ValidationError, match="MAIN 角色不支持 openai_tts"):
        ProviderCreate(name="Broken", role="main", kind="openai_tts", base_url="https://example.com", model="tts-1")
    assert providers.normalize_provider_base_url("openai", "https://example.com/proxy/v1/") == "https://example.com/proxy"
    assert providers.normalize_provider_base_url("ollama", "http://localhost:11434/api") == "http://localhost:11434"
    legacy = ProviderCreate(name="Audio", role="tts", kind="sandevistan_tts", base_url="http://localhost:20810")
    assert (legacy.role, legacy.kind) == ("audio", "sandevistan_audio")
    with pytest.raises(ValidationError, match="TTS_ONLY 角色不支持 openai_tts"):
        ProviderCreate(name="Legacy", role="tts", kind="openai_tts", base_url="https://example.com", model="tts-1")


def test_audio_host_voice_configuration_rejects_collisions() -> None:
    with pytest.raises(ValidationError, match="同一个预置音色"):
        ProviderCreate(
            name="Audio", role="audio", kind="sandevistan_audio", base_url="http://localhost:20810",
            config={"host_a": "Vivian", "host_b": "Vivian"},
        )
    with pytest.raises(ValidationError, match="同一个声纹人员"):
        ProviderCreate(
            name="Audio", role="audio", kind="sandevistan_audio", base_url="http://localhost:20810",
            config={
                "host_a_voice_mode": "voiceprint", "host_b_voice_mode": "voiceprint",
                "host_a_voiceprint_person_id": "person_1", "host_b_voiceprint_person_id": "person_1",
            },
        )


def test_v5_migrates_audio_and_preserves_tts_only_profiles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from sandevistan_read import database as database_module

    path = tmp_path / "legacy-audio.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_versions (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
            INSERT INTO schema_versions VALUES (4, 'old');
            CREATE TABLE provider_profiles (
                id TEXT PRIMARY KEY,name TEXT NOT NULL,role TEXT NOT NULL,kind TEXT NOT NULL,base_url TEXT NOT NULL,
                model TEXT NOT NULL,secret_enc TEXT NOT NULL,capabilities_json TEXT NOT NULL,config_json TEXT NOT NULL,
                active INTEGER NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL
            );
            INSERT INTO provider_profiles VALUES ('audio','Sandevistan Audio','tts','sandevistan_tts','http://localhost:20810','tts','secret','{}','{}',1,'old','old');
            INSERT INTO provider_profiles VALUES ('cloud','Cloud Voice','tts','openai_tts','https://example.com','tts-1','secret','{}','{}',1,'old','old');
            """
        )
    monkeypatch.setattr(database_module, "PATHS", SimpleNamespace(backups=tmp_path / "backups"))
    database = Database(path)
    database._migrate_v5()
    audio = database.fetchone("SELECT role,kind,active,config_json FROM provider_profiles WHERE id='audio'")
    cloud = database.fetchone("SELECT role,kind,active,secret_enc FROM provider_profiles WHERE id='cloud'")
    assert audio and (audio["role"], audio["kind"], audio["active"]) == ("audio", "sandevistan_audio", 1)
    assert json.loads(audio["config_json"])["asr_auto_select"] is True
    assert cloud == {"role": "tts_only", "kind": "openai_tts", "active": 0, "secret_enc": "secret"}
    assert database.fetchone("SELECT MAX(version) AS version FROM schema_versions")["version"] == 5


@pytest.mark.asyncio
async def test_ollama_catalog_is_normalized_without_empty_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen:latest", "model": "qwen:latest", "details": {"parameter_size": "2B"}}]})
        return httpx.Response(404)

    mock_client(monkeypatch, handler)
    result = await providers.inspect_provider(candidate())
    assert result["status"] == "warning"
    assert result["activation_eligible"] is True
    assert result["models"][0]["id"] == "qwen:latest"
    assert result["capabilities"]["token_limits"]["effective_context_tokens"] == 4096
    assert result["capabilities"]["token_limits"]["context_source"] == "fallback"


@pytest.mark.asyncio
async def test_authentication_failure_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer bad-key"
        return httpx.Response(401, json={"error": "invalid"})

    mock_client(monkeypatch, handler)
    result = await providers.inspect_provider(candidate(kind="openai", base_url="https://api.example.com/v1", api_key="bad-key"))
    assert result["status"] == "failed"
    assert result["connection_ok"] is True
    assert result["error"]["code"] == "unauthorized"
    assert "bad-key" not in str(result)


@pytest.mark.asyncio
async def test_deep_verification_can_replace_missing_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(404)
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    mock_client(monkeypatch, handler)
    profile = candidate(kind="openai", base_url="https://compatible.example.com/v1", model="hidden-chat")
    catalog = await providers.inspect_provider(profile, "catalog")
    verified = await providers.inspect_provider(profile, "deep")
    assert catalog["status"] == "warning" and catalog["activation_eligible"] is False
    assert verified["status"] == "warning" and verified["activation_eligible"] is True
    assert verified["capabilities"]["token_limits"]["context_source"] == "fallback"


@pytest.mark.asyncio
async def test_ollama_distinguishes_model_runtime_and_output_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen:latest"}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"qwen.context_length": 32768}, "parameters": "num_ctx 8192"})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen:latest", "context_length": 6144}]})
        raise AssertionError(request.url.path)

    mock_client(monkeypatch, handler)
    result = await providers.inspect_provider(candidate())
    limits = result["capabilities"]["token_limits"]
    assert result["status"] == "passed"
    assert limits["model_context_tokens"] == 32768
    assert limits["effective_context_tokens"] == 6144
    assert limits["max_output_tokens"] == 1536
    assert limits["context_source"] == "ollama_runtime"


def test_context_overrides_are_validated() -> None:
    with pytest.raises(ValidationError, match="最大输出必须小于上下文窗口"):
        ProviderCreate(
            name="Broken",
            role="main",
            kind="openai",
            base_url="https://example.com",
            model="chat",
            config={"context_window_tokens": 4096, "max_output_tokens": 4096},
        )


@pytest.mark.parametrize("value", [-0.01, 2.01, True, "1", float("nan")])
def test_temperature_override_is_validated(value) -> None:
    with pytest.raises(ValidationError, match="Temperature 必须是 0 到 2 之间的数字"):
        ProviderCreate(
            name="Broken",
            role="main",
            kind="openai",
            base_url="https://example.com",
            model="chat",
            config={"temperature": value},
        )
    with pytest.raises(ValidationError, match="Temperature 必须是 0 到 2 之间的数字"):
        ProviderUpdate(config={"temperature": value})


def test_temperature_override_accepts_explicit_zero() -> None:
    created = ProviderCreate(
        name="Deterministic",
        role="main",
        kind="openai",
        base_url="https://example.com",
        model="chat",
        config={"temperature": 0},
    )
    assert created.config["temperature"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("config", "expected"), [({}, 0.25), ({"temperature": 1}, 1.0)])
async def test_chat_uses_provider_temperature_override(monkeypatch: pytest.MonkeyPatch, config: dict, expected: float) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["temperature"] == expected
        assert "reasoning_effort" not in payload
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 30, "prompt_tokens_details": {"cached_tokens": 10}, "completion_tokens_details": {"reasoning_tokens": 20}},
        })

    mock_client(monkeypatch, handler)
    profile = candidate(kind="openai", base_url="https://compatible.example.com", model="chat", config=config)
    result = await providers._chat_once(
        profile,
        [{"role": "user", "content": "test"}],
        json_mode=False,
        timeout=5,
        max_tokens=128,
        temperature=0.25,
    )
    assert result.content == "OK"
    assert result.reasoning_tokens == 20
    assert result.cached_tokens == 10
    assert result.temperature_source == ("provider" if config else "task_default")


@pytest.mark.asyncio
async def test_ollama_chat_uses_provider_temperature_override(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["options"]["temperature"] == 1
        return httpx.Response(200, json={"message": {"content": "OK"}, "done_reason": "stop"})

    mock_client(monkeypatch, handler)
    profile = candidate(config={"temperature": 1})
    result = await providers._chat_once(
        profile,
        [{"role": "user", "content": "test"}],
        json_mode=False,
        timeout=5,
        max_tokens=128,
        temperature=0.25,
    )
    assert result.content == "OK"


@pytest.mark.asyncio
async def test_deep_verification_uses_temperature_override_and_requires_text(monkeypatch: pytest.MonkeyPatch) -> None:
    response_text = "OK"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal response_text
        if request.url.path == "/v1/models":
            return httpx.Response(404)
        payload = json.loads(request.content)
        assert payload["temperature"] == 1
        assert payload["max_tokens"] == 64
        return httpx.Response(200, json={"choices": [{"message": {"content": response_text}}]})

    mock_client(monkeypatch, handler)
    profile = candidate(
        kind="openai",
        base_url="https://compatible.example.com",
        model="reasoning-chat",
        config={"temperature": 1},
    )
    verified = await providers.inspect_provider(profile, "deep")
    assert verified["activation_eligible"] is True

    response_text = ""
    rejected = await providers.inspect_provider(profile, "deep")
    assert rejected["activation_eligible"] is False
    assert rejected["error"]["message"] == "Provider 未返回验证文本"


@pytest.mark.asyncio
async def test_deep_verification_surfaces_upstream_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(404)
        return httpx.Response(400, json={"error": {"message": "only temperature 1 is allowed", "type": "invalid_request_error"}})

    mock_client(monkeypatch, handler)
    profile = candidate(kind="openai", base_url="https://compatible.example.com", model="reasoning-chat")
    result = await providers.inspect_provider(profile, "deep")
    assert result["activation_eligible"] is False
    assert result["error"]["message"] == "only temperature 1 is allowed"
    assert result["error"]["upstream_status"] == 400


@pytest.mark.asyncio
async def test_sandevistan_audio_catalog_recommends_tts_and_asr_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/capabilities"
        assert request.headers["authorization"] == "Bearer local-key"
        return httpx.Response(
            200,
            json={
                "tts": {
                    "model_capabilities": [
                        {
                            "id": "voice-1.7b",
                            "name": "Voice 1.7B",
                            "installed": True,
                            "compute_devices": [{"id": "gpu", "available": True, "precision": "bf16"}],
                            "controls": {"instruction_voice_modes": ["preset"]},
                        }
                    ],
                    "preset_speakers": ["Vivian", "Dylan"],
                    "preset_speaker_native_languages": {"Vivian": "zh-CN"},
                    "languages": ["Chinese"],
                },
                "asr": {
                    "default_model": "qwen3-asr-0.6b",
                    "models": [{
                        "id": "qwen3-asr-0.6b", "name": "Qwen3 ASR 0.6B", "installed": True, "default": True,
                        "compute_devices": [{"id": "gpu", "available": True, "default": True, "precision": "bf16"}],
                    }],
                    "diarization": "CAM++", "languages": ["Chinese", "English"],
                    "aligner_languages": ["Chinese", "English"], "timestamp_precisions": ["segment", "word_or_character"],
                },
            },
        )

    mock_client(monkeypatch, handler)
    result = await providers.inspect_provider(candidate(role="audio", kind="sandevistan_audio", base_url="http://localhost:20810", model="", api_key="local-key", config={"auto_select": True, "asr_auto_select": True}))
    assert result["activation_eligible"] is True
    assert result["recommended"] == {"model": "voice-1.7b", "compute_device": "gpu", "reason": "installed_fallback"}
    assert result["capabilities"]["voices"][0]["id"] == "Vivian"
    assert result["capabilities"]["asr"]["recommended"] == {"model": "qwen3-asr-0.6b", "compute_device": "gpu"}


@pytest.mark.asyncio
async def test_audio_catalog_exposes_people_and_locks_latest_voiceprint_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/capabilities":
            return httpx.Response(200, json={
                "tts": {
                    "model_capabilities": [{
                        "id": "voice-1.7b", "name": "Voice 1.7B", "installed": True,
                        "voice_modes": ["preset", "voiceprint"],
                        "compute_devices": [{"id": "cpu", "available": True, "default": True}],
                        "controls": {"instruction_voice_modes": ["preset"]},
                    }],
                    "preset_speakers": ["Vivian", "Dylan"],
                },
                "asr": {
                    "default_model": "asr", "models": [{
                        "id": "asr", "name": "ASR", "installed": True, "default": True,
                        "compute_devices": [{"id": "cpu", "available": True, "default": True}],
                    }],
                    "diarization": "CAM++", "languages": ["Chinese", "English"],
                    "aligner_languages": ["Chinese", "English"], "timestamp_precisions": ["segment"],
                },
            })
        if request.url.path == "/api/v1/voiceprints/people":
            return httpx.Response(200, json={"items": [{
                "id": "person_1", "name": "Sample Person", "note": "host",
                "samples": [
                    {"id": "sample_old", "tts_eligible": True, "language": "Chinese", "duration": 8.0, "created_at": "2026-01-01", "transcript": "private"},
                    {"id": "sample_new", "tts_eligible": True, "language": "Chinese", "duration": 18.0, "created_at": "2026-02-01", "transcript": "private"},
                ],
            }]})
        raise AssertionError(request.url.path)

    mock_client(monkeypatch, handler)
    result = await providers.inspect_provider(candidate(
        role="audio", kind="sandevistan_audio", base_url="http://localhost:20810", model="voice-1.7b",
        config={
            "auto_select": False, "compute_device": "cpu", "asr_auto_select": True,
            "host_a_voice_mode": "voiceprint", "host_a_voiceprint_person_id": "person_1",
            "host_b": "Dylan",
        },
    ))
    assert result["activation_eligible"] is True
    person = result["voiceprint_library"]["people"][0]
    assert person["latest_sample"]["id"] == "sample_new"
    assert "transcript" not in person["latest_sample"]
    assert result["resolved_audio_config"]["host_a_voiceprint_sample_id"] == "sample_new"


@pytest.mark.asyncio
async def test_voiceprint_synthesis_uses_sample_and_omits_preset_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    submitted: dict[str, list[str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/tts/jobs":
            submitted.update(parse_qs(request.content.decode()))
            return httpx.Response(202, json={"id": "tts-job"})
        if request.method == "GET" and request.url.path == "/api/v1/jobs/tts-job":
            return httpx.Response(200, json={"state": "succeeded", "result": {"artifacts": [{"name": "voice.wav", "mime_type": "audio/wav"}]}})
        if request.method == "GET" and request.url.path == "/api/v1/jobs/tts-job/artifacts/voice.wav":
            return httpx.Response(200, content=b"RIFFaudio", headers={"content-type": "audio/wav"})
        if request.method == "DELETE":
            return httpx.Response(204)
        raise AssertionError(f"{request.method} {request.url.path}")

    mock_client(monkeypatch, handler)
    output = tmp_path / "clone.wav"
    await providers._synthesize_sandevistan(
        candidate(role="audio", kind="sandevistan_audio", base_url="http://localhost:20810", config={}),
        "测试克隆", None, output, language="Chinese", model="voice-1.7b", compute_device="cpu",
        voice_mode="voiceprint", voiceprint_sample_id="sample_new", instruct="不应发送",
        idempotency_key="voiceprint-test", cancel_check=None,
    )
    assert output.read_bytes() == b"RIFFaudio"
    assert submitted["voice_mode"] == ["voiceprint"]
    assert submitted["voiceprint_sample_id"] == ["sample_new"]
    assert "speaker" not in submitted and "instruct" not in submitted


def test_host_instruction_is_stable_and_clone_mode_omits_it() -> None:
    config = {"host_a_instruct": "自然沉稳", "host_a_voice_mode": "preset"}
    first = providers.host_voice_instruction(config, "host_a", "zh-CN", supported=True)
    second = providers.host_voice_instruction(config, "host_a", "zh-CN", supported=True)
    assert first == second
    assert "突然兴奋" in str(first)
    assert providers.host_voice_instruction({**config, "host_a_voice_mode": "voiceprint"}, "host_a", "zh-CN", supported=True) is None


@pytest.mark.asyncio
async def test_local_asr_retries_device_failure_once_on_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = candidate(role="audio", kind="sandevistan_audio", base_url="http://localhost:20810", model="qwen3-tts-1.7b", capabilities=audio_capabilities(), config={"compute_device": "gpu", "asr_auto_select": True})
    monkeypatch.setattr(providers, "active_provider", lambda role: profile)
    devices: list[str] = []

    async def fake_once(provider, path, *, language, model, compute_device, idempotency_key, cancel_check):
        assert model == "qwen3-asr-0.6b"
        devices.append(compute_device)
        if compute_device == "gpu":
            raise providers.ProviderError("CUDA out of memory", code="insufficient_gpu_memory")
        return {"text": "ok", "segments": [], "compute_device": "cpu"}

    monkeypatch.setattr(providers, "_transcribe_sandevistan_once", fake_once)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    result = await providers.transcribe_audio(audio, language="Chinese", idempotency_key="12345678")
    assert devices == ["gpu", "cpu"]
    assert result["fallback_used"] is True
    assert result["compute_device"] == "cpu"


@pytest.mark.asyncio
async def test_local_asr_does_not_retry_non_device_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = candidate(role="audio", kind="sandevistan_audio", base_url="http://localhost:20810", model="qwen3-tts-1.7b", capabilities=audio_capabilities(), config={"compute_device": "gpu", "asr_auto_select": True})
    monkeypatch.setattr(providers, "active_provider", lambda role: profile)
    devices: list[str] = []

    async def fake_once(provider, path, *, language, model, compute_device, idempotency_key, cancel_check):
        devices.append(compute_device)
        raise providers.ProviderError("invalid language", code="validation_error")

    monkeypatch.setattr(providers, "_transcribe_sandevistan_once", fake_once)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(providers.ProviderError, match="invalid language"):
        await providers.transcribe_audio(audio, language="Chinese", idempotency_key="12345678")
    assert devices == ["gpu"]


@pytest.mark.asyncio
async def test_local_asr_uses_manual_model_and_device_without_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = candidate(
        role="audio",
        kind="sandevistan_audio",
        base_url="http://localhost:20810",
        model="qwen3-tts-1.7b",
        capabilities=audio_capabilities(),
        config={
            "compute_device": "gpu",
            "asr_auto_select": False,
            "asr_model": "qwen3-asr-0.6b",
            "asr_compute_device": "cpu",
            "asr_allow_device_fallback": False,
        },
    )
    monkeypatch.setattr(providers, "active_provider", lambda role: profile)
    executions: list[tuple[str, str]] = []

    async def fake_once(provider, path, *, language, model, compute_device, idempotency_key, cancel_check):
        executions.append((model, compute_device))
        return {"text": "ok", "segments": []}

    monkeypatch.setattr(providers, "_transcribe_sandevistan_once", fake_once)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    result = await providers.transcribe_audio(audio, language="Chinese")
    assert executions == [("qwen3-asr-0.6b", "cpu")]
    assert result["compute_device"] == "cpu" and result["fallback_used"] is False


def test_audio_readiness_requires_asr_acceptance_capabilities() -> None:
    profile = candidate(
        role="audio",
        kind="sandevistan_audio",
        model="qwen3-tts-1.7b",
        config={"compute_device": "gpu"},
        capabilities={"models": audio_capabilities()["models"]},
    )
    ready, message = providers.audio_provider_readiness(profile)
    assert ready is False
    assert "ASR 模型" in message


@pytest.mark.asyncio
async def test_sequence_synthesis_preserves_item_mapping_and_downloads_each_wav(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    submitted: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/tts/sequence-jobs":
            submitted.update(json.loads(request.content))
            return httpx.Response(202, json={"id": "sequence-job"})
        if request.method == "GET" and request.url.path == "/api/v1/jobs/sequence-job":
            return httpx.Response(200, json={
                "state": "succeeded", "result": {
                    "sequence": {"items": [
                        {"id": "turn-0000", "artifact_name": "item-0000.wav"},
                        {"id": "turn-0001", "artifact_name": "item-0001.wav"},
                    ]},
                    "acceleration": {
                        "active": True,
                        "stage_batch_sizes": {"generation": 2, "decoder": 1},
                        "oom_fallbacks": [{"stage": "generation", "from": 4, "to": 2}],
                    },
                },
            })
        if request.method == "GET" and request.url.path.startswith("/api/v1/jobs/sequence-job/artifacts/"):
            return httpx.Response(200, content=b"RIFFsequence", headers={"content-type": "audio/wav"})
        if request.method == "DELETE":
            return httpx.Response(204)
        raise AssertionError((request.method, request.url.path))

    mock_client(monkeypatch, handler)
    profile = candidate(
        role="audio", kind="sandevistan_audio", base_url="http://localhost:20810",
        model="qwen3-tts-0.6b", config={"compute_device": "gpu", "podcast_sequence_tts": True},
        capabilities={"sequence_jobs": {"supported": True, "contract_version": 1}},
    )
    outputs = {"turn-0000": tmp_path / "0.wav", "turn-0001": tmp_path / "1.wav"}
    execution: dict = {}
    result = await providers.synthesize_sequence(
        [
            {"id": "turn-0000", "text": "第一句", "speaker": "Vivian"},
            {"id": "turn-0001", "text": "第二句", "speaker": "Dylan"},
        ],
        outputs,
        provider=profile,
        model="qwen3-tts-0.6b",
        compute_device="gpu",
        voice_mode="preset",
        idempotency_key="sequence-test-key",
        execution=execution,
    )
    assert submitted["items"][1]["speaker"] == "Dylan"
    assert "response_format" not in submitted
    assert list(result) == ["turn-0000", "turn-0001"]
    assert all(path.read_bytes().startswith(b"RIFF") for path in outputs.values())
    assert execution["generation_batch_size"] == 2
    assert execution["oom_fallbacks"] == [{"stage": "generation", "from": 4, "to": 2}]


def test_podcast_is_rejected_before_enqueue_when_audio_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    app_module = importlib.import_module("sandevistan_read.app")
    enqueued: list[str] = []
    monkeypatch.setattr(app_module, "_require_notebook", lambda notebook_id: None)
    monkeypatch.setattr(app_module, "active_provider", lambda role: None)
    monkeypatch.setattr(app_module, "enqueue", lambda *args: enqueued.append("called"))
    with pytest.raises(HTTPException, match="AUDIO Provider") as captured:
        app_module.podcast("n1", PodcastRequest(minutes=5))
    assert captured.value.status_code == 409
    assert enqueued == []


@pytest.mark.asyncio
async def test_openai_tts_is_retained_as_ineligible_tts_only_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "tts-1", "owned_by": "vendor"}]})

    mock_client(monkeypatch, handler)
    profile = candidate(role="tts_only", kind="openai_tts", base_url="https://speech.example.com", model="tts-1")
    incomplete = await providers.inspect_provider(profile)
    complete = await providers.inspect_provider({**profile, "config": {"host_a": "alloy", "host_b": "nova"}})
    assert incomplete["activation_eligible"] is False
    assert "两位主持人" in incomplete["warning"]
    assert complete["activation_eligible"] is False
    assert "不能用于 Podcast" in complete["warning"]


@pytest.mark.asyncio
async def test_activation_switch_is_atomic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app_module = importlib.import_module("sandevistan_read.app")
    database = Database(tmp_path / "providers.sqlite3")
    database.initialize()
    now = utc_now()
    for identifier, name, active in (("provider_old", "Old", 1), ("provider_new", "New", 0)):
        database.execute(
            """INSERT INTO provider_profiles
            (id,name,role,kind,base_url,model,secret_enc,capabilities_json,config_json,active,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (identifier, name, "main", "openai", "https://api.example.com", "chat-model", "", json_dump({}), json_dump({}), active, now, now),
        )
    monkeypatch.setattr(app_module, "DB", database)
    monkeypatch.setattr(providers, "DB", database)

    async def failed(candidate, mode):
        return {"activation_eligible": False, "warning": "offline", "error": None}

    monkeypatch.setattr(app_module, "inspect_provider", failed)
    with pytest.raises(HTTPException):
        await app_module.update_provider("provider_new", ProviderUpdate(active=True))
    assert database.fetchone("SELECT active FROM provider_profiles WHERE id='provider_old'")["active"] == 1
    assert database.fetchone("SELECT active FROM provider_profiles WHERE id='provider_new'")["active"] == 0

    async def passed(candidate, mode):
        return {"activation_eligible": True, "models": [], "capabilities": {}, "recommended": None}

    monkeypatch.setattr(app_module, "inspect_provider", passed)
    await app_module.update_provider("provider_new", ProviderUpdate(active=True))
    assert database.fetchone("SELECT active FROM provider_profiles WHERE id='provider_old'")["active"] == 0
    assert database.fetchone("SELECT active FROM provider_profiles WHERE id='provider_new'")["active"] == 1


def _stub_budgeted_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(providers, "_chat_provider", lambda role: {"model": "m"})
    monkeypatch.setattr(providers.TokenLimits, "from_provider", staticmethod(lambda provider: None))
    monkeypatch.setattr(
        providers,
        "prompt_budget",
        lambda limits, max_tokens, minimum, scale: SimpleNamespace(input_tokens=100_000, output_tokens=1000, image_tokens_per_image=0),
    )
    monkeypatch.setattr(providers, "estimate_messages_tokens", lambda messages, images: 10)


@pytest.mark.asyncio
async def test_budgeted_chat_retries_transport_error_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    _stub_budgeted_chat(monkeypatch)
    calls = {"count": 0}

    async def flaky(provider, messages, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("connection reset")
        return SimpleNamespace(content="OK", finish_reason="stop")

    monkeypatch.setattr(providers, "_chat_once", flaky)
    build = SimpleNamespace(messages=[{"role": "user", "content": "hi"}], total_segments=1, included_segments=1, truncated_segments=0)
    result = await providers.budgeted_chat(lambda budget: build)
    assert result.content == "OK"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_budgeted_chat_gives_up_after_single_transport_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    _stub_budgeted_chat(monkeypatch)
    calls = {"count": 0}

    async def always_down(provider, messages, **kwargs):
        calls["count"] += 1
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(providers, "_chat_once", always_down)
    build = SimpleNamespace(messages=[{"role": "user", "content": "hi"}], total_segments=1, included_segments=1, truncated_segments=0)
    with pytest.raises(httpx.TimeoutException):
        await providers.budgeted_chat(lambda budget: build)
    assert calls["count"] == 2
