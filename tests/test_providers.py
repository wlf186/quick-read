import importlib
import json
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from sandevistan_read.database import Database, json_dump, utc_now
from sandevistan_read import providers
from sandevistan_read.schemas import ProviderCreate, ProviderUpdate


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


def test_provider_role_kind_and_base_url_are_normalized() -> None:
    with pytest.raises(ValidationError, match="MAIN 角色不支持 openai_tts"):
        ProviderCreate(name="Broken", role="main", kind="openai_tts", base_url="https://example.com", model="tts-1")
    assert providers.normalize_provider_base_url("openai", "https://example.com/proxy/v1/") == "https://example.com/proxy"
    assert providers.normalize_provider_base_url("ollama", "http://localhost:11434/api") == "http://localhost:11434"


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
async def test_sandevistan_tts_catalog_recommends_installed_gpu_model(monkeypatch: pytest.MonkeyPatch) -> None:
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
                }
            },
        )

    mock_client(monkeypatch, handler)
    result = await providers.inspect_provider(candidate(role="tts", kind="sandevistan_tts", base_url="http://localhost:20810", model="", api_key="local-key", config={"auto_select": True}))
    assert result["activation_eligible"] is True
    assert result["recommended"] == {"model": "voice-1.7b", "compute_device": "gpu"}
    assert result["capabilities"]["voices"][0]["id"] == "Vivian"


@pytest.mark.asyncio
async def test_local_asr_retries_device_failure_once_on_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = candidate(role="tts", kind="sandevistan_tts", base_url="http://localhost:20810", model="qwen3-tts-1.7b")
    monkeypatch.setattr(providers, "active_provider", lambda role: profile)
    devices: list[str] = []

    async def fake_once(provider, path, *, language, compute_device, idempotency_key, cancel_check):
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
    profile = candidate(role="tts", kind="sandevistan_tts", base_url="http://localhost:20810", model="qwen3-tts-1.7b")
    monkeypatch.setattr(providers, "active_provider", lambda role: profile)
    devices: list[str] = []

    async def fake_once(provider, path, *, language, compute_device, idempotency_key, cancel_check):
        devices.append(compute_device)
        raise providers.ProviderError("invalid language", code="validation_error")

    monkeypatch.setattr(providers, "_transcribe_sandevistan_once", fake_once)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(providers.ProviderError, match="invalid language"):
        await providers.transcribe_audio(audio, language="Chinese", idempotency_key="12345678")
    assert devices == ["gpu"]


@pytest.mark.asyncio
async def test_openai_tts_requires_both_host_voices(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "tts-1", "owned_by": "vendor"}]})

    mock_client(monkeypatch, handler)
    profile = candidate(role="tts", kind="openai_tts", base_url="https://speech.example.com", model="tts-1")
    incomplete = await providers.inspect_provider(profile)
    complete = await providers.inspect_provider({**profile, "config": {"host_a": "alloy", "host_b": "nova"}})
    assert incomplete["activation_eligible"] is False
    assert "两位主持人" in incomplete["warning"]
    assert complete["activation_eligible"] is True


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
