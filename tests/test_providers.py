import importlib
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
        assert request.url.path == "/api/tags"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"models": [{"name": "qwen:latest", "model": "qwen:latest", "details": {"parameter_size": "2B"}}]})

    mock_client(monkeypatch, handler)
    result = await providers.inspect_provider(candidate())
    assert result["status"] == "passed"
    assert result["activation_eligible"] is True
    assert result["models"][0]["id"] == "qwen:latest"


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
    assert verified["status"] == "passed" and verified["activation_eligible"] is True


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
