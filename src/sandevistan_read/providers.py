from __future__ import annotations

import asyncio
import base64
import re
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx

from .database import DB, json_dump, json_load, utc_now
from .security import VAULT


class ProviderError(RuntimeError):
    pass


def active_provider(role: str) -> dict[str, Any] | None:
    row = DB.fetchone("SELECT * FROM provider_profiles WHERE role=? AND active=1 ORDER BY updated_at DESC LIMIT 1", (role,))
    if not row:
        return None
    row["capabilities"] = json_load(row.pop("capabilities_json"), {})
    row["config"] = json_load(row.pop("config_json"), {})
    row["api_key"] = VAULT.decrypt(row.pop("secret_enc", "")) if row.get("secret_enc") else ""
    return row


def provider_by_id(provider_id: str) -> dict[str, Any] | None:
    row = DB.fetchone("SELECT * FROM provider_profiles WHERE id=?", (provider_id,))
    if not row:
        return None
    row["capabilities"] = json_load(row.pop("capabilities_json"), {})
    row["config"] = json_load(row.pop("config_json"), {})
    secret = row.pop("secret_enc", "")
    row["api_key"] = VAULT.decrypt(secret) if secret else ""
    return row


async def chat(
    messages: list[dict[str, Any]],
    *,
    role: str = "main",
    json_mode: bool = False,
    timeout: float = 180,
    max_tokens: int = 1400,
    temperature: float = 0.15,
) -> str:
    provider = active_provider(role)
    if not provider and role == "vlm":
        provider = active_provider("main")
    if not provider:
        raise ProviderError(f"No active {role} provider")
    headers: dict[str, str] = {}
    if provider["api_key"]:
        headers["Authorization"] = f"Bearer {provider['api_key']}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        if provider["kind"] == "ollama":
            payload: dict[str, Any] = {
                "model": provider["model"],
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            if json_mode:
                payload["format"] = "json"
            response = await client.post(f"{provider['base_url'].rstrip('/')}/api/chat", json=payload, headers=headers)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        if provider["kind"] == "openai":
            payload = {"model": provider["model"], "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            response = await client.post(f"{provider['base_url'].rstrip('/')}/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    raise ProviderError(f"Provider {provider['kind']} cannot serve chat")


async def describe_image(path: Path, nearby_text: str) -> str:
    provider = active_provider("vlm") or active_provider("main")
    if not provider or not provider.get("capabilities", {}).get("vision"):
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    prompt = "仅描述图中可验证的文字、图表关系和版面信息；不要补充图外事实。附近提取文本：\n" + nearby_text[:3000]
    if provider["kind"] == "ollama":
        messages = [{"role": "user", "content": prompt, "images": [encoded]}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}]}]
    return await chat(messages, role="vlm")


async def health(role: str, provider_id: str | None = None) -> dict[str, Any]:
    provider = provider_by_id(provider_id) if provider_id else active_provider(role)
    if not provider:
        return {"ok": False, "message": "未配置"}
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            if provider["kind"] == "ollama":
                response = await client.get(f"{provider['base_url'].rstrip('/')}/api/tags")
            elif provider["kind"] == "sandevistan_tts":
                response = await client.get(f"{provider['base_url'].rstrip('/')}/api/v1/capabilities")
            else:
                response = await client.get(f"{provider['base_url'].rstrip('/')}/v1/models", headers={"Authorization": f"Bearer {provider['api_key']}"})
            return {"ok": response.is_success, "status": response.status_code, "message": provider["name"]}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


def _quality_score(model: dict[str, Any]) -> tuple[float, int, str]:
    name = f"{model.get('id', '')} {model.get('name', '')}".lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", name)
    parameters = float(match.group(1)) if match else 0.0
    controls = model.get("controls") or {}
    return parameters, int(bool(controls.get("instruction_voice_modes"))), str(model.get("id", ""))


async def probe_tts_provider(provider_id: str, *, apply_defaults: bool = False) -> dict[str, Any]:
    """Read live TTS capabilities and optionally apply the highest-quality safe default."""
    provider = provider_by_id(provider_id)
    if not provider:
        raise ProviderError("TTS provider does not exist")
    if provider["kind"] != "sandevistan_tts":
        return {"ok": True, "kind": provider["kind"], "discovery": False, "models": [], "voices": [], "recommended": None}
    headers = {"Authorization": f"Bearer {provider['api_key']}"} if provider["api_key"] else {}
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(f"{provider['base_url'].rstrip('/')}/api/v1/capabilities", headers=headers)
        response.raise_for_status()
    tts = response.json().get("tts") or {}
    models: list[dict[str, Any]] = []
    for item in tts.get("model_capabilities") or []:
        models.append(
            {
                "id": item.get("id"),
                "name": item.get("name") or item.get("id"),
                "installed": bool(item.get("installed", True)),
                "devices": [
                    {
                        "id": device.get("id"),
                        "available": bool(device.get("available")),
                        "precision": device.get("precision"),
                        "reason": device.get("unavailable_reason"),
                        "reason_code": device.get("unavailable_reason_code"),
                    }
                    for device in item.get("compute_devices") or []
                ],
                "voice_modes": item.get("voice_modes") or [],
                "controls": item.get("controls") or {},
            }
        )
    installed = [item for item in models if item["installed"]]
    best = max(installed, key=_quality_score) if installed else None
    recommended = None
    if best:
        available = {device["id"] for device in best["devices"] if device["available"]}
        device = "gpu" if "gpu" in available else "cpu" if "cpu" in available else next(iter(available), None)
        recommended = {"model": best["id"], "compute_device": device}
    native = tts.get("preset_speaker_native_languages") or {}
    voices = [{"id": name, "native_language": native.get(name)} for name in tts.get("preset_speakers") or []]
    normalized = {
        "async": True,
        "discovery": True,
        "models": models,
        "voices": voices,
        "languages": tts.get("languages") or [],
        "recommended": recommended,
    }
    config = dict(provider.get("config") or {})
    legacy_default = (
        provider.get("model") == "qwen3-tts-0.6b"
        and config.get("compute_device") == "cpu"
        and config.get("host_a", "Vivian") == "Vivian"
        and config.get("host_b", "Dylan") == "Dylan"
        and "auto_select" not in config
    )
    auto_select = bool(config.get("auto_select", legacy_default or not provider.get("model")))
    if apply_defaults:
        config.setdefault("host_a", "Vivian")
        config.setdefault("host_b", "Dylan")
        config.setdefault("host_a_en", "Ryan")
        config.setdefault("host_b_en", "Aiden")
        config.setdefault("host_a_instruct", "自然、沉稳、有叙事感的知识播客主持人口吻，语速适中。")
        config.setdefault("host_b_instruct", "敏锐、亲切、善于追问和澄清的知识播客主持人口吻，语速适中。")
        config.setdefault("allow_device_fallback", True)
        config.setdefault("cleanup_remote_jobs", True)
        config["auto_select"] = auto_select
        model = provider.get("model") or ""
        if auto_select and recommended:
            model = recommended["model"]
            config["compute_device"] = recommended["compute_device"]
        DB.execute(
            "UPDATE provider_profiles SET model=?,capabilities_json=?,config_json=?,updated_at=? WHERE id=?",
            (model, json_dump(normalized), json_dump(config), utc_now(), provider_id),
        )
    return {"ok": True, **normalized, "auto_select": auto_select}


async def _synthesize_sandevistan(
    provider: dict[str, Any],
    text: str,
    voice: str,
    output: Path,
    *,
    language: str,
    model: str,
    compute_device: str,
    instruct: str | None,
    idempotency_key: str,
    cancel_check: Callable[[], bool] | None,
) -> Path:
    config = provider.get("config", {})
    headers = {"Authorization": f"Bearer {provider['api_key']}"} if provider["api_key"] else {}
    headers["Idempotency-Key"] = idempotency_key
    data = {
        "text": text,
        "speaker": voice,
        "model": model,
        "language": language,
        "voice_mode": "preset",
        "response_format": "wav",
        "compute_device": compute_device,
    }
    if instruct:
        data["instruct"] = instruct
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(f"{provider['base_url'].rstrip('/')}/api/v1/tts/jobs", data=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        job_id = result.get("id") or result.get("job_id")
        status_url = result.get("status_url") or f"/api/v1/jobs/{job_id}"
        for _ in range(7200):
            if cancel_check and cancel_check():
                try:
                    await client.post(f"{provider['base_url'].rstrip('/')}/api/v1/jobs/{job_id}/cancel", headers=headers)
                finally:
                    raise ProviderError("任务已取消")
            poll_url = status_url if status_url.startswith("http") else provider["base_url"].rstrip("/") + status_url
            poll = await client.get(poll_url, headers=headers)
            poll.raise_for_status()
            state = poll.json()
            job_state = state.get("state") or state.get("status")
            if job_state in {"completed", "succeeded", "done"}:
                result_data = state.get("result") or {}
                artifacts = result_data.get("artifacts") or []
                audio_artifact = next((item for item in artifacts if str(item.get("mime_type", "")).startswith("audio/")), None)
                if audio_artifact and audio_artifact.get("name"):
                    url = f"/api/v1/jobs/{job_id}/artifacts/{quote(str(audio_artifact['name']), safe='')}"
                else:
                    url = state.get("download_url") or result_data.get("download_url") or f"/api/v1/jobs/{job_id}/result"
                audio = await client.get(url if url.startswith("http") else provider["base_url"].rstrip("/") + url, headers=headers)
                audio.raise_for_status()
                if not audio.headers.get("content-type", "").startswith("audio/"):
                    raise ProviderError("TTS provider returned metadata instead of an audio artifact")
                output.write_bytes(audio.content)
                if config.get("cleanup_remote_jobs", True):
                    try:
                        await client.delete(f"{provider['base_url'].rstrip('/')}/api/v1/jobs/{job_id}", params={"purge": "true"}, headers=headers)
                    except Exception:
                        pass
                return output
            if job_state in {"failed", "error", "cancelled"}:
                raise ProviderError(state.get("error_message") or state.get("error") or "TTS job failed")
            await asyncio.sleep(max(0.5, min(float(state.get("poll_after_seconds") or 1), 5)))
    raise ProviderError("TTS job timed out")


async def synthesize(
    text: str,
    voice: str,
    output: Path,
    *,
    language: str = "Chinese",
    cancel_check: Callable[[], bool] | None = None,
    model: str | None = None,
    compute_device: str | None = None,
    instruct: str | None = None,
    idempotency_key: str | None = None,
) -> Path:
    provider = active_provider("tts")
    if not provider:
        raise ProviderError("TTS provider is not configured")
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {provider['api_key']}"} if provider["api_key"] else {}
    if provider["kind"] == "openai_tts":
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{provider['base_url'].rstrip('/')}/v1/audio/speech",
                json={"model": model or provider["model"], "voice": voice, "input": text, "response_format": "wav"},
                headers=headers,
            )
            response.raise_for_status()
            output.write_bytes(response.content)
            return output
    if provider["kind"] == "sandevistan_tts":
        config = provider.get("config", {})
        selected_model = model or provider["model"]
        selected_device = compute_device or config.get("compute_device", "gpu")
        key = idempotency_key or str(uuid.uuid4())
        try:
            return await _synthesize_sandevistan(
                provider,
                text,
                voice,
                output,
                language=language,
                model=selected_model,
                compute_device=selected_device,
                instruct=instruct,
                idempotency_key=key,
                cancel_check=cancel_check,
            )
        except (ProviderError, httpx.HTTPError):
            if selected_device != "gpu" or not config.get("allow_device_fallback", True):
                raise
            return await _synthesize_sandevistan(
                provider,
                text,
                voice,
                output,
                language=language,
                model=selected_model,
                compute_device="cpu",
                instruct=instruct,
                idempotency_key=(key + "-cpu")[:128],
                cancel_check=cancel_check,
            )
    raise ProviderError(f"Provider {provider['kind']} cannot serve TTS")
