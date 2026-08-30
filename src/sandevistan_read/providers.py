from __future__ import annotations

import asyncio
import base64
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from .database import DB, json_dump, json_load, utc_now
from .security import VAULT


class ProviderError(RuntimeError):
    pass


MAX_CATALOG_BYTES = 4 * 1024 * 1024
TEST_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII="


def normalize_provider_base_url(kind: str, value: str) -> str:
    raw = value.strip().rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProviderError("服务地址必须是有效的 HTTP 或 HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ProviderError("服务地址不能包含账号、密码、查询参数或片段")
    path = parsed.path.rstrip("/")
    suffix = "/v1" if kind in {"openai", "openai_tts"} else "/api/v1" if kind == "sandevistan_tts" else "/api"
    if path.endswith(suffix):
        path = path[: -len(suffix)]
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))


def _headers(provider: dict[str, Any]) -> dict[str, str]:
    key = str(provider.get("api_key") or "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _inspection_error(
    *,
    started: float,
    code: str,
    stage: str,
    message: str,
    hint: str,
    connection_ok: bool = False,
    upstream_status: int | None = None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "connection_ok": connection_ok,
        "activation_eligible": False,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "catalog_supported": False,
        "models": [],
        "capabilities": {},
        "recommended": None,
        "error": {"code": code, "stage": stage, "message": message, "hint": hint, "upstream_status": upstream_status},
    }


def _normalized_tts_capabilities(payload: dict[str, Any]) -> dict[str, Any]:
    tts = payload.get("tts") or {}
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
        available = [device["id"] for device in best["devices"] if device["available"]]
        device = "gpu" if "gpu" in available else "cpu" if "cpu" in available else available[0] if available else None
        recommended = {"model": best["id"], "compute_device": device}
    native = tts.get("preset_speaker_native_languages") or {}
    voices = [{"id": name, "native_language": native.get(name)} for name in tts.get("preset_speakers") or []]
    return {
        "models": models,
        "voices": voices,
        "languages": tts.get("languages") or [],
        "recommended": recommended,
        "async": True,
        "discovery": True,
    }


async def _provider_catalog(provider: dict[str, Any]) -> tuple[bool, list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    base_url = provider["base_url"]
    headers = _headers(provider)
    async with httpx.AsyncClient(timeout=8, follow_redirects=False) as client:
        if provider["kind"] == "ollama":
            response = await client.get(f"{base_url}/api/tags", headers=headers)
        elif provider["kind"] == "sandevistan_tts":
            response = await client.get(f"{base_url}/api/v1/capabilities", headers=headers)
        else:
            response = await client.get(f"{base_url}/v1/models", headers=headers)
        if response.status_code in {404, 405, 501}:
            return False, [], {}, None
        response.raise_for_status()
        if len(response.content) > MAX_CATALOG_BYTES:
            raise ProviderError("Provider 返回的能力清单过大")
        payload = response.json()
    if not isinstance(payload, dict):
        raise ProviderError("Provider 返回了无法识别的能力清单")
    if provider["kind"] == "sandevistan_tts":
        normalized = _normalized_tts_capabilities(payload)
        models = normalized.pop("models")
        recommended = normalized.pop("recommended")
        return True, models, normalized, recommended
    source = payload.get("models") if provider["kind"] == "ollama" else payload.get("data")
    if not isinstance(source, list):
        raise ProviderError("Provider 返回了无法识别的模型清单")
    models_by_id: dict[str, dict[str, Any]] = {}
    for item in source:
        if not isinstance(item, dict):
            continue
        identifier = str(item.get("model") or item.get("id") or item.get("name") or "").strip()
        if not identifier:
            continue
        models_by_id[identifier] = {
            "id": identifier,
            "name": str(item.get("name") or identifier),
            "details": item.get("details") or {"owned_by": item.get("owned_by")},
        }
    return True, sorted(models_by_id.values(), key=lambda item: item["id"].lower()), {}, None


async def _deep_verify(provider: dict[str, Any]) -> None:
    model = str(provider.get("model") or "").strip()
    if not model:
        raise ProviderError("请先选择或填写模型")
    headers = _headers(provider)
    role, kind = provider["role"], provider["kind"]
    if role == "tts" and kind == "sandevistan_tts":
        config = provider.get("config") or {}
        device = str(config.get("compute_device") or "cpu")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output = Path(handle.name)
        try:
            await _synthesize_sandevistan(
                provider,
                "连接测试",
                str(config.get("host_a") or "Vivian"),
                output,
                language="Chinese",
                model=model,
                compute_device=device,
                instruct=None,
                idempotency_key=f"provider-test-{uuid.uuid4()}",
                cancel_check=None,
            )
            if not output.exists() or output.stat().st_size == 0:
                raise ProviderError("TTS Provider 未返回音频")
        finally:
            output.unlink(missing_ok=True)
        return
    async with httpx.AsyncClient(timeout=180, follow_redirects=False) as client:
        if role == "tts":
            voice = str((provider.get("config") or {}).get("host_a") or "").strip()
            if not voice:
                raise ProviderError("OpenAI TTS 需要配置 Host A 音色")
            response = await client.post(
                f"{provider['base_url']}/v1/audio/speech",
                json={"model": model, "voice": voice, "input": "连接测试", "response_format": "wav"},
                headers=headers,
            )
            response.raise_for_status()
            if not response.content:
                raise ProviderError("TTS Provider 未返回音频")
            return
        if kind == "ollama":
            message: dict[str, Any] = {"role": "user", "content": "只回复 OK"}
            if role == "vlm":
                message["images"] = [TEST_IMAGE_BASE64]
            response = await client.post(
                f"{provider['base_url']}/api/chat",
                json={"model": model, "messages": [message], "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 4}},
                headers=headers,
            )
        else:
            content: Any = "Reply with OK only."
            if role == "vlm":
                content = [
                    {"type": "text", "text": "Reply with OK only."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_BASE64}"}},
                ]
            response = await client.post(
                f"{provider['base_url']}/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 4},
                headers=headers,
            )
        response.raise_for_status()
        if not response.content:
            raise ProviderError("Provider 未返回验证结果")


async def inspect_provider(candidate: dict[str, Any], mode: str = "catalog") -> dict[str, Any]:
    started = time.perf_counter()
    provider = {**candidate, "config": dict(candidate.get("config") or {})}
    try:
        provider["base_url"] = normalize_provider_base_url(provider["kind"], str(provider.get("base_url") or ""))
    except ProviderError as exc:
        return _inspection_error(started=started, code="invalid_url", stage="url", message=str(exc), hint="请输入 Provider 服务根地址")
    try:
        supported, models, capabilities, recommended = await _provider_catalog(provider)
    except httpx.TimeoutException:
        return _inspection_error(started=started, code="timeout", stage="connection", message="连接 Provider 超时", hint="确认服务已启动，并检查地址、防火墙或代理")
    except httpx.ConnectError:
        return _inspection_error(started=started, code="unreachable", stage="connection", message="无法连接 Provider", hint="确认主机名、端口和网络可达")
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        code = "unauthorized" if status == 401 else "forbidden" if status == 403 else "upstream_error"
        message = "API Key 无效或缺少权限" if status in {401, 403} else f"Provider 返回 HTTP {status}"
        return _inspection_error(started=started, code=code, stage="authentication" if status in {401, 403} else "catalog", message=message, hint="检查 API Key 与服务地址", connection_ok=True, upstream_status=status)
    except (ValueError, ProviderError) as exc:
        return _inspection_error(started=started, code="bad_response", stage="catalog", message=str(exc), hint="该服务可能未完整实现所选兼容协议", connection_ok=True)
    except httpx.HTTPError:
        return _inspection_error(started=started, code="network_error", stage="connection", message="Provider 连接失败", hint="检查 TLS、代理与服务日志")

    model = str(provider.get("model") or "").strip()
    model_ids = {str(item.get("id") or "") for item in models}
    if provider["kind"] == "sandevistan_tts" and provider["config"].get("auto_select") and recommended:
        model = str(recommended.get("model") or model)
        provider["model"] = model
        provider["config"]["compute_device"] = recommended.get("compute_device")
    warning = None
    eligible = supported and bool(model) and model in model_ids
    if not supported:
        warning = "服务可连接，但不支持实时模型清单；请手填模型并执行深度验证"
    elif not model:
        warning = "连接成功，请选择或填写模型"
    elif model not in model_ids:
        warning = "当前模型未出现在实时清单中；可手填后执行深度验证"
    if provider["kind"] == "openai_tts":
        config = provider["config"]
        if not str(config.get("host_a") or "").strip() or not str(config.get("host_b") or "").strip():
            eligible = False
            warning = "OpenAI TTS 启用前需要配置两位主持人的音色"
    def verification_error(**kwargs: Any) -> dict[str, Any]:
        failure = _inspection_error(started=started, connection_ok=True, stage="verification", **kwargs)
        failure.update({"catalog_supported": supported, "models": models, "capabilities": capabilities, "recommended": recommended})
        return failure
    if mode == "deep":
        try:
            await _deep_verify(provider)
        except httpx.TimeoutException:
            return verification_error(code="timeout", message="深度验证超时", hint="模型可能仍在加载，请稍后重试")
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            return verification_error(code="verification_failed", message=f"模型调用失败（HTTP {status}）", hint="检查模型、能力和角色是否匹配", upstream_status=status)
        except (httpx.HTTPError, ProviderError) as exc:
            return verification_error(code="verification_failed", message=str(exc) or "深度验证失败", hint="检查模型、音色、设备及 Provider 日志")
        eligible, warning = True, None
    return {
        "status": "passed" if eligible else "warning",
        "connection_ok": True,
        "activation_eligible": eligible,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "catalog_supported": supported,
        "models": models,
        "capabilities": capabilities,
        "recommended": recommended,
        "warning": warning,
        "error": None,
    }


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
    inspection = await inspect_provider(provider, "catalog")
    error = inspection.get("error") or {}
    return {
        "ok": bool(inspection.get("activation_eligible")),
        "message": error.get("message") or inspection.get("warning") or provider["name"],
        "latency_ms": inspection.get("latency_ms"),
        "status": inspection.get("status"),
    }


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
    normalized = _normalized_tts_capabilities(response.json())
    recommended = normalized.get("recommended")
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
