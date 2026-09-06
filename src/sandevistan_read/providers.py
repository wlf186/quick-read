from __future__ import annotations

import asyncio
import base64
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from .database import DB, json_dump, json_load, utc_now
from .context_budget import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_IMAGE_TOKENS,
    MIN_OUTPUT_WINDOW_TOKENS,
    RETRY_SCALES,
    ContextUsage,
    PromptBudget,
    TokenLimits,
    estimate_messages_tokens,
    is_context_error,
    positive_int,
    prompt_budget,
    resolve_temperature,
    truncate_text_tokens,
    validate_token_overrides,
)
from .security import VAULT


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, code: str = "provider_error", status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class ContextOverflowError(ProviderError):
    pass


MAX_CATALOG_BYTES = 4 * 1024 * 1024
TEST_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII="
PODCAST_TTS_CANDIDATE_REVISIONS = {
    "qwen3-tts-0.6b": {
        "base": "5d83992436eae1d760afd27aff78a71d676296fc",
        "custom_voice": "85e237c12c027371202489a0ec509ded67b5e4b5",
    },
}
# A candidate moves here only after the repeatable objective and dual-baseline
# acoustic gates pass. Human blind listening remains available as an optional
# confirmation, but is not required for an automated release qualification.
# Qualification is device-scoped because CPU and GPU inference use different
# precision paths; an untested device must not inherit another device's result.
PODCAST_TTS_QUALIFIED_TARGETS: dict[str, dict[str, Any]] = {
    "qwen3-tts-0.6b": {
        "checkpoints": PODCAST_TTS_CANDIDATE_REVISIONS["qwen3-tts-0.6b"],
        "devices": ["gpu"],
        "method": "automated_dual_baseline_v1",
    },
}


def normalize_provider_base_url(kind: str, value: str) -> str:
    raw = value.strip().rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProviderError("服务地址必须是有效的 HTTP 或 HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ProviderError("服务地址不能包含账号、密码、查询参数或片段")
    path = parsed.path.rstrip("/")
    suffix = "/v1" if kind in {"openai", "openai_tts"} else "/api/v1" if kind in {"sandevistan_audio", "sandevistan_tts"} else "/api"
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


def _normalized_devices(item: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": device.get("id"),
            "available": bool(device.get("available")),
            "default": bool(device.get("default")),
            "precision": device.get("precision"),
            "reason": device.get("unavailable_reason"),
            "reason_code": device.get("unavailable_reason_code"),
        }
        for device in item.get("compute_devices") or []
        if isinstance(device, dict) and device.get("id")
    ]


def _normalized_audio_capabilities(payload: dict[str, Any]) -> dict[str, Any]:
    tts = payload.get("tts") or {}
    default_tts_model = str(tts.get("default_model") or "")
    models: list[dict[str, Any]] = []
    for item in tts.get("model_capabilities") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        models.append(
            {
                "id": item.get("id"),
                "name": item.get("name") or item.get("id"),
                "installed": bool(item.get("installed", True)),
                "default": bool(item.get("default") or item.get("id") == default_tts_model),
                "devices": _normalized_devices(item),
                "voice_modes": item.get("voice_modes") or [],
                "controls": item.get("controls") or {},
                "checkpoints": item.get("checkpoints") or [],
            }
        )
    installed = [item for item in models if item["installed"]]
    default_candidate = next((item for item in installed if item["id"] == default_tts_model), None)
    disqualified_default_id = ""
    qualification = PODCAST_TTS_QUALIFIED_TARGETS.get(str(default_candidate["id"])) if default_candidate else None
    qualified_default_device = None
    if default_candidate and qualification:
        reported = {
            str(checkpoint.get("variant") or ""): str(checkpoint.get("revision") or "")
            for checkpoint in default_candidate.get("checkpoints") or [] if isinstance(checkpoint, dict)
        }
        expected = qualification.get("checkpoints") or {}
        qualified_devices = {str(value) for value in qualification.get("devices") or []}
        available_qualified_devices = [
            str(device["id"])
            for device in default_candidate.get("devices") or []
            if device.get("available") and str(device.get("id") or "") in qualified_devices
        ]
        if (
            any(reported.get(str(variant)) != str(revision) for variant, revision in expected.items())
            or not available_qualified_devices
        ):
            disqualified_default_id = str(default_candidate["id"])
            default_candidate = None
        else:
            qualified_default_device = (
                "gpu" if "gpu" in available_qualified_devices
                else "cpu" if "cpu" in available_qualified_devices
                else available_qualified_devices[0]
            )
    else:
        default_candidate = None
    best = default_candidate
    fallback_installed = [item for item in installed if item["id"] != disqualified_default_id]
    best = best or (max(fallback_installed, key=_quality_score) if fallback_installed else None)
    recommended = None
    if best:
        available = [device["id"] for device in best["devices"] if device["available"]]
        device = qualified_default_device if best is default_candidate else (
            "gpu" if "gpu" in available else "cpu" if "cpu" in available else available[0] if available else None
        )
        recommended = {
            "model": best["id"], "compute_device": device,
            "reason": "service_default" if best is default_candidate else "installed_fallback",
        }
    native = tts.get("preset_speaker_native_languages") or {}
    voices = [{"id": name, "native_language": native.get(name)} for name in tts.get("preset_speakers") or []]
    raw_sequence = tts.get("sequence_jobs") or {}
    sequence_jobs = {
        "supported": bool(raw_sequence.get("supported")),
        "contract_version": int(raw_sequence.get("contract_version") or 0),
        "endpoint": str(raw_sequence.get("endpoint") or ""),
        "voice_modes": [str(value) for value in raw_sequence.get("voice_modes") or []],
        "artifact_mode": str(raw_sequence.get("artifact_mode") or ""),
        "format": str(raw_sequence.get("format") or ""),
        "max_items": max(1, min(100, int(raw_sequence.get("max_items") or 100))),
        "max_total_chars": max(1, int(raw_sequence.get("max_total_chars") or 1)),
    }
    if not (
        sequence_jobs["supported"]
        and sequence_jobs["contract_version"] == 1
        and sequence_jobs["endpoint"] == "/api/v1/tts/sequence-jobs"
        and sequence_jobs["artifact_mode"] == "per_item"
        and sequence_jobs["format"] == "wav"
    ):
        sequence_jobs["supported"] = False
    asr = payload.get("asr") or {}
    asr_models = []
    for item in asr.get("models") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        asr_models.append(
            {
                "id": item.get("id"),
                "name": item.get("name") or item.get("id"),
                "installed": bool(item.get("installed", item.get("installation_state") == "installed")),
                "default": bool(item.get("default")),
                "devices": _normalized_devices(item),
            }
        )
    default_model = str(asr.get("default_model") or "")
    preferred = next((item for item in asr_models if item["id"] == default_model and item["installed"]), None)
    preferred = preferred or next((item for item in asr_models if item["default"] and item["installed"]), None)
    preferred = preferred or next((item for item in asr_models if item["installed"]), None)
    asr_recommended = None
    if preferred:
        available_devices = [device for device in preferred["devices"] if device["available"]]
        selected_device = next((device for device in available_devices if device["default"]), None)
        selected_device = selected_device or next((device for device in available_devices if device["id"] == "gpu"), None)
        selected_device = selected_device or next((device for device in available_devices if device["id"] == "cpu"), None)
        selected_device = selected_device or (available_devices[0] if available_devices else None)
        asr_recommended = {"model": preferred["id"], "compute_device": selected_device["id"] if selected_device else None}
    return {
        "models": models,
        "voices": voices,
        "languages": tts.get("languages") or [],
        "default_model": default_tts_model or None,
        "sequence_jobs": sequence_jobs,
        "recommended": recommended,
        "async": True,
        "discovery": True,
        "asr": {
            "default_model": default_model or None,
            "models": asr_models,
            "recommended": asr_recommended,
            "diarization": asr.get("diarization"),
            "speaker_count": asr.get("speaker_count") or {},
            "languages": asr.get("languages") or [],
            "default_language": asr.get("default_language"),
            "timestamp_precisions": asr.get("timestamp_precisions") or [],
            "aligner_languages": asr.get("aligner_languages") or [],
            "single_task_acceleration": asr.get("single_task_acceleration") or {},
        } if asr else {},
    }


def _instruction_safe_tts_recommendation(
    normalized: dict[str, Any], config: dict[str, Any], recommended: dict[str, Any] | None,
) -> dict[str, Any] | None:
    defaults = {
        "host_a_instruct": "自然、沉稳、有叙事感的知识播客主持人口吻，语速适中。",
        "host_b_instruct": "敏锐、亲切、善于追问和澄清的知识播客主持人口吻，语速适中。",
    }
    customized = any(str(config.get(key) or "").strip() not in {"", value} for key, value in defaults.items())
    selected = next(
        (item for item in normalized.get("models") or [] if item.get("id") == (recommended or {}).get("model")), {}
    )
    if not customized or "preset" in set((selected.get("controls") or {}).get("instruction_voice_modes") or []):
        return recommended
    preserving = [
        item for item in normalized.get("models") or []
        if item.get("installed") and "preset" in set((item.get("controls") or {}).get("instruction_voice_modes") or [])
    ]
    if not preserving:
        return recommended
    selected = max(preserving, key=_quality_score)
    available = [item["id"] for item in selected.get("devices") or [] if item.get("available")]
    return {
        "model": selected["id"],
        "compute_device": "gpu" if "gpu" in available else "cpu" if "cpu" in available else available[0] if available else None,
        "reason": "preserve_custom_instructions",
    }


async def _voiceprint_library(provider: dict[str, Any], models: list[dict[str, Any]]) -> dict[str, Any]:
    if not any("voiceprint" in set(item.get("voice_modes") or []) for item in models):
        return {"status": "unsupported", "people": [], "message": "当前 TTS 模型不支持声纹克隆"}
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
            response = await client.get(
                f"{provider['base_url'].rstrip('/')}/api/v1/voiceprints/people",
                headers=_headers(provider),
            )
        if response.status_code in {404, 405, 501}:
            return {"status": "unsupported", "people": [], "message": "AUDIO Provider 未提供声纹库接口"}
        if not response.is_success:
            return {
                "status": "unavailable",
                "people": [],
                "message": f"声纹库读取失败（HTTP {response.status_code}）",
            }
        if len(response.content) > MAX_CATALOG_BYTES:
            return {"status": "unavailable", "people": [], "message": "声纹库清单过大"}
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return {"status": "unavailable", "people": [], "message": "声纹库暂时无法读取"}
    source = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(source, list):
        return {"status": "unavailable", "people": [], "message": "声纹库返回格式无法识别"}
    people: list[dict[str, Any]] = []
    for person in source:
        if not isinstance(person, dict) or not person.get("id") or not person.get("name"):
            continue
        eligible = [
            sample for sample in person.get("samples") or []
            if isinstance(sample, dict) and sample.get("id") and sample.get("tts_eligible")
        ]
        latest = max(
            eligible,
            key=lambda sample: (str(sample.get("created_at") or ""), str(sample.get("id") or "")),
            default=None,
        )
        item: dict[str, Any] = {
            "id": str(person["id"]),
            "name": str(person["name"]),
            "note": str(person.get("note") or "") or None,
            "eligible_sample_count": len(eligible),
        }
        if latest:
            item["latest_sample"] = {
                "id": str(latest["id"]),
                "language": str(latest.get("language") or "Auto"),
                "duration": latest.get("duration"),
                "created_at": latest.get("created_at"),
            }
        people.append(item)
    return {"status": "ready", "people": people, "message": None}


def host_voice_selection(
    config: dict[str, Any], host: str, language: str, *, preset_override: str | None = None,
) -> dict[str, Any]:
    mode = str(config.get(f"{host}_voice_mode") or "preset")
    if mode == "voiceprint":
        person_id = str(config.get(f"{host}_voiceprint_person_id") or "").strip()
        sample_id = str(config.get(f"{host}_voiceprint_sample_id") or "").strip()
        if not person_id or not sample_id:
            raise ProviderError(f"{host.upper()} 尚未选择可用的声纹人员")
        return {"mode": "voiceprint", "person_id": person_id, "sample_id": sample_id, "label": person_id}
    defaults = {"host_a": ("Vivian", "Ryan"), "host_b": ("Dylan", "Aiden")}
    chinese, english = defaults[host]
    field = f"{host}_en" if language == "en" else host
    speaker = str(preset_override or config.get(field) or (english if language == "en" else chinese)).strip()
    if not speaker:
        raise ProviderError(f"{host.upper()} 尚未选择预置音色")
    return {"mode": "preset", "speaker": speaker, "label": speaker}


def host_voice_instruction(
    config: dict[str, Any], host: str, language: str, *, supported: bool,
) -> str | None:
    if not supported or config.get(f"{host}_voice_mode", "preset") != "preset":
        return None
    base = str(config.get(f"{host}_instruct") or "").strip()
    if language == "en":
        guardrail = (
            "Keep the same restrained knowledge-podcast voice throughout the episode: use an even medium pace, "
            "stable volume and pitch, and only a slight natural rise for questions. Do not become excited, angry, "
            "shout, rush, or make abrupt changes in intensity."
        )
    else:
        guardrail = (
            "整集保持同一位知识播客主持人的克制表达基线：使用均匀的中等语速、稳定的音量和音高，"
            "疑问句只做轻微自然上扬；不要突然兴奋、愤怒、喊叫、加速或大幅改变强弱。"
        )
    return f"{base} {guardrail}".strip()


def _resolve_audio_voice_config(
    provider: dict[str, Any], capabilities: dict[str, Any], voiceprint_library: dict[str, Any],
) -> str | None:
    config = provider.setdefault("config", {})
    config.setdefault("host_a_voice_mode", "preset")
    config.setdefault("host_b_voice_mode", "preset")
    config.setdefault("host_a", "Vivian")
    config.setdefault("host_b", "Dylan")
    config.setdefault("host_a_en", "Ryan")
    config.setdefault("host_b_en", "Aiden")
    selected = next(
        (item for item in capabilities.get("models") or [] if item.get("id") == provider.get("model")),
        None,
    )
    if selected:
        supported_modes = set(selected.get("voice_modes") or ["preset"])
        for host in ("host_a", "host_b"):
            mode = str(config.get(f"{host}_voice_mode") or "preset")
            if mode not in supported_modes:
                return f"所选 TTS 模型不支持 {mode} 音色模式"
    people = {item.get("id"): item for item in voiceprint_library.get("people") or []}
    for host in ("host_a", "host_b"):
        if config.get(f"{host}_voice_mode") != "voiceprint":
            continue
        if voiceprint_library.get("status") != "ready":
            return voiceprint_library.get("message") or "声纹库不可用"
        person_id = str(config.get(f"{host}_voiceprint_person_id") or "").strip()
        person = people.get(person_id)
        if not person:
            return f"{host.upper()} 选择的声纹人员不存在"
        latest = person.get("latest_sample") or {}
        if not latest.get("id"):
            return f"{host.upper()} 选择的声纹人员没有可用于 TTS 的样本"
        config[f"{host}_voiceprint_sample_id"] = latest["id"]
    try:
        a = host_voice_selection(config, "host_a", "zh-CN")
        b = host_voice_selection(config, "host_b", "zh-CN")
    except ProviderError as exc:
        return str(exc)
    if a["mode"] == b["mode"] == "preset" and a["speaker"] == b["speaker"]:
        return "Host A 与 Host B 不能使用同一个预置音色"
    if a["mode"] == b["mode"] == "voiceprint" and a["person_id"] == b["person_id"]:
        return "Host A 与 Host B 不能使用同一个声纹人员"
    return None


def _asr_execution(provider: dict[str, Any]) -> tuple[str, str, bool]:
    config = provider.get("config") or {}
    asr = (provider.get("capabilities") or {}).get("asr") or {}
    recommended = asr.get("recommended") or {}
    auto_select = bool(config.get("asr_auto_select", True))
    model = str((recommended.get("model") if auto_select else config.get("asr_model")) or config.get("asr_model") or asr.get("default_model") or "qwen3-asr-0.6b")
    device = str((recommended.get("compute_device") if auto_select else config.get("asr_compute_device")) or config.get("asr_compute_device") or "gpu")
    return model, device, bool(config.get("asr_allow_device_fallback", True))


def audio_provider_readiness(provider: dict[str, Any] | None) -> tuple[bool, str]:
    if not provider:
        return False, "请先配置并启用 AUDIO Provider"
    if provider.get("role") != "audio" or provider.get("kind") not in {"sandevistan_audio", "sandevistan_tts"}:
        return False, "当前 Provider 不同时提供 Podcast 所需的 TTS 与 ASR"
    capabilities = provider.get("capabilities") or {}
    tts_model = str(provider.get("model") or "")
    tts_selected = next((item for item in capabilities.get("models") or [] if item.get("id") == tts_model), None)
    if not tts_selected or tts_selected.get("installed") is False:
        return False, "AUDIO Provider 没有可用的 TTS 模型"
    config = provider.get("config") or {}
    supported_modes = set(tts_selected.get("voice_modes") or ["preset"])
    for host in ("host_a", "host_b"):
        mode = str(config.get(f"{host}_voice_mode") or "preset")
        if mode not in supported_modes:
            return False, f"所选 TTS 模型不支持 {mode} 音色模式"
        if mode == "voiceprint" and (
            not str(config.get(f"{host}_voiceprint_person_id") or "").strip()
            or not str(config.get(f"{host}_voiceprint_sample_id") or "").strip()
        ):
            return False, f"{host.upper()} 尚未锁定可用的声纹样本"
    if (
        config.get("host_a_voice_mode", "preset") == config.get("host_b_voice_mode", "preset") == "preset"
        and str(config.get("host_a") or "Vivian") == str(config.get("host_b") or "Dylan")
    ):
        return False, "Host A 与 Host B 不能使用同一个预置音色"
    if (
        config.get("host_a_voice_mode") == config.get("host_b_voice_mode") == "voiceprint"
        and str(config.get("host_a_voiceprint_person_id") or "")
        == str(config.get("host_b_voiceprint_person_id") or "")
    ):
        return False, "Host A 与 Host B 不能使用同一个声纹人员"
    tts_device = str((provider.get("config") or {}).get("compute_device") or "")
    if tts_selected.get("devices") and not any(
        item.get("id") == tts_device and item.get("available") for item in tts_selected.get("devices") or []
    ):
        return False, "所选 TTS 设备不可用"
    asr = capabilities.get("asr") or {}
    models = asr.get("models") or []
    model, device, _ = _asr_execution(provider)
    selected = next((item for item in models if item.get("id") == model), None)
    if not selected or not selected.get("installed"):
        return False, "AUDIO Provider 没有可用的 ASR 模型"
    if not any(item.get("id") == device and item.get("available") for item in selected.get("devices") or []):
        return False, "所选 ASR 设备不可用"
    if not asr.get("diarization"):
        return False, "ASR Provider 未声明说话人分离能力"
    if "segment" not in set(asr.get("timestamp_precisions") or []):
        return False, "ASR Provider 未声明分段时间戳能力"
    languages = set(asr.get("languages") or [])
    aligners = set(asr.get("aligner_languages") or [])
    if not {"Chinese", "English"}.issubset(languages) or not {"Chinese", "English"}.issubset(aligners):
        return False, "ASR Provider 必须支持中英文识别与时间对齐"
    return True, "AUDIO Provider 已就绪"


CONTEXT_LIMIT_FIELDS = ("context_window_tokens", "context_window", "context_length", "max_context_length", "max_model_len")
OUTPUT_LIMIT_FIELDS = ("max_output_tokens", "max_completion_tokens")
LIMIT_CONTAINERS = ("token_limits", "limits", "capabilities", "top_provider", "metadata", "details")


def _limit_from_metadata(payload: dict[str, Any], fields: tuple[str, ...]) -> int | None:
    candidates = [payload]
    candidates.extend(payload.get(name) for name in LIMIT_CONTAINERS if isinstance(payload.get(name), dict))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for field_name in fields:
            value = positive_int(candidate.get(field_name))
            if value:
                return value
    return None


def _catalog_token_limits(payload: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    context = _limit_from_metadata(payload, CONTEXT_LIMIT_FIELDS)
    max_input = _limit_from_metadata(payload, ("max_input_tokens",))
    max_output = _limit_from_metadata(payload, OUTPUT_LIMIT_FIELDS)
    if context:
        result["model_context_tokens"] = context
    if max_input:
        result["max_input_tokens"] = max_input
    if max_output:
        result["max_output_tokens"] = max_output
    return result


def _ollama_show_limits(payload: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    model_info = payload.get("model_info") or {}
    contexts = [positive_int(value) for key, value in model_info.items() if str(key).endswith(".context_length")]
    contexts = [value for value in contexts if value]
    if contexts:
        result["model_context_tokens"] = max(contexts)
    image_values = [positive_int(value) for key, value in model_info.items() if str(key).endswith(".mm.tokens_per_image")]
    image_values = [value for value in image_values if value]
    if image_values:
        result["image_tokens_per_image"] = max(image_values)
    parameters = str(payload.get("parameters") or "")
    match = re.search(r"(?:^|\n)\s*num_ctx\s+(\d+)\b", parameters)
    if match:
        result["modelfile_context_tokens"] = int(match.group(1))
    return result


def _parameter_count(payload: dict[str, Any], model: str = "") -> int | None:
    model_info = payload.get("model_info") or {}
    for key, value in model_info.items():
        if str(key).endswith(".parameter_count"):
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count > 0:
                return count
    details = payload.get("details") or {}
    text = str(details.get("parameter_size") or model)
    match = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])\b", text)
    if not match:
        return None
    scale = 1_000_000_000 if match.group(2).lower() == "b" else 1_000_000
    return round(float(match.group(1)) * scale)


def study_generation_profile(provider: dict[str, Any]) -> dict[str, Any]:
    """Select a transparent study pipeline tier without confusing context size with model quality."""
    config = provider.get("config") or {}
    requested = str(config.get("study_generation_tier") or "auto")
    if requested not in {"auto", "lite", "full"}:
        requested = "auto"
    capabilities = provider.get("capabilities") or {}
    model_profile = capabilities.get("model_profile") or {}
    parameter_count = model_profile.get("parameter_count")
    try:
        parameter_count = int(parameter_count) if parameter_count else None
    except (TypeError, ValueError):
        parameter_count = None
    limits = TokenLimits.from_provider(provider)
    if requested != "auto":
        tier, source = requested, "manual"
        reason = "Provider 人工指定"
    else:
        constraints: list[str] = []
        if limits.effective_context_tokens < 8192:
            constraints.append("上下文小于 8K")
        if limits.max_output_tokens < 1536:
            constraints.append("输出小于 1.5K")
        if parameter_count is not None and parameter_count < 7_000_000_000:
            constraints.append("参数量小于 7B")
        tier = "lite" if constraints else "full"
        source = "auto"
        reason = "、".join(constraints) if constraints else "上下文与模型能力满足完整审校管线"
    return {
        "tier": tier,
        "source": source,
        "reason": reason,
        "parameter_count": parameter_count,
        "supports_difficulties": ["easy", "medium", "mixed"] if tier == "lite" else ["easy", "medium", "hard", "mixed"],
    }


def _effective_token_limits(
    provider: dict[str, Any],
    detected: dict[str, int],
    *,
    context_source: str | None = None,
) -> dict[str, Any]:
    config = provider.get("config") or {}
    validate_token_overrides(config)
    model_max = positive_int(detected.get("model_context_tokens"))
    manual_context = positive_int(config.get("context_window_tokens"))
    if manual_context and model_max and manual_context > model_max:
        raise ProviderError(f"人工上下文窗口 {manual_context} 超过模型报告的最大值 {model_max}")
    if manual_context:
        effective, selected_context_source = manual_context, "manual"
    else:
        runtime = positive_int(detected.get("runtime_context_tokens"))
        modelfile = positive_int(detected.get("modelfile_context_tokens"))
        provider_context = positive_int(detected.get("provider_context_tokens"))
        effective = runtime or modelfile or provider_context or DEFAULT_CONTEXT_WINDOW_TOKENS
        selected_context_source = (
            "ollama_runtime" if runtime else "ollama_modelfile" if modelfile else context_source or "provider_metadata" if provider_context else "fallback"
        )
    if model_max:
        effective = min(effective, model_max)
    manual_output = positive_int(config.get("max_output_tokens"))
    detected_output = positive_int(detected.get("max_output_tokens"))
    if manual_output and manual_output >= effective:
        raise ProviderError("人工最大输出必须小于有效上下文窗口")
    max_output = manual_output or detected_output or max(MIN_OUTPUT_WINDOW_TOKENS, min(4096, effective // 4))
    max_output = min(max_output, effective - 1)
    return {
        "model_context_tokens": model_max,
        "effective_context_tokens": effective,
        "max_input_tokens": positive_int(detected.get("max_input_tokens")),
        "max_output_tokens": max_output,
        "context_source": selected_context_source,
        "output_source": "manual" if manual_output else "provider_metadata" if detected_output else "derived",
        "image_tokens_per_image": positive_int(detected.get("image_tokens_per_image")) or DEFAULT_IMAGE_TOKENS,
        "probed_at": utc_now(),
    }


async def _discover_chat_capabilities(provider: dict[str, Any], models: list[dict[str, Any]]) -> dict[str, Any]:
    model = str(provider.get("model") or "").strip()
    if provider.get("role") not in {"main", "vlm"} or not model:
        return {}
    selected = next((item for item in models if item.get("id") == model), {})
    detected = dict(selected.get("token_limits") or {})
    headers = _headers(provider)
    if provider["kind"] == "ollama":
        model_profile: dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=8, follow_redirects=False) as client:
            show_result, running_result = await asyncio.gather(
                client.post(f"{provider['base_url']}/api/show", json={"model": model, "verbose": False}, headers=headers),
                client.get(f"{provider['base_url']}/api/ps", headers=headers),
                return_exceptions=True,
            )
        if isinstance(show_result, httpx.Response) and show_result.is_success and len(show_result.content) <= MAX_CATALOG_BYTES:
            show_payload = show_result.json()
            if isinstance(show_payload, dict):
                detected.update(_ollama_show_limits(show_payload))
                parameter_count = _parameter_count(show_payload, model)
                if parameter_count:
                    model_profile["parameter_count"] = parameter_count
                details = show_payload.get("details") or {}
                if details.get("quantization_level"):
                    model_profile["quantization"] = details["quantization_level"]
        if isinstance(running_result, httpx.Response) and running_result.is_success and len(running_result.content) <= MAX_CATALOG_BYTES:
            for item in (running_result.json().get("models") or []):
                if isinstance(item, dict) and model in {item.get("model"), item.get("name")}:
                    running_context = positive_int(item.get("context_length"))
                    if running_context:
                        detected["runtime_context_tokens"] = running_context
                    break
        capabilities = {"token_limits": _effective_token_limits(provider, detected), "model_profile": model_profile}
        capabilities["study_generation"] = study_generation_profile({**provider, "capabilities": capabilities})
        return capabilities

    if selected:
        detail_url = f"{provider['base_url']}/v1/models/{quote(model, safe='')}"
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=False) as client:
                response = await client.get(detail_url, headers=headers)
            if response.is_success and len(response.content) <= MAX_CATALOG_BYTES:
                detail = response.json()
                if isinstance(detail, dict):
                    detail_limits = _catalog_token_limits(detail)
                    for key, value in detail_limits.items():
                        detected.setdefault(key, value)
        except (httpx.HTTPError, ValueError):
            pass
    if detected.get("model_context_tokens"):
        detected.setdefault("provider_context_tokens", detected["model_context_tokens"])
    capabilities = {"token_limits": _effective_token_limits(provider, detected, context_source="provider_metadata")}
    capabilities["study_generation"] = study_generation_profile({**provider, "capabilities": capabilities})
    return capabilities


async def _provider_catalog(provider: dict[str, Any]) -> tuple[bool, list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    base_url = provider["base_url"]
    headers = _headers(provider)
    async with httpx.AsyncClient(timeout=8, follow_redirects=False) as client:
        if provider["kind"] == "ollama":
            response = await client.get(f"{base_url}/api/tags", headers=headers)
        elif provider["kind"] in {"sandevistan_audio", "sandevistan_tts"}:
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
    if provider["kind"] in {"sandevistan_audio", "sandevistan_tts"}:
        normalized = _normalized_audio_capabilities(payload)
        recommended = _instruction_safe_tts_recommendation(
            normalized, provider.get("config") or {}, normalized.get("recommended"),
        )
        models = normalized.pop("models")
        normalized.pop("recommended")
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
            "token_limits": _catalog_token_limits(item),
        }
    return True, sorted(models_by_id.values(), key=lambda item: item["id"].lower()), {}, None


async def _deep_verify(provider: dict[str, Any]) -> None:
    model = str(provider.get("model") or "").strip()
    if not model:
        raise ProviderError("请先选择或填写模型")
    headers = _headers(provider)
    role, kind = provider["role"], provider["kind"]
    if role == "audio" and kind in {"sandevistan_audio", "sandevistan_tts"}:
        config = provider.get("config") or {}
        device = str(config.get("compute_device") or "cpu")
        outputs: list[Path] = []
        try:
            for host, text in (("host_a", "主持人甲连接测试"), ("host_b", "主持人乙连接测试")):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                    output = Path(handle.name)
                outputs.append(output)
                selection = host_voice_selection(config, host, "zh-CN")
                await _synthesize_sandevistan(
                    provider,
                    text,
                    selection.get("speaker"),
                    output,
                    language="Chinese",
                    model=model,
                    compute_device=device,
                    voice_mode=selection["mode"],
                    voiceprint_sample_id=selection.get("sample_id"),
                    instruct=None,
                    idempotency_key=f"provider-test-{uuid.uuid4()}",
                    cancel_check=None,
                )
                if not output.exists() or output.stat().st_size == 0:
                    raise ProviderError(f"{host.upper()} TTS Provider 未返回音频")
            result = await _transcribe_with_provider(
                provider,
                outputs[0],
                language="Chinese",
                idempotency_key=f"provider-asr-test-{uuid.uuid4()}",
                cancel_check=None,
            )
            segments = [item for item in result.get("segments") or [] if isinstance(item, dict)]
            if not segments or not all(
                item.get("start") is not None
                and item.get("end") is not None
                and (item.get("speaker") or item.get("speaker_label"))
                for item in segments
            ):
                raise ProviderError("ASR Provider 未返回带时间戳和说话人标签的转写")
        finally:
            for output in outputs:
                output.unlink(missing_ok=True)
        return
    async with httpx.AsyncClient(timeout=180, follow_redirects=False) as client:
        if role == "tts_only":
            voice = str((provider.get("config") or {}).get("host_a") or "").strip()
            if not voice:
                raise ProviderError("OpenAI TTS 需要配置 Host A 音色")
            response = await client.post(
                f"{provider['base_url']}/v1/audio/speech",
                json={"model": model, "voice": voice, "input": "连接测试", "response_format": "wav"},
                headers=headers,
            )
            if not response.is_success:
                raise _provider_response_error(response)
            if not response.content:
                raise ProviderError("TTS Provider 未返回音频")
            return
        try:
            temperature = resolve_temperature(provider.get("config") or {}, 0.0)
        except ValueError as exc:
            raise ProviderError(str(exc)) from exc
        if kind == "ollama":
            message: dict[str, Any] = {"role": "user", "content": "只回复 OK"}
            if role == "vlm":
                message["images"] = [TEST_IMAGE_BASE64]
            limits = TokenLimits.from_provider(provider)
            response = await client.post(
                f"{provider['base_url']}/api/chat",
                json={"model": model, "messages": [message], "stream": False, "think": False, "options": {"temperature": temperature, "num_predict": 64, "num_ctx": limits.effective_context_tokens}},
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
                json={"model": model, "messages": [{"role": "user", "content": content}], "temperature": temperature, "max_tokens": 64},
                headers=headers,
            )
        if not response.is_success:
            raise _provider_response_error(response)
        try:
            result = response.json()
        except ValueError as exc:
            raise ProviderError("Provider 返回了无法识别的验证结果") from exc
        if not isinstance(result, dict):
            raise ProviderError("Provider 返回了无法识别的验证结果")
        if kind == "ollama":
            message = result.get("message")
        else:
            choices = result.get("choices")
            first = choices[0] if isinstance(choices, list) and choices else None
            message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(message, dict) or not str(message.get("content") or "").strip():
            raise ProviderError("Provider 未返回验证文本")


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

    voiceprint_library: dict[str, Any] | None = None
    if provider["role"] == "audio" and provider["kind"] in {"sandevistan_audio", "sandevistan_tts"}:
        voiceprint_library = await _voiceprint_library(provider, models)
    model = str(provider.get("model") or "").strip()
    model_ids = {str(item.get("id") or "") for item in models}
    if provider["kind"] in {"sandevistan_audio", "sandevistan_tts"} and provider["config"].get("auto_select") and recommended:
        model = str(recommended.get("model") or model)
        provider["model"] = model
        provider["config"]["compute_device"] = recommended.get("compute_device")
    warning = None
    context_warning = None
    eligible = supported and bool(model) and model in model_ids
    if not supported:
        warning = "服务可连接，但不支持实时模型清单；请手填模型并执行深度验证"
    elif not model:
        warning = "连接成功，请选择或填写模型"
    elif model not in model_ids:
        warning = "当前模型未出现在实时清单中；可手填后执行深度验证"
    def verification_error(**kwargs: Any) -> dict[str, Any]:
        failure = _inspection_error(started=started, connection_ok=True, stage="verification", **kwargs)
        failure.update({"catalog_supported": supported, "models": models, "capabilities": capabilities, "recommended": recommended})
        if voiceprint_library is not None:
            failure["voiceprint_library"] = voiceprint_library
        return failure
    if provider["role"] in {"main", "vlm"} and model:
        try:
            capabilities.update(await _discover_chat_capabilities(provider, models))
            provider["capabilities"] = capabilities
        except (ValueError, ProviderError) as exc:
            return verification_error(code="invalid_context_limits", message=str(exc), hint="调整上下文或最大输出覆盖值")
        token_limits = capabilities.get("token_limits") or {}
        if token_limits.get("context_source") == "fallback":
            context_warning = "Provider 未报告上下文窗口，当前按 4K 安全预算运行；可在模型设置中人工覆盖"
            warning = f"{warning}；{context_warning}" if warning else context_warning
    if provider["role"] == "audio":
        provider["capabilities"] = {**capabilities, "models": models}
        asr_recommended = (capabilities.get("asr") or {}).get("recommended") or {}
        if provider["config"].get("asr_auto_select", True) and asr_recommended.get("model"):
            provider["config"]["asr_model"] = asr_recommended["model"]
            provider["config"]["asr_compute_device"] = asr_recommended.get("compute_device")
        voice_error = _resolve_audio_voice_config(
            provider,
            provider["capabilities"],
            voiceprint_library or {"status": "unsupported", "people": []},
        )
        if voice_error:
            eligible = False
            warning = voice_error
        else:
            audio_ready, audio_message = audio_provider_readiness(provider)
            if not audio_ready:
                eligible = False
                warning = audio_message
    if provider["kind"] == "openai_tts":
        config = provider["config"]
        if not str(config.get("host_a") or "").strip() or not str(config.get("host_b") or "").strip():
            warning = "OpenAI TTS 旧配置缺少两位主持人的音色"
        eligible = False
        warning = f"{warning}；仅提供 TTS，不能用于 Podcast" if warning else "仅提供 TTS，不能用于 Podcast"
    if mode == "deep":
        try:
            await _deep_verify(provider)
        except httpx.TimeoutException:
            return verification_error(code="timeout", message="深度验证超时", hint="模型可能仍在加载，请稍后重试")
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            return verification_error(code="verification_failed", message=f"模型调用失败（HTTP {status}）", hint="检查模型、能力和角色是否匹配", upstream_status=status)
        except ProviderError as exc:
            return verification_error(code="verification_failed", message=str(exc) or "深度验证失败", hint="检查模型、温度、音色、设备及 Provider 日志", upstream_status=exc.status)
        except httpx.HTTPError as exc:
            return verification_error(code="verification_failed", message=str(exc) or "深度验证失败", hint="检查模型、音色、设备及 Provider 日志")
        if provider["role"] in {"main", "vlm"}:
            eligible = True
            warning = context_warning
        elif eligible:
            warning = context_warning
    result = {
        "status": "warning" if warning else "passed" if eligible else "warning",
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
    if voiceprint_library is not None:
        result["voiceprint_library"] = voiceprint_library
        result["resolved_audio_config"] = {
            key: value for key, value in provider["config"].items()
            if key.startswith("host_")
        }
    return result


def active_provider(role: str) -> dict[str, Any] | None:
    setting = DB.fetchone("SELECT enabled FROM provider_role_settings WHERE role=?", (role,))
    if setting and not setting["enabled"]:
        return None
    row = DB.fetchone(
        "SELECT * FROM provider_profiles WHERE role=? AND COALESCE(selected,active)=1 ORDER BY updated_at DESC LIMIT 1",
        (role,),
    )
    if not row:
        return None
    row["capabilities"] = json_load(row.pop("capabilities_json"), {})
    row["config"] = json_load(row.pop("config_json"), {})
    row["api_key"] = VAULT.decrypt(row.pop("secret_enc", "")) if row.get("secret_enc") else ""
    if role in {"main", "vlm"}:
        row["capabilities"]["study_generation"] = study_generation_profile(row)
    return row


def provider_by_id(provider_id: str) -> dict[str, Any] | None:
    row = DB.fetchone("SELECT * FROM provider_profiles WHERE id=?", (provider_id,))
    if not row:
        return None
    row["capabilities"] = json_load(row.pop("capabilities_json"), {})
    row["config"] = json_load(row.pop("config_json"), {})
    secret = row.pop("secret_enc", "")
    row["api_key"] = VAULT.decrypt(secret) if secret else ""
    if row.get("role") in {"main", "vlm"}:
        row["capabilities"]["study_generation"] = study_generation_profile(row)
    return row


@dataclass
class ChatCompletion:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    temperature: float | None = None
    temperature_source: str | None = None


@dataclass
class PromptBuild:
    messages: list[dict[str, Any]]
    total_segments: int = 0
    included_segments: int = 0
    truncated_segments: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetedCompletion:
    content: str
    build: PromptBuild
    budget: PromptBudget
    finish_reason: str | None = None


def _chat_provider(role: str) -> dict[str, Any]:
    provider = active_provider(role)
    if not provider and role == "vlm":
        provider = active_provider("main")
    if not provider:
        raise ProviderError(f"No active {role} provider")
    return provider


def _provider_response_error(response: httpx.Response) -> ProviderError:
    code, message = "upstream_error", f"Provider 返回 HTTP {response.status_code}"
    try:
        payload = response.json()
        error = payload.get("error", payload) if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            code = str(error.get("code") or error.get("type") or code)
            message = str(error.get("message") or error.get("detail") or message)
        elif error:
            message = str(error)
    except ValueError:
        pass
    if is_context_error(response.status_code, code, message):
        return ContextOverflowError(message, code=code, status=response.status_code)
    return ProviderError(message, code=code, status=response.status_code)


async def _chat_once(
    provider: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    json_mode: bool,
    timeout: float,
    max_tokens: int,
    temperature: float,
) -> ChatCompletion:
    headers: dict[str, str] = {}
    if provider["api_key"]:
        headers["Authorization"] = f"Bearer {provider['api_key']}"
    limits = TokenLimits.from_provider(provider)
    try:
        request_temperature = resolve_temperature(provider.get("config") or {}, temperature)
    except ValueError as exc:
        raise ProviderError(str(exc)) from exc
    async with httpx.AsyncClient(timeout=timeout) as client:
        if provider["kind"] == "ollama":
            payload: dict[str, Any] = {
                "model": provider["model"],
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": request_temperature, "num_predict": max_tokens, "num_ctx": limits.effective_context_tokens},
            }
            if json_mode:
                payload["format"] = "json"
            response = await client.post(f"{provider['base_url'].rstrip('/')}/api/chat", json=payload, headers=headers)
            if not response.is_success:
                raise _provider_response_error(response)
            result = response.json()
            return ChatCompletion(
                str(result.get("message", {}).get("content", "")),
                positive_int(result.get("prompt_eval_count")),
                positive_int(result.get("eval_count")),
                str(result.get("done_reason") or "") or None,
                temperature=request_temperature,
                temperature_source="provider" if (provider.get("config") or {}).get("temperature") is not None else "task_default",
            )
        if provider["kind"] == "openai":
            payload = {"model": provider["model"], "messages": messages, "temperature": request_temperature, "max_tokens": max_tokens}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            response = await client.post(f"{provider['base_url'].rstrip('/')}/v1/chat/completions", json=payload, headers=headers)
            if not response.is_success:
                raise _provider_response_error(response)
            result = response.json()
            usage = result.get("usage") or {}
            completion_details = usage.get("completion_tokens_details") or {}
            prompt_details = usage.get("prompt_tokens_details") or {}
            choice = result["choices"][0]
            return ChatCompletion(
                str(choice["message"]["content"] or ""),
                positive_int(usage.get("prompt_tokens")),
                positive_int(usage.get("completion_tokens")),
                str(choice.get("finish_reason") or "") or None,
                positive_int(completion_details.get("reasoning_tokens")),
                positive_int(prompt_details.get("cached_tokens")) or positive_int(usage.get("cached_tokens")),
                request_temperature,
                "provider" if (provider.get("config") or {}).get("temperature") is not None else "task_default",
            )
    raise ProviderError(f"Provider {provider['kind']} cannot serve chat")


async def chat(
    messages: list[dict[str, Any]],
    *,
    role: str = "main",
    json_mode: bool = False,
    timeout: float = 180,
    max_tokens: int = 1400,
    temperature: float = 0.15,
) -> str:
    provider = _chat_provider(role)
    limits = TokenLimits.from_provider(provider)
    budget = prompt_budget(limits, max_tokens, min(128, max_tokens), 1.0)
    estimated = estimate_messages_tokens(messages, limits.image_tokens_per_image)
    if estimated > budget.input_tokens:
        raise ContextOverflowError(
            f"本地估算输入 {estimated} tokens 超过安全预算 {budget.input_tokens}", code="local_context_budget", status=422
        )
    return (await _chat_once(provider, messages, json_mode=json_mode, timeout=timeout, max_tokens=budget.output_tokens, temperature=temperature)).content


async def budgeted_chat(
    builder: Callable[[PromptBudget], PromptBuild],
    *,
    role: str = "main",
    json_mode: bool = False,
    timeout: float = 180,
    max_tokens: int = 1400,
    minimum_output_tokens: int = 128,
    temperature: float = 0.15,
    trace: ContextUsage | None = None,
    stage: str = "generation",
    provider_override: dict[str, Any] | None = None,
) -> BudgetedCompletion:
    provider = provider_override or _chat_provider(role)
    limits = TokenLimits.from_provider(provider)
    last_error: ContextOverflowError | None = None
    for attempt, scale in enumerate(RETRY_SCALES, start=1):
        budget = prompt_budget(limits, max_tokens, minimum_output_tokens, scale)
        build = builder(budget)
        estimated = estimate_messages_tokens(build.messages, budget.image_tokens_per_image)
        if estimated > budget.input_tokens:
            raise ContextOverflowError(
                f"提示构建结果 {estimated} tokens 超过安全预算 {budget.input_tokens}", code="local_context_budget", status=422
            )
        try:
            if trace:
                trace.begin_request(estimated_tokens=estimated + budget.output_tokens)
            try:
                completion = await _chat_once(
                    provider,
                    build.messages,
                    json_mode=json_mode,
                    timeout=timeout,
                    max_tokens=budget.output_tokens,
                    temperature=temperature,
                )
            except (httpx.ConnectError, httpx.TimeoutException):
                # 传输层错误与提示预算无关，同一预算下只重试一次；不消耗溢出降档
                if trace:
                    trace.record_failure()
                completion = await _chat_once(
                    provider,
                    build.messages,
                    json_mode=json_mode,
                    timeout=timeout,
                    max_tokens=budget.output_tokens,
                    temperature=temperature,
                )
        except ContextOverflowError as exc:
            if trace:
                trace.record_failure()
                trace.overflow_retries += 1
            last_error = exc
            continue
        except Exception:
            if trace:
                trace.record_failure()
            raise
        if trace:
            trace.record(
                limits=limits,
                requested_output=max_tokens,
                output_tokens=budget.output_tokens,
                estimated_prompt=estimated,
                actual_prompt=completion.prompt_tokens,
                actual_completion=completion.completion_tokens,
                reasoning_tokens=completion.reasoning_tokens,
                cached_tokens=completion.cached_tokens,
                temperature=completion.temperature,
                temperature_source=completion.temperature_source,
                stage=stage,
                total_segments=build.total_segments,
                included_segments=build.included_segments,
                truncated_segments=build.truncated_segments,
            )
            if completion.finish_reason in {"length", "max_tokens"}:
                trace.output_limited_calls += 1
        return BudgetedCompletion(completion.content, build, budget, completion.finish_reason)
    raise last_error or ContextOverflowError("Provider 上下文窗口不足", code="context_window_exceeded", status=422)


async def describe_image(path: Path, nearby_text: str, provider: dict[str, Any] | None = None) -> str:
    provider = provider or active_provider("vlm") or active_provider("main")
    if not provider or not provider.get("capabilities", {}).get("vision"):
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    prefix = "仅描述图中可验证的文字、图表关系和版面信息；不要补充图外事实。附近提取文本：\n"

    def build(budget: PromptBudget) -> PromptBuild:
        text_budget = max(0, budget.input_tokens - budget.image_tokens_per_image - estimate_messages_tokens([{"role": "user", "content": prefix}]))
        nearby, clipped = truncate_text_tokens(nearby_text, text_budget)
        prompt = prefix + nearby
        if provider["kind"] == "ollama":
            messages = [{"role": "user", "content": prompt, "images": [encoded]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}]}]
        return PromptBuild(messages, 1, 1, int(clipped))

    return (await budgeted_chat(build, role=provider["role"], max_tokens=700, minimum_output_tokens=128, provider_override=provider)).content


async def health(role: str, provider_id: str | None = None) -> dict[str, Any]:
    if provider_id is None:
        setting = DB.fetchone("SELECT enabled FROM provider_role_settings WHERE role=?", (role,))
        if setting and not setting["enabled"]:
            return {"ok": False, "status": "disabled", "message": "已暂停"}
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


async def probe_chat_provider(provider_id: str, *, apply: bool = True) -> dict[str, Any]:
    provider = provider_by_id(provider_id)
    if not provider:
        raise ProviderError("Provider does not exist")
    if provider["role"] not in {"main", "vlm"}:
        raise ProviderError("Only MAIN or VLM providers expose chat token limits")
    inspection = await inspect_provider(provider, "catalog")
    discovered = inspection.get("capabilities") or {}
    token_limits = discovered.get("token_limits")
    if apply and token_limits:
        merged = dict(provider.get("capabilities") or {})
        merged.update(discovered)
        DB.execute(
            "UPDATE provider_profiles SET capabilities_json=?,updated_at=? WHERE id=?",
            (json_dump(merged), utc_now(), provider_id),
        )
    return {"ok": bool(inspection.get("connection_ok")), **inspection}


async def refresh_active_chat_capabilities() -> None:
    providers = [provider for role in ("main", "vlm") if (provider := active_provider(role))]
    pending = []
    for provider in providers:
        token_limits = (provider.get("capabilities") or {}).get("token_limits") or {}
        if not token_limits.get("probed_at"):
            pending.append(probe_chat_provider(provider["id"], apply=True))
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _quality_score(model: dict[str, Any]) -> tuple[float, int, str]:
    name = f"{model.get('id', '')} {model.get('name', '')}".lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", name)
    parameters = float(match.group(1)) if match else 0.0
    controls = model.get("controls") or {}
    return parameters, int(bool(controls.get("instruction_voice_modes"))), str(model.get("id", ""))


async def probe_audio_provider(provider_id: str, *, apply_defaults: bool = False) -> dict[str, Any]:
    """Read live TTS/ASR capabilities and optionally apply safe execution defaults."""
    provider = provider_by_id(provider_id)
    if not provider:
        raise ProviderError("AUDIO provider does not exist")
    if provider["kind"] not in {"sandevistan_audio", "sandevistan_tts"}:
        return {"ok": True, "kind": provider["kind"], "discovery": False, "models": [], "voices": [], "recommended": None}
    headers = {"Authorization": f"Bearer {provider['api_key']}"} if provider["api_key"] else {}
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(f"{provider['base_url'].rstrip('/')}/api/v1/capabilities", headers=headers)
        response.raise_for_status()
    normalized = _normalized_audio_capabilities(response.json())
    recommended = _instruction_safe_tts_recommendation(
        normalized, provider.get("config") or {}, normalized.get("recommended"),
    )
    normalized["recommended"] = recommended
    asr_recommended = (normalized.get("asr") or {}).get("recommended") or {}
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
        config.setdefault("host_a_voice_mode", "preset")
        config.setdefault("host_b_voice_mode", "preset")
        config.setdefault("host_a", "Vivian")
        config.setdefault("host_b", "Dylan")
        config.setdefault("host_a_en", "Ryan")
        config.setdefault("host_b_en", "Aiden")
        config.setdefault("host_a_instruct", "自然、沉稳、有叙事感的知识播客主持人口吻，语速适中。")
        config.setdefault("host_b_instruct", "敏锐、亲切、善于追问和澄清的知识播客主持人口吻，语速适中。")
        config.setdefault("allow_device_fallback", True)
        config.setdefault("asr_auto_select", True)
        config.setdefault("asr_allow_device_fallback", True)
        config.setdefault("cleanup_remote_jobs", True)
        config.setdefault("podcast_sequence_tts", True)
        config["auto_select"] = auto_select
        model = provider.get("model") or ""
        if auto_select and recommended:
            model = recommended["model"]
            config["compute_device"] = recommended["compute_device"]
        if config["asr_auto_select"] and asr_recommended.get("model"):
            config["asr_model"] = asr_recommended["model"]
            config["asr_compute_device"] = asr_recommended.get("compute_device")
        DB.execute(
            "UPDATE provider_profiles SET model=?,capabilities_json=?,config_json=?,updated_at=? WHERE id=?",
            (model, json_dump(normalized), json_dump(config), utc_now(), provider_id),
        )
    return {"ok": True, **normalized, "auto_select": auto_select}


async def probe_tts_provider(provider_id: str, *, apply_defaults: bool = False) -> dict[str, Any]:
    """Deprecated compatibility alias for callers that still use the old name."""
    return await probe_audio_provider(provider_id, apply_defaults=apply_defaults)


async def _synthesize_sandevistan(
    provider: dict[str, Any],
    text: str,
    voice: str | None,
    output: Path,
    *,
    language: str,
    model: str,
    compute_device: str,
    voice_mode: str,
    voiceprint_sample_id: str | None,
    instruct: str | None,
    idempotency_key: str,
    cancel_check: Callable[[], bool] | None,
) -> Path:
    config = provider.get("config", {})
    headers = {"Authorization": f"Bearer {provider['api_key']}"} if provider["api_key"] else {}
    headers["Idempotency-Key"] = idempotency_key
    data = {
        "text": text,
        "model": model,
        "language": language,
        "voice_mode": voice_mode,
        "response_format": "wav",
        "compute_device": compute_device,
    }
    if voice_mode == "voiceprint":
        if not voiceprint_sample_id:
            raise ProviderError("声纹克隆缺少可用样本，请刷新 AUDIO Provider 配置")
        data["voiceprint_sample_id"] = voiceprint_sample_id
    elif voice:
        data["speaker"] = voice
    else:
        raise ProviderError("预置音色不能为空")
    if instruct and voice_mode == "preset":
        data["instruct"] = instruct
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(f"{provider['base_url'].rstrip('/')}/api/v1/tts/jobs", data=data, headers=headers)
        if not response.is_success:
            raise _provider_response_error(response)
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
            if not poll.is_success:
                raise _provider_response_error(poll)
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
                failure = ProviderError(
                    str(state.get("error_message") or state.get("error") or "TTS job failed"),
                    code=str(state.get("error_code") or "tts_failed"),
                )
                if config.get("cleanup_remote_jobs", True):
                    try:
                        await client.delete(
                            f"{provider['base_url'].rstrip('/')}/api/v1/jobs/{job_id}",
                            params={"purge": "true"},
                            headers=headers,
                        )
                    except Exception:
                        pass
                raise failure
            await asyncio.sleep(max(0.5, min(float(state.get("poll_after_seconds") or 1), 5)))
    raise ProviderError("TTS job timed out")


async def synthesize(
    text: str,
    voice: str | None,
    output: Path,
    *,
    language: str = "Chinese",
    cancel_check: Callable[[], bool] | None = None,
    model: str | None = None,
    compute_device: str | None = None,
    voice_mode: str = "preset",
    voiceprint_sample_id: str | None = None,
    instruct: str | None = None,
    idempotency_key: str | None = None,
    execution: dict[str, Any] | None = None,
    provider: dict[str, Any] | None = None,
) -> Path:
    provider = provider or active_provider("audio")
    if not provider:
        raise ProviderError("AUDIO provider is not configured")
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {provider['api_key']}"} if provider["api_key"] else {}
    if provider["kind"] in {"sandevistan_audio", "sandevistan_tts"}:
        config = provider.get("config", {})
        selected_model = model or provider["model"]
        selected_device = compute_device or config.get("compute_device", "gpu")
        key = idempotency_key or str(uuid.uuid4())
        try:
            result = await _synthesize_sandevistan(
                provider,
                text,
                voice,
                output,
                language=language,
                model=selected_model,
                compute_device=selected_device,
                voice_mode=voice_mode,
                voiceprint_sample_id=voiceprint_sample_id,
                instruct=instruct,
                idempotency_key=key,
                cancel_check=cancel_check,
            )
            if execution is not None:
                execution.update({"compute_device": selected_device, "fallback_used": False})
            return result
        except (ProviderError, httpx.HTTPError) as exc:
            if selected_device != "gpu" or not config.get("allow_device_fallback", True) or not _device_failure(exc):
                raise
            result = await _synthesize_sandevistan(
                provider,
                text,
                voice,
                output,
                language=language,
                model=selected_model,
                compute_device="cpu",
                voice_mode=voice_mode,
                voiceprint_sample_id=voiceprint_sample_id,
                instruct=instruct,
                idempotency_key=(key + "-cpu")[:128],
                cancel_check=cancel_check,
            )
            if execution is not None:
                execution.update({"compute_device": "cpu", "fallback_used": True, "fallback_reason": str(exc)[:300]})
            return result
    raise ProviderError(f"Provider {provider['kind']} cannot serve TTS")


async def _synthesize_sequence_sandevistan(
    provider: dict[str, Any],
    items: list[dict[str, Any]],
    outputs: dict[str, Path],
    *,
    language: str,
    model: str,
    compute_device: str,
    voice_mode: str,
    idempotency_key: str,
    cancel_check: Callable[[], bool] | None,
    execution: dict[str, Any] | None = None,
) -> dict[str, Path]:
    headers = _headers(provider)
    headers["Idempotency-Key"] = idempotency_key
    request_items: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item.get("id") or "")
        if item_id not in outputs:
            raise ProviderError("批量 TTS 输出映射不完整", code="sequence_contract_error")
        request_item = {"id": item_id, "text": str(item.get("text") or "")}
        if voice_mode == "voiceprint":
            sample_id = str(item.get("voiceprint_sample_id") or "")
            if not sample_id:
                raise ProviderError("声纹克隆缺少可用样本，请刷新 AUDIO Provider 配置")
            request_item["voiceprint_sample_id"] = sample_id
        else:
            speaker = str(item.get("speaker") or "")
            if not speaker:
                raise ProviderError("预置音色不能为空")
            request_item["speaker"] = speaker
            if item.get("instruct"):
                request_item["instruct"] = str(item["instruct"])
        request_items.append(request_item)
    data = {
        "model": model,
        "language": language,
        "voice_mode": voice_mode,
        "compute_device": compute_device,
        "items": request_items,
    }
    job_id = ""
    config = provider.get("config") or {}
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            f"{provider['base_url'].rstrip('/')}/api/v1/tts/sequence-jobs", json=data, headers=headers,
        )
        if not response.is_success:
            error = _provider_response_error(response)
            if response.status_code in {404, 405, 501}:
                error.code = "tts_sequence_unsupported"
            raise error
        submission = response.json()
        job_id = str(submission.get("id") or submission.get("job_id") or "")
        if not job_id:
            raise ProviderError("批量 TTS Provider 未返回任务 ID", code="sequence_contract_error")
        status_url = submission.get("status_url") or f"/api/v1/jobs/{job_id}"
        try:
            for _ in range(7200):
                if cancel_check and cancel_check():
                    try:
                        await client.post(f"{provider['base_url'].rstrip('/')}/api/v1/jobs/{job_id}/cancel", headers=headers)
                    finally:
                        raise ProviderError("任务已取消", code="cancelled")
                poll_url = status_url if str(status_url).startswith("http") else provider["base_url"].rstrip("/") + str(status_url)
                poll = await client.get(poll_url, headers=headers)
                if not poll.is_success:
                    raise _provider_response_error(poll)
                state = poll.json()
                job_state = state.get("state") or state.get("status")
                if job_state in {"completed", "succeeded", "done"}:
                    result = state.get("result") or {}
                    sequence = result.get("sequence") or {}
                    result_items = sequence.get("items") or []
                    artifact_names = {
                        str(item.get("id") or ""): str(item.get("artifact_name") or "")
                        for item in result_items if isinstance(item, dict)
                    }
                    expected_ids = [str(item["id"]) for item in request_items]
                    if list(artifact_names) != expected_ids or any(not artifact_names[value] for value in expected_ids):
                        raise ProviderError("批量 TTS 返回的条目顺序或音频映射无效", code="sequence_contract_error")
                    if execution is not None:
                        acceleration = result.get("acceleration") or {}
                        execution.update({
                            "sequence_item_count": len(result_items),
                            "provider_acceleration": acceleration,
                            "generation_batch_size": int(
                                ((acceleration.get("stage_batch_sizes") or {}).get("generation") or 1)
                            ),
                            "oom_fallbacks": list(acceleration.get("oom_fallbacks") or []),
                        })
                    for item_id in expected_ids:
                        name = artifact_names[item_id]
                        url = f"/api/v1/jobs/{job_id}/artifacts/{quote(name, safe='')}"
                        audio = await client.get(provider["base_url"].rstrip("/") + url, headers=headers)
                        if not audio.is_success:
                            raise _provider_response_error(audio)
                        if not audio.headers.get("content-type", "").startswith("audio/") or not audio.content.startswith(b"RIFF"):
                            raise ProviderError("批量 TTS Provider 返回了无效 WAV 音频", code="sequence_contract_error")
                        destination = outputs[item_id]
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_suffix(destination.suffix + ".tmp")
                        temporary.write_bytes(audio.content)
                        temporary.replace(destination)
                    return {item_id: outputs[item_id] for item_id in expected_ids}
                if job_state in {"failed", "error", "cancelled"}:
                    raise ProviderError(
                        str(state.get("error_message") or state.get("error") or "批量 TTS 任务失败"),
                        code=str(state.get("error_code") or "tts_failed"),
                    )
                await asyncio.sleep(max(0.5, min(float(state.get("poll_after_seconds") or 1), 5)))
        finally:
            if job_id and config.get("cleanup_remote_jobs", True):
                try:
                    await client.delete(
                        f"{provider['base_url'].rstrip('/')}/api/v1/jobs/{job_id}", params={"purge": "true"}, headers=headers,
                    )
                except Exception:
                    pass
    raise ProviderError(
        f"批量 TTS 任务超时（已等待 {round(time.perf_counter() - started)} 秒）", code="timeout",
    )


async def synthesize_sequence(
    items: list[dict[str, Any]],
    outputs: dict[str, Path],
    *,
    language: str = "Chinese",
    cancel_check: Callable[[], bool] | None = None,
    model: str | None = None,
    compute_device: str | None = None,
    voice_mode: str = "preset",
    idempotency_key: str | None = None,
    execution: dict[str, Any] | None = None,
    provider: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Synthesize an ordered homogeneous sequence, retaining same-model GPU-to-CPU fallback."""
    provider = provider or active_provider("audio")
    if not provider:
        raise ProviderError("AUDIO provider is not configured")
    sequence = (provider.get("capabilities") or {}).get("sequence_jobs") or {}
    if not (
        provider.get("kind") in {"sandevistan_audio", "sandevistan_tts"}
        and sequence.get("supported")
        and int(sequence.get("contract_version") or 0) == 1
        and provider.get("config", {}).get("podcast_sequence_tts", True)
    ):
        raise ProviderError("AUDIO Provider 不支持批量 TTS", code="tts_sequence_unsupported")
    if not items:
        return {}
    config = provider.get("config") or {}
    selected_model = str(model or provider.get("model") or "")
    selected_device = str(compute_device or config.get("compute_device") or "gpu")
    key = idempotency_key or str(uuid.uuid4())
    started = time.perf_counter()
    try:
        result = await _synthesize_sequence_sandevistan(
            provider, items, outputs, language=language, model=selected_model,
            compute_device=selected_device, voice_mode=voice_mode, idempotency_key=key,
            cancel_check=cancel_check, execution=execution,
        )
        if execution is not None:
            execution.update({
                "compute_device": selected_device, "fallback_used": False,
                "provider_processing_seconds": round(time.perf_counter() - started, 3),
            })
        return result
    except (ProviderError, httpx.HTTPError) as exc:
        if selected_device != "gpu" or not config.get("allow_device_fallback", True) or not _device_failure(exc):
            raise
        result = await _synthesize_sequence_sandevistan(
            provider, items, outputs, language=language, model=selected_model,
            compute_device="cpu", voice_mode=voice_mode, idempotency_key=(key + "-cpu")[:128],
            cancel_check=cancel_check, execution=execution,
        )
        if execution is not None:
            execution.update({
                "compute_device": "cpu", "fallback_used": True, "fallback_reason": str(exc)[:300],
                "provider_processing_seconds": round(time.perf_counter() - started, 3),
            })
        return result


def _device_failure(error: BaseException) -> bool:
    code = str(getattr(error, "code", "") or "").lower()
    message = str(error).lower()
    return code in {"gpu_unavailable", "insufficient_gpu_memory", "cuda_error", "gpu_out_of_memory", "worker_oom"} or any(
        marker in message for marker in ("cuda", "out of memory", "gpu unavailable", "gpu is unavailable", "显存不足")
    )


async def _transcribe_sandevistan_once(
    provider: dict[str, Any],
    path: Path,
    *,
    language: str,
    model: str,
    compute_device: str,
    idempotency_key: str,
    cancel_check: Callable[[], bool] | None,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {provider['api_key']}"} if provider["api_key"] else {}
    headers["Idempotency-Key"] = idempotency_key
    data = {
        "model": model,
        "language": language,
        "speaker_count": "2",
        "diarize": "true",
        "align": "true",
        "export_formats": "json",
        "compute_device": compute_device,
        "use_voiceprint_library": "false",
        "accelerate_single_task": "true",
    }
    job_id: str | None = None
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            with path.open("rb") as audio:
                response = await client.post(
                    f"{provider['base_url'].rstrip('/')}/api/v1/asr/jobs",
                    data=data,
                    files={"file": (path.name, audio, "audio/mp4" if path.suffix.lower() == ".m4a" else "audio/wav")},
                    headers=headers,
                )
            if not response.is_success:
                raise _provider_response_error(response)
            submitted = response.json()
            job_id = str(submitted.get("id") or submitted.get("job_id") or "")
            if not job_id:
                raise ProviderError("ASR provider did not return a job id")
            status_url = submitted.get("status_url") or f"/api/v1/jobs/{job_id}"
            for _ in range(7200):
                if cancel_check and cancel_check():
                    await client.post(f"{provider['base_url'].rstrip('/')}/api/v1/jobs/{job_id}/cancel", headers=headers)
                    raise ProviderError("任务已取消", code="cancelled")
                poll_url = status_url if str(status_url).startswith("http") else provider["base_url"].rstrip("/") + str(status_url)
                poll = await client.get(poll_url, headers=headers)
                if not poll.is_success:
                    raise _provider_response_error(poll)
                state = poll.json()
                job_state = str(state.get("state") or state.get("status") or "")
                if job_state in {"completed", "succeeded", "done"}:
                    result_url = state.get("result_url") or f"/api/v1/jobs/{job_id}/result"
                    result_response = await client.get(
                        result_url if str(result_url).startswith("http") else provider["base_url"].rstrip("/") + str(result_url),
                        headers=headers,
                    )
                    if not result_response.is_success:
                        raise _provider_response_error(result_response)
                    result = result_response.json()
                    return result.get("result") if isinstance(result.get("result"), dict) else result
                if job_state in {"failed", "error", "cancelled"}:
                    raise ProviderError(
                        str(state.get("error_message") or state.get("error") or "ASR job failed"),
                        code=str(state.get("error_code") or "asr_failed"),
                    )
                await asyncio.sleep(max(0.5, min(float(state.get("poll_after_seconds") or 1), 5)))
            raise ProviderError("ASR job timed out", code="timeout")
        finally:
            if job_id:
                try:
                    await client.delete(
                        f"{provider['base_url'].rstrip('/')}/api/v1/jobs/{job_id}", params={"purge": "true"}, headers=headers
                    )
                except Exception:
                    pass


async def _transcribe_with_provider(
    provider: dict[str, Any],
    path: Path,
    *,
    language: str,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    model, device, allow_fallback = _asr_execution(provider)
    key = idempotency_key or str(uuid.uuid4())
    try:
        result = await _transcribe_sandevistan_once(
            provider, path, language=language, model=model, compute_device=device, idempotency_key=key,
            cancel_check=cancel_check,
        )
        return {**result, "model": result.get("model") or model, "compute_device": device, "fallback_used": False}
    except Exception as exc:
        if device != "gpu" or not allow_fallback or not _device_failure(exc):
            raise
        result = await _transcribe_sandevistan_once(
            provider, path, language=language, model=model, compute_device="cpu", idempotency_key=(key + "-cpu")[:128],
            cancel_check=cancel_check,
        )
        return {
            **result,
            "model": result.get("model") or model,
            "compute_device": "cpu",
            "fallback_used": True,
            "fallback_reason": str(exc)[:300],
        }


async def transcribe_audio(
    path: Path,
    *,
    language: str,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
    provider: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the supplied task provider, or resolve the active AUDIO for standalone calls."""
    provider = provider if provider is not None else active_provider("audio")
    ready, message = audio_provider_readiness(provider)
    if not ready or not provider:
        raise ProviderError(message, code="asr_unsupported")
    return await _transcribe_with_provider(
        provider,
        path,
        language=language,
        cancel_check=cancel_check,
        idempotency_key=idempotency_key,
    )
