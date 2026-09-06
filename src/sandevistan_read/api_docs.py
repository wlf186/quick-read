"""OpenAPI metadata only: these schemas never validate or filter runtime payloads."""

from typing import Any


def _object(properties: dict[str, Any], description: str = "") -> dict[str, Any]:
    return {"type": "object", "additionalProperties": True, "properties": properties, "description": description}


def _array(items: dict[str, Any], description: str = "") -> dict[str, Any]:
    return {"type": "array", "items": items, "description": description}


PROVIDER_CONFIG_DOCS = {
    "description": (
        "Provider 扩展配置，未列出的字段仍保留。以下 AUDIO 开关若提供，必须是 JSON 布尔值，"
        "字符串、数字和 null 不可替代。说明中的缺省行为不代表保存时自动补齐所有键。"
    ),
    "properties": {
        "auto_select": {"type": "boolean", "description": "TTS 自动推荐开关；显式 false 保留人工模型/设备。缺省由已有模型及旧配置迁移规则决定。"},
        "compute_device": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "TTS 请求设备，例如 gpu/cpu；尚未解析出设备时可为 null，执行时仍可能按允许的策略回退。"},
        "allow_device_fallback": {"type": "boolean", "description": "TTS 设备故障时允许同模型 GPU→CPU 回退，缺省按 true 处理。"},
        "podcast_sequence_tts": {
            "type": "boolean",
            "description": (
                "缺省按 true 处理；false 同时关闭 Podcast 批量合成与脚本/TTS 重叠。"
                "开启仍需兼容 sequence_jobs 能力，重叠还受 MAIN/AUDIO hostname 判定限制。"
                "此项保存于 Provider config，不是 config.toml 或单次 Podcast 请求字段。"
            ),
        },
        "asr_auto_select": {"type": "boolean", "description": "使用 AUDIO 服务推荐的 ASR 模型/设备，缺省按 true 处理。"},
        "asr_model": {"type": "string", "maxLength": 240, "description": "ASR 验收模型 ID。"},
        "asr_compute_device": {"type": "string", "maxLength": 64, "description": "ASR 请求设备，与 TTS 分别配置。"},
        "asr_allow_device_fallback": {"type": "boolean", "description": "ASR 设备故障时允许同模型 GPU→CPU 回退，缺省按 true 处理。"},
    },
    "examples": [{"auto_select": True, "podcast_sequence_tts": True, "allow_device_fallback": True, "asr_auto_select": True}],
}
PROVIDER_CONFIG_SCHEMA = {"type": "object", "additionalProperties": True, **PROVIDER_CONFIG_DOCS}

SEQUENCE_JOBS_SCHEMA = _object({
    "supported": {"type": "boolean", "description": "经 quick-read 协议校验后的批量能力；旧服务或协议不兼容时为 false。"},
    "contract_version": {"type": "integer", "description": "当前支持 1；缺失时规范化为 0。"},
    "endpoint": {"type": "string", "description": "兼容值为 /api/v1/tts/sequence-jobs，这是 AUDIO 上游接口。"},
    "voice_modes": _array({"type": "string"}, "上游声明的音色模式，例如 preset、voiceprint。"),
    "artifact_mode": {"type": "string", "description": "兼容值为 per_item，每轮保留独立音频。"},
    "format": {"type": "string", "description": "兼容值为 wav。"},
    "max_items": {"type": "integer", "description": "规范化后的每批条数上限，限制在 1–100。"},
    "max_total_chars": {"type": "integer", "description": "上游声明的每批总字符预算，缺失时保守规范化为 1。"},
})

MODEL_SCHEMA = _object({
    "id": {"type": "string"},
    "name": {"type": "string"},
    "installed": {"type": "boolean"},
    "default": {"type": "boolean", "description": "AUDIO 服务报告的默认模型标记，不等同于 quick-read 最终推荐。"},
    "voice_modes": _array({"type": "string"}),
    "devices": _array(_object({"id": {"type": "string"}, "available": {"type": "boolean"}})),
    "controls": _object({}, "模型公开的控制能力；instruction_voice_modes 表示支持表达指令的音色模式。"),
    "checkpoints": _array(_object({
        "variant": {"type": "string", "description": "checkpoint 变体，例如 base、custom_voice。"},
        "revision": {"type": "string", "description": "服务报告的固定 checkpoint revision，用于默认资格匹配和音频复用哈希。"},
    }), "AUDIO 模型 checkpoint 清单；服务未公开时可能为空。"),
})

RECOMMENDATION_SCHEMA = {"anyOf": [_object({
    "model": {"type": "string"},
    "compute_device": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "reason": {
        "type": "string",
        "description": (
            "AUDIO 推荐原因：service_default 表示服务默认组合匹配 checkpoint/device 资格表；"
            "installed_fallback 表示按已安装模型质量排序回退，不保证均在资格表内；"
            "preserve_custom_instructions 表示存在支持自定义表达指令的已安装备选。"
        ),
        "examples": ["service_default", "installed_fallback", "preserve_custom_instructions"],
    },
}), {"type": "null"}]}

CAPABILITIES_SCHEMA = _object({
    "default_model": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "AUDIO 服务报告的默认 TTS 模型。"},
    "sequence_jobs": SEQUENCE_JOBS_SCHEMA,
    "models": _array(MODEL_SCHEMA, "已保存的 AUDIO 能力包含模型快照；inspect 的模型清单位于顶层 models。"),
    "recommended": RECOMMENDATION_SCHEMA,
    "asr": _object({}, "ASR 模型、设备、推荐与验收能力，与 TTS 分别解析。"),
}, "按 Provider 角色和协议变化的能力对象，以下主要列出 AUDIO 字段。")

INSPECTION_SCHEMA = _object({
    "status": {"type": "string", "enum": ["passed", "warning", "failed"]},
    "connection_ok": {"type": "boolean"},
    "activation_eligible": {"type": "boolean", "description": "能否启用该角色；HTTP 200 本身不代表验证通过。"},
    "latency_ms": {"type": "number"},
    "catalog_supported": {"type": "boolean"},
    "models": _array(MODEL_SCHEMA),
    "capabilities": CAPABILITIES_SCHEMA,
    "recommended": RECOMMENDATION_SCHEMA,
    "resolved_audio_config": PROVIDER_CONFIG_SCHEMA,
    "voiceprint_library": _object({}, "脱敏的人员与可用样本元数据，不包含原始声纹音频。"),
    "warning": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "error": {"anyOf": [_object({
        "code": {"type": "string"}, "stage": {"type": "string"},
        "message": {"type": "string"}, "hint": {"type": "string"},
        "upstream_status": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    }), {"type": "null"}]},
}, "能力检查结果；字段可能随角色、服务能力或失败阶段缺省。")

PROVIDER_SCHEMA = _object({
    "id": {"type": "string"}, "name": {"type": "string"},
    "role": {"type": "string"}, "kind": {"type": "string"},
    "base_url": {"type": "string"}, "model": {"type": "string"},
    "active": {"type": "integer"}, "selected": {"type": "integer"},
    "has_api_key": {"type": "boolean"},
    "config": PROVIDER_CONFIG_SCHEMA, "capabilities": CAPABILITIES_SCHEMA,
}, "保存的 Provider；凭据仅以 has_api_key 表示，响应保留其他扩展元数据。")

PERFORMANCE_SCHEMA = _object({
    "script_seconds": {"type": "number", "description": "本次脚本阶段墙钟秒数，复用文字稿时接近零。"},
    "tts_seconds": {"type": "number", "description": "本次常规合成/修复耗时加提前合成活动区间，可能含等待；不是纯 GPU 计算时间。"},
    "total_seconds": {"type": "number", "description": "本次 Podcast 执行至成品验收后的墙钟秒数，包含 FFmpeg 和 ASR。"},
    "overlap_enabled": {"type": "boolean", "description": "是否启用提前合成；不表示一定产生了可复用音频或重叠收益。"},
    "overlap_seconds": {"type": "number", "description": "脚本阶段与提前合成活动区间的交集秒数。"},
    "serial_estimate_seconds": {"type": "number", "description": "total_seconds + overlap_seconds，仅为推算串行耗时。"},
    "overlap_gain_ratio": {"type": "number", "description": "overlap_seconds / serial_estimate_seconds 的估计节省比例，不是独立串行基准实测。"},
}, "Podcast 执行统计；旧产物或其他类型产物可以没有此对象。阶段耗时可能重叠，不能直接相加。")

TTS_EXECUTION_SCHEMA = _object({
    "name": {"type": "string"}, "kind": {"type": "string"}, "model": {"type": "string"},
    "requested_device": {"type": "string"}, "compute_device": {"type": "string"},
    "fallback_used": {"type": "boolean"}, "fallback_reason": {"type": "string"},
    "sequence_supported": {"type": "boolean"},
    "sequence_enabled": {"type": "boolean", "description": "由配置与能力决定是否尝试批量；实际执行仍可能回退逐轮。"},
    "sequence_jobs": {"type": "integer", "description": "常规/修复阶段成功批量任务数，不含提前合成任务。"},
    "single_jobs": {"type": "integer", "description": "常规/修复阶段的逐轮合成调用数。"},
    "batch_sizes": _array({"type": "integer"}, "每个成功批量任务的轮次数。"),
    "generation_batch_sizes": _array({"type": "integer"}, "上游报告的 generation 阶段批大小，不等于请求轮次数。"),
    "oom_fallbacks": _array(_object({}), "上游显存不足时的批大小调整记录。"),
    "reused_turns": {"type": "integer", "description": "本次命中音频片段缓存的计数。"},
    "sequence_fallback_reason": {"type": "string", "description": "批量失败并回退逐轮时记录的原因，未发生时缺省。"},
    "speculative_jobs": {"type": "integer", "description": "提前合成成功任务数。"},
    "speculative_turns": {"type": "integer", "description": "提前合成成功轮次数。"},
    "speculative_generation_batch_sizes": _array({"type": "integer"}),
    "speculative_oom_fallbacks": _array(_object({})),
    "speculative_reused_turns": {"type": "integer", "description": "最终文字稿与配置哈希仍匹配的提前合成轮次数。"},
    "speculative_reuse_ratio": {"type": "number", "description": "最终复用轮数 / 提前合成的不同轮次索引数，无提前合成时为 0。"},
    "asr": _object({}, "ASR 模型、设备及回退元数据。"),
}, "Podcast TTS 执行记录，统计属于本次运行，部分字段仅在对应路径执行时出现。")

ARTIFACT_SCHEMA = _object({
    "id": {"type": "string"}, "notebook_id": {"type": "string"},
    "type": {"type": "string"}, "title": {"type": "string"},
    "language": {"type": "string"}, "status": {"type": "string"},
    "payload": _object({"performance": PERFORMANCE_SCHEMA, "provider": TTS_EXECUTION_SCHEMA},
                       "按产物类型变化的扩展内容；这里列出 Podcast 执行元数据，其他内容原样保留。"),
    "citations": _array(_object({})),
    "media_url": {"type": "string", "description": "存在媒体时返回。"},
}, "产物公开视图；Quiz 作答前仍隐藏答案、解释与引用。旧 Podcast 不要求存在新增统计字段。")


def json_response(schema: dict[str, Any], description: str) -> dict[int, Any]:
    """Describe an HTTP 200 body without installing a FastAPI response_model."""
    return {200: {"description": description, "content": {"application/json": {"schema": schema}}}}


PROVIDER_CREATE_RESPONSES = json_response(_object({
    "id": {"type": "string"}, "active": {"type": "boolean"},
    "inspection": {"anyOf": [INSPECTION_SCHEMA, {"type": "null"}]},
}), "创建结果；保存为停用状态时 inspection 为 null。")
PROVIDER_UPDATE_RESPONSES = json_response(_object({
    "ok": {"type": "boolean"}, "active": {"type": "boolean"},
}), "更新结果；config 若提供则替换配置对象，请保留需要沿用的键。")
PROVIDER_LIST_RESPONSES = json_response(_array(PROVIDER_SCHEMA), "已保存的 Provider 列表。")
INSPECTION_RESPONSES = json_response(INSPECTION_SCHEMA, "请检查 status、activation_eligible 与 error，不仅检查 HTTP 状态。")
PROVIDER_TEST_RESPONSES = json_response(_object({
    **INSPECTION_SCHEMA["properties"], "ok": {"type": "boolean"},
}), "已保存 Provider 的 catalog 检查结果，不保存配置。")
PROVIDER_PROBE_RESPONSES = json_response(_object({
    **CAPABILITIES_SCHEMA["properties"], "ok": {"type": "boolean"}, "auto_select": {"type": "boolean"},
}), "重新探测并持久化能力；AUDIO 还应用默认配置和自动推荐，MAIN/VLM 返回各自窗口能力。")
ARTIFACT_RESPONSES = json_response(ARTIFACT_SCHEMA, "产物详情及可选的 Podcast 执行统计。")
ARTIFACT_LIST_RESPONSES = json_response(_array(ARTIFACT_SCHEMA), "产物列表；view=summary 时 payload 为 {}，citations 为 []，不返回媒体链接。")
