from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator

from .api_docs import PROVIDER_CONFIG_DOCS
from .context_budget import resolve_temperature, validate_token_overrides


class NotebookCreate(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=1000)


class NotebookUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)


class NotebookBatchDelete(BaseModel):
    notebook_ids: list[str] = Field(min_length=1, max_length=100)


class SourceSelection(BaseModel):
    selected: bool


class ImageProcessingPolicy(BaseModel):
    mode: Literal["process", "off"] = "process"
    processors: list[Literal["vlm", "main", "ocr"]] = Field(default_factory=lambda: ["vlm", "main", "ocr"])

    @model_validator(mode="after")
    def validate_processors(self):
        if len(self.processors) != len(set(self.processors)):
            raise ValueError("图片处理步骤不能重复")
        if self.mode == "process" and not self.processors:
            raise ValueError("启用图片处理时至少选择一个处理步骤")
        return self


class ProviderRoleUpdate(BaseModel):
    enabled: bool | None = None
    selected_provider_id: str | None = None
    validation_mode: Literal["catalog", "deep"] = "catalog"


ProviderRole = Literal["main", "vlm", "audio", "tts_only", "tts"]
ProviderKind = Literal["ollama", "openai", "sandevistan_audio", "openai_tts", "sandevistan_tts"]
ProviderValidationMode = Literal["catalog", "deep"]
StudyDifficulty = Literal["easy", "medium", "hard", "mixed"]


def _normalize_provider_pair(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    role, kind = normalized.get("role"), normalized.get("kind")
    if role == "tts" and kind == "sandevistan_tts":
        normalized.update({"role": "audio", "kind": "sandevistan_audio"})
    elif role == "tts" and kind == "openai_tts":
        normalized["role"] = "tts_only"
    return normalized


def _validate_provider_pair(role: str, kind: str, *, allow_tts_only: bool = False) -> None:
    allowed = {"main": {"ollama", "openai"}, "vlm": {"ollama", "openai"}, "audio": {"sandevistan_audio"}}
    if allow_tts_only:
        allowed["tts_only"] = {"openai_tts"}
    if kind not in allowed.get(role, set()):
        raise ValueError(f"{role.upper()} 角色不支持 {kind} Provider")


def _validate_provider_config(config: dict[str, Any]) -> None:
    validate_token_overrides(config)
    resolve_temperature(config, 0.0)
    tier = config.get("study_generation_tier", "auto")
    if tier not in {"auto", "lite", "full"}:
        raise ValueError("学习生成档位必须是 auto、lite 或 full")
    for field in ("auto_select", "allow_device_fallback", "podcast_sequence_tts", "asr_auto_select", "asr_allow_device_fallback"):
        if field in config and not isinstance(config[field], bool):
            raise ValueError(f"{field} 必须是布尔值")
    for field, maximum in (
        ("asr_model", 240),
        ("asr_compute_device", 64),
        ("host_a_voiceprint_person_id", 240),
        ("host_a_voiceprint_sample_id", 240),
        ("host_b_voiceprint_person_id", 240),
        ("host_b_voiceprint_sample_id", 240),
    ):
        if field in config and (not isinstance(config[field], str) or len(config[field]) > maximum):
            raise ValueError(f"{field} 必须是长度不超过 {maximum} 的字符串")
    for host in ("host_a", "host_b"):
        mode = config.get(f"{host}_voice_mode", "preset")
        if mode not in {"preset", "voiceprint"}:
            raise ValueError(f"{host}_voice_mode 必须是 preset 或 voiceprint")
    if (
        config.get("host_a_voice_mode", "preset") == "preset"
        and config.get("host_b_voice_mode", "preset") == "preset"
        and str(config.get("host_a") or "").strip()
        and str(config.get("host_a") or "").strip() == str(config.get("host_b") or "").strip()
    ):
        raise ValueError("Host A 与 Host B 不能使用同一个预置音色")
    if (
        config.get("host_a_voice_mode") == "voiceprint"
        and config.get("host_b_voice_mode") == "voiceprint"
        and str(config.get("host_a_voiceprint_person_id") or "").strip()
        and str(config.get("host_a_voiceprint_person_id") or "").strip()
        == str(config.get("host_b_voiceprint_person_id") or "").strip()
    ):
        raise ValueError("Host A 与 Host B 不能使用同一个声纹人员")


class ProviderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    role: ProviderRole
    kind: ProviderKind
    base_url: str = Field(min_length=1, max_length=2048)
    model: str = Field(default="", max_length=240)
    api_key: str = ""
    capabilities: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict, json_schema_extra=PROVIDER_CONFIG_DOCS)
    active: bool = True
    validation_mode: ProviderValidationMode = "catalog"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_pair(cls, value: Any):
        return _normalize_provider_pair(value)

    @model_validator(mode="after")
    def validate_provider_pair(self):
        _validate_provider_pair(self.role, self.kind)
        _validate_provider_config(self.config)
        return self


class ProviderUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    base_url: str | None = Field(default=None, min_length=1, max_length=2048)
    model: str | None = Field(default=None, max_length=240)
    api_key: str | None = None
    capabilities: dict[str, Any] | None = None
    config: dict[str, Any] | None = Field(default=None, json_schema_extra=PROVIDER_CONFIG_DOCS)
    active: bool | None = None
    validation_mode: ProviderValidationMode = "catalog"

    @model_validator(mode="after")
    def validate_context_overrides(self):
        if self.config is not None:
            _validate_provider_config(self.config)
        return self


class ProviderInspectionRequest(BaseModel):
    provider_id: str | None = None
    role: ProviderRole
    kind: ProviderKind
    base_url: str = Field(min_length=1, max_length=2048)
    model: str = Field(default="", max_length=240)
    api_key: str | None = None
    config: dict[str, Any] = Field(default_factory=dict, json_schema_extra=PROVIDER_CONFIG_DOCS)
    mode: ProviderValidationMode = "catalog"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_pair(cls, value: Any):
        return _normalize_provider_pair(value)

    @model_validator(mode="after")
    def validate_provider_pair(self):
        _validate_provider_pair(self.role, self.kind, allow_tts_only=True)
        _validate_provider_config(self.config)
        return self


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None
    source_ids: list[str] | None = None
    language: Literal["auto", "zh-CN", "en"] = "auto"


class SummaryRequest(BaseModel):
    source_ids: list[str] | None = None
    language: Literal["auto", "zh-CN", "en"] = "auto"


class QuizRequest(BaseModel):
    source_ids: list[str] | None = None
    count: int = Field(default=10, ge=1, le=30)
    difficulty: StudyDifficulty = "mixed"
    language: Literal["auto", "zh-CN", "en"] = "auto"
    custom_prompt: str = Field(default="", max_length=1000)


class FlashcardRequest(BaseModel):
    source_ids: list[str] | None = None
    count: int = Field(default=20, ge=1, le=50)
    difficulty: StudyDifficulty = "mixed"
    language: Literal["auto", "zh-CN", "en"] = "auto"
    custom_prompt: str = Field(default="", max_length=1000)


class PodcastRequest(BaseModel):
    source_ids: list[str] | None = None
    duration_mode: Literal["auto", "fixed"] = "auto"
    minutes: Literal[5, 10, 20, 30] | None = None
    language: Literal["auto", "zh-CN", "en"] = "zh-CN"
    focus: str = Field(default="", max_length=1000)
    host_a: str | None = None
    host_b: str | None = None

    @model_validator(mode="after")
    def normalize_duration(self) -> "PodcastRequest":
        if self.minutes is not None and "duration_mode" not in self.model_fields_set:
            self.duration_mode = "fixed"
        if self.duration_mode == "fixed" and self.minutes is None:
            raise ValueError("固定时长模式必须提供 minutes")
        return self


class QuizSubmission(BaseModel):
    answers: dict[str, int] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def validate_answers(self) -> "QuizSubmission":
        if any(answer not in range(4) for answer in self.answers.values()):
            raise ValueError("Quiz 选项必须是 0 到 3")
        return self


class StudySessionCreate(BaseModel):
    mode: Literal["all", "missed", "due", "same"] = "all"
    shuffle: bool = False


class QuizAnswer(BaseModel):
    item_id: str = Field(min_length=1, max_length=120)
    option_index: int = Field(ge=0, le=3)


class FlashcardSessionReview(BaseModel):
    item_id: str = Field(min_length=1, max_length=120)
    rating: Literal["again", "hard", "good", "easy"]


class FlashcardReview(BaseModel):
    card_id: str
    rating: Literal["again", "hard", "good", "easy", "mastered"]
    session_id: str | None = None


class LoginRequest(BaseModel):
    access_key: str
