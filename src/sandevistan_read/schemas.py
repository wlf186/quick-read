from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator

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


ProviderRole = Literal["main", "vlm", "tts"]
ProviderKind = Literal["ollama", "openai", "sandevistan_tts", "openai_tts"]
ProviderValidationMode = Literal["catalog", "deep"]
StudyDifficulty = Literal["easy", "medium", "hard", "mixed"]


def _validate_provider_pair(role: str, kind: str) -> None:
    allowed = {"main": {"ollama", "openai"}, "vlm": {"ollama", "openai"}, "tts": {"sandevistan_tts", "openai_tts"}}
    if kind not in allowed.get(role, set()):
        raise ValueError(f"{role.upper()} 角色不支持 {kind} Provider")


def _validate_provider_config(config: dict[str, Any]) -> None:
    validate_token_overrides(config)
    resolve_temperature(config, 0.0)
    tier = config.get("study_generation_tier", "auto")
    if tier not in {"auto", "lite", "full"}:
        raise ValueError("学习生成档位必须是 auto、lite 或 full")


class ProviderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    role: ProviderRole
    kind: ProviderKind
    base_url: str = Field(min_length=1, max_length=2048)
    model: str = Field(default="", max_length=240)
    api_key: str = ""
    capabilities: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    active: bool = True
    validation_mode: ProviderValidationMode = "catalog"

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
    config: dict[str, Any] | None = None
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
    config: dict[str, Any] = Field(default_factory=dict)
    mode: ProviderValidationMode = "catalog"

    @model_validator(mode="after")
    def validate_provider_pair(self):
        _validate_provider_pair(self.role, self.kind)
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
