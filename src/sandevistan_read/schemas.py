from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class NotebookCreate(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=1000)


class NotebookUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)


class SourceSelection(BaseModel):
    selected: bool


class ProviderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    role: Literal["main", "vlm", "tts"]
    kind: Literal["ollama", "openai", "sandevistan_tts", "openai_tts"]
    base_url: str
    model: str = ""
    api_key: str = ""
    capabilities: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    active: bool = True


class ProviderUpdate(BaseModel):
    name: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    capabilities: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    active: bool | None = None


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
    difficulty: Literal["easy", "medium", "hard", "mixed"] = "mixed"
    language: Literal["auto", "zh-CN", "en"] = "auto"


class FlashcardRequest(BaseModel):
    source_ids: list[str] | None = None
    count: int = Field(default=20, ge=1, le=50)
    language: Literal["auto", "zh-CN", "en"] = "auto"


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
    answers: dict[str, int]


class FlashcardReview(BaseModel):
    card_id: str
    rating: Literal["again", "hard", "mastered"]


class LoginRequest(BaseModel):
    access_key: str
