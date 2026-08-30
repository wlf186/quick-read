from __future__ import annotations

import ipaddress
import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .paths import PATHS


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 20830
    log_level: str = "info"


@dataclass
class SecurityConfig:
    access_key: str = ""


@dataclass
class RuntimeConfig:
    max_upload_mib: int = 512
    minimum_free_mib: int = 2048
    job_poll_seconds: float = 0.5


@dataclass
class ModelConfig:
    embedding: str = "intfloat/multilingual-e5-small"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    offline: bool = True


@dataclass
class ToolConfig:
    ffmpeg: str = ""
    libreoffice: str = ""

    def resolve(self, value: str, project_candidates: tuple[Path, ...], system_candidates: tuple[str, ...]) -> str | None:
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = PATHS.root / path
            return str(path) if path.exists() else None
        for candidate in project_candidates:
            if candidate.exists():
                return str(candidate)
        for candidate in system_candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None

    @property
    def ffmpeg_path(self) -> str | None:
        return self.resolve(
            self.ffmpeg,
            (PATHS.tools / "ffmpeg" / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg"),),
            ("ffmpeg", "ffmpeg.exe"),
        )

    @property
    def libreoffice_path(self) -> str | None:
        return self.resolve(
            self.libreoffice,
            (PATHS.tools / "libreoffice" / "program" / ("soffice.exe" if os.name == "nt" else "soffice"),),
            ("soffice", "libreoffice", "soffice.exe"),
        )

    @staticmethod
    def version(executable: str | None) -> str | None:
        if not executable:
            return None
        for flag in ("-version", "--version"):
            try:
                completed = subprocess.run([executable, flag], capture_output=True, text=True, timeout=8, check=False)
                line = (completed.stdout or completed.stderr).splitlines()
                if completed.returncode == 0 and line:
                    return line[0].strip()
            except OSError:
                return None
        return None

    @staticmethod
    def scope(executable: str | None) -> str:
        if not executable:
            return "missing"
        try:
            Path(executable).resolve().relative_to(PATHS.root)
            return "project"
        except ValueError:
            return "system"


@dataclass
class DevelopmentConfig:
    ollama_url: str = "http://iollama:11434"
    ollama_model: str = "qwen3.5:2b"
    tts_url: str = "http://localhost:20810"


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)

    def validate(self) -> None:
        try:
            is_loopback = ipaddress.ip_address(self.server.host).is_loopback
        except ValueError:
            is_loopback = self.server.host.lower() == "localhost"
        if not is_loopback and not self.security.access_key:
            raise RuntimeError("A security.access_key is required when binding outside localhost")
        if not 1 <= self.server.port <= 65535:
            raise RuntimeError("server.port must be between 1 and 65535")


def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def load_config() -> AppConfig:
    if not PATHS.config.exists():
        example = PATHS.root / "config.example.toml"
        if example.exists():
            PATHS.config.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    data: dict = {}
    if PATHS.config.exists():
        with PATHS.config.open("rb") as handle:
            data = tomllib.load(handle)
    config = AppConfig(
        server=ServerConfig(**_section(data, "server")),
        security=SecurityConfig(**_section(data, "security")),
        runtime=RuntimeConfig(**_section(data, "runtime")),
        models=ModelConfig(**_section(data, "models")),
        tools=ToolConfig(**_section(data, "tools")),
        development=DevelopmentConfig(**_section(data, "development")),
    )
    config.validate()
    return config


CONFIG = load_config()
