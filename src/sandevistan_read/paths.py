from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    tools: Path
    runtime: Path
    data: Path
    models: Path
    cache: Path
    logs: Path
    run: Path
    temp: Path
    downloads: Path
    libreoffice_profiles: Path
    blobs: Path
    renders: Path
    artifacts: Path
    job_work: Path
    backups: Path
    indexes: Path
    secrets: Path
    config: Path
    database: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        override = os.environ.get("SANDEVISTAN_PROJECT_ROOT")
        root = Path(override).resolve() if override else Path(__file__).resolve().parents[2]
        runtime = root / "runtime"
        data = runtime / "data"
        return cls(
            root=root,
            tools=root / ".tools",
            runtime=runtime,
            data=data,
            models=runtime / "models",
            cache=runtime / "cache",
            logs=runtime / "logs",
            run=runtime / "run",
            temp=runtime / "tmp",
            downloads=runtime / "cache" / "downloads",
            libreoffice_profiles=runtime / "tmp" / "libreoffice-profiles",
            blobs=data / "blobs",
            renders=data / "renders",
            artifacts=data / "artifacts",
            job_work=data / "job-work",
            backups=data / "backups",
            indexes=data / "indexes",
            secrets=data / "secrets",
            config=runtime / "config.toml",
            database=data / "sandevistan-read.db",
        )

    def ensure(self) -> None:
        for path in (
            self.runtime,
            self.tools,
            self.data,
            self.models,
            self.cache,
            self.logs,
            self.run,
            self.temp,
            self.downloads,
            self.libreoffice_profiles,
            self.blobs,
            self.renders,
            self.artifacts,
            self.job_work,
            self.backups,
            self.indexes,
            self.secrets,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def install_environment(self) -> None:
        """Keep third-party caches and temporary files inside the project."""
        values = {
            "UV_PROJECT_ENVIRONMENT": str(self.root / ".venv"),
            "UV_CACHE_DIR": str(self.cache / "uv"),
            "COREPACK_HOME": str(self.cache / "corepack"),
            "PNPM_HOME": str(self.root / ".tools" / "pnpm"),
            "HF_HOME": str(self.models / "huggingface"),
            "SENTENCE_TRANSFORMERS_HOME": str(self.models / "sentence-transformers"),
            "TORCH_HOME": str(self.models / "torch"),
            "XDG_CACHE_HOME": str(self.cache),
            "PLAYWRIGHT_BROWSERS_PATH": str(self.cache / "ms-playwright"),
            "TMPDIR": str(self.temp),
            "TEMP": str(self.temp),
            "TMP": str(self.temp),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "ANONYMIZED_TELEMETRY": "False",
        }
        for key, value in values.items():
            os.environ.setdefault(key, value)


PATHS = ProjectPaths.discover()
PATHS.ensure()
PATHS.install_environment()
