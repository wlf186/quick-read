from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "intfloat/multilingual-e5-small"
TARGET = ROOT / "runtime" / "models" / "sentence-transformers" / MODEL_ID.replace("/", "--")
CACHE_SNAPSHOT = TARGET.parent / ("models--" + MODEL_ID.replace("/", "--"))
os.environ.setdefault("HF_HOME", str(ROOT / "runtime" / "models" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ROOT / "runtime" / "models" / "sentence-transformers"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main() -> None:
    from sentence_transformers import SentenceTransformer

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not (TARGET / "config.json").exists():
        model = SentenceTransformer(MODEL_ID, device="cpu", cache_folder=str(TARGET.parent))
        model.save(str(TARGET))
    manifest = {
        "model": MODEL_ID,
        "path": str(TARGET.relative_to(ROOT)),
        "installed_at": datetime.now(UTC).isoformat(),
        "runtime_offline": True,
    }
    (TARGET / "sandevistan-model.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(CACHE_SNAPSHOT, ignore_errors=True)
    print(f"Project-local embedding model ready: {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
