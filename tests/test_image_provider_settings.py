import shutil
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from sandevistan_read import app as app_module
from sandevistan_read.database import Database
from sandevistan_read.documents import parse_document
from sandevistan_read.paths import PATHS


def initialized_database(path: Path) -> Database:
    database = Database(path)
    database.initialize()
    return database


def test_v6_defaults_keep_main_required_and_image_chain(monkeypatch, tmp_path: Path):
    database = initialized_database(tmp_path / "settings.sqlite3")
    monkeypatch.setattr(app_module, "DB", database)
    monkeypatch.setattr(app_module.CONFIG.security, "access_key", "")
    client = TestClient(app_module.api)

    roles = client.get("/provider-roles").json()
    assert roles[0] == {"role": "main", "enabled": True, "required": True, "selected_provider_id": None}
    assert client.get("/settings/image-processing").json() == {
        "mode": "process", "processors": ["vlm", "main", "ocr"]
    }
    assert client.put("/settings/image-processing", json={"mode": "off", "processors": []}).json()["mode"] == "off"
    response = client.patch("/provider-roles/main", json={"enabled": False})
    assert response.status_code == 409


def test_standalone_image_becomes_a_visual_candidate(tmp_path: Path):
    path = tmp_path / "diagram.png"
    Image.new("RGB", (8, 8), "white").save(path)
    try:
        parsed = parse_document(path, "test-image")
        assert parsed.parser == "pymupdf-image"
        assert parsed.blocks[0].visual_needed is True
        assert parsed.blocks[0].locator["visual_only"] is True
    finally:
        shutil.rmtree(PATHS.renders / "test-image", ignore_errors=True)
