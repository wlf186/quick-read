import shutil
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from sandevistan_read import app as app_module
from sandevistan_read.database import Database, new_id, utc_now
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


def seeded_vlm_provider(database: Database) -> str:
    now = utc_now()
    provider_id = new_id("provider")
    database.execute(
        "INSERT INTO provider_profiles (id,name,role,kind,base_url,model,secret_enc,capabilities_json,config_json,active,created_at,updated_at,selected) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (provider_id, "Vision", "vlm", "ollama", "http://localhost:11434", "qwen-vl:latest", "", "{}", "{}", 1, now, now, 1),
    )
    database.execute("UPDATE provider_role_settings SET enabled=1,updated_at=? WHERE role='vlm'", (now,))
    return provider_id


def test_disable_role_keeps_selected_provider_id(monkeypatch, tmp_path: Path):
    database = initialized_database(tmp_path / "settings.sqlite3")
    monkeypatch.setattr(app_module, "DB", database)
    monkeypatch.setattr(app_module.CONFIG.security, "access_key", "")
    provider_id = seeded_vlm_provider(database)
    client = TestClient(app_module.api)

    response = client.patch("/provider-roles/vlm", json={"enabled": False})
    assert response.status_code == 200
    assert response.json() == {"role": "vlm", "enabled": False, "selected_provider_id": provider_id}
    roles = {item["role"]: item for item in client.get("/provider-roles").json()}
    assert roles["vlm"]["enabled"] is False
    assert roles["vlm"]["selected_provider_id"] == provider_id


def test_reenable_role_defaults_to_catalog_validation(monkeypatch, tmp_path: Path):
    database = initialized_database(tmp_path / "settings.sqlite3")
    monkeypatch.setattr(app_module, "DB", database)
    monkeypatch.setattr(app_module.CONFIG.security, "access_key", "")
    provider_id = seeded_vlm_provider(database)
    database.execute("UPDATE provider_role_settings SET enabled=0 WHERE role='vlm'")
    monkeypatch.setattr(app_module, "active_provider", lambda role: None)
    monkeypatch.setattr(app_module, "provider_by_id", lambda pid: {"id": provider_id, "role": "vlm"})
    modes: list[str] = []

    async def fake_inspect(provider, mode="catalog"):
        modes.append(mode)
        return {"activation_eligible": True}

    monkeypatch.setattr(app_module, "inspect_provider", fake_inspect)
    client = TestClient(app_module.api)

    response = client.patch("/provider-roles/vlm", json={"enabled": True})
    assert response.status_code == 200
    assert modes == ["catalog"]
    assert response.json() == {"role": "vlm", "enabled": True, "selected_provider_id": provider_id}


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
