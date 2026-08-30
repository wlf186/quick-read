from pathlib import Path

from fastapi.testclient import TestClient

from sandevistan_read import app as app_module
from sandevistan_read import cleanup
from sandevistan_read.database import Database, utc_now


def cleanup_database(path: Path) -> Database:
    database = Database(path)
    with database.transaction() as connection:
        connection.executescript(
            """
            CREATE TABLE notebooks (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                deletion_requested_at TEXT,
                cleanup_error TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                notebook_id TEXT NOT NULL,
                state TEXT NOT NULL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE cleanup_operations (
                id TEXT PRIMARY KEY,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                state TEXT NOT NULL,
                phase TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT
            );
            """
        )
    return database


def test_request_notebook_deletes_queues_active_targets(monkeypatch, tmp_path: Path):
    database = cleanup_database(tmp_path / "cleanup.sqlite3")
    now = utc_now()
    with database.transaction() as connection:
        connection.executemany(
            "INSERT INTO notebooks(id,state,updated_at) VALUES(?,?,?)",
            [("nb_one", "active", now), ("nb_two", "active", now), ("nb_busy", "deleting", now)],
        )
        connection.executemany(
            "INSERT INTO jobs(id,notebook_id,state,updated_at) VALUES(?,?,?,?)",
            [("job_running", "nb_one", "running", now), ("job_queued", "nb_two", "queued", now)],
        )
    monkeypatch.setattr(cleanup, "DB", database)

    results = cleanup.request_notebook_deletes(["nb_one", "missing", "nb_busy", "nb_one", "nb_two"])

    assert [item["id"] for item in results] == ["nb_one", "missing", "nb_busy", "nb_two"]
    assert [item["accepted"] for item in results] == [True, False, False, True]
    assert all(item.get("operation_id") for item in (results[0], results[3]))
    assert database.fetchone("SELECT state FROM notebooks WHERE id='nb_one'")["state"] == "deleting"
    assert database.fetchone("SELECT state FROM notebooks WHERE id='nb_two'")["state"] == "deleting"
    assert database.fetchone("SELECT state FROM notebooks WHERE id='nb_busy'")["state"] == "deleting"
    assert database.fetchone("SELECT state,cancel_requested FROM jobs WHERE id='job_running'") == {"state": "cancelling", "cancel_requested": 1}
    assert database.fetchone("SELECT state,cancel_requested FROM jobs WHERE id='job_queued'") == {"state": "queued", "cancel_requested": 1}
    assert len(database.fetchall("SELECT id FROM cleanup_operations")) == 2


def test_batch_delete_notebooks_endpoint_reports_each_result(monkeypatch):
    expected = [
        {"id": "nb_one", "accepted": True, "operation_id": "cleanup_one"},
        {"id": "missing", "accepted": False, "error": "Notebook 不存在"},
    ]
    monkeypatch.setattr(app_module, "request_notebook_deletes", lambda notebook_ids: expected)
    monkeypatch.setattr(app_module.CONFIG.security, "access_key", "")

    response = TestClient(app_module.api).post(
        "/notebooks/batch-delete",
        json={"notebook_ids": ["nb_one", "missing"]},
    )

    assert response.status_code == 202
    assert response.json() == {"items": expected}


def test_batch_delete_notebooks_rejects_empty_request(monkeypatch):
    monkeypatch.setattr(app_module.CONFIG.security, "access_key", "")
    response = TestClient(app_module.api).post("/notebooks/batch-delete", json={"notebook_ids": []})
    assert response.status_code == 422
