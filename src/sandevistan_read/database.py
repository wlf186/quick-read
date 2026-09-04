from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .paths import PATHS


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_load(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class Database:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or PATHS.database
        self._write_lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            connection = self.connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS schema_versions (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS notebooks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
            revision_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            media_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            blob_path TEXT NOT NULL,
            preview_path TEXT,
            state TEXT NOT NULL,
            selected INTEGER NOT NULL DEFAULT 1,
            page_count INTEGER NOT NULL DEFAULT 0,
            parser TEXT,
            error TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sources_notebook ON sources(notebook_id, selected, state);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_revision ON sources(revision_id);
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            source_revision_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            content TEXT NOT NULL,
            locator_json TEXT NOT NULL,
            embedding_json TEXT,
            checksum TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id, ordinal);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            source_id UNINDEXED,
            content,
            tokenize='unicode61'
        );
        CREATE TABLE IF NOT EXISTS summaries (
            id TEXT PRIMARY KEY,
            notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
            scope_hash TEXT NOT NULL,
            content TEXT NOT NULL,
            citations_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_summaries_scope ON summaries(notebook_id, scope_hash, created_at DESC);
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            citations_json TEXT NOT NULL DEFAULT '[]',
            scope_hash TEXT,
            state TEXT NOT NULL DEFAULT 'complete',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
        CREATE TABLE IF NOT EXISTS provider_profiles (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            kind TEXT NOT NULL,
            base_url TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            secret_enc TEXT NOT NULL DEFAULT '',
            capabilities_json TEXT NOT NULL DEFAULT '{}',
            config_json TEXT NOT NULL DEFAULT '{}',
            active INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_provider_role ON provider_profiles(role, active);
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            state TEXT NOT NULL,
            stage TEXT NOT NULL,
            progress REAL NOT NULL,
            notebook_id TEXT,
            parent_id TEXT,
            payload_json TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            retryable INTEGER NOT NULL DEFAULT 1,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, created_at);
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            scope_json TEXT NOT NULL,
            language TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            citations_json TEXT NOT NULL,
            media_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_artifacts_notebook ON artifacts(notebook_id, type, created_at DESC);
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
            answers_json TEXT NOT NULL,
            score REAL NOT NULL,
            session_id TEXT,
            results_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS flashcard_reviews (
            id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
            card_id TEXT NOT NULL,
            rating TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS study_sessions (
            id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            mode TEXT NOT NULL,
            item_ids_json TEXT NOT NULL,
            state_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_study_sessions_artifact ON study_sessions(artifact_id, kind, status, updated_at DESC);
        CREATE TABLE IF NOT EXISTS flashcard_states (
            artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
            card_id TEXT NOT NULL,
            fsrs_json TEXT NOT NULL,
            due_at TEXT NOT NULL,
            last_rating TEXT,
            suspended INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(artifact_id, card_id)
        );
        CREATE INDEX IF NOT EXISTS idx_flashcard_states_due ON flashcard_states(artifact_id, suspended, due_at);
        """
        with self.connect() as connection:
            connection.executescript(schema)
            connection.execute(
                "INSERT OR IGNORE INTO schema_versions(version, applied_at) VALUES(1, ?)",
                (utc_now(),),
            )
            connection.commit()
        self._migrate_v2()
        self._migrate_v3()
        self._migrate_v4()
        self._migrate_v5()
        self._migrate_v6()
        with self.transaction() as connection:
            connection.execute("""UPDATE jobs SET processing_seconds=MAX(0,(julianday(finished_at)-julianday(started_at))*86400)
                WHERE processing_seconds=0 AND started_at IS NOT NULL AND finished_at IS NOT NULL""")
            connection.execute("UPDATE jobs SET stage_progress=progress WHERE stage_progress=0 AND progress>0")

    def _migrate_v2(self) -> None:
        """Add observable jobs and resumable cleanup. Safe to run on every start."""
        current = self.fetchone("SELECT MAX(version) AS version FROM schema_versions") or {}
        if int(current.get("version") or 0) >= 2:
            return
        if self.path.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = PATHS.backups / f"sandevistan-read.pre-v2.{stamp}.db"
            source = sqlite3.connect(self.path)
            target = sqlite3.connect(backup)
            try:
                source.backup(target)
            finally:
                target.close(); source.close()
        with self.transaction() as connection:
            notebook_columns = {row[1] for row in connection.execute("PRAGMA table_info(notebooks)")}
            job_columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
            notebook_additions = {
                "state": "TEXT NOT NULL DEFAULT 'active'",
                "deletion_requested_at": "TEXT",
                "cleanup_error": "TEXT",
            }
            job_additions = {
                "display_name": "TEXT NOT NULL DEFAULT ''",
                "stage_code": "TEXT NOT NULL DEFAULT 'queued'",
                "stage_progress": "REAL NOT NULL DEFAULT 0",
                "progress_basis": "TEXT NOT NULL DEFAULT 'observed'",
                "stage_current": "REAL",
                "stage_total": "REAL",
                "stage_unit": "TEXT",
                "activity_json": "TEXT NOT NULL DEFAULT '{}'",
                "workload_json": "TEXT NOT NULL DEFAULT '{}'",
                "execution_profile_json": "TEXT NOT NULL DEFAULT '{}'",
                "processing_seconds": "REAL NOT NULL DEFAULT 0",
            }
            for name, definition in notebook_additions.items():
                if name not in notebook_columns:
                    connection.execute(f"ALTER TABLE notebooks ADD COLUMN {name} {definition}")
            for name, definition in job_additions.items():
                if name not in job_columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    stage_code TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    stage_current REAL,
                    stage_total REAL,
                    stage_unit TEXT,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, id);
                CREATE TABLE IF NOT EXISTS local_resources (
                    id TEXT PRIMARY KEY,
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    notebook_id TEXT,
                    kind TEXT NOT NULL,
                    relative_path TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL DEFAULT 'active',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    transferred_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_local_resources_owner ON local_resources(owner_type, owner_id, state);
                CREATE TABLE IF NOT EXISTS cleanup_operations (
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
                CREATE INDEX IF NOT EXISTS idx_cleanup_state ON cleanup_operations(state, created_at);
            """)
            connection.execute("UPDATE jobs SET display_name=CASE kind WHEN 'ingest' THEN '文档解析' WHEN 'summary' THEN '生成摘要' WHEN 'quiz' THEN 'Quiz 题库' WHEN 'flashcard' THEN 'Flashcard 闪卡' WHEN 'podcast' THEN '双人音频播客' ELSE kind END WHERE display_name='' OR display_name IS NULL")
            connection.execute("UPDATE jobs SET stage_code=CASE WHEN state='complete' THEN 'complete' WHEN state='failed' THEN 'failed' WHEN state='cancelled' THEN 'cancelled' WHEN state='running' THEN 'recovering' ELSE 'queued' END")
            connection.execute("INSERT INTO schema_versions(version, applied_at) VALUES(2, ?)", (utc_now(),))

    def _migrate_v3(self) -> None:
        """Persist per-message generation metadata without rewriting history."""
        current = self.fetchone("SELECT MAX(version) AS version FROM schema_versions") or {}
        if int(current.get("version") or 0) >= 3:
            return
        if self.path.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = PATHS.backups / f"sandevistan-read.pre-v3.{stamp}.db"
            source = sqlite3.connect(self.path)
            target = sqlite3.connect(backup)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
        with self.transaction() as connection:
            message_columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
            if "metadata_json" not in message_columns:
                connection.execute("ALTER TABLE messages ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")
            connection.execute("INSERT INTO schema_versions(version, applied_at) VALUES(3, ?)", (utc_now(),))

    def _migrate_v4(self) -> None:
        """Add resumable study sessions and FSRS state while preserving study history."""
        current = self.fetchone("SELECT MAX(version) AS version FROM schema_versions") or {}
        if int(current.get("version") or 0) >= 4:
            return
        if self.path.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = PATHS.backups / f"sandevistan-read.pre-v4.{stamp}.db"
            source = sqlite3.connect(self.path)
            target = sqlite3.connect(backup)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
        with self.transaction() as connection:
            attempt_columns = {row[1] for row in connection.execute("PRAGMA table_info(quiz_attempts)")}
            if "session_id" not in attempt_columns:
                connection.execute("ALTER TABLE quiz_attempts ADD COLUMN session_id TEXT")
            if "results_json" not in attempt_columns:
                connection.execute("ALTER TABLE quiz_attempts ADD COLUMN results_json TEXT NOT NULL DEFAULT '{}'")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS study_sessions (
                    id TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    item_ids_json TEXT NOT NULL,
                    state_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_study_sessions_artifact ON study_sessions(artifact_id, kind, status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS flashcard_states (
                    artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
                    card_id TEXT NOT NULL,
                    fsrs_json TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    last_rating TEXT,
                    suspended INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(artifact_id, card_id)
                );
                CREATE INDEX IF NOT EXISTS idx_flashcard_states_due ON flashcard_states(artifact_id, suspended, due_at);
            """)
            connection.execute("INSERT INTO schema_versions(version, applied_at) VALUES(4, ?)", (utc_now(),))

    def _migrate_v5(self) -> None:
        """Promote the co-located speech service to AUDIO while retaining TTS-only profiles."""
        current = self.fetchone("SELECT MAX(version) AS version FROM schema_versions") or {}
        if int(current.get("version") or 0) >= 5:
            return
        if self.path.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = PATHS.backups / f"sandevistan-read.pre-v5.{stamp}.db"
            backup.parent.mkdir(parents=True, exist_ok=True)
            source = sqlite3.connect(self.path)
            target = sqlite3.connect(backup)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
        with self.transaction() as connection:
            audio_rows = connection.execute(
                "SELECT id,config_json FROM provider_profiles WHERE role='tts' AND kind='sandevistan_tts'"
            ).fetchall()
            for row in audio_rows:
                config = json_load(row["config_json"], {}) or {}
                config.setdefault("asr_auto_select", True)
                config.setdefault("asr_allow_device_fallback", True)
                connection.execute(
                    "UPDATE provider_profiles SET role='audio',kind='sandevistan_audio',config_json=?,updated_at=? WHERE id=?",
                    (json_dump(config), utc_now(), row["id"]),
                )
            connection.execute(
                "UPDATE provider_profiles SET role='tts_only',active=0,updated_at=? WHERE role='tts' AND kind='openai_tts'",
                (utc_now(),),
            )
            connection.execute("INSERT INTO schema_versions(version, applied_at) VALUES(5, ?)", (utc_now(),))

    def _migrate_v6(self) -> None:
        """Separate provider role state and persist image-processing settings/results."""
        current = self.fetchone("SELECT MAX(version) AS version FROM schema_versions") or {}
        if int(current.get("version") or 0) >= 6:
            return
        if self.path.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = PATHS.backups / f"sandevistan-read.pre-v6.{stamp}.db"
            backup.parent.mkdir(parents=True, exist_ok=True)
            source = sqlite3.connect(self.path)
            target = sqlite3.connect(backup)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
        now = utc_now()
        with self.transaction() as connection:
            provider_columns = {row[1] for row in connection.execute("PRAGMA table_info(provider_profiles)")}
            if "selected" not in provider_columns:
                connection.execute("ALTER TABLE provider_profiles ADD COLUMN selected INTEGER NOT NULL DEFAULT 0")
            connection.execute("UPDATE provider_profiles SET selected=active")
            for role in ("main", "vlm", "audio"):
                rows = connection.execute(
                    "SELECT id FROM provider_profiles WHERE role=? AND selected=1 ORDER BY updated_at DESC", (role,)
                ).fetchall()
                for row in rows[1:]:
                    connection.execute("UPDATE provider_profiles SET selected=0,active=0 WHERE id=?", (row["id"],))
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS provider_role_settings (
                    role TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_visuals (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    locator_json TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    status TEXT NOT NULL,
                    processor TEXT,
                    description TEXT NOT NULL DEFAULT '',
                    attempts_json TEXT NOT NULL DEFAULT '[]',
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_source_visuals_source ON source_visuals(source_id, ordinal);
            """)
            for role in ("main", "vlm", "audio"):
                active = connection.execute(
                    "SELECT 1 FROM provider_profiles WHERE role=? AND selected=1 LIMIT 1", (role,)
                ).fetchone()
                enabled = 1 if role == "main" or active else 0
                connection.execute(
                    "INSERT OR IGNORE INTO provider_role_settings(role,enabled,updated_at) VALUES(?,?,?)",
                    (role, enabled, now),
                )
            connection.execute(
                "INSERT OR IGNORE INTO app_settings(key,value_json,updated_at) VALUES(?,?,?)",
                ("image_processing", json_dump({"mode": "process", "processors": ["vlm", "main", "ocr"]}), now),
            )
            connection.execute("INSERT INTO schema_versions(version, applied_at) VALUES(6, ?)", (now,))

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> None:
        with self.transaction() as connection:
            connection.execute(sql, parameters)

    def fetchone(self, sql: str, parameters: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(sql, parameters).fetchone()
            return dict(row) if row else None

    def fetchall(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(sql, parameters).fetchall()]

    def seed(self, ollama_url: str, ollama_model: str, audio_url: str) -> None:
        now = utc_now()
        with self.transaction() as connection:
            if not connection.execute("SELECT 1 FROM notebooks LIMIT 1").fetchone():
                connection.execute(
                    "INSERT INTO notebooks(id, title, description, created_at, updated_at) VALUES(?,?,?,?,?)",
                    (new_id("nb"), "Project Relic / 产品研究", "本地资料研究笔记本", now, now),
                )
            if not connection.execute("SELECT 1 FROM provider_profiles LIMIT 1").fetchone():
                connection.executemany(
                    """INSERT INTO provider_profiles
                    (id,name,role,kind,base_url,model,secret_enc,capabilities_json,config_json,active,created_at,updated_at,selected)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [
                        (
                            new_id("provider"), "Local Ollama", "main", "ollama", ollama_url.rstrip("/"),
                            ollama_model, "", json_dump({"vision": True, "json": True}), "{}", 1, now, now, 1,
                        ),
                        (
                            new_id("provider"), "Local Vision", "vlm", "ollama", ollama_url.rstrip("/"),
                            ollama_model, "", json_dump({"vision": True, "json": True}), "{}", 1, now, now, 1,
                        ),
                        (
                            new_id("provider"), "Sandevistan Audio", "audio", "sandevistan_audio", audio_url.rstrip("/"),
                            "qwen3-tts-0.6b", "", json_dump({"async": True}),
                            json_dump({"host_a": "Vivian", "host_b": "Dylan", "language": "Chinese", "response_format": "wav", "compute_device": "cpu", "asr_auto_select": True, "asr_allow_device_fallback": True}),
                            1, now, now, 1,
                        ),
                    ],
                )
                connection.execute("UPDATE provider_role_settings SET enabled=1,updated_at=? WHERE role IN ('vlm','audio')", (now,))

    def reset_running_jobs(self) -> None:
        now = utc_now()
        self.execute(
            "UPDATE jobs SET state='queued', stage='服务重启后恢复排队', stage_code='recovering', updated_at=? WHERE state IN ('running','cancelling')",
            (now,),
        )


DB = Database()
