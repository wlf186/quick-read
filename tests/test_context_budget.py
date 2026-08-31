import sqlite3

import pytest

from sandevistan_read import providers
from sandevistan_read.context_budget import ContextUsage, TokenLimits, estimate_text_tokens, pack_items
from sandevistan_read.database import Database


def test_token_limits_apply_manual_overrides_and_conservative_defaults() -> None:
    fallback = TokenLimits.from_provider({"capabilities": {}, "config": {}})
    assert fallback.effective_context_tokens == 4096
    assert fallback.max_output_tokens == 1024
    assert fallback.context_source == "fallback"

    manual = TokenLimits.from_provider(
        {
            "capabilities": {"token_limits": {"model_context_tokens": 32768, "effective_context_tokens": 8192}},
            "config": {"context_window_tokens": 16384, "max_output_tokens": 2048},
        }
    )
    assert manual.effective_context_tokens == 16384
    assert manual.max_output_tokens == 2048
    assert manual.context_source == "manual"


def test_packing_preserves_source_coverage_within_budget() -> None:
    items = [
        {"source": "a", "text": "甲" * 120},
        {"source": "a", "text": "乙" * 120},
        {"source": "b", "text": "丙" * 120},
    ]
    packed = pack_items(items, lambda item: item["text"], 400, group_key=lambda item: item["source"])
    assert {item["source"] for item in packed.items} == {"a", "b"}
    assert sum(estimate_text_tokens(text) + 2 for text in packed.texts) <= 400


@pytest.mark.asyncio
async def test_budgeted_chat_retries_context_overflow_at_smaller_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = {
        "kind": "openai",
        "model": "chat",
        "base_url": "https://example.com",
        "api_key": "",
        "capabilities": {"token_limits": {"effective_context_tokens": 8192, "max_output_tokens": 2048}},
        "config": {},
    }
    monkeypatch.setattr(providers, "active_provider", lambda role: provider)
    calls = 0
    scales: list[float] = []

    async def fake_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise providers.ContextOverflowError("maximum context length", status=400, code="context_length_exceeded")
        return providers.ChatCompletion("ok", prompt_tokens=100, completion_tokens=8)

    monkeypatch.setattr(providers, "_chat_once", fake_once)
    trace = ContextUsage()

    def build(budget):
        scales.append(budget.scale)
        return providers.PromptBuild([{"role": "user", "content": "short"}], 10, int(10 * budget.scale))

    result = await providers.budgeted_chat(build, max_tokens=1000, trace=trace)
    assert result.content == "ok"
    assert scales == [1.0, 0.5, 0.25]
    assert trace.as_dict()["overflow_retries"] == 2
    assert trace.as_dict()["adjusted"] is True


def test_message_metadata_migration_preserves_legacy_rows(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_versions (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
            INSERT INTO schema_versions VALUES (1, 'old');
            INSERT INTO schema_versions VALUES (2, 'old');
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations_json TEXT NOT NULL DEFAULT '[]',
                scope_hash TEXT,
                state TEXT NOT NULL DEFAULT 'complete',
                created_at TEXT NOT NULL
            );
            INSERT INTO messages VALUES ('m1','c1','assistant','legacy','[]',NULL,'complete','old');
            """
        )
    database = Database(path)
    database._migrate_v3()
    row = database.fetchone("SELECT content,metadata_json FROM messages WHERE id='m1'")
    assert row == {"content": "legacy", "metadata_json": "{}"}
    assert database.fetchone("SELECT MAX(version) AS version FROM schema_versions")["version"] == 3
