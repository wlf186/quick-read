from __future__ import annotations

import json
import sqlite3

from sandevistan_read.database import Database, json_dump
from sandevistan_read.context_budget import PromptBudget
from sandevistan_read.providers import BudgetedCompletion, study_generation_profile
from sandevistan_read.study import generate_study_artifact, validate_flashcard_item, validate_quiz_item
from sandevistan_read import study
from sandevistan_read import study_sessions


def evidence() -> dict[str, dict]:
    return {"S1": {"content": "光合作用把光能转化为化学能，并在叶绿体中产生有机物。"}}


def test_quiz_validator_accepts_grounded_understanding_item() -> None:
    item, reason = validate_quiz_item(
        {
            "question": "植物细胞通过光合作用实现了哪种能量转换？",
            "options": ["光能转化为化学能", "化学能转化为声能", "热能转化为核能", "电能转化为机械能"],
            "answer_index": 0,
            "hint": "考虑叶绿体吸收的能量来源。",
            "explanation": "资料明确说明光合作用把光能转化为化学能。",
            "citations": ["S1"],
            "difficulty": "medium",
            "cognitive_level": "understand",
        },
        {"S1"},
        evidence(),
    )
    assert reason is None
    assert item and item["answer_index"] == 0 and item["citations"] == ["S1"]


def test_quiz_validator_rejects_source_location_and_generic_distractors() -> None:
    base = {
        "question": "以下哪项内容直接出现在第 2 页？",
        "options": ["光能转化为化学能", "资料没有讨论这一主题", "外部知识", "以上皆是"],
        "answer_index": 0,
        "hint": "回忆原文。",
        "explanation": "光合作用把光能转化为化学能。",
        "citations": ["S1"],
    }
    item, reason = validate_quiz_item(base, {"S1"}, evidence())
    assert item is None
    assert reason == "source_location_stem"


def test_flashcard_validator_keeps_item_specific_citations() -> None:
    item, reason = validate_flashcard_item(
        {
            "front": "光合作用完成怎样的能量转换？",
            "back": "把光能转化为化学能。",
            "explanation": "这一过程发生在叶绿体，并产生有机物。",
            "citations": ["S1", "S999"],
            "card_type": "concept",
        },
        {"S1"},
        evidence(),
    )
    assert reason is None
    assert item and item["citations"] == ["S1"]


def test_study_generation_tier_uses_model_and_both_token_windows() -> None:
    provider = {
        "kind": "ollama",
        "config": {},
        "capabilities": {
            "model_profile": {"parameter_count": 2_000_000_000},
            "token_limits": {"effective_context_tokens": 32768, "max_output_tokens": 4096},
        },
    }
    assert study_generation_profile(provider)["tier"] == "lite"
    provider["capabilities"]["model_profile"]["parameter_count"] = 8_000_000_000
    assert study_generation_profile(provider)["tier"] == "full"
    provider["capabilities"]["token_limits"]["max_output_tokens"] = 1024
    assert study_generation_profile(provider)["tier"] == "lite"


async def test_lite_quiz_generation_uses_model_items_without_template_fillers(tmp_path, monkeypatch) -> None:
    database = _study_database(tmp_path)
    database.execute(
        """INSERT INTO sources(id,notebook_id,revision_id,filename,media_type,size_bytes,sha256,blob_path,preview_path,state,selected,page_count,parser,error,metadata_json,created_at,updated_at)
        VALUES('s1','n1','r1','biology.md','text/markdown',100,'hash','runtime/data/blobs/s1/biology.md',NULL,'ready',1,1,'text',NULL,'{}','now','now')"""
    )
    for index in range(1, 5):
        database.execute(
            "INSERT INTO chunks(id,source_id,source_revision_id,ordinal,content,locator_json,embedding_json,checksum,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (f"ch{index}", "s1", "r1", index, f"光合作用把光能转化为化学能，并由叶绿体产生有机物。过程阶段 {index}。", json_dump({"section": "光合作用"}), "[]", f"h{index}", "now"),
        )
    monkeypatch.setattr(study, "DB", database)
    monkeypatch.setattr(study, "active_provider", lambda role: {"kind": "ollama", "config": {}, "capabilities": {"model_profile": {"parameter_count": 2_000_000_000}, "token_limits": {"effective_context_tokens": 4096, "max_output_tokens": 1024}}})
    calls = 0

    async def fake_chat(builder, **kwargs):
        nonlocal calls
        calls += 1
        budget = PromptBudget(4096, 2800, 900, 2048, 1.0)
        build = builder(budget)
        label = build.metadata["labels"][0]
        content = json.dumps({"items": [{
            "question": f"植物进行光合作用时，能量转换的主要结果是什么？（情境 {calls}）",
            "options": ["光能转化为化学能", "化学能转化为声能", "机械能转化为核能", "电能转化为热能"],
            "answer_index": 0,
            "hint": "考虑叶绿体吸收的能量来源。",
            "explanation": "资料说明光合作用把光能转化为化学能并产生有机物。",
            "citations": [label],
            "difficulty": "medium",
            "cognitive_level": "understand",
        }]}, ensure_ascii=False)
        return BudgetedCompletion(content, build, budget)

    monkeypatch.setattr(study, "budgeted_chat", fake_chat)
    monkeypatch.setattr(study, "_semantic_unique", lambda candidate, accepted, kind: True)
    result = await generate_study_artifact("n1", "quiz", 3, ["s1"], "zh-CN", "medium")
    assert result["status"] == "ready"
    assert len(result["payload"]["items"]) == 3
    assert all("直接出现在" not in item["question"] for item in result["payload"]["items"])
    assert [item["answer_index"] for item in result["payload"]["items"]] == [0, 1, 2]
    assert all(len(item["citations"]) == 1 for item in result["payload"]["items"])


def _study_database(tmp_path) -> Database:
    database = Database(tmp_path / "study.sqlite3")
    database.initialize()
    database.execute("INSERT INTO notebooks(id,title,description,created_at,updated_at) VALUES('n1','Study','','now','now')")
    return database


def test_quiz_session_hides_answer_then_returns_grounded_feedback(tmp_path, monkeypatch) -> None:
    database = _study_database(tmp_path)
    payload = {
        "version": 2,
        "items": [{
            "id": "q1", "question": "能量如何转换？", "options": ["光到化学", "化学到声", "热到核", "电到机械"],
            "answer_index": 0, "hint": "考虑光。", "explanation": "资料说明光能转化为化学能。", "citations": ["S1"],
        }],
    }
    citations = [{"id": "S1", "source_id": "s1", "filename": "biology.md", "locator": {"section": "光合作用"}, "quote": "光能转化为化学能"}]
    database.execute(
        """INSERT INTO artifacts(id,notebook_id,type,title,scope_json,language,status,payload_json,citations_json,media_path,created_at,updated_at)
        VALUES('a1','n1','quiz','Quiz','[]','zh-CN','ready',?,?,NULL,'now','now')""",
        (json_dump(payload), json_dump(citations)),
    )
    monkeypatch.setattr(study_sessions, "DB", database)
    session = study_sessions.create_session("a1", "all")
    assert "answer_index" not in session["items"][0]
    result = study_sessions.answer_quiz(session["id"], "q1", 0)
    assert result["result"]["correct"] is True
    assert result["result"]["citation_details"][0]["id"] == "S1"
    assert result["session"]["status"] == "complete"
    assert database.fetchone("SELECT COUNT(*) AS count FROM quiz_attempts")["count"] == 1


def test_flashcard_review_persists_fsrs_state_and_resumes(tmp_path, monkeypatch) -> None:
    database = _study_database(tmp_path)
    payload = {"items": [{"id": "c1", "front": "正面", "back": "背面", "citations": []}]}
    database.execute(
        """INSERT INTO artifacts(id,notebook_id,type,title,scope_json,language,status,payload_json,citations_json,media_path,created_at,updated_at)
        VALUES('a2','n1','flashcard','Cards','[]','zh-CN','ready',?,'[]',NULL,'now','now')""",
        (json_dump(payload),),
    )
    monkeypatch.setattr(study_sessions, "DB", database)
    session = study_sessions.create_session("a2", "due")
    response = study_sessions.review_flashcard(session["id"], "c1", "good")
    assert response["rating"] == "good"
    assert response["session"]["status"] == "complete"
    state = database.fetchone("SELECT last_rating,due_at FROM flashcard_states WHERE artifact_id='a2' AND card_id='c1'")
    assert state and state["last_rating"] == "good" and state["due_at"]


def test_v4_migration_preserves_attempts_and_reviews(tmp_path) -> None:
    path = tmp_path / "legacy-study.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_versions (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
            INSERT INTO schema_versions VALUES (1, 'old'); INSERT INTO schema_versions VALUES (2, 'old'); INSERT INTO schema_versions VALUES (3, 'old');
            CREATE TABLE quiz_attempts (id TEXT PRIMARY KEY,artifact_id TEXT NOT NULL,answers_json TEXT NOT NULL,score REAL NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE flashcard_reviews (id TEXT PRIMARY KEY,artifact_id TEXT NOT NULL,card_id TEXT NOT NULL,rating TEXT NOT NULL,created_at TEXT NOT NULL);
            INSERT INTO quiz_attempts VALUES ('qa','a','{}',1.0,'old');
            INSERT INTO flashcard_reviews VALUES ('fr','a','c','mastered','old');
            """
        )
    database = Database(path)
    database._migrate_v4()
    attempt = database.fetchone("SELECT id,results_json FROM quiz_attempts WHERE id='qa'")
    assert attempt == {"id": "qa", "results_json": "{}"}
    assert database.fetchone("SELECT rating FROM flashcard_reviews WHERE id='fr'")["rating"] == "mastered"
    assert database.fetchone("SELECT MAX(version) AS version FROM schema_versions")["version"] == 4
