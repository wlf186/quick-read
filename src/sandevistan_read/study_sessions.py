from __future__ import annotations

import csv
import io
import random
from datetime import UTC, datetime
from typing import Any

from fsrs import Card, Rating, Scheduler

from .database import DB, json_dump, json_load, new_id, utc_now


SCHEDULER = Scheduler(desired_retention=0.9, maximum_interval=365)
RATINGS = {"again": Rating.Again, "hard": Rating.Hard, "good": Rating.Good, "easy": Rating.Easy, "mastered": Rating.Good}


def _artifact(artifact_id: str, expected: str | None = None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    row = DB.fetchone("SELECT * FROM artifacts WHERE id=?", (artifact_id,))
    if not row or (expected and row["type"] != expected):
        raise ValueError("学习产物不存在")
    payload = json_load(row["payload_json"], {})
    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        items = []
    return row, payload, items


def public_artifact(item: dict[str, Any]) -> dict[str, Any]:
    """Hide answer keys and post-answer evidence from public Quiz payloads."""
    if item.get("type") != "quiz":
        return item
    payload = dict(item.get("payload") or {})
    public_items = []
    for source in payload.get("items") or []:
        question = dict(source)
        for field in ("answer", "answer_index", "explanation", "citations"):
            question.pop(field, None)
        public_items.append(question)
    payload["items"] = public_items
    return {**item, "payload": payload, "citations": []}


def _parse_time(value: str | None) -> datetime:
    try:
        parsed = datetime.fromisoformat(value or "")
    except ValueError:
        parsed = datetime.now(UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _load_card_state(artifact_id: str, card_id: str) -> tuple[Card, str | None]:
    state = DB.fetchone("SELECT * FROM flashcard_states WHERE artifact_id=? AND card_id=?", (artifact_id, card_id))
    if state:
        return Card.from_dict(json_load(state["fsrs_json"], {})), state.get("last_rating")
    reviews = DB.fetchall("SELECT rating,created_at FROM flashcard_reviews WHERE artifact_id=? AND card_id=? ORDER BY created_at", (artifact_id, card_id))
    first_review = _parse_time(reviews[0]["created_at"]) if reviews else datetime.now(UTC)
    card = Card(due=first_review)
    last_rating = None
    for review in reviews:
        last_rating = "good" if review["rating"] == "mastered" else review["rating"]
        rating = RATINGS.get(review["rating"])
        if rating:
            card, _ = SCHEDULER.review_card(card, rating, review_datetime=_parse_time(review["created_at"]))
    _save_card_state(artifact_id, card_id, card, last_rating)
    return card, last_rating


def _save_card_state(artifact_id: str, card_id: str, card: Card, last_rating: str | None) -> None:
    now = utc_now()
    DB.execute(
        """INSERT INTO flashcard_states(artifact_id,card_id,fsrs_json,due_at,last_rating,suspended,created_at,updated_at)
        VALUES(?,?,?,?,?,0,?,?)
        ON CONFLICT(artifact_id,card_id) DO UPDATE SET fsrs_json=excluded.fsrs_json,due_at=excluded.due_at,last_rating=excluded.last_rating,updated_at=excluded.updated_at""",
        (artifact_id, card_id, json_dump(card.to_dict()), card.due.isoformat(), last_rating, now, now),
    )


def _latest_missed_quiz(artifact_id: str) -> list[str]:
    row = DB.fetchone("SELECT results_json FROM quiz_attempts WHERE artifact_id=? ORDER BY created_at DESC LIMIT 1", (artifact_id,))
    results = json_load((row or {}).get("results_json"), {})
    values = results.get("items", []) if isinstance(results, dict) else []
    return [str(item.get("item_id")) for item in values if isinstance(item, dict) and not item.get("correct")]


def _session_items(artifact_id: str, kind: str, mode: str, items: list[dict[str, Any]]) -> list[str]:
    available = [str(item.get("id")) for item in items if item.get("id")]
    if mode == "same":
        previous = DB.fetchone("SELECT item_ids_json FROM study_sessions WHERE artifact_id=? AND kind=? ORDER BY updated_at DESC LIMIT 1", (artifact_id, kind))
        same = [value for value in json_load((previous or {}).get("item_ids_json"), []) if value in available]
        return same or available
    if kind == "quiz":
        return [item_id for item_id in _latest_missed_quiz(artifact_id) if item_id in available] if mode == "missed" else available
    states = {row["card_id"]: row for row in DB.fetchall("SELECT card_id,due_at,last_rating,suspended FROM flashcard_states WHERE artifact_id=?", (artifact_id,))}
    unsuspended = [item_id for item_id in available if not states.get(item_id, {}).get("suspended")]
    if mode == "missed":
        return [item_id for item_id in unsuspended if states.get(item_id, {}).get("last_rating") == "again"]
    if mode == "due":
        now = datetime.now(UTC)
        return [item_id for item_id in unsuspended if item_id not in states or _parse_time(states[item_id].get("due_at")) <= now]
    return unsuspended


def create_session(artifact_id: str, mode: str = "all", shuffle: bool = False) -> dict[str, Any]:
    row, _, items = _artifact(artifact_id)
    kind = row["type"]
    if kind not in {"quiz", "flashcard"}:
        raise ValueError("该产物不支持学习会话")
    allowed = {"all", "missed", "same"} | ({"due"} if kind == "flashcard" else set())
    if mode not in allowed:
        raise ValueError("当前学习模式不适用于该产物")
    active = DB.fetchone(
        "SELECT id FROM study_sessions WHERE artifact_id=? AND kind=? AND mode=? AND status='active' ORDER BY updated_at DESC LIMIT 1",
        (artifact_id, kind, mode),
    )
    if active:
        return get_session(active["id"])
    session_id, now = new_id("study"), utc_now()
    item_ids = _session_items(artifact_id, kind, mode, items)
    if shuffle:
        random.Random(session_id).shuffle(item_ids)
    state = {"answers": {}, "results": {}, "reviews": {}, "shuffle": shuffle}
    status = "complete" if not item_ids else "active"
    DB.execute(
        """INSERT INTO study_sessions(id,artifact_id,kind,mode,item_ids_json,state_json,status,created_at,updated_at,completed_at)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (session_id, artifact_id, kind, mode, json_dump(item_ids), json_dump(state), status, now, now, now if status == "complete" else None),
    )
    return get_session(session_id)


def _citation_details(row: dict[str, Any], labels: list[str]) -> list[dict[str, Any]]:
    citations = json_load(row.get("citations_json"), [])
    by_id = {item.get("id"): item for item in citations if isinstance(item, dict)}
    return [by_id[label] for label in labels if label in by_id]


def get_session(session_id: str) -> dict[str, Any]:
    session = DB.fetchone("SELECT * FROM study_sessions WHERE id=?", (session_id,))
    if not session:
        raise ValueError("学习会话不存在")
    row, _, items = _artifact(session["artifact_id"], session["kind"])
    by_id = {str(item.get("id")): item for item in items}
    item_ids = json_load(session["item_ids_json"], [])
    state = json_load(session["state_json"], {})
    rendered = []
    for item_id in item_ids:
        source = dict(by_id.get(item_id) or {})
        if session["kind"] == "quiz":
            public = {key: value for key, value in source.items() if key not in {"answer", "answer_index", "explanation", "citations"}}
            if item_id in state.get("results", {}):
                public["result"] = state["results"][item_id]
            rendered.append(public)
        else:
            source["citation_details"] = _citation_details(row, list(source.get("citations") or []))
            if item_id in state.get("reviews", {}):
                source["review"] = state["reviews"][item_id]
            rendered.append(source)
    completed_count = len(state.get("results", {})) if session["kind"] == "quiz" else len(state.get("reviews", {}))
    return {
        "id": session["id"],
        "artifact_id": session["artifact_id"],
        "kind": session["kind"],
        "mode": session["mode"],
        "status": session["status"],
        "items": rendered,
        "progress": {"current": completed_count, "total": len(item_ids)},
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
    }


def answer_quiz(session_id: str, item_id: str, option_index: int) -> dict[str, Any]:
    session = DB.fetchone("SELECT * FROM study_sessions WHERE id=? AND kind='quiz'", (session_id,))
    if not session:
        raise ValueError("Quiz 学习会话不存在")
    row, _, items = _artifact(session["artifact_id"], "quiz")
    item_ids = json_load(session["item_ids_json"], [])
    item = next((value for value in items if value.get("id") == item_id), None)
    if not item or item_id not in item_ids:
        raise ValueError("题目不属于当前学习会话")
    state = json_load(session["state_json"], {})
    if item_id in state.get("results", {}):
        return {"result": state["results"][item_id], "session": get_session(session_id)}
    answer_index = int(item.get("answer_index", item.get("answer", -1)))
    if answer_index not in range(4):
        raise ValueError("题目缺少有效答案")
    result = {
        "item_id": item_id,
        "selected_index": option_index,
        "answer_index": answer_index,
        "correct": option_index == answer_index,
        "explanation": item.get("explanation") or "正确答案由对应资料引用支持。",
        "citations": item.get("citations") or [],
        "citation_details": _citation_details(row, list(item.get("citations") or [])),
    }
    state.setdefault("answers", {})[item_id] = option_index
    state.setdefault("results", {})[item_id] = result
    complete = len(state["results"]) == len(item_ids)
    now = utc_now()
    DB.execute(
        "UPDATE study_sessions SET state_json=?,status=?,updated_at=?,completed_at=? WHERE id=?",
        (json_dump(state), "complete" if complete else "active", now, now if complete else None, session_id),
    )
    if complete:
        correct = sum(bool(value["correct"]) for value in state["results"].values())
        result_items = [state["results"][value] for value in item_ids]
        DB.execute(
            """INSERT INTO quiz_attempts(id,artifact_id,answers_json,score,session_id,results_json,created_at)
            VALUES(?,?,?,?,?,?,?)""",
            (new_id("attempt"), session["artifact_id"], json_dump(state["answers"]), correct / max(1, len(item_ids)), session_id, json_dump({"items": result_items}), now),
        )
    return {"result": result, "session": get_session(session_id)}


def review_flashcard(session_id: str, item_id: str, rating: str) -> dict[str, Any]:
    session = DB.fetchone("SELECT * FROM study_sessions WHERE id=? AND kind='flashcard'", (session_id,))
    if not session:
        raise ValueError("Flashcard 学习会话不存在")
    _, _, items = _artifact(session["artifact_id"], "flashcard")
    item_ids = json_load(session["item_ids_json"], [])
    if item_id not in item_ids or item_id not in {item.get("id") for item in items}:
        raise ValueError("闪卡不属于当前学习会话")
    normalized = "good" if rating == "mastered" else rating
    selected = RATINGS.get(normalized)
    if not selected:
        raise ValueError("不支持的复习评分")
    card, _ = _load_card_state(session["artifact_id"], item_id)
    reviewed_at = datetime.now(UTC)
    card, _ = SCHEDULER.review_card(card, selected, review_datetime=reviewed_at)
    scheduled_days = round(max(0.0, (card.due - reviewed_at).total_seconds() / 86400), 4)
    _save_card_state(session["artifact_id"], item_id, card, normalized)
    DB.execute(
        "INSERT INTO flashcard_reviews(id,artifact_id,card_id,rating,created_at) VALUES(?,?,?,?,?)",
        (new_id("review"), session["artifact_id"], item_id, normalized, reviewed_at.isoformat()),
    )
    state = json_load(session["state_json"], {})
    state.setdefault("reviews", {})[item_id] = {"rating": normalized, "due_at": card.due.isoformat(), "scheduled_days": scheduled_days}
    complete = len(state["reviews"]) == len(item_ids)
    now = utc_now()
    DB.execute(
        "UPDATE study_sessions SET state_json=?,status=?,updated_at=?,completed_at=? WHERE id=?",
        (json_dump(state), "complete" if complete else "active", now, now if complete else None, session_id),
    )
    return {"card_id": item_id, "rating": normalized, "due_at": card.due.isoformat(), "scheduled_days": scheduled_days, "session": get_session(session_id)}


def suspend_flashcard(artifact_id: str, card_id: str) -> None:
    _, _, items = _artifact(artifact_id, "flashcard")
    if card_id not in {item.get("id") for item in items}:
        raise ValueError("闪卡不存在")
    card, last_rating = _load_card_state(artifact_id, card_id)
    _save_card_state(artifact_id, card_id, card, last_rating)
    DB.execute("UPDATE flashcard_states SET suspended=1,updated_at=? WHERE artifact_id=? AND card_id=?", (utc_now(), artifact_id, card_id))


def flashcards_csv(artifact_id: str) -> str:
    _, _, items = _artifact(artifact_id, "flashcard")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["front", "back", "explanation", "citations"])
    for item in items:
        writer.writerow([item.get("front", ""), item.get("back", ""), item.get("explanation", ""), " ".join(item.get("citations") or [])])
    return output.getvalue()
