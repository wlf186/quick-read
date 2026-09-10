"""Bounded native-Ollama response shapes; stored Podcast artifacts stay compatible."""
from typing import Any


def record(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def string() -> dict[str, Any]:
    return {"type": "string"}


def array(items: dict[str, Any], maximum: int, minimum: int = 0) -> dict[str, Any]:
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


def scene_schema(role: str, claim_ids: list[str], chapter_ids: list[str] | None = None, exchange_count: int = 2) -> dict[str, Any]:
    turn = record({"speaker": {"type": "string", "enum": ["A", "B"]},
                   "act_code": {"type": "string", "enum": list("IFBQAXEMCSO")},
                   "text": {"type": "string", "minLength": 1},
                   "claim_ids": array({"type": "string", "enum": claim_ids}, 6),
                   "example_kind": {"type": "string", "enum": ["none", "source", "illustrative"]}})
    exchange = array(turn, 4, 2)
    if chapter_ids:
        return record({"opening": exchange, "chapter_bodies": record({key: array(turn, 2, 2) for key in chapter_ids}), "closing": exchange})
    if role == "core":
        return record({key: exchange for key in ("opening", "body", "closing")})
    return record({"exchanges": array(record({"turns": exchange}), max(1, min(6, exchange_count)), 1)})


def notes_schema() -> dict[str, Any]:
    return record({"notes": array(record({key: string() for key in
        ("chunk_id", "claim", "qualification", "statement_kind", "quote")}), 8)})


def plan_schema(maximum: int, unit_ids: list[str] | None = None) -> dict[str, Any]:
    chapter = record({**{key: string() for key in ("title", "purpose", "tension", "bridge_in", "bridge_out", "new_information", "question", "mechanism", "required_conditions", "optional_example")},
                      "lead_host": {"type": "string", "enum": ["HOST_A", "HOST_B"]},
                      "claim_ids": array(string(), 10, 1)})
    properties = {"episode_thesis": string(), "answer_path": string(), "chapters": array(chapter, maximum, 1)}
    if unit_ids:
        properties["assignments"] = record({key: {"type": "integer", "minimum": 1, "maximum": maximum} for key in unit_ids})
    return record(properties)


def audit_schema() -> dict[str, Any]:
    verdict = {"type": "string", "enum": ["connected", "broken", "uncertain"]}
    closure = record({"opening_quote": string(), "closing_quote": string(), "verdict": verdict, "reason": string()})
    check = record({"index": {"type": "integer"}, "question_quote": string(), "answer_quote": string(), "verdict": verdict, "reason": string()})
    fact = record({"index": {"type": "integer"}, "claim_id": string(), "script_quote": string(), "source_quote": string(),
                   "verdict": {"type": "string", "enum": ["supported", "contradicted", "uncertain"]}, "reason": string()})
    duplicate = record({"index": {"type": "integer"}, "prior_index": {"type": "integer"},
                        "quote": string(), "prior_quote": string(), "reason": string()})
    return record({"closure": closure, "checks": array(check, 12), "facts": array(fact, 12), "duplicates": array(duplicate, 6)})
