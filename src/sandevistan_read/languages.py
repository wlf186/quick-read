from __future__ import annotations

import re
from typing import Any

from .database import Database, json_load


def _text_vote(text: str) -> tuple[str | None, int, int]:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cjk == latin == 0:
        return None, cjk, latin
    return ("zh-CN" if cjk >= latin else "en"), cjk, latin


def resolve_output_language(database: Database, source_ids: list[str], requested: str) -> tuple[str, dict[str, Any]]:
    if requested in {"zh-CN", "en"}:
        return requested, {"requested": requested, "effective": requested, "source_votes": {"zh-CN": 0, "en": 0, "unknown": 0}, "fallback": False}
    votes = {"zh-CN": 0, "en": 0, "unknown": 0}
    aggregate_cjk = aggregate_latin = 0
    for source_id in source_ids:
        row = database.fetchone("SELECT metadata_json FROM sources WHERE id=?", (source_id,)) or {}
        declared = str(json_load(row.get("metadata_json"), {}).get("language") or "").lower()
        vote: str | None = "en" if declared.startswith("en") else "zh-CN" if declared.startswith("zh") else None
        if vote is None:
            chunks = database.fetchall("SELECT content FROM chunks WHERE source_id=? ORDER BY ordinal", (source_id,))
            if chunks:
                indexes = sorted({0, len(chunks) // 2, len(chunks) - 1})
                sample = "\n".join(str(chunks[index].get("content") or "")[:3000] for index in indexes)
                vote, cjk, latin = _text_vote(sample)
                aggregate_cjk += cjk
                aggregate_latin += latin
        votes[vote or "unknown"] += 1
    if votes["en"] != votes["zh-CN"]:
        effective = "en" if votes["en"] > votes["zh-CN"] else "zh-CN"
        fallback = False
    elif aggregate_latin != aggregate_cjk:
        effective = "en" if aggregate_latin > aggregate_cjk else "zh-CN"
        fallback = False
    else:
        effective, fallback = "zh-CN", True
    return effective, {"requested": requested, "effective": effective, "source_votes": votes, "fallback": fallback}


def text_matches_language(text: str, language: str) -> bool:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    significant = cjk + latin
    if significant < 12:
        return True
    return cjk >= max(4, latin // 3) if language != "en" else latin >= max(8, cjk * 2)
