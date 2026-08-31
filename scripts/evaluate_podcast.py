#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sandevistan_read.database import DB, json_load
from sandevistan_read.paths import PATHS
from sandevistan_read.podcast import PodcastQualityError, _content_minutes, _repeated_stem_ratio, build_podcast_script


def transcript(payload: dict[str, Any]) -> str:
    turns = payload.get("turns") or []
    if turns:
        return "\n".join(f"{turn.get('speaker', '?')}: {turn.get('text', '')}" for turn in turns)
    return str(payload.get("script") or "")


def metrics(payload: dict[str, Any]) -> dict[str, Any]:
    turns = payload.get("turns") or []
    return {
        "version": payload.get("version", 1),
        "turn_count": len(turns),
        "estimated_minutes": round(_content_minutes(turns), 2) if turns else None,
        "repeated_stem_ratio": round(_repeated_stem_ratio(turns), 3) if turns and all("dialogue_act" in turn for turn in turns) else None,
        "cited_turn_ratio": round(sum(bool(turn.get("citation_ids")) for turn in turns) / max(1, len(turns)), 3),
        "quality": payload.get("quality") or {},
    }


def load_baseline(artifact_id: str | None) -> dict[str, Any] | None:
    if not artifact_id:
        return None
    row = DB.fetchone("SELECT payload_json FROM artifacts WHERE id=? AND type='podcast'", (artifact_id,))
    if not row:
        raise SystemExit(f"找不到 Podcast 产物：{artifact_id}")
    return json_load(row["payload_json"], {})


def review_sheet(items: list[tuple[str, str]]) -> str:
    rubric = """# Podcast blind review

请不要查看 `mapping.json`，先分别为 A/B 打分。每项 1–5 分：

- 连贯性：每轮是否回应上一轮，跨章是否自然。
- 自然度：是否像真实双人交流，而不是模板采访或事实清单。
- 资料忠实度：是否存在无依据的数字、实体、案例或因果。
- 角色稳定：两位主持人的职责是否清晰且不僵硬。
- 重复控制：是否反复重启话题、复述相同事实或使用相同句式。

记录格式：`A: 连贯/自然/忠实/角色/重复 = _/_/_/_/_`。
"""
    sections = "\n\n".join(f"## Transcript {label}\n\n{text}" for label, text in items)
    return rubric + "\n\n" + sections + "\n"


async def run(args: argparse.Namespace) -> tuple[Path, bool]:
    baseline = load_baseline(args.baseline_artifact)
    payload = {
        "source_ids": args.source_id or None,
        "duration_mode": "fixed",
        "minutes": args.minutes,
        "language": args.language,
        "focus": args.focus,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = PATHS.runtime / "evals" / f"podcast-v3-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    try:
        candidate = await build_podcast_script(args.notebook_id, payload)
    except PodcastQualityError as exc:
        failure = {"status": "failed", "message": str(exc), "quality_report": exc.report}
        (output / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        return output, False
    (output / "candidate.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = {"candidate": metrics(candidate), "baseline": metrics(baseline) if baseline else None}
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    if baseline:
        candidate_text, baseline_text = transcript(candidate), transcript(baseline)
        candidate_first = int(hashlib.sha256(candidate_text.encode()).hexdigest(), 16) % 2 == 0
        ordered = [("A", candidate_text), ("B", baseline_text)] if candidate_first else [("A", baseline_text), ("B", candidate_text)]
        mapping = {"A": "candidate" if candidate_first else "baseline", "B": "baseline" if candidate_first else "candidate"}
        (output / "blind-review.md").write_text(review_sheet(ordered), encoding="utf-8")
        (output / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, True


def main() -> None:
    parser = argparse.ArgumentParser(description="生成脚本级 Podcast V3 评测，不调用 TTS。")
    parser.add_argument("--notebook-id", required=True)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--minutes", type=int, choices=(5, 10, 20, 30), default=5)
    parser.add_argument("--language", choices=("auto", "zh-CN", "en"), default="zh-CN")
    parser.add_argument("--focus", default="")
    parser.add_argument("--baseline-artifact")
    args = parser.parse_args()
    destination, passed = asyncio.run(run(args))
    print(json.dumps({"status": "passed" if passed else "failed", "output": str(destination)}, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
