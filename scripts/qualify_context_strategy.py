#!/usr/bin/env python3
"""Validate local paired reviews, optionally register one Provider × feature.

No model calls. Dry run by default. Raw sources and evaluation artifacts stay local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sandevistan_read.context_qualification import assess, fingerprint
from sandevistan_read.context_budget import CONTEXT_STRATEGY_VERSION
from sandevistan_read.database import DB, json_dump, json_load, utc_now
from sandevistan_read.providers import provider_by_id


def validate_artifacts(report: dict, directory: Path) -> list[str]:
    errors = []
    artifacts = report.get("artifacts", {})
    required = {"reference", "baseline", "candidate", "fixture", "review"}
    if not required <= artifacts.keys():
        return ["Reference, source manifests, fixture and detailed review artifacts are required"]
    for name, entry in artifacts.items():
        path = (directory / entry["path"]).resolve()
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            errors.append(f"Artifact missing or changed: {name}")
        if name != "review" and entry.get("sha256") != report.get(f"{name}_sha256"):
            errors.append(f"Artifact identity differs from report: {name}")
    # The candidate source manifest must describe the code being activated.
    candidate = artifacts.get("candidate", {})
    path = directory / candidate.get("path", "")
    if path.is_file():
        manifest = json.loads(path.read_text())
        root = Path(__file__).resolve().parents[1]
        actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (root / "src").rglob("*.py")}
        if manifest != actual:
            errors.append("Installed candidate source differs from qualified snapshot")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--provider-id", required=True)
    parser.add_argument("--kind", required=True, choices=["summary", "chat"])
    parser.add_argument("--apply", action="store_true", help="Register a passing report in local app settings")
    args = parser.parse_args()
    provider = provider_by_id(args.provider_id)
    if not provider or provider.get("role") != "main":
        parser.error("A configured MAIN provider is required")
    raw = args.report.read_bytes()
    report = json.loads(raw)
    result = assess(report, provider, args.kind)
    result["reasons"].extend(validate_artifacts(report, args.report.resolve().parent))
    result["passed"] = not result["reasons"]
    result["applied"] = False
    if result["passed"] and args.apply:
        key = fingerprint(provider)
        row = DB.fetchone("SELECT value_json FROM app_settings WHERE key='context_qualifications'")
        records = json_load(row["value_json"], []) if row else []
        if not isinstance(records, list):
            raise ValueError("Invalid qualification registry")
        records = [r for r in records if not (r.get("fingerprint") == key and r.get("kind") == args.kind)]
        records.append({"fingerprint": key, "kind": args.kind, "passed": True,
                        "strategy_version": CONTEXT_STRATEGY_VERSION,
                        "report_sha256": hashlib.sha256(raw).hexdigest(), "qualified_at": utc_now()})
        DB.execute("INSERT OR REPLACE INTO app_settings(key,value_json,updated_at) VALUES(?,?,?)",
                   ("context_qualifications", json_dump(records), utc_now()))
        result["applied"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
