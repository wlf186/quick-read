import importlib.util
import sqlite3
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["evaluate_generation", "evaluate_context_strategy"])
def test_fresh_checkout_creates_shared_lock_and_serializes_calls(tmp_path, monkeypatch, script):
    spec = importlib.util.spec_from_file_location(script, Path(__file__).parents[1] / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path / "checkout")
    first = module.lock_database(".context-audio.sqlite")
    second = module.lock_database(".context-audio.sqlite")
    try:
        first.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            second.execute("BEGIN IMMEDIATE")
        first.rollback()
        second.execute("BEGIN IMMEDIATE")
    finally:
        first.close()
        second.close()
