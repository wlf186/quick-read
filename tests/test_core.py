from pathlib import Path

from sandevistan_read.documents import chunk_blocks, parse_text, sanitize_filename
from sandevistan_read.retrieval import reciprocal_rank_fusion


def test_filename_is_confined():
    assert sanitize_filename("../../secret.md") == "secret.md"


def test_text_parse_and_chunk(tmp_path: Path):
    path = tmp_path / "demo.md"
    path.write_text("# 标题\n" + "本地可追溯资料。" * 500, encoding="utf-8")
    parsed = parse_text(path, ".md")
    chunks = chunk_blocks(parsed.blocks)
    assert parsed.blocks and len(chunks) > 1
    assert all(chunk.locator["section"] == "标题" for chunk in chunks)


def test_rrf_combines_rankings():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert scores["a"] == scores["b"]
