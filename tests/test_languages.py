from sandevistan_read.database import Database, json_dump
from sandevistan_read.languages import resolve_output_language


def _database(tmp_path) -> Database:
    database = Database(tmp_path / "language.sqlite3")
    database.initialize()
    database.execute("INSERT INTO notebooks(id,title,description,created_at,updated_at) VALUES('n1','Languages','','now','now')")
    return database


def _source(database: Database, source_id: str, metadata: dict, chunks: list[str]) -> None:
    database.execute(
        """INSERT INTO sources(id,notebook_id,revision_id,filename,media_type,size_bytes,sha256,blob_path,preview_path,state,selected,page_count,parser,error,metadata_json,created_at,updated_at)
        VALUES(?, 'n1', ?, 'source.md', 'text/markdown', 1, ?, 'blob', NULL, 'ready', 1, 1, 'text', NULL, ?, 'now', 'now')""",
        (source_id, f"r-{source_id}", f"h-{source_id}", json_dump(metadata)),
    )
    for index, content in enumerate(chunks):
        database.execute(
            "INSERT INTO chunks(id,source_id,source_revision_id,ordinal,content,locator_json,embedding_json,checksum,created_at) VALUES(?,?,?,?,?,'{}','[]',?,'now')",
            (f"c-{source_id}-{index}", source_id, f"r-{source_id}", index, content, f"hc-{source_id}-{index}"),
        )


def test_auto_language_prefers_declared_metadata(tmp_path) -> None:
    database = _database(tmp_path)
    _source(database, "s1", {"language": "en-US"}, ["中文正文不应覆盖明确的元数据。"])
    language, selection = resolve_output_language(database, ["s1"], "auto")
    assert language == "en"
    assert selection["source_votes"]["en"] == 1


def test_auto_language_samples_body_when_metadata_missing(tmp_path) -> None:
    database = _database(tmp_path)
    _source(database, "s1", {}, ["A long English passage about systems and evidence." * 20])
    language, selection = resolve_output_language(database, ["s1"], "auto")
    assert language == "en"
    assert selection["fallback"] is False
