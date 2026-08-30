from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from sandevistan_read.documents import chunk_blocks, parse_epub


CONTAINER = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
PACKAGE = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
 <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>测试书</dc:title><dc:creator>本地作者</dc:creator><dc:language>zh</dc:language></metadata>
 <manifest><item id="b" href="b.xhtml" media-type="application/xhtml+xml"/><item id="a" href="a.xhtml" media-type="application/xhtml+xml"/></manifest>
 <spine><itemref idref="a"/><itemref idref="b"/></spine>
</package>"""


def make_epub(path: Path) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", CONTAINER)
        archive.writestr("OEBPS/content.opf", PACKAGE)
        archive.writestr("OEBPS/a.xhtml", "<html><head><title>A</title><script>bad()</script></head><body><h1>第一章</h1><p>第一章正文，可验证内容。</p></body></html>")
        archive.writestr("OEBPS/b.xhtml", "<html><head><title>B</title></head><body><h1>第二章</h1><p>第二章正文。</p></body></html>")


def test_epub_uses_spine_order_and_safe_text(tmp_path: Path):
    path = tmp_path / "book.epub"
    make_epub(path)
    parsed = parse_epub(path)
    assert parsed.parser == "epub-spine"
    assert parsed.page_count == 2
    assert parsed.metadata["title"] == "测试书"
    assert parsed.blocks[0].locator["section"] == "第一章"
    assert parsed.blocks[1].locator["section"] == "第二章"
    assert "bad()" not in parsed.blocks[0].text
    assert chunk_blocks(parsed.blocks)


def test_epub_rejects_missing_container(tmp_path: Path):
    path = tmp_path / "broken.epub"
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
    with pytest.raises(ValueError, match="container"):
        parse_epub(path)
