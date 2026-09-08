from __future__ import annotations

import hashlib
import html
import base64
import io
import os
import posixpath
import re
import shutil
import subprocess
import uuid
import zipfile
from datetime import date, datetime, time as datetime_time
from contextvars import ContextVar
from itertools import zip_longest
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote
from typing import Any
from xml.etree import ElementTree

import fitz
from bs4 import BeautifulSoup
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from pptx import Presentation
from PIL import Image

from .config import CONFIG
from .paths import PATHS


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".epub", ".txt", ".md", ".markdown", ".html", ".htm", ".png", ".jpg", ".jpeg", ".webp"}
EPUB_MAX_ENTRIES = 10_000
EPUB_MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
EPUB_MAX_TEXT_MEMBER_BYTES = 32 * 1024 * 1024
_PARSE_TIMINGS: ContextVar[dict[str, float] | None] = ContextVar("document_parse_timings", default=None)


@dataclass
class ParsedBlock:
    text: str
    locator: dict[str, Any]
    image_path: str | None = None
    visual_needed: bool = False


@dataclass
class ParsedDocument:
    blocks: list[ParsedBlock] = field(default_factory=list)
    page_count: int = 0
    parser: str = ""
    preview_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def sanitize_filename(filename: str) -> str:
    filename = Path(filename).name.replace("\x00", "")
    cleaned = re.sub(r"[^\w.()\-\u4e00-\u9fff ]+", "_", filename, flags=re.UNICODE).strip(" .")
    return cleaned[:180] or "document"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _render_pdf_page(page: fitz.Page, destination: Path) -> str:
    started = perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    pixmap.save(destination)
    if (timings := _PARSE_TIMINGS.get()) is not None:
        timings["render_seconds"] = timings.get("render_seconds", 0.0) + perf_counter() - started
    return str(destination.relative_to(PATHS.root))


def _store_visual(data: bytes, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.convert("RGB").save(destination, "PNG")
    except Exception as exc:
        raise ValueError("图片格式无法解码") from exc
    return str(destination.relative_to(PATHS.root))


def _store_svg(data: bytes, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if len(data) > 16 * 1024 * 1024 or re.search(br"<!DOCTYPE|<script|<foreignObject", data, re.I):
            raise ValueError("SVG 包含不允许的内容")
        if re.search(br"(?:href|xlink:href)\s*=\s*['\"](?!data:|#)", data, re.I):
            raise ValueError("SVG 不允许引用外部资源")
        with fitz.open(stream=data, filetype="svg") as document:
            pixmap = document[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            pixmap.save(destination)
    except Exception as exc:
        raise ValueError("SVG 无法安全渲染") from exc
    return str(destination.relative_to(PATHS.root))


def _page_needs_vision(page: fitz.Page, text: str) -> bool:
    """Ignore only unmistakable backgrounds/rules; uncertain graphics stay visual."""
    # get_images() includes shared resources that are never painted on this page.
    if len(text) < 80 or page.get_image_info():
        return True
    bounds = page.rect
    uncertain = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        simple = len(drawing["items"]) == 1 and drawing["items"][0][0] in {"re", "l"}
        background = simple and rect.contains(bounds + (2, 2, -2, -2))
        horizontal_rule = simple and rect.height <= 3 and rect.width >= bounds.width * 0.7
        margin_rule = simple and min(rect.width, rect.height) <= 6 and (
            rect.y1 <= bounds.height * 0.04 or rect.y0 >= bounds.height * 0.96)
        if not (background or horizontal_rule or margin_rule):
            uncertain.append(drawing)
    if len(uncertain) == 1:
        drawing = uncertain[0]
        rect = drawing["rect"]
        # A lone short horizontal accent beneath text, without chart axes or
        # other shapes, is a heading underline. Bars in plots remain uncertain.
        if (len(drawing["items"]) == 1 and drawing["items"][0][0] == "re"
                and 0 < rect.height <= 3 and 0 < rect.width < bounds.width * 0.2
                and any(abs(block[0] - rect.x0) <= 3 and 0 <= rect.y0 - block[3] <= 24
                        for block in page.get_text("blocks") if len(block) > 4)):
            return False
    return bool(uncertain)


def parse_pdf(path: Path, source_id: str) -> ParsedDocument:
    document = fitz.open(path)
    result = ParsedDocument(page_count=len(document), parser="pymupdf", preview_path=str(path.relative_to(PATHS.root)), metadata={"locator_unit": "page"})
    render_dir = PATHS.renders / source_id
    for index, page in enumerate(document):
        blocks = page.get_text("blocks", sort=True)
        text = _clean_text("\n".join(str(block[4]) for block in blocks if len(block) > 4))
        visual_needed = _page_needs_vision(page, text)
        image_path = None
        if visual_needed:
            image_path = _render_pdf_page(page, render_dir / f"page-{index + 1:04d}.png")
        if text:
            locator = {
                "kind": "page",
                "page": index + 1,
                "bboxes": [list(map(float, block[:4])) for block in blocks[:24]],
            }
            result.blocks.append(ParsedBlock(text=text, locator=locator, image_path=image_path, visual_needed=visual_needed))
        elif image_path:
            result.blocks.append(
                ParsedBlock(text="", locator={"kind": "page", "page": index + 1, "bboxes": []}, image_path=image_path, visual_needed=True)
            )
    document.close()
    return result


def _convert_office_to_pdf(path: Path, source_id: str) -> Path | None:
    executable = CONFIG.tools.libreoffice_path
    if not executable:
        return None
    output_dir = PATHS.renders / source_id / "office-preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = PATHS.libreoffice_profiles / f"{source_id}-{uuid.uuid4().hex}"
    profile.mkdir(parents=True, exist_ok=False)
    command = [
        executable,
        f"-env:UserInstallation={profile.as_uri()}",
        "--headless",
        "--nologo",
        "--nodefault",
        "--nofirststartwizard",
        "--nolockcheck",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(path),
    ]
    environment = os.environ.copy()
    environment["SAL_USE_VCLPLUGIN"] = "svp"
    try:
        completed = subprocess.run(command, capture_output=True, timeout=180, check=False, env=environment)
        converted = output_dir / f"{path.stem}.pdf"
        return converted if completed.returncode == 0 and converted.exists() else None
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def _office_preview(path: Path, source_id: str, metadata: dict[str, Any]) -> Path | None:
    started = perf_counter()
    try:
        preview = _convert_office_to_pdf(path, source_id)
        if preview:
            with fitz.open(preview) as pdf:
                if not len(pdf):
                    raise ValueError("预览 PDF 没有页面")
            return preview
        reason = "converter_unavailable_or_failed"
    except (OSError, subprocess.SubprocessError, UnicodeError, RuntimeError, ValueError) as exc:
        reason = type(exc).__name__
    finally:
        metadata.setdefault("ingest_timings", {})["office_conversion_seconds"] = round(perf_counter() - started, 4)
    metadata.setdefault("warnings", []).append({"code": "office_preview_unavailable", "reason": reason,
        "message": "正文已提取，预览转换未完成；依赖预览的图表或图片可能未被识别。"})
    return None


def parse_docx(path: Path, source_id: str) -> ParsedDocument:
    document = Document(path)
    blocks: list[ParsedBlock] = []
    section = "文档"
    ordinal = 0
    table_index = 0
    for element in document.element.body:
        if element.tag.endswith("}tbl"):
            table_index += 1
            table = Table(element, document)
            rows = [" | ".join(_clean_text(cell.text) for cell in row.cells) for row in table.rows]
            if any(rows):
                blocks.append(ParsedBlock("\n".join(rows), {"kind": "table", "table": table_index, "section": section, "structured": True}))
            continue
        if not element.tag.endswith("}p"):
            continue
        paragraph = Paragraph(element, document)
        text = _clean_text(paragraph.text)
        if not text:
            continue
        style = paragraph.style.name.lower() if paragraph.style else ""
        if "heading" in style or "标题" in style:
            section = text
        blocks.append(ParsedBlock(text=text, locator={"kind": "section", "section": section, "paragraph": ordinal + 1}))
        ordinal += 1
    metadata: dict[str, Any] = {"locator_unit": "section"}
    preview = _office_preview(path, source_id, metadata)
    page_count = 0
    if preview:
        with fitz.open(preview) as pdf:
            page_count = len(pdf)
            render_dir = PATHS.renders / source_id
            for index, page in enumerate(pdf):
                if not _page_needs_vision(page, _clean_text(page.get_text())):
                    continue
                image_path = _render_pdf_page(page, render_dir / f"page-{index + 1:04d}.png")
                blocks.append(ParsedBlock(text="", locator={"kind": "page", "page": index + 1, "visual_only": True}, image_path=image_path, visual_needed=True))
    return ParsedDocument(
        blocks=blocks,
        page_count=page_count or max(1, len(document.sections)),
        parser="python-docx+libreoffice" if preview else "python-docx",
        preview_path=str(preview.relative_to(PATHS.root)) if preview else None,
        metadata={**metadata, "visual_preview": bool(preview)},
    )


def parse_pptx(path: Path, source_id: str) -> ParsedDocument:
    presentation = Presentation(path)
    blocks: list[ParsedBlock] = []
    for slide_number, slide in enumerate(presentation.slides, start=1):
        parts: list[str] = []
        image_count = 0
        def shapes_in(shapes: Any) -> Any:
            for shape in shapes:
                yield shape
                if hasattr(shape, "shapes"):
                    yield from shapes_in(shape.shapes)

        for shape in shapes_in(slide.shapes):
            if getattr(shape, "has_text_frame", False):
                value = _clean_text(shape.text)
                if value:
                    parts.append(value)
            if getattr(shape, "shape_type", None) == 13:
                image_count += 1
            if getattr(shape, "has_chart", False):
                image_count += 1
            if getattr(shape, "has_table", False):
                rows = [" | ".join(_clean_text(cell.text) for cell in row.cells) for row in shape.table.rows]
                table_text = _clean_text("\n".join(rows))
                if table_text:
                    parts.append("[表格]\n" + table_text)
        text = _clean_text("\n".join(parts))
        blocks.append(
            ParsedBlock(
                text=text,
                locator={"kind": "slide", "slide": slide_number},
                visual_needed=image_count > 0 or len(text) < 80,
            )
        )
    metadata: dict[str, Any] = {"locator_unit": "slide"}
    preview = _office_preview(path, source_id, metadata)
    if preview:
        with fitz.open(preview) as pdf:
            render_dir = PATHS.renders / source_id
            for index, page in enumerate(pdf):
                if index < len(blocks) and blocks[index].visual_needed:
                    blocks[index].image_path = _render_pdf_page(page, render_dir / f"slide-{index + 1:04d}.png")
    return ParsedDocument(
        blocks=blocks,
        page_count=len(presentation.slides),
        parser="python-pptx+libreoffice" if preview else "python-pptx",
        preview_path=str(preview.relative_to(PATHS.root)) if preview else None,
        metadata={**metadata, "visual_preview": bool(preview)},
    )


def _spreadsheet_value(cell: Any, cached: Any) -> str:
    value = cached.value if cell.data_type == "f" else cell.value
    if value is None:
        return f"[公式未缓存：{cell.value}]" if cell.data_type == "f" else ""
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and "%" in cell.number_format:
        decimals = re.search(r"\.([0#]+)%", cell.number_format)
        return f"{value * 100:.{len(decimals[1]) if decimals else 0}f}%"
    return _clean_text(str(value))


def _cached_chart_block(xml: bytes, name: str) -> ParsedBlock | None:
    """Keep complete standard chart caches as native evidence; otherwise render."""
    root = ElementTree.fromstring(xml)
    ns = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    plot = root.find(".//c:plotArea", ns)
    if plot is None:
        return None
    charts = [item for item in plot if item.tag.rsplit("}", 1)[-1].endswith("Chart")]
    if len(charts) != 1 or charts[0].tag.rsplit("}", 1)[-1] not in {"barChart", "lineChart", "pieChart"}:
        return None
    chart = charts[0]
    # Cached series do not represent fitted trends, uncertainty bars or custom
    # labels. Keep the visual path for those additional chart semantics.
    if any(node.tag.rsplit("}", 1)[-1] in {"trendline", "errBars", "dLbl", "extLst"}
           for node in chart.iter()):
        return None
    title = " · ".join(node.text or "" for node in root.findall(".//c:title//a:t", ns)) or "图表"
    lines = [title, "类别 | 系列 | 图表缓存原始数值"]
    refs = [node.text or "" for node in chart.findall(".//c:f", ns)]

    def values(parent: Any) -> list[str] | None:
        if parent is None:
            return None
        cache = next((item for item in parent.iter() if item.tag.rsplit("}", 1)[-1] in {"strCache", "numCache", "strLit", "numLit"}), None)
        if cache is None:
            return None
        points = cache.findall("c:pt", ns)
        count = cache.find("c:ptCount", ns)
        if count is None or count.get("val") != str(len(points)) or not points:
            return None
        by_index = {int(item.attrib["idx"]): item.findtext("c:v", default="", namespaces=ns) for item in points}
        if set(by_index) != set(range(len(points))) or not all(by_index.values()):
            return None
        return [by_index[index] for index in range(len(points))]

    series = chart.findall("c:ser", ns)
    if not series:
        return None
    for item in series:
        categories, numbers = values(item.find("c:cat", ns)), values(item.find("c:val", ns))
        if not categories or not numbers or len(categories) != len(numbers):
            return None
        label = item.findtext("c:tx//c:v", default="数值", namespaces=ns)
        lines.extend(f"{category} | {label} | {number}" for category, number in zip(categories, numbers))
    lines.append("图表原始数值按原文件保存；百分比显示请对照工作表单元格。")
    sheet = refs[0].split("!")[0].strip("'").replace("''", "'") if refs and "!" in refs[0] else ""
    return ParsedBlock("\n".join(lines), {"kind": "chart", "chart": name, "sheet": sheet,
        "data_ranges": refs, "structured": True})


def parse_xlsx(path: Path, source_id: str) -> ParsedDocument:
    """Read values and formula caches without evaluating formulas or external links."""
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > EPUB_MAX_ENTRIES or sum(member.file_size for member in members) > EPUB_MAX_UNCOMPRESSED_BYTES:
            raise ValueError("XLSX 解压规模超过解析限制")
        if any(member.filename.endswith(".xml") and member.file_size > EPUB_MAX_TEXT_MEMBER_BYTES for member in members):
            raise ValueError("XLSX XML 部件超过解析限制")
        relations = {item.attrib["Id"]: item.attrib.get("Target", "") for item in
            ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            if item.attrib.get("TargetMode") != "External"}
        merges: dict[str, list[str]] = {}
        for sheet in ElementTree.fromstring(archive.read("xl/workbook.xml")).iter():
            if sheet.tag.rsplit("}", 1)[-1] != "sheet":
                continue
            relationship = next((value for key, value in sheet.attrib.items() if key.endswith("}id")), "")
            target = relations.get(relationship, "")
            member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            if member in archive.namelist():
                merges[sheet.attrib["name"]] = [element.attrib["ref"] for element in
                    ElementTree.fromstring(archive.read(member)).iter()
                    if element.tag.rsplit("}", 1)[-1] == "mergeCell"]
        native_charts = []
        has_visuals = any(member.filename.startswith("xl/media/") for member in members)
        for member in members:
            if re.fullmatch(r"xl/charts/chart\d+\.xml", member.filename):
                try:
                    block = _cached_chart_block(archive.read(member), Path(member.filename).name)
                except (ValueError, KeyError, ElementTree.ParseError):
                    block = None
                if block:
                    native_charts.append(block)
                else:
                    has_visuals = True
            elif re.fullmatch(r"xl/drawings/drawing\d+\.xml", member.filename):
                if any(element.tag.rsplit("}", 1)[-1] in {"sp", "cxnSp", "pic"} for element in ElementTree.fromstring(archive.read(member.filename)).iter()):
                    has_visuals = True
    result = ParsedDocument(parser="openpyxl", metadata={"locator_unit": "sheet", "merged_cells": merges})
    formulas = load_workbook(path, read_only=True, data_only=False, keep_links=False)
    try:
        cached = load_workbook(path, read_only=True, data_only=True, keep_links=False)
        try:
            result.page_count = len(formulas.worksheets)
            missing = 0
            for sheet in formulas:
                rows: list[tuple[int, str]] = []
                max_column = 1
                # Ignore inflated producer dimensions; iterate actual worksheet XML.
                sheet.reset_dimensions()
                cached_sheet = cached[sheet.title]
                cached_sheet.reset_dimensions()
                for row_number, (row, cached_row) in enumerate(zip_longest(sheet.rows, cached_sheet.rows, fillvalue=()), 1):
                    values = []
                    for column, (cell, value_cell) in enumerate(zip(row, cached_row), 1):
                        if cell.value is None:
                            continue
                        value = _spreadsheet_value(cell, value_cell)
                        missing += int(cell.data_type == "f" and value_cell.value is None)
                        values.append(f"{get_column_letter(column)}{row_number}: {value}")
                        max_column = max(max_column, column)
                    if values:
                        rows.append((row_number, " | ".join(values)))
                if not rows:
                    continue
                header = f"工作表：{sheet.title}\n表头/起始行：" + "\n".join(value for _, value in rows[:2])
                if merges.get(sheet.title):
                    header += "\n合并单元格：" + ", ".join(merges[sheet.title])
                batch: list[tuple[int, str]] = []
                size = len(header)

                def flush() -> None:
                    if batch:
                        result.blocks.append(ParsedBlock(header + "\n" + "\n".join(value for _, value in batch),
                            {"kind": "sheet", "sheet": sheet.title, "cell_range": f"A{batch[0][0]}:{get_column_letter(max_column)}{batch[-1][0]}",
                             "header_range": f"A{rows[0][0]}:{get_column_letter(max_column)}{rows[min(1, len(rows) - 1)][0]}", "structured": True}))

                for row in rows:
                    if batch and size + len(row[1]) > 1700:
                        flush()
                        batch = []
                        size = len(header)
                    batch.append(row)
                    size += len(row[1]) + 1
                flush()
            if missing:
                result.metadata.setdefault("warnings", []).append({"code": "formula_cache_missing", "count": missing,
                    "message": "部分公式没有保存计算结果，已保留公式并标注，未推算数值。"})
        finally:
            cached.close()
    finally:
        formulas.close()
    result.blocks.extend(native_charts)
    result.metadata["native_charts"] = len(native_charts)
    if has_visuals:
        preview = _office_preview(path, source_id, result.metadata)
        if preview:
            result.preview_path = str(preview.relative_to(PATHS.root))
            with fitz.open(preview) as pdf:
                for index, page in enumerate(pdf):
                    if _page_needs_vision(page, _clean_text(page.get_text())):
                        image = _render_pdf_page(page, PATHS.renders / source_id / f"preview-{index + 1:04d}.png")
                        result.blocks.append(ParsedBlock("", {"kind": "page", "page": index + 1, "visual_only": True}, image, True))
    return result


def _epub_member(base: str, href: str) -> str:
    href = unquote(href.split("#", 1)[0]).replace("\\", "/")
    member = posixpath.normpath(posixpath.join(base, href))
    if not member or member.startswith("../") or member.startswith("/") or "/../" in f"/{member}/":
        raise ValueError("EPUB contains an unsafe resource path")
    return member


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_epub(path: Path, source_id: str = "preview") -> ParsedDocument:
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError("EPUB 文件损坏或不是有效 ZIP 容器") from exc
    with archive:
        entries = archive.infolist()
        if len(entries) > EPUB_MAX_ENTRIES or sum(item.file_size for item in entries) > EPUB_MAX_UNCOMPRESSED_BYTES:
            raise ValueError("EPUB 解压后体积或文件数量超过安全限制")
        names = {item.filename for item in entries}
        if "META-INF/container.xml" not in names:
            raise ValueError("EPUB 缺少 META-INF/container.xml")
        try:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            rootfiles = [node.attrib.get("full-path", "") for node in container.iter() if _xml_local_name(node.tag) == "rootfile"]
        except ElementTree.ParseError as exc:
            raise ValueError("EPUB container.xml 无法解析") from exc
        opf_name = next((name for name in rootfiles if name in names), "")
        if not opf_name:
            raise ValueError("EPUB 未找到 package document")
        try:
            package = ElementTree.fromstring(archive.read(opf_name))
        except ElementTree.ParseError as exc:
            raise ValueError("EPUB package document 无法解析") from exc
        base = posixpath.dirname(opf_name)
        manifest: dict[str, dict[str, str]] = {}
        spine: list[str] = []
        metadata: dict[str, Any] = {"locator_unit": "chapter"}
        creators: list[str] = []
        for node in package.iter():
            local = _xml_local_name(node.tag)
            if local == "item" and node.attrib.get("id"):
                manifest[node.attrib["id"]] = dict(node.attrib)
            elif local == "itemref" and node.attrib.get("idref"):
                spine.append(node.attrib["idref"])
            elif local in {"title", "language"} and (node.text or "").strip() and local not in metadata:
                metadata[local] = _clean_text(node.text or "")
            elif local == "creator" and (node.text or "").strip():
                creators.append(_clean_text(node.text or ""))
        if creators:
            metadata["creators"] = creators
        encrypted: set[str] = set()
        if "META-INF/encryption.xml" in names:
            try:
                encryption = ElementTree.fromstring(archive.read("META-INF/encryption.xml"))
                encrypted = {unquote(node.attrib.get("URI", "")) for node in encryption.iter() if _xml_local_name(node.tag) == "CipherReference"}
            except ElementTree.ParseError:
                encrypted = set()
        blocks: list[ParsedBlock] = []
        for spine_index, item_id in enumerate(spine, start=1):
            item = manifest.get(item_id, {})
            href = item.get("href", "")
            media_type = item.get("media-type", "")
            if not href or media_type not in {"application/xhtml+xml", "text/html"}:
                continue
            member = _epub_member(base, href)
            if member not in names:
                continue
            if href in encrypted or member in encrypted:
                raise ValueError("EPUB 正文受 DRM/加密保护，无法在本地解析")
            info = archive.getinfo(member)
            if info.file_size > EPUB_MAX_TEXT_MEMBER_BYTES:
                raise ValueError("EPUB 单个正文文件超过安全限制")
            raw = archive.read(member).decode("utf-8", errors="replace")
            soup = BeautifulSoup(raw, "html.parser")
            for image_index, image in enumerate(soup.find_all("img"), start=1):
                src = str(image.get("src") or "")
                if not src or src.startswith(("http://", "https://")):
                    continue
                try:
                    image_member = _epub_member(posixpath.dirname(member), src)
                    if image_member not in names or image_member in encrypted:
                        continue
                    image_data = archive.read(image_member)
                    destination = PATHS.renders / source_id / f"epub-{spine_index:04d}-{image_index:04d}.png"
                    stored = _store_svg(image_data, destination) if image_member.lower().endswith(".svg") else _store_visual(image_data, destination)
                    blocks.append(ParsedBlock(
                        text="", image_path=stored, visual_needed=True,
                        locator={"kind": "epub-image", "spine": spine_index, "href": href, "asset": src, "visual_only": True},
                    ))
                except (KeyError, ValueError):
                    continue
            for svg_index, svg in enumerate(soup.find_all("svg"), start=1):
                try:
                    stored = _store_svg(str(svg).encode(), PATHS.renders / source_id / f"epub-svg-{spine_index:04d}-{svg_index:04d}.png")
                    blocks.append(ParsedBlock("", {"kind": "epub-svg", "spine": spine_index, "href": href, "visual_only": True}, stored, True))
                except ValueError:
                    continue
            for unsafe in soup(["script", "style", "iframe", "object", "embed", "svg"]):
                unsafe.decompose()
            heading = soup.find(re.compile(r"^h[1-6]$"))
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            section = _clean_text(heading.get_text(" ", strip=True) if heading else title) or f"章节 {spine_index}"
            text = _clean_text(soup.get_text("\n"))
            if text:
                blocks.append(ParsedBlock(text=text, locator={"kind": "epub", "spine": spine_index, "href": href, "section": section}))
        if not blocks:
            raise ValueError("EPUB 没有可读取的正文")
        metadata["spine_items"] = len(spine)
        try:
            preview_path = str(path.relative_to(PATHS.root))
        except ValueError:
            preview_path = None
        return ParsedDocument(blocks=blocks, page_count=len(blocks), parser="epub-spine", preview_path=preview_path, metadata=metadata)


def parse_text(path: Path, extension: str, source_id: str = "preview") -> ParsedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    visual_blocks: list[ParsedBlock] = []
    if extension in {".html", ".htm", ".md", ".markdown"}:
        pattern = re.compile(r"data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=\s]+)", re.I)
        for index, match in enumerate(pattern.finditer(raw), start=1):
            try:
                data = base64.b64decode(re.sub(r"\s+", "", match.group(2)), validate=True)
                stored = _store_visual(data, PATHS.renders / source_id / f"inline-{index:04d}.png")
                line = raw.count("\n", 0, match.start()) + 1
                visual_blocks.append(ParsedBlock("", {"kind": "inline-image", "line": line, "visual_only": True}, stored, True))
            except (ValueError, base64.binascii.Error):
                continue
        if extension in {".html", ".htm"}:
            inline_soup = BeautifulSoup(raw, "html.parser")
            for index, svg in enumerate(inline_soup.find_all("svg"), start=1):
                try:
                    stored = _store_svg(str(svg).encode(), PATHS.renders / source_id / f"inline-svg-{index:04d}.png")
                    visual_blocks.append(ParsedBlock("", {"kind": "inline-svg", "visual_only": True}, stored, True))
                except ValueError:
                    continue
    if extension in {".html", ".htm"}:
        soup = BeautifulSoup(raw, "html.parser")
        for unsafe in soup(["script", "style", "iframe", "object", "embed"]):
            unsafe.decompose()
        raw = soup.get_text("\n")
    lines = raw.splitlines()
    blocks: list[ParsedBlock] = []
    buffer: list[str] = []
    start_line = 1
    section = "文档"
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if extension in {".md", ".markdown"} and stripped.startswith("#"):
            if buffer:
                blocks.append(ParsedBlock(_clean_text("\n".join(buffer)), {"kind": "lines", "line_start": start_line, "line_end": line_number - 1, "section": section}))
                buffer = []
            section = stripped.lstrip("# ") or section
            start_line = line_number
        if not buffer:
            start_line = line_number
        buffer.append(line)
        if sum(len(item) for item in buffer) >= 2200:
            blocks.append(ParsedBlock(_clean_text("\n".join(buffer)), {"kind": "lines", "line_start": start_line, "line_end": line_number, "section": section}))
            buffer = []
    if buffer:
        blocks.append(ParsedBlock(_clean_text("\n".join(buffer)), {"kind": "lines", "line_start": start_line, "line_end": len(lines), "section": section}))
    return ParsedDocument(blocks=[block for block in blocks if block.text] + visual_blocks, page_count=max(1, len(blocks)), parser=f"text-{extension.lstrip('.')}", metadata={"locator_unit": "section", "inline_visuals": len(visual_blocks)})


def parse_image(path: Path, source_id: str) -> ParsedDocument:
    stored = _store_visual(path.read_bytes(), PATHS.renders / source_id / "image-0001.png")
    return ParsedDocument(
        blocks=[ParsedBlock("", {"kind": "image", "visual_only": True}, stored, True)],
        page_count=1,
        parser="pymupdf-image",
        preview_path=stored,
        metadata={"locator_unit": "image"},
    )


def parse_document(path: Path, source_id: str) -> ParsedDocument:
    measured: dict[str, float] = {}
    token = _PARSE_TIMINGS.set(measured)
    started = perf_counter()
    try:
        result = _parse_document(path, source_id)
        timings = result.metadata.setdefault("ingest_timings", {})
        timings.update({key: round(value, 4) for key, value in measured.items()})
        timings["native_seconds"] = round(max(0.0, perf_counter() - started
            - measured.get("render_seconds", 0.0) - timings.get("office_conversion_seconds", 0.0)), 4)
        return result
    finally:
        _PARSE_TIMINGS.reset(token)


def _parse_document(path: Path, source_id: str) -> ParsedDocument:
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {extension}")
    if extension == ".pdf":
        return parse_pdf(path, source_id)
    if extension == ".docx":
        return parse_docx(path, source_id)
    if extension == ".pptx":
        return parse_pptx(path, source_id)
    if extension == ".xlsx":
        return parse_xlsx(path, source_id)
    if extension == ".epub":
        return parse_epub(path, source_id)
    if extension in {".png", ".jpg", ".jpeg", ".webp"}:
        return parse_image(path, source_id)
    return parse_text(path, extension, source_id)


def chunk_blocks(blocks: list[ParsedBlock], target_chars: int = 1800, overlap_chars: int = 260) -> list[ParsedBlock]:
    grouped: list[ParsedBlock] = []
    for block in blocks:
        previous = grouped[-1] if grouped else None
        if (previous and block.locator.get("kind") == previous.locator.get("kind") == "section"
                and block.locator.get("section") == previous.locator.get("section")
                and block.locator.get("paragraph") and previous.locator.get("paragraph")
                and not block.image_path and not previous.image_path
                and len(previous.text) + len(block.text) + 1 <= target_chars):
            previous.text += "\n" + block.text
            previous.locator["paragraph_end"] = block.locator["paragraph"]
        else:
            grouped.append(ParsedBlock(block.text, dict(block.locator), block.image_path, block.visual_needed))
    chunks: list[ParsedBlock] = []
    for block in grouped:
        text = _clean_text(block.text)
        if not text:
            continue
        if len(text) <= target_chars:
            chunks.append(block)
            continue
        if block.locator.get("structured"):
            lines = text.splitlines()
            header = lines[0]
            batch = header
            for line in lines[1:]:
                if len(batch) + len(line) + 1 > target_chars and batch != header:
                    chunks.append(ParsedBlock(batch, dict(block.locator)))
                    batch = header
                batch += "\n" + line
            if batch.strip():
                chunks.append(ParsedBlock(batch, dict(block.locator)))
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + target_chars)
            if end < len(text):
                boundary = max(text.rfind("。", start, end), text.rfind("\n", start, end), text.rfind(". ", start, end))
                if boundary > start + target_chars // 2:
                    end = boundary + 1
            chunks.append(ParsedBlock(text=text[start:end].strip(), locator=dict(block.locator), image_path=block.image_path, visual_needed=False))
            if end >= len(text):
                break
            start = max(end - overlap_chars, start + 1)
    return chunks
