"""
extract/docx.py
---------------
Extract text chunks from DOCX files using python-docx.

Reading order:
  Walks document.element.body children in DOM order so paragraphs
  and tables are interleaved exactly as they appear in the document.

Covers:
  - Body paragraphs (headings, normal text)
  - Tables (row/col coordinates, header-row detection)
  - Headers and footers (from the first section)
  - Footnotes
  - Core properties (author, last_modified_by)
  - Embedded raster images (PNG/JPEG) → OCR
  - EMF/WMF/SVG: attempts PIL open; emits BLOCKED chunk if impossible
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import List, Optional
from lxml import etree

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from extract.models import Chunk
from extract.ocr import ocr_bytes

log = logging.getLogger(__name__)

# Word XML namespaces
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_REL_IMG = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"

# If a table's first row contains any of these keywords we treat it as a header row
_HEADER_KEYWORDS = {
    "name", "email", "phone", "employee", "contact", "owner", "approver", "id",
    "staff", "emp", "user", "users", "member", "members", "reviewer", "author",
    "assignee", "respondent", "custodian", "delegate", "stakeholder", "participant",
    "operator", "signoff", "lead", "manager", "attendee", "client contact",
    "peer reviewer", "qa reviewer"
}


def _row_is_header(row) -> bool:
    """Heuristic: first row of a table is a header if any cell text matches header keywords."""
    for cell in row.cells:
        if cell.text.strip().lower() in _HEADER_KEYWORDS:
            return True
    return False


def _table_to_chunks(table: Table, file: str, table_idx: int,
                     order_start: int, config: dict) -> List[Chunk]:
    """
    Render a DOCX table as Chunk objects.

    Each row becomes one Chunk. The location encodes table index and
    row number; the text is pipe-separated cell values with the header
    context prepended (if detected) for context-aware detection.
    """
    chunks: List[Chunk] = []
    order = order_start
    rows = table.rows
    if not rows:
        return chunks

    header_texts: List[str] = []
    header_row_idx: Optional[int] = None

    # Detect header row: either tblHeader XML property OR heuristic keyword match
    for ri, row in enumerate(rows):
        has_tbl_header = any(
            cell._tc.find(qn("w:trPr")) is not None and
            cell._tc.find(qn("w:trPr")).find(qn("w:tblHeader")) is not None
            for cell in row.cells
        )
        if has_tbl_header or (ri == 0 and _row_is_header(row)):
            header_texts = [c.text.strip() for c in row.cells]
            header_row_idx = ri
            break

    for ri, row in enumerate(rows):
        cell_texts = [c.text.strip() for c in row.cells]
        row_text = " | ".join(cell_texts)
        is_header = ri == header_row_idx

        # Build location string including column context for context.py
        location = f"table:{table_idx + 1}/r{ri + 1}"
        if header_texts:
            location += f"[headers: {' | '.join(header_texts)}]"

        chunks.append(Chunk(
            file=file,
            location=location,
            text=row_text,
            method="native",
            ocr_confidence=None,
            order_index=order,
        ))
        order += 1
    return chunks


def _extract_images(doc: Document, doc_path: str, file: str,
                    order_start: int, config: dict) -> List[Chunk]:
    """
    Extract all embedded raster images from the DOCX zip and OCR them.
    Returns Chunk objects with location="image:N".
    """
    chunks: List[Chunk] = []
    order = order_start

    profile_name = config.get("active_profile", "balanced")
    ocr_min_conf = config["profiles"][profile_name]["ocr_min_confidence"]

    raster_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"}

    try:
        with zipfile.ZipFile(doc_path, "r") as zf:
            image_names = [
                n for n in zf.namelist()
                if n.startswith("word/media/") and
                Path(n).suffix.lower() in raster_exts
            ]
            for img_idx, img_name in enumerate(sorted(image_names)):
                raw = zf.read(img_name)
                text, mean_conf = ocr_bytes(raw)
                location = f"image:{img_idx + 1}"
                if mean_conf < ocr_min_conf or not text.strip():
                    reason = (
                        f"OCR confidence {mean_conf:.1f} below threshold {ocr_min_conf}"
                        if mean_conf > 0
                        else "No text detected or OCR unavailable"
                    )
                    chunks.append(Chunk(
                        file=file,
                        location=location,
                        text="",
                        method="blocked",
                        ocr_confidence=mean_conf,
                        order_index=order,
                        blocked_reason=reason,
                    ))
                else:
                    chunks.append(Chunk(
                        file=file,
                        location=location,
                        text=text,
                        method="ocr",
                        ocr_confidence=mean_conf,
                        order_index=order,
                    ))
                order += 1

            # Non-raster formats (EMF, WMF, SVG) → BLOCKED
            non_raster = [
                n for n in zf.namelist()
                if n.startswith("word/media/") and
                Path(n).suffix.lower() not in raster_exts and
                Path(n).suffix.lower() in {".emf", ".wmf", ".svg"}
            ]
            for img_name in non_raster:
                log.warning("Cannot OCR vector image %s → BLOCKED", img_name)
                chunks.append(Chunk(
                    file=file,
                    location=f"image:{Path(img_name).name}",
                    text="",
                    method="blocked",
                    ocr_confidence=None,
                    order_index=order,
                    blocked_reason=f"Vector image format not supported: {Path(img_name).suffix}",
                ))
                order += 1
    except Exception as exc:
        log.error("Could not extract images from %s: %s", doc_path, exc)

    return chunks


def _paragraph_text(para: Paragraph) -> str:
    """Return the full text of a paragraph, preserving runs in order."""
    return para.text


def extract_docx(file_path: str, config: dict) -> List[Chunk]:
    """
    Extract all text chunks from a DOCX file.

    Parameters
    ----------
    file_path : Absolute path to the DOCX file.
    config    : Loaded config dict.

    Returns
    -------
    List of Chunk objects in document order.
    """
    path = Path(file_path)
    chunks: List[Chunk] = []
    order = 0

    try:
        doc = Document(str(path))
    except Exception as exc:
        log.error("Cannot open DOCX %s: %s", file_path, exc)
        return [Chunk(
            file=file_path, location="file", text="",
            method="blocked", ocr_confidence=None, order_index=0,
            blocked_reason=f"Cannot open DOCX: {exc}",
        )]

    # --- Core properties (metadata) ---
    props = doc.core_properties
    for key in ("author", "last_modified_by", "title", "subject"):
        val = getattr(props, key, None)
        if val and str(val).strip():
            chunks.append(Chunk(
                file=file_path,
                location=f"metadata/{key}",
                text=str(val).strip(),
                method="native",
                ocr_confidence=None,
                order_index=order,
            ))
            order += 1

    # --- Headers and footers (first section) ---
    section = doc.sections[0]
    for part_name, part in [
        ("header", section.header),
        ("footer", section.footer),
    ]:
        if part is not None:
            text = "\n".join(p.text for p in part.paragraphs if p.text.strip())
            if text.strip():
                chunks.append(Chunk(
                    file=file_path,
                    location=part_name,
                    text=text.strip(),
                    method="native",
                    ocr_confidence=None,
                    order_index=order,
                ))
                order += 1

    # --- Body: paragraphs and tables in DOM order ---
    table_idx = 0
    body = doc.element.body
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "p":
            # Paragraph
            para = Paragraph(child, doc)
            text = _paragraph_text(para).strip()
            if text:
                chunks.append(Chunk(
                    file=file_path,
                    location="body",
                    text=text,
                    method="native",
                    ocr_confidence=None,
                    order_index=order,
                ))
                order += 1

        elif tag == "tbl":
            # Table
            table = Table(child, doc)
            table_chunks = _table_to_chunks(table, file_path, table_idx, order, config)
            chunks.extend(table_chunks)
            order += len(table_chunks)
            table_idx += 1

    # --- Footnotes ---
    try:
        footnotes_part = doc.part.footnotes
        if footnotes_part is not None:
            xml_root = etree.fromstring(footnotes_part._blob)
            fn_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            for fn_idx, fn_elem in enumerate(xml_root.findall(f"{{{fn_ns}}}footnote")):
                fn_id = fn_elem.get(f"{{{fn_ns}}}id", str(fn_idx))
                # Skip separator / continuation footnotes (id -1, 0)
                try:
                    if int(fn_id) < 1:
                        continue
                except ValueError:
                    pass
                texts = [
                    t.text or ""
                    for t in fn_elem.iter(f"{{{fn_ns}}}t")
                ]
                fn_text = "".join(texts).strip()
                if fn_text:
                    chunks.append(Chunk(
                        file=file_path,
                        location=f"footnote:{fn_id}",
                        text=fn_text,
                        method="native",
                        ocr_confidence=None,
                        order_index=order,
                    ))
                    order += 1
    except Exception as exc:
        log.debug("Could not extract footnotes from %s: %s", path.name, exc)

    # --- Embedded images ---
    img_chunks = _extract_images(doc, str(path), file_path, order, config)
    chunks.extend(img_chunks)
    order += len(img_chunks)

    log.info("DOCX %s → %d chunks (%d blocked)",
             path.name, len(chunks), sum(1 for c in chunks if c.is_blocked()))
    return chunks
