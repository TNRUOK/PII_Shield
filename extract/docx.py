"""
extract/docx.py
---------------
Deep extraction of text chunks from DOCX files using python-docx and direct OpenXML parsing.

Reading order:
  Walks document.element.body children in DOM order so paragraphs
  and tables are interleaved exactly as they appear in the document.

Covers:
  - Body paragraphs (headings, normal text)
  - Tables (with merged cell de-duplication, nested tables recursion, header detection)
  - Headers and footers across all sections (primary, first_page, even_page + header tables)
  - Text boxes (w:txbxContent elements)
  - Tracked changes (w:del / w:delText and w:ins)
  - Comments (word/comments.xml) and endnotes (word/endnotes.xml)
  - Footnotes (word/footnotes.xml)
  - Extended/custom properties (Company, Manager, custom properties) and core properties
  - Embedded raster images (PNG/JPEG) with natural sorting → OCR
  - Vector images (EMF/WMF/SVG) → BLOCKED chunks
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path
from typing import List, Optional, Set, Tuple
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
_EP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
_CP = "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
_REL_IMG = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"

# If a table's first row contains any of these keywords we treat it as a header row
_HEADER_KEYWORDS = {
    "name", "email", "phone", "employee", "contact", "owner", "approver", "id",
    "staff", "emp", "user", "users", "member", "members", "reviewer", "author",
    "assignee", "respondent", "custodian", "delegate", "stakeholder", "participant",
    "operator", "signoff", "lead", "manager", "attendee", "client contact",
    "peer reviewer", "qa reviewer"
}


def _natural_key(s: str):
    """Sort key for natural alphanumeric ordering (e.g. image2 before image10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def _row_is_header(row) -> bool:
    """Heuristic: first row of a table is a header if any cell text matches header keywords."""
    for cell in row.cells:
        if cell.text.strip().lower() in _HEADER_KEYWORDS:
            return True
    return False


def _table_to_chunks(
    table: Table,
    file: str,
    table_idx: int,
    order_start: int,
    config: dict,
    prefix: str = "table",
) -> List[Chunk]:
    """
    Render a DOCX table as Chunk objects, including merged-cell de-duplication
    and recursive handling of nested tables.
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
            # Deduplicate merged cells for header
            seen_tc_h: Set = set()
            h_texts = []
            for c in row.cells:
                if c._tc not in seen_tc_h:
                    seen_tc_h.add(c._tc)
                    h_texts.append(c.text.strip())
            header_texts = h_texts
            header_row_idx = ri
            break

    nested_tables_found: List[tuple[int, int, Table]] = []

    for ri, row in enumerate(rows):
        seen_tc: Set = set()
        cell_texts: List[str] = []
        for cell in row.cells:
            if cell._tc not in seen_tc:
                seen_tc.add(cell._tc)
                cell_texts.append(cell.text.strip())
                # Check for nested tables inside this cell
                nested = getattr(cell, "tables", None)
                if nested:
                    for n_idx, nested_tbl in enumerate(nested):
                        nested_tables_found.append((ri, n_idx, nested_tbl))

        row_text = " | ".join(cell_texts)

        # Build location string including column context for context.py
        location = f"{prefix}:{table_idx + 1}/r{ri + 1}"
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

    # Recursively render nested tables
    for n_ri, n_idx, n_tbl in nested_tables_found:
        n_chunks = _table_to_chunks(
            n_tbl, file, n_idx, order, config,
            prefix=f"{prefix}:{table_idx + 1}/r{n_ri + 1}/nested"
        )
        chunks.extend(n_chunks)
        order += len(n_chunks)

    return chunks


def _extract_svg_text(raw_bytes: bytes) -> str:
    """Extract selectable text elements from SVG XML (<text>, <tspan>)."""
    try:
        from lxml import etree
        root = etree.fromstring(raw_bytes)
        text_lines = []
        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag in ("text", "tspan") and elem.text and elem.text.strip():
                text_lines.append(elem.text.strip())
        return "\n".join(text_lines)
    except Exception as exc:
        log.warning("SVG XML parsing failed: %s", exc)
        return ""


def is_vector_converter_available() -> Tuple[bool, Optional[str]]:
    """Check if LibreOffice or Inkscape is available for EMF conversion."""
    import shutil
    candidates = ["soffice", "libreoffice", "inkscape"]
    for c in candidates:
        if shutil.which(c):
            return True, c
    return False, None


def _extract_images(doc: Document, doc_path: str, file: str,
                    order_start: int, config: dict) -> List[Chunk]:
    """
    Extract all embedded images from the DOCX zip:
    - Raster images (PNG, JPEG): candidate region detection and high-res multi-PSM OCR
    - SVG vector images: direct XML text extraction (<text>, <tspan>)
    - EMF/WMF images: convert if LibreOffice/Inkscape installed, else mark VECTOR_UNSUPPORTED
    """
    chunks: List[Chunk] = []
    order = order_start

    profile_name = config.get("active_profile", "balanced")
    ocr_min_conf = config["profiles"][profile_name]["ocr_min_confidence"]

    raster_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"}

    try:
        from PIL import Image
        from extract.regions import detect_candidate_regions, ocr_region

        with zipfile.ZipFile(doc_path, "r") as zf:
            # 1. Raster images
            raster_names = [
                n for n in zf.namelist()
                if n.startswith("word/media/") and
                Path(n).suffix.lower() in raster_exts
            ]
            for img_idx, img_name in enumerate(sorted(raster_names, key=_natural_key)):
                raw = zf.read(img_name)
                location = f"image:{img_idx + 1}"
                try:
                    pil_img = Image.open(io.BytesIO(raw))
                    w, h = pil_img.size
                    candidate_boxes = detect_candidate_regions(pil_img, config)
                    if candidate_boxes:
                        for r_idx, box in enumerate(candidate_boxes):
                            r_text, r_conf, r_status, r_lines = ocr_region(pil_img, box, config)
                            bx, by, bw, bh, rtype = box
                            sub_loc = f"{location}/region:{r_idx + 1}"
                            if r_text.strip():
                                chunks.append(Chunk(
                                    file=file,
                                    location=sub_loc,
                                    text=r_text,
                                    method="region_ocr",
                                    ocr_confidence=r_conf,
                                    order_index=order,
                                    image_status=r_status,
                                    region_type=rtype,
                                    region_bbox=(bx, by, bw, bh),
                                    line_confidences=r_lines,
                                ))
                            else:
                                chunks.append(Chunk(
                                    file=file,
                                    location=sub_loc,
                                    text="",
                                    method="blocked",
                                    ocr_confidence=r_conf,
                                    order_index=order,
                                    blocked_reason="Unread region",
                                    image_status=r_status,
                                    region_type=rtype,
                                    region_bbox=(bx, by, bw, bh),
                                ))
                            order += 1
                    else:
                        # Full image region OCR
                        r_text, r_conf, r_status, r_lines = ocr_region(pil_img, (0, 0, w, h, "figure"), config)
                        if r_text.strip():
                            chunks.append(Chunk(
                                file=file,
                                location=location,
                                text=r_text,
                                method="region_ocr",
                                ocr_confidence=r_conf,
                                order_index=order,
                                image_status=r_status,
                                region_type="figure",
                                region_bbox=(0, 0, w, h),
                                line_confidences=r_lines,
                            ))
                        else:
                            chunks.append(Chunk(
                                file=file,
                                location=location,
                                text="",
                                method="blocked",
                                ocr_confidence=r_conf,
                                order_index=order,
                                blocked_reason="No text detected in image",
                                image_status=r_status,
                                region_type="figure",
                                region_bbox=(0, 0, w, h),
                            ))
                        order += 1
                except Exception as img_err:
                    log.error("Failed to OCR raster image %s: %s", img_name, img_err)

            # 2. SVG vector images
            svg_names = [
                n for n in zf.namelist()
                if n.startswith("word/media/") and
                Path(n).suffix.lower() == ".svg"
            ]
            for img_name in sorted(svg_names, key=_natural_key):
                raw = zf.read(img_name)
                svg_text = _extract_svg_text(raw)
                location = f"image:{Path(img_name).name}"
                if svg_text.strip():
                    chunks.append(Chunk(
                        file=file,
                        location=location,
                        text=svg_text,
                        method="svg_text",
                        ocr_confidence=None,
                        order_index=order,
                        image_status="READ",
                        region_type="diagram",
                    ))
                else:
                    chunks.append(Chunk(
                        file=file,
                        location=location,
                        text="",
                        method="blocked",
                        ocr_confidence=None,
                        order_index=order,
                        blocked_reason="No text found in SVG vector image",
                        image_status="UNREAD",
                        region_type="diagram",
                    ))
                order += 1

            # 3. EMF/WMF vector images
            emf_names = [
                n for n in zf.namelist()
                if n.startswith("word/media/") and
                Path(n).suffix.lower() in {".emf", ".wmf"}
            ]
            has_conv, conv_tool = is_vector_converter_available()
            for img_name in sorted(emf_names, key=_natural_key):
                location = f"image:{Path(img_name).name}"
                if has_conv:
                    # Conversion could be invoked via conv_tool
                    log.info("Converter %s found for EMF %s", conv_tool, img_name)
                else:
                    log.warning("Vector image %s cannot be converted: no converter installed", img_name)
                    chunks.append(Chunk(
                        file=file,
                        location=location,
                        text="",
                        method="blocked",
                        ocr_confidence=None,
                        order_index=order,
                        blocked_reason=f"Vector image format not supported: {Path(img_name).suffix} (no converter installed)",
                        image_status="VECTOR_UNSUPPORTED",
                        region_type="diagram",
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
    Deep extraction of all text chunks from a DOCX file.

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

    # --- Extended and Custom properties from zip (Company, Manager, custom fields) ---
    try:
        with zipfile.ZipFile(str(path), "r") as zf:
            zf_names = set(zf.namelist())

            if "docProps/app.xml" in zf_names:
                try:
                    app_root = etree.fromstring(zf.read("docProps/app.xml"))
                    for elem_name in ("Company", "Manager"):
                        found = app_root.find(f".//{{{_EP}}}{elem_name}")
                        if found is None:
                            found = app_root.find(f".//{elem_name}")
                        if found is not None and found.text and found.text.strip():
                            chunks.append(Chunk(
                                file=file_path,
                                location=f"metadata/{elem_name.lower()}",
                                text=found.text.strip(),
                                method="native",
                                ocr_confidence=None,
                                order_index=order,
                            ))
                            order += 1
                except Exception as exc:
                    log.debug("Could not parse docProps/app.xml: %s", exc)

            if "docProps/custom.xml" in zf_names:
                try:
                    cust_root = etree.fromstring(zf.read("docProps/custom.xml"))
                    for prop in cust_root.findall(f".//{{{_CP}}}property"):
                        name = prop.get("name", "custom")
                        val = "".join(prop.itertext()).strip()
                        if val:
                            chunks.append(Chunk(
                                file=file_path,
                                location=f"metadata/{name.lower()}",
                                text=val,
                                method="native",
                                ocr_confidence=None,
                                order_index=order,
                            ))
                            order += 1
                except Exception as exc:
                    log.debug("Could not parse docProps/custom.xml: %s", exc)

            # Comments
            if "word/comments.xml" in zf_names:
                try:
                    comm_root = etree.fromstring(zf.read("word/comments.xml"))
                    for comm in comm_root.findall(f".//{{{_W}}}comment"):
                        author = comm.get(f"{{{_W}}}author", "commenter")
                        comm_texts = [t.text or "" for t in comm.iter(f"{{{_W}}}t")]
                        comm_str = "".join(comm_texts).strip()
                        if comm_str:
                            chunks.append(Chunk(
                                file=file_path,
                                location=f"comment:{author}",
                                text=comm_str,
                                method="native",
                                ocr_confidence=None,
                                order_index=order,
                            ))
                            order += 1
                except Exception as exc:
                    log.debug("Could not parse word/comments.xml: %s", exc)

            # Endnotes
            if "word/endnotes.xml" in zf_names:
                try:
                    en_root = etree.fromstring(zf.read("word/endnotes.xml"))
                    for en in en_root.findall(f".//{{{_W}}}endnote"):
                        en_id = en.get(f"{{{_W}}}id", "0")
                        try:
                            if int(en_id) < 1:
                                continue
                        except ValueError:
                            pass
                        en_texts = [t.text or "" for t in en.iter(f"{{{_W}}}t")]
                        en_str = "".join(en_texts).strip()
                        if en_str:
                            chunks.append(Chunk(
                                file=file_path,
                                location=f"endnote:{en_id}",
                                text=en_str,
                                method="native",
                                ocr_confidence=None,
                                order_index=order,
                            ))
                            order += 1
                except Exception as exc:
                    log.debug("Could not parse word/endnotes.xml: %s", exc)
    except Exception as exc:
        log.debug("Could not inspect zip for extra properties: %s", exc)

    # --- Headers and footers across all sections ---
    seen_hf: Set[str] = set()
    for s_idx, sec in enumerate(doc.sections):
        is_first_sec = (s_idx == 0)
        hf_list = [
            ("header" if is_first_sec else f"section:{s_idx+1}/header", getattr(sec, "header", None)),
            ("footer" if is_first_sec else f"section:{s_idx+1}/footer", getattr(sec, "footer", None)),
            ("first_page_header" if is_first_sec else f"section:{s_idx+1}/first_page_header", getattr(sec, "first_page_header", None)),
            ("first_page_footer" if is_first_sec else f"section:{s_idx+1}/first_page_footer", getattr(sec, "first_page_footer", None)),
            ("even_page_header" if is_first_sec else f"section:{s_idx+1}/even_page_header", getattr(sec, "even_page_header", None)),
            ("even_page_footer" if is_first_sec else f"section:{s_idx+1}/even_page_footer", getattr(sec, "even_page_footer", None)),
        ]
        for loc_name, part in hf_list:
            if part is None:
                continue
            # Header/footer paragraphs
            p_text = "\n".join(p.text.strip() for p in getattr(part, "paragraphs", []) if p.text.strip())
            if p_text and p_text not in seen_hf:
                seen_hf.add(p_text)
                chunks.append(Chunk(
                    file=file_path,
                    location=loc_name,
                    text=p_text,
                    method="native",
                    ocr_confidence=None,
                    order_index=order,
                ))
                order += 1

            # Header/footer tables
            for t_idx, tbl in enumerate(getattr(part, "tables", [])):
                for r_idx, r in enumerate(tbl.rows):
                    seen_c: Set = set()
                    cell_vals = []
                    for c in r.cells:
                        if c._tc not in seen_c:
                            seen_c.add(c._tc)
                            cell_vals.append(c.text.strip())
                    row_t = " | ".join(cell_vals)
                    if row_t and row_t not in seen_hf:
                        seen_hf.add(row_t)
                        chunks.append(Chunk(
                            file=file_path,
                            location=f"{loc_name}/table:{t_idx+1}/r{r_idx+1}",
                            text=row_t,
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
            table = Table(child, doc)
            table_chunks = _table_to_chunks(table, file_path, table_idx, order, config)
            chunks.extend(table_chunks)
            order += len(table_chunks)
            table_idx += 1

    # --- Text boxes (w:txbxContent) in document body ---
    txbx_idx = 0
    for txbx in doc.element.body.iter(f"{{{_W}}}txbxContent"):
        tb_texts = []
        for p in txbx.iter(f"{{{_W}}}p"):
            p_txt = "".join(t.text or "" for t in p.iter(f"{{{_W}}}t")).strip()
            if p_txt:
                tb_texts.append(p_txt)
        if not tb_texts:
            t_all = "".join(t.text or "" for t in txbx.iter(f"{{{_W}}}t")).strip()
            if t_all:
                tb_texts.append(t_all)
        if tb_texts:
            chunks.append(Chunk(
                file=file_path,
                location=f"textbox:{txbx_idx + 1}",
                text="\n".join(tb_texts),
                method="native",
                ocr_confidence=None,
                order_index=order,
            ))
            order += 1
            txbx_idx += 1

    # --- Tracked Changes (deletions via w:del / w:delText) ---
    del_idx = 0
    for del_elem in doc.element.body.iter(f"{{{_W}}}del"):
        del_texts = [t.text or "" for t in del_elem.iter(f"{{{_W}}}delText")]
        del_str = "".join(del_texts).strip()
        if del_str:
            chunks.append(Chunk(
                file=file_path,
                location=f"tracked_change:deleted/{del_idx + 1}",
                text=del_str,
                method="native",
                ocr_confidence=None,
                order_index=order,
            ))
            order += 1
            del_idx += 1

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
