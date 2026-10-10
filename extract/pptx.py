"""
extract/pptx.py
---------------
Extract text chunks from PPTX files using python-pptx.

Covers per slide:
  - Text frames (including shapes inside GroupShapes, recursively)
  - Tables (row/col coordinates)
  - Picture shapes → OCR
  - Speaker notes
Core properties are extracted once at the top.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from pptx import Presentation
from pptx.util import Pt
from pptx.enum.shapes import MSO_SHAPE_TYPE

from extract.models import Chunk
from extract.ocr import ocr_bytes

log = logging.getLogger(__name__)


def _table_location(slide_idx: int, table_idx: int, ri: int,
                    header_texts: List[str]) -> str:
    loc = f"slide:{slide_idx + 1}/table:{table_idx + 1}/r{ri + 1}"
    if header_texts:
        loc += f"[headers: {' | '.join(header_texts)}]"
    return loc


def _extract_shape_chunks(
    shape,
    file: str,
    slide_idx: int,
    table_counter: List[int],  # mutable counter shared across recursion
    order_ref: List[int],      # mutable order counter
    config: dict,
) -> List[Chunk]:
    """
    Recursively extract chunks from a shape (handles GroupShapes).
    table_counter and order_ref are single-element lists used as mutable ints.
    """
    chunks: List[Chunk] = []
    profile_name = config.get("active_profile", "balanced")
    ocr_min_conf = config["profiles"][profile_name]["ocr_min_confidence"]

    # --- Group shape: recurse into children ---
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for child in shape.shapes:
            chunks.extend(_extract_shape_chunks(
                child, file, slide_idx, table_counter, order_ref, config))
        return chunks

    # --- Text frame ---
    if shape.has_text_frame:
        text = shape.text_frame.text.strip()
        if text:
            chunks.append(Chunk(
                file=file,
                location=f"slide:{slide_idx + 1}/shape:{shape.shape_id}",
                text=text,
                method="native",
                ocr_confidence=None,
                order_index=order_ref[0],
            ))
            order_ref[0] += 1

    # --- Table ---
    if shape.has_table:
        table = shape.table
        rows = table.rows
        header_texts: List[str] = []
        t_idx = table_counter[0]

        # Detect header: first row
        if rows:
            first_row_texts = [cell.text.strip() for cell in rows[0].cells]
            kw = {
                "name", "email", "phone", "employee", "contact", "owner", "approver",
                "id", "staff", "role", "title", "user", "users", "member", "members",
                "reviewer", "author", "assignee", "respondent", "custodian",
                "delegate", "stakeholder", "participant", "operator", "signoff",
                "client contact", "peer reviewer", "qa reviewer"
            }
            if any(t.lower() in kw for t in first_row_texts):
                header_texts = first_row_texts

        for ri, row in enumerate(rows):
            cell_texts = [cell.text.strip() for cell in row.cells]
            row_text = " | ".join(cell_texts)
            if not row_text.strip():
                continue
            loc = _table_location(slide_idx, t_idx, ri, header_texts)
            chunks.append(Chunk(
                file=file,
                location=loc,
                text=row_text,
                method="native",
                ocr_confidence=None,
                order_index=order_ref[0],
            ))
            order_ref[0] += 1

        table_counter[0] += 1

    # --- Picture: OCR ---
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        try:
            img_bytes = shape.image.blob
            text, mean_conf = ocr_bytes(img_bytes)
            location = f"slide:{slide_idx + 1}/image:{shape.shape_id}"
            if mean_conf < ocr_min_conf or not text.strip():
                if mean_conf == 0.0 and not text.strip():
                    # No text in image (decorative)
                    log.debug("Picture shape %d on slide %d: no text detected",
                              shape.shape_id, slide_idx + 1)
                else:
                    chunks.append(Chunk(
                        file=file,
                        location=location,
                        text="",
                        method="blocked",
                        ocr_confidence=mean_conf,
                        order_index=order_ref[0],
                        blocked_reason=(
                            f"OCR confidence {mean_conf:.1f} below threshold {ocr_min_conf}"
                        ),
                    ))
                    order_ref[0] += 1
            else:
                chunks.append(Chunk(
                    file=file,
                    location=location,
                    text=text,
                    method="ocr",
                    ocr_confidence=mean_conf,
                    order_index=order_ref[0],
                ))
                order_ref[0] += 1
        except Exception as exc:
            log.warning("Could not OCR picture on slide %d: %s", slide_idx + 1, exc)

    return chunks


def extract_pptx(file_path: str, config: dict) -> List[Chunk]:
    """
    Extract all text chunks from a PPTX file.

    Parameters
    ----------
    file_path : Absolute path to the PPTX file.
    config    : Loaded config dict.

    Returns
    -------
    List of Chunk objects in presentation order.
    """
    path = Path(file_path)
    chunks: List[Chunk] = []

    try:
        prs = Presentation(str(path))
    except Exception as exc:
        log.error("Cannot open PPTX %s: %s", file_path, exc)
        return [Chunk(
            file=file_path, location="file", text="",
            method="blocked", ocr_confidence=None, order_index=0,
            blocked_reason=f"Cannot open PPTX: {exc}",
        )]

    order_ref = [0]  # mutable order counter

    # --- Core properties ---
    props = prs.core_properties
    for key in ("author", "last_modified_by", "title", "subject"):
        val = getattr(props, key, None)
        if val and str(val).strip():
            chunks.append(Chunk(
                file=file_path,
                location=f"metadata/{key}",
                text=str(val).strip(),
                method="native",
                ocr_confidence=None,
                order_index=order_ref[0],
            ))
            order_ref[0] += 1

    # --- Slides ---
    for slide_idx, slide in enumerate(prs.slides):
        table_counter = [0]

        for shape in slide.shapes:
            shape_chunks = _extract_shape_chunks(
                shape, file_path, slide_idx, table_counter, order_ref, config)
            chunks.extend(shape_chunks)

        # Speaker notes
        if slide.has_notes_slide:
            notes_tf = slide.notes_slide.notes_text_frame
            notes_text = notes_tf.text.strip() if notes_tf else ""
            if notes_text:
                chunks.append(Chunk(
                    file=file_path,
                    location=f"slide:{slide_idx + 1}/notes",
                    text=notes_text,
                    method="native",
                    ocr_confidence=None,
                    order_index=order_ref[0],
                ))
                order_ref[0] += 1

    log.info("PPTX %s → %d chunks (%d blocked)",
             path.name, len(chunks), sum(1 for c in chunks if c.is_blocked()))
    return chunks
