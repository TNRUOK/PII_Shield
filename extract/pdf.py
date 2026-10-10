"""
extract/pdf.py
--------------
Extract text chunks from PDF files using PyMuPDF.

Strategy per page:
  1. Try native text layer (page.get_text).
  2. If < config.pdf_native_min_chars: render at 300 DPI and OCR.
  3. If OCR confidence < config.ocr_min_confidence: emit BLOCKED chunk.

Also extracts PDF metadata (author, title, subject) as Chunk(location="metadata").
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image

from extract.models import Chunk
from extract.ocr import ocr_image, is_tesseract_available

log = logging.getLogger(__name__)

# Pixels per inch when rendering a scanned page for OCR
OCR_RENDER_DPI = 300
_INCH_TO_PT = 72  # PyMuPDF uses points; 1 inch = 72 pt → scale = DPI/72


def _render_page(page: fitz.Page, dpi: int = OCR_RENDER_DPI) -> Image.Image:
    """Render a PDF page to a PIL Image at the given DPI."""
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _metadata_chunks(doc: fitz.Document, file: str, order_start: int) -> List[Chunk]:
    """Extract PDF metadata fields as individual Chunk objects."""
    meta = doc.metadata or {}
    fields = {
        "author": meta.get("author", ""),
        "title": meta.get("title", ""),
        "subject": meta.get("subject", ""),
        "creator": meta.get("creator", ""),
    }
    chunks = []
    idx = order_start
    for key, value in fields.items():
        if value and value.strip():
            chunks.append(Chunk(
                file=file,
                location=f"metadata/{key}",
                text=value.strip(),
                method="native",
                ocr_confidence=None,
                order_index=idx,
            ))
            idx += 1
    return chunks


def extract_pdf(file_path: str, config: dict) -> List[Chunk]:
    """
    Extract all text chunks from a PDF file.

    Parameters
    ----------
    file_path : Absolute path to the PDF.
    config    : Loaded config dict (from config.yaml).

    Returns
    -------
    List of Chunk objects in reading order (metadata first, then page order).
    """
    path = Path(file_path)
    chunks: List[Chunk] = []
    order = 0

    min_chars: int = config.get("pdf_native_min_chars", 50)
    profile_name: str = config.get("active_profile", "balanced")
    profile: dict = config["profiles"][profile_name]
    ocr_min_conf: float = profile["ocr_min_confidence"]

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        log.error("Cannot open PDF %s: %s", file_path, exc)
        return [Chunk(
            file=file_path,
            location="file",
            text="",
            method="blocked",
            ocr_confidence=None,
            order_index=0,
            blocked_reason=f"Cannot open PDF: {exc}",
        )]

    # --- Metadata ---
    meta_chunks = _metadata_chunks(doc, file_path, order)
    chunks.extend(meta_chunks)
    order += len(meta_chunks)

    # --- Pages ---
    for page_num in range(len(doc)):
        page = doc[page_num]
        location = f"page:{page_num + 1}"

        # Try native text layer first
        native_text = page.get_text("text").strip()

        if len(native_text) >= min_chars:
            # Good native text layer
            chunks.append(Chunk(
                file=file_path,
                location=location,
                text=native_text,
                method="native",
                ocr_confidence=None,
                order_index=order,
            ))
            order += 1
        else:
            # Scanned or minimal text — try OCR
            log.info("Page %d of %s: native text too short (%d chars), trying OCR",
                     page_num + 1, path.name, len(native_text))
            try:
                pil_img = _render_page(page, dpi=OCR_RENDER_DPI)
                ocr_text, mean_conf = ocr_image(
                    pil_img, min_confidence=ocr_min_conf, config=config
                )
            except Exception as exc:
                log.error("OCR failed on page %d of %s: %s", page_num + 1, path.name, exc)
                chunks.append(Chunk(
                    file=file_path,
                    location=location,
                    text="",
                    method="blocked",
                    ocr_confidence=None,
                    order_index=order,
                    blocked_reason=f"OCR error: {exc}",
                ))
                order += 1
                continue

            if not is_tesseract_available():
                chunks.append(Chunk(
                    file=file_path,
                    location=location,
                    text="",
                    method="blocked",
                    ocr_confidence=0.0,
                    order_index=order,
                    blocked_reason="Tesseract not installed",
                ))
            elif not ocr_text.strip() and mean_conf == 0.0:
                chunks.append(Chunk(
                    file=file_path,
                    location=location,
                    text="",
                    method="blocked",
                    ocr_confidence=0.0,
                    order_index=order,
                    blocked_reason="no text detected",
                ))
            elif mean_conf < ocr_min_conf:
                log.warning("Page %d of %s: OCR confidence %.1f < %.1f → BLOCKED",
                            page_num + 1, path.name, mean_conf, ocr_min_conf)
                chunks.append(Chunk(
                    file=file_path,
                    location=location,
                    text=ocr_text,
                    method="blocked",
                    ocr_confidence=mean_conf,
                    order_index=order,
                    blocked_reason=f"OCR confidence {mean_conf:.1f} below threshold {ocr_min_conf}",
                    low_confidence=True,
                ))
            else:
                chunks.append(Chunk(
                    file=file_path,
                    location=location,
                    text=ocr_text,
                    method="ocr",
                    ocr_confidence=mean_conf,
                    order_index=order,
                ))
            order += 1

    doc.close()
    log.info("PDF %s → %d chunks (%d blocked)",
             path.name,
             len(chunks),
             sum(1 for c in chunks if c.is_blocked()))
    return chunks
