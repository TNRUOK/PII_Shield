"""
extract/pdf.py
--------------
Extract text chunks from PDF files using PyMuPDF.

Strategy per page:
  1. Try native text layer (page.get_text).
  2. If < config.pdf_native_min_chars: render at 300 DPI and OCR.
  3. If OCR confidence < config.ocr_min_confidence: emit BLOCKED chunk.

Also extracts PDF metadata (author, title, subject) as Chunk(location="metadata").
Supports parallelized OCR via ProcessPoolExecutor and in-memory SHA1 page caching.
"""
from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from extract.models import Chunk
from extract.ocr import ocr_image, is_tesseract_available

log = logging.getLogger(__name__)

# Pixels per inch when rendering a scanned page for OCR
OCR_RENDER_DPI = 300
_INCH_TO_PT = 72  # PyMuPDF uses points; 1 inch = 72 pt → scale = DPI/72

# In-memory SHA1 page cache
_PDF_PAGE_CACHE: Dict[str, Tuple[str, float]] = {}


def get_pdf_cache_size() -> int:
    """Return the number of cached OCR pages."""
    return len(_PDF_PAGE_CACHE)


def clear_pdf_cache() -> None:
    """Clear the in-memory OCR page cache."""
    _PDF_PAGE_CACHE.clear()


def _render_page(page: fitz.Page, dpi: int = OCR_RENDER_DPI) -> Tuple[Image.Image, str]:
    """Render a PDF page to a PIL Image at given DPI and compute its sha1 hash."""
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    sha1 = hashlib.sha1(pix.samples).hexdigest()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img, sha1


def _ocr_page_worker(args: tuple) -> tuple[int, str, float, str, Optional[str], Optional[list]]:
    """Worker function for parallel OCR execution across multiple worker processes."""
    file_path, page_num, ocr_min_conf, config, dpi = args
    os.environ["OMP_THREAD_LIMIT"] = "1"
    try:
        import fitz
        from PIL import Image
        import hashlib
        from extract.ocr import ocr_image
        doc = fitz.open(file_path)
        page = doc[page_num]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        sha1 = hashlib.sha1(pix.samples).hexdigest()
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        ocr_res = ocr_image(pil_img, min_confidence=ocr_min_conf, config=config)
        ocr_text, mean_conf = ocr_res
        line_confs = getattr(ocr_res, "line_confidences", None)
        return page_num, ocr_text, mean_conf, sha1, None, line_confs
    except Exception as exc:
        return page_num, "", 0.0, "", str(exc), None


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
    Extract all text chunks from a PDF file with parallel OCR and SHA1 caching.

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

    # First pass: check native text layer and check sha1 cache
    native_pages: Dict[int, str] = {}
    pages_to_ocr: List[int] = []
    cached_ocr_results: Dict[int, Tuple[str, float]] = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        native_text = page.get_text("text").strip()
        if len(native_text) >= min_chars:
            native_pages[page_num] = native_text
        else:
            # Check sha1 cache
            zoom = OCR_RENDER_DPI / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
            sha1 = hashlib.sha1(pix.samples).hexdigest()
            if sha1 in _PDF_PAGE_CACHE:
                cached_ocr_results[page_num] = _PDF_PAGE_CACHE[sha1]
            else:
                pages_to_ocr.append(page_num)

    doc.close()

    # Second pass: parallel OCR for uncached pages
    ocr_results: Dict[int, Tuple[str, float, Optional[str], Optional[list]]] = {}
    for p_num, entry in cached_ocr_results.items():
        text = entry[0]
        conf = entry[1]
        line_confs = entry[2] if len(entry) > 2 else None
        ocr_results[p_num] = (text, conf, None, line_confs)

    if pages_to_ocr:
        if len(pages_to_ocr) == 1:
            p_num = pages_to_ocr[0]
            _, ocr_text, mean_conf, sha1, err, line_confs = _ocr_page_worker(
                (file_path, p_num, ocr_min_conf, config, OCR_RENDER_DPI)
            )
            if sha1:
                _PDF_PAGE_CACHE[sha1] = (ocr_text, mean_conf, line_confs)
            ocr_results[p_num] = (ocr_text, mean_conf, err, line_confs)
        else:
            workers = min(4, os.cpu_count() or 1, len(pages_to_ocr))
            worker_args = [
                (file_path, p, ocr_min_conf, config, OCR_RENDER_DPI)
                for p in pages_to_ocr
            ]
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_ocr_page_worker, worker_args))
            for p_num, ocr_text, mean_conf, sha1, err, line_confs in results:
                if sha1:
                    _PDF_PAGE_CACHE[sha1] = (ocr_text, mean_conf, line_confs)
                ocr_results[p_num] = (ocr_text, mean_conf, err, line_confs)

    # Reconstruct chunks in strict deterministic page order
    total_pages = len(native_pages) + len(ocr_results)
    for page_num in range(total_pages):
        location = f"page:{page_num + 1}"
        if page_num in native_pages:
            chunks.append(Chunk(
                file=file_path,
                location=location,
                text=native_pages[page_num],
                method="native",
                ocr_confidence=None,
                order_index=order,
            ))
            order += 1
        elif page_num in ocr_results:
            ocr_text, mean_conf, err, line_confs = ocr_results[page_num]
            if err:
                log.error("OCR failed on page %d of %s: %s", page_num + 1, path.name, err)
                chunks.append(Chunk(
                    file=file_path,
                    location=location,
                    text="",
                    method="blocked",
                    ocr_confidence=None,
                    order_index=order,
                    blocked_reason=f"OCR error: {err}",
                ))
            elif not is_tesseract_available():
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
                    line_confidences=line_confs,
                ))
            else:
                chunks.append(Chunk(
                    file=file_path,
                    location=location,
                    text=ocr_text,
                    method="ocr",
                    ocr_confidence=mean_conf,
                    order_index=order,
                    line_confidences=line_confs,
                ))
            order += 1

    log.info("PDF %s → %d chunks (%d blocked)",
             path.name,
             len(chunks),
             sum(1 for c in chunks if c.is_blocked()))
    return chunks
