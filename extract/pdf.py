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


def _ocr_page_worker(args: tuple) -> tuple:
    """Worker function for parallel OCR execution across multiple worker processes."""
    file_path, page_num, ocr_min_conf, config, dpi = args
    os.environ["OMP_THREAD_LIMIT"] = "1"
    try:
        import fitz
        from PIL import Image
        import hashlib
        from extract.ocr import ocr_image
        from extract.regions import detect_candidate_regions, ocr_region, is_duplicate_region_text

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

        # Region detection and high-res multi-PSM OCR
        regions_data = []
        try:
            candidate_boxes = detect_candidate_regions(pil_img, config)
            for r_idx, box in enumerate(candidate_boxes):
                reg_text, reg_conf, reg_status, reg_lines = ocr_region(pil_img, box, config)
                x, y, w, h, rtype = box
                # Deduplication against full-page text
                if reg_text.strip() and is_duplicate_region_text(reg_text, ocr_text):
                    continue
                regions_data.append({
                    "sub_idx": r_idx + 1,
                    "text": reg_text,
                    "conf": reg_conf,
                    "status": reg_status,
                    "type": rtype,
                    "bbox": (x, y, w, h),
                    "line_confs": reg_lines,
                })
        except Exception as r_exc:
            pass

        return page_num, ocr_text, mean_conf, sha1, None, line_confs, regions_data
    except Exception as exc:
        return page_num, "", 0.0, "", str(exc), None, []


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
    native_embedded_images: Dict[int, List[dict]] = {}
    pages_to_ocr: List[int] = []
    cached_ocr_results: Dict[int, tuple] = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        native_text = page.get_text("text").strip()
        if len(native_text) >= min_chars:
            native_pages[page_num] = native_text
            # Extract embedded images on native pages for region OCR
            try:
                from extract.regions import ocr_region
                img_list = page.get_images(full=True)
                emb_regs = []
                for img_idx, img_info in enumerate(img_list):
                    xref = img_info[0]
                    base_img = doc.extract_image(xref)
                    if base_img and "image" in base_img:
                        w, h = base_img.get("width", 0), base_img.get("height", 0)
                        if w >= 120 and h >= 60:
                            import io
                            from PIL import Image
                            pil_emb = Image.open(io.BytesIO(base_img["image"]))
                            bbox = (0, 0, w, h, "figure")
                            r_text, r_conf, r_status, r_lines = ocr_region(pil_emb, bbox, config)
                            emb_regs.append({
                                "sub_idx": img_idx + 1,
                                "text": r_text,
                                "conf": r_conf,
                                "status": r_status,
                                "type": "figure",
                                "bbox": (0, 0, w, h),
                                "line_confs": r_lines,
                            })
                if emb_regs:
                    native_embedded_images[page_num] = emb_regs
            except Exception:
                pass
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
    ocr_results: Dict[int, tuple] = {}
    for p_num, entry in cached_ocr_results.items():
        text = entry[0]
        conf = entry[1]
        line_confs = entry[2] if len(entry) > 2 else None
        reg_data = entry[3] if len(entry) > 3 else []
        ocr_results[p_num] = (text, conf, None, line_confs, reg_data)

    if pages_to_ocr:
        if len(pages_to_ocr) == 1:
            p_num = pages_to_ocr[0]
            _, ocr_text, mean_conf, sha1, err, line_confs, reg_data = _ocr_page_worker(
                (file_path, p_num, ocr_min_conf, config, OCR_RENDER_DPI)
            )
            if sha1:
                _PDF_PAGE_CACHE[sha1] = (ocr_text, mean_conf, line_confs, reg_data)
            ocr_results[p_num] = (ocr_text, mean_conf, err, line_confs, reg_data)
        else:
            workers = min(4, os.cpu_count() or 1, len(pages_to_ocr))
            worker_args = [
                (file_path, p, ocr_min_conf, config, OCR_RENDER_DPI)
                for p in pages_to_ocr
            ]
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    results = list(executor.map(_ocr_page_worker, worker_args))
            except Exception as pool_exc:
                log.warning("ProcessPoolExecutor fallback to synchronous worker: %s", pool_exc)
                results = [_ocr_page_worker(wa) for wa in worker_args]

            for p_num, ocr_text, mean_conf, sha1, err, line_confs, reg_data in results:
                if sha1:
                    _PDF_PAGE_CACHE[sha1] = (ocr_text, mean_conf, line_confs, reg_data)
                ocr_results[p_num] = (ocr_text, mean_conf, err, line_confs, reg_data)

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
            # Add embedded image region chunks for this native page
            if page_num in native_embedded_images:
                for reg in native_embedded_images[page_num]:
                    reg_loc = f"page:{page_num + 1}/region:{reg['sub_idx']}"
                    if reg["text"].strip():
                        chunks.append(Chunk(
                            file=file_path,
                            location=reg_loc,
                            text=reg["text"],
                            method="region_ocr",
                            ocr_confidence=reg["conf"],
                            order_index=order,
                            image_status=reg["status"],
                            region_type=reg["type"],
                            region_bbox=reg["bbox"],
                            line_confidences=reg["line_confs"],
                        ))
                    else:
                        chunks.append(Chunk(
                            file=file_path,
                            location=reg_loc,
                            text="",
                            method="blocked",
                            ocr_confidence=reg["conf"],
                            order_index=order,
                            blocked_reason="Unread region",
                            image_status=reg["status"],
                            region_type=reg["type"],
                            region_bbox=reg["bbox"],
                        ))
                    order += 1
        elif page_num in ocr_results:
            ocr_text, mean_conf, err, line_confs, reg_data = ocr_results[page_num]
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
                order += 1
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
                order += 1
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
                order += 1
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
                order += 1
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

            # Append detected region chunks for scanned page
            for reg in reg_data:
                reg_loc = f"page:{page_num + 1}/region:{reg['sub_idx']}"
                if reg["text"].strip():
                    chunks.append(Chunk(
                        file=file_path,
                        location=reg_loc,
                        text=reg["text"],
                        method="region_ocr",
                        ocr_confidence=reg["conf"],
                        order_index=order,
                        image_status=reg["status"],
                        region_type=reg["type"],
                        region_bbox=reg["bbox"],
                        line_confidences=reg["line_confs"],
                    ))
                else:
                    chunks.append(Chunk(
                        file=file_path,
                        location=reg_loc,
                        text="",
                        method="blocked",
                        ocr_confidence=reg["conf"],
                        order_index=order,
                        blocked_reason="Unread region",
                        image_status=reg["status"],
                        region_type=reg["type"],
                        region_bbox=reg["bbox"],
                    ))
                order += 1

    log.info("PDF %s → %d chunks (%d blocked)",
             path.name,
             len(chunks),
             sum(1 for c in chunks if c.is_blocked()))
    return chunks
