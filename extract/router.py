"""
extract/router.py
-----------------
Dispatch a file to the correct extractor based on extension and magic bytes.

Magic-byte checks take priority over extension so a renamed file is handled
correctly. Falls back to extension-only if the file is too short for a header.

Supported formats: PDF, DOCX, PPTX, PNG/JPEG/TIFF/BMP (single image → OCR).
Everything else returns a single BLOCKED chunk.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Callable, Dict, Tuple

from extract.models import Chunk
from extract.ocr import ocr_image, is_tesseract_available

log = logging.getLogger(__name__)

# Magic byte signatures: (offset, bytes) → format name
_MAGIC: List[Tuple[int, bytes, str]] = [
    (0, b"%PDF",                 "pdf"),
    (0, b"PK\x03\x04",          "zip"),  # DOCX/PPTX are ZIP-based
    (0, b"\x89PNG\r\n\x1a\n",   "png"),
    (0, b"\xff\xd8\xff",        "jpeg"),
    (0, b"II\x2a\x00",          "tiff"),  # little-endian TIFF
    (0, b"MM\x00\x2a",          "tiff"),  # big-endian TIFF
    (0, b"BM",                  "bmp"),
]


def _detect_format(path: Path) -> str:
    """
    Read the first 8 bytes and try magic-byte matching.
    Returns a canonical format string: "pdf", "docx", "pptx", "image", "unknown".
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(8)
    except Exception:
        return "unknown"

    for offset, magic, fmt in _MAGIC:
        if header[offset: offset + len(magic)] == magic:
            if fmt == "zip":
                # Distinguish DOCX from PPTX by extension (both are ZIP)
                ext = path.suffix.lower()
                if ext == ".docx":
                    return "docx"
                if ext in (".pptx", ".ppt"):
                    return "pptx"
                return "zip"  # unsupported ZIP variant
            if fmt in ("png", "jpeg", "tiff", "bmp"):
                return "image"
            return fmt

    # Fall back to extension
    ext = path.suffix.lower()
    return {
        ".pdf": "pdf",
        ".docx": "docx",
        ".pptx": "pptx",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".tiff": "image",
        ".tif": "image",
        ".bmp": "image",
    }.get(ext, "unknown")


def _blocked(file_path: str, reason: str) -> List[Chunk]:
    return [Chunk(
        file=file_path,
        location="file",
        text="",
        method="blocked",
        ocr_confidence=None,
        order_index=0,
        blocked_reason=reason,
    )]


def route(file_path: str, config: dict) -> List[Chunk]:
    """
    Route a file to the appropriate extractor and return its Chunk list.

    Parameters
    ----------
    file_path : Absolute path to the input file.
    config    : Loaded config dict.
    """
    path = Path(file_path)

    if not path.exists():
        log.error("File not found: %s", file_path)
        return _blocked(file_path, "File not found")

    if path.stat().st_size == 0:
        log.error("Empty file: %s", file_path)
        return _blocked(file_path, "Empty file (0 bytes)")

    fmt = _detect_format(path)
    log.info("Routing %s as format=%s", path.name, fmt)

    if fmt == "pdf":
        from extract.pdf import extract_pdf
        return extract_pdf(file_path, config)

    elif fmt == "docx":
        from extract.docx import extract_docx
        return extract_docx(file_path, config)

    elif fmt == "pptx":
        from extract.pptx import extract_pptx
        return extract_pptx(file_path, config)

    elif fmt == "image":
        # Single image file: OCR directly
        from PIL import Image
        profile_name = config.get("active_profile", "balanced")
        ocr_min_conf = config["profiles"][profile_name]["ocr_min_confidence"]
        try:
            img = Image.open(str(path))
            text, mean_conf = ocr_image(img, min_confidence=ocr_min_conf, config=config)
            if not is_tesseract_available():
                return _blocked(file_path, "Tesseract not installed")
            if not text.strip() and mean_conf == 0.0:
                return _blocked(file_path, "no text detected")
            if mean_conf < ocr_min_conf:
                return [Chunk(
                    file=file_path,
                    location="page:1",
                    text=text,
                    method="blocked",
                    ocr_confidence=mean_conf,
                    order_index=0,
                    blocked_reason=f"OCR confidence {mean_conf:.1f} below threshold {ocr_min_conf}",
                    low_confidence=True,
                )]
            return [Chunk(
                file=file_path,
                location="page:1",
                text=text,
                method="ocr",
                ocr_confidence=mean_conf,
                order_index=0,
            )]
        except Exception as exc:
            return _blocked(file_path, f"Cannot open image: {exc}")

    else:
        log.warning("Unsupported format for %s (detected: %s)", path.name, fmt)
        return _blocked(file_path, f"Unsupported file format: {fmt or path.suffix}")
