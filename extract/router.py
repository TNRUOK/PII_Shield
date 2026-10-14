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
    (0, b"PK\x03\x04",          "zip"),   # DOCX/PPTX/XLSX are ZIP-based
    (0, b"\x89PNG\r\n\x1a\n",   "png"),
    (0, b"\xff\xd8\xff",        "jpeg"),
    (0, b"II\x2a\x00",          "tiff"),  # little-endian TIFF
    (0, b"MM\x00\x2a",          "tiff"),  # big-endian TIFF
    (0, b"BM",                  "bmp"),
    (0, b"GIF87a",              "image"),
    (0, b"GIF89a",              "image"),
    (0, b"RIFF",                "riff"),  # WebP begins with RIFF....WEBP
]

# Extensions recognized as plain text, source code, tabular, config, or markup
_TEXT_EXTENSIONS = {
    ".txt", ".csv", ".tsv", ".json", ".xml", ".html", ".htm",
    ".md", ".markdown", ".rst",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".sql", ".sh", ".bash", ".bat", ".ps1", ".cmd",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".log",
    ".properties", ".rtf", ".tex", ".diff", ".patch",
}


def _detect_format(path: Path) -> str:
    """
    Read the first 12 bytes and try magic-byte matching.
    Returns a canonical format string: "pdf", "docx", "pptx", "xlsx", "image", "text", "unknown".
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(12)
    except Exception:
        return "unknown"

    ext = path.suffix.lower()

    for offset, magic, fmt in _MAGIC:
        if header[offset : offset + len(magic)] == magic:
            if fmt == "zip":
                if ext == ".docx":
                    return "docx"
                if ext in (".pptx", ".ppt"):
                    return "pptx"
                if ext in (".xlsx", ".xlsm", ".xltx"):
                    return "xlsx"
                return "zip"
            if fmt == "riff":
                if header[8:12] == b"WEBP" or ext == ".webp":
                    return "image"
                return "unknown"
            if fmt in ("png", "jpeg", "tiff", "bmp", "image"):
                return "image"
            return fmt

    # Known extensions
    if ext in (".pdf",):
        return "pdf"
    if ext in (".docx", ".doc"):
        return "docx"
    if ext in (".pptx", ".ppt"):
        return "pptx"
    if ext in (".xlsx", ".xls"):
        return "xlsx"
    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"):
        return "image"
    if ext in _TEXT_EXTENSIONS:
        return "text"

    # Universal heuristic: if not binary, treat as text
    from extract.text import is_binary_string
    if not is_binary_string(header):
        return "text"

    return "unknown"


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

    elif fmt == "xlsx":
        from extract.text import extract_spreadsheet
        return extract_spreadsheet(file_path, config)

    elif fmt == "text":
        from extract.text import extract_plain_text
        return extract_plain_text(file_path, config)

    elif fmt == "image":
        # Single image file: OCR directly
        from PIL import Image
        profile_name = config.get("active_profile", "balanced")
        ocr_min_conf = config["profiles"][profile_name]["ocr_min_confidence"]
        try:
            img = Image.open(str(path))
            res = ocr_image(img, min_confidence=ocr_min_conf, config=config)
            text, mean_conf = res
            line_confs = getattr(res, "line_confidences", None)
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
                    line_confidences=line_confs,
                )]
            return [Chunk(
                file=file_path,
                location="page:1",
                text=text,
                method="ocr",
                ocr_confidence=mean_conf,
                order_index=0,
                line_confidences=line_confs,
            )]
        except Exception as exc:
            return _blocked(file_path, f"Cannot open image: {exc}")

    else:
        # Final fallback: attempt plain text extraction if not binary
        from extract.text import is_binary_string, extract_plain_text
        try:
            sample = path.read_bytes()[:8192]
            if not is_binary_string(sample):
                log.info("Treating unrecognized format %s as plain text", path.name)
                return extract_plain_text(file_path, config)
        except Exception:
            pass

        log.warning("Unsupported format for %s (detected: %s)", path.name, fmt)
        return _blocked(file_path, f"Unsupported file format: {fmt or path.suffix}")
