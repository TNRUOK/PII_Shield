"""
extract/regions.py
------------------
Targeted region detection, high-resolution multi-PSM OCR, and multimodal safety
for scanned documents, figures, diagrams, and tables (Issues 9 and 10).
"""
from __future__ import annotations

import difflib
import logging
import math
import os
import re
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
from pytesseract import Output

from extract.models import Chunk
from extract.ocr import (
    pil_to_cv2,
    cv2_to_pil,
    _deskew,
    _clean_ocr_line,
    _reconstruct_text_with_confs,
    is_tesseract_available,
    OCRResult,
)

log = logging.getLogger(__name__)


def detect_candidate_regions(
    pil_img: Image.Image,
    config: Optional[dict] = None,
) -> List[Tuple[int, int, int, int, str]]:
    """
    Detect candidate visual regions (tables, figures, stamps, diagrams) on an image.

    Returns list of tuples: (x, y, w, h, region_type)
    where region_type is 'table', 'stamp', 'diagram', or 'dense_block'.
    """
    cfg = config or {}
    min_size = cfg.get("ocr_region_min_px", [120, 60])
    min_w, min_h = min_size[0], min_size[1]
    max_regions = 12

    img_w, img_h = pil_img.size
    total_area = img_w * img_h
    arr = pil_to_cv2(pil_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    candidates: List[Tuple[int, int, int, int, str]] = []

    # 1. Dark panels / inverted screenshots (dark theme admin console, colored boxes)
    try:
        dark_mask = (gray < 160).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if w >= min_w and h >= min_h and area < 0.90 * total_area:
                crop = gray[y:y+h, x:x+w]
                if np.mean(crop) < 150:
                    candidates.append((x, y, w, h, "figure"))
    except Exception as exc:
        log.debug("Dark panel detection failed: %s", exc)

    # 2. Structural / Table detection (morphological horizontal & vertical lines)
    try:
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h_size = max(img_w // 30, 25)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
        h_lines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, h_kernel)

        v_size = max(img_h // 30, 25)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        v_lines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, v_kernel)

        grid = cv2.bitwise_and(h_lines, v_lines)
        if cv2.countNonZero(grid) > 4:
            table_mask = cv2.add(h_lines, v_lines)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
            table_contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in table_contours:
                x, y, w, h = cv2.boundingRect(c)
                if w >= min_w and h >= min_h and (w * h) < 0.90 * total_area:
                    candidates.append((x, y, w, h, "table"))
    except Exception as exc:
        log.debug("Table morphology detection failed: %s", exc)

    # 3. Dense diagrams / figures / stamps (non-white background or high edge density)
    try:
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if w >= min_w and h >= min_h and area < 0.85 * total_area:
                crop = gray[y:y+h, x:x+w]
                crop_edges = edges[y:y+h, x:x+w]
                edge_density = cv2.countNonZero(crop_edges) / float(area)
                mean_lum = float(np.mean(crop))
                # Text paragraphs have mean_lum > 190 and edge_density < 0.08
                # Real figures/stamps/badges have non-white background or high edge density
                if mean_lum < 160 or (edge_density > 0.15 and 0.6 <= (float(w)/max(h,1)) <= 1.6):
                    aspect = float(w) / max(h, 1)
                    if 0.7 <= aspect <= 1.4 and (area < 0.20 * total_area):
                        candidates.append((x, y, w, h, "stamp"))
                    else:
                        candidates.append((x, y, w, h, "figure"))
    except Exception as exc:
        log.debug("Edge/diagram detection failed: %s", exc)

    if not candidates:
        return []

    # 3. Merge overlapping / nested bounding boxes
    merged: List[Tuple[int, int, int, int, str]] = []

    def _boxes_overlap(b1, b2) -> bool:
        x1, y1, w1, h1, _ = b1
        x2, y2, w2, h2, _ = b2
        # Check intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        if xi2 <= xi1 or yi2 <= yi1:
            return False
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        # Overlap if intersection is > 20% of either box or one contains the other
        return (inter_area / min(area1, area2)) > 0.20

    def _union_box(b1, b2) -> Tuple[int, int, int, int, str]:
        x1, y1, w1, h1, t1 = b1
        x2, y2, w2, h2, t2 = b2
        nx1 = min(x1, x2)
        ny1 = min(y1, y2)
        nx2 = max(x1 + w1, x2 + w2)
        ny2 = max(y1 + h1, y2 + h2)
        chosen_type = "table" if "table" in (t1, t2) else ("stamp" if "stamp" in (t1, t2) else "figure")
        return (nx1, ny1, nx2 - nx1, ny2 - ny1, chosen_type)

    for box in candidates:
        to_merge_idx = -1
        for idx, m_box in enumerate(merged):
            if _boxes_overlap(box, m_box):
                to_merge_idx = idx
                break
        if to_merge_idx >= 0:
            merged[to_merge_idx] = _union_box(merged[to_merge_idx], box)
        else:
            merged.append(box)

    # Filter out boxes that cover almost the whole page (>90% area)
    filtered = [b for b in merged if (b[2] * b[3]) < 0.90 * total_area and b[2] >= min_w and b[3] >= min_h]

    # Sort by area descending and cap at max_regions
    filtered.sort(key=lambda b: b[2] * b[3], reverse=True)
    return filtered[:max_regions]


def preprocess_region(crop_img: Image.Image) -> Tuple[Image.Image, float]:
    """
    High-resolution preprocessing for a cropped region:
    - Upscale with Lanczos interpolation if longest edge < 2400 (cap at 5000)
    - Check luminance: invert if mean < 110 (dark background / inverted text)
    - Deskew small angles
    - CLAHE + Otsu contrast enhancement
    Returns (preprocessed_pil_image, scale_factor).
    """
    w, h = crop_img.size
    longest = max(w, h)
    scale = 1.0

    if longest < 2400:
        scale = min(2400.0 / max(longest, 1), 4.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if max(new_w, new_h) <= 5000:
            crop_img = crop_img.resize((new_w, new_h), Image.LANCZOS)
        else:
            scale = 1.0

    arr = pil_to_cv2(crop_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    # Inversion if dark background
    mean_lum = float(np.mean(gray))
    if mean_lum < 110.0:
        gray = cv2.bitwise_not(gray)

    # Deskew
    gray = _deskew(gray)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu binarization
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2_to_pil(binary), scale


def _estimate_char_height(pil_bin: Image.Image, scale: float) -> float:
    """Estimate median character height in original unscaled pixels."""
    arr = np.array(pil_bin.convert("L"))
    # Invert binary so text is white
    _, inv = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = []
    for c in contours:
        _, _, _, ch = cv2.boundingRect(c)
        if ch > 4:  # filter noise
            heights.append(ch / max(scale, 1.0))
    if not heights:
        return 0.0
    return float(np.median(heights))


def ocr_region(
    pil_img: Image.Image,
    bbox: Tuple[int, int, int, int, str],
    config: Optional[dict] = None,
    lang: str = "eng",
) -> Tuple[str, float, str, Optional[List[Tuple[str, float]]]]:
    """
    Run multi-PSM high-resolution OCR on a candidate region.

    Parameters
    ----------
    pil_img : Full page or parent PIL image.
    bbox    : (x, y, w, h, region_type)
    config  : Loaded config dict.
    lang    : Tesseract language string.

    Returns
    -------
    (text, mean_confidence, image_status, line_confidences)
    image_status in ("READ", "LOW_CONFIDENCE", "UNREAD", "UNVERIFIED")
    """
    if not is_tesseract_available():
        return "", 0.0, "UNREAD", None

    cfg = config or {}
    conf_thresh = float(cfg.get("ocr_region_conf", 70))
    min_w_per_mp = float(cfg.get("ocr_min_words_per_megapixel", 10))
    oem = cfg.get("ocr_oem", 3)

    x, y, w, h, rtype = bbox
    img_w, img_h = pil_img.size

    # Crop with 20px padding clamped to image bounds
    pad = 20
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_w, x + w + pad)
    y1 = min(img_h, y + h + pad)

    crop = pil_img.crop((x0, y0, x1, y1))
    preprocessed_bin, scale = preprocess_region(crop)

    # Check font height before running full multi-psm
    char_height = _estimate_char_height(preprocessed_bin, scale)

    # Multi-PSM strategy: PSM 6 (uniform block), PSM 11 (sparse text), and PSM 4 (column/table)
    psms = [6, 11]
    if rtype == "table":
        psms = [4, 6, 11]

    candidates: List[OCRResult] = []

    for psm in psms:
        tess_config = f"--oem {oem} --psm {psm}"
        try:
            data = pytesseract.image_to_data(
                preprocessed_bin,
                lang=lang,
                config=tess_config,
                output_type=Output.DATAFRAME,
            )
            if data is not None and not data.empty:
                valid = data[data["conf"] > 0].copy()
                if not valid.empty:
                    conf = float(valid["conf"].mean())
                    text, line_confs = _reconstruct_text_with_confs(valid)
                    if text.strip():
                        candidates.append(OCRResult(text.strip(), conf, line_confs))
        except Exception as exc:
            log.debug("Region OCR failed (psm=%d): %s", psm, exc)

    if not candidates:
        status = "UNREAD" if char_height >= 7.0 else "UNVERIFIED"
        return "", 0.0, status, None

    # Pick candidate with highest valid word yield and best confidence
    # If a candidate captures numbers/emails or significantly more words, prefer it
    candidates.sort(
        key=lambda c: (len(c.text.split()), c.confidence),
        reverse=True
    )
    best = candidates[0]

    # Combine text lines from other PSMs if they contain critical tokens (emails, phones, IDs) missed by best
    merged_lines = list(best.text.split("\n"))
    merged_text_set = set(best.text.split())
    for other in candidates[1:]:
        for line in other.text.split("\n"):
            words = [w for w in line.split() if w]
            # Check if this line has an email, phone, or ID not present in best
            has_special = any("@" in w or "-" in w or "+" in w for w in words)
            if has_special and not any(w in merged_text_set for w in words):
                merged_lines.append(line)
                merged_text_set.update(words)

    final_text = "\n".join(merged_lines).strip()
    final_conf = best.confidence

    # Compute yield metrics
    crop_w = x1 - x0
    crop_h = y1 - y0
    mp = (crop_w * crop_h) / 1_000_000.0
    word_count = len(final_text.split())
    words_per_mp = word_count / max(mp, 0.01)

    # Determine status
    if char_height < 7.0 and char_height > 0:
        image_status = "UNVERIFIED"
    elif final_conf >= conf_thresh and words_per_mp >= min_w_per_mp and word_count > 0:
        image_status = "READ"
    elif word_count > 0 and final_conf > 0:
        image_status = "LOW_CONFIDENCE"
    else:
        image_status = "UNREAD"

    return final_text, final_conf, image_status, best.line_confidences


def is_duplicate_region_text(region_text: str, page_text: str, threshold: float = 0.50) -> bool:
    """
    Check if region text is already substantially represented in the page-level OCR text.
    Prevents creating redundant duplicate chunks for text already cleanly read.
    """
    clean_reg = " ".join(region_text.split()).strip().lower()
    clean_page = " ".join(page_text.split()).strip().lower()

    if not clean_reg:
        return True

    # 1. Direct substring match
    if clean_reg in clean_page:
        return True

    # 2. Check informative tokens (tokens >= 4 chars or containing digits like IDs, phones)
    reg_tokens = [w for w in clean_reg.split() if len(w) >= 4 or any(c.isdigit() for c in w)]
    if reg_tokens:
        page_token_set = set(clean_page.split())
        overlap = sum(1 for t in reg_tokens if t in page_token_set or t in clean_page)
        if (overlap / len(reg_tokens)) >= threshold:
            return True

    # 3. Sequence matcher on normalized strings
    ratio = difflib.SequenceMatcher(None, clean_reg, clean_page).quick_ratio()
    return ratio >= 0.80


def blank_regions(
    page_img: Image.Image,
    bboxes: List[Tuple[int, int, int, int]],
    fill_color: Tuple[int, int, int] = (220, 220, 220),
) -> Image.Image:
    """
    Create a safe derivative page image by blanking out unverified / unread visual regions.
    """
    safe_img = page_img.copy()
    draw = ImageDraw.Draw(safe_img)
    for (x, y, w, h) in bboxes:
        draw.rectangle([x, y, x + w, y + h], fill=fill_color, outline=(150, 150, 150), width=2)
    return safe_img
