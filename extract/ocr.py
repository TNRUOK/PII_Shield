"""
extract/ocr.py
--------------
Shared OCR preprocessing and Tesseract wrapper used by pdf.py, docx.py, pptx.py.

Pipeline:
  PIL Image → grayscale → deskew → denoise → adaptive threshold → pytesseract
"""
from __future__ import annotations

import io
import math
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output

log = logging.getLogger(__name__)

# Auto-detect Tesseract executable on Windows if not on PATH
def _setup_tesseract_cmd() -> None:
    candidates = [
        os.path.expanduser(r"~\scoop\apps\tesseract\current\tesseract.exe"),
        os.path.expanduser(r"~\scoop\shims\tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    if not shutil.which("tesseract"):
        for c in candidates:
            if Path(c).is_file():
                pytesseract.pytesseract.tesseract_cmd = str(Path(c).resolve())
                log.info("Configured Tesseract binary at %s", c)
                break

    # Ensure TESSDATA_PREFIX is set if tessdata is in scoop location
    if "TESSDATA_PREFIX" not in os.environ:
        scoop_tessdata = os.path.expanduser(r"~\scoop\apps\tesseract\current\tessdata")
        if Path(scoop_tessdata).is_dir():
            os.environ["TESSDATA_PREFIX"] = scoop_tessdata

_setup_tesseract_cmd()


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV numpy array (BGR)."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _upsample_if_small(img: Image.Image, min_dim: int = 1800) -> Image.Image:
    """
    If the smallest dimension of the image is below `min_dim`, upscale
    proportionally using Lanczos resampling (equivalent to raising a low-DPI
    scan to ~300 dpi for Tesseract).
    """
    w, h = img.size
    small = min(w, h)
    if small < min_dim:
        scale = min_dim / small
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        log.debug("Upsampled image from %dx%d to %dx%d", w, h, new_w, new_h)
    return img


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert an OpenCV array (BGR or grayscale) to PIL Image."""
    if len(arr.shape) == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew using the angle of the largest contour's
    minimum-area bounding box. Handles up to ±45° rotation.
    """
    # Threshold to find text blobs
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return gray  # not enough content to estimate angle

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    if abs(angle) < 0.5:
        return gray  # negligible skew — skip rotation to avoid degradation

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    log.debug("Deskewed by %.2f°", angle)
    return rotated


def preprocess(img: Image.Image) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a PIL image and return a
    binary numpy array ready for Tesseract.

    Steps:
      1. Upsample if low resolution (<1800px)
      2. Convert to grayscale
      3. Deskew
      4. Otsu binarization (preserves stroke fidelity and punctuation better than adaptive)
    """
    img = _upsample_if_small(img, min_dim=1800)
    arr = pil_to_cv2(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_adaptive(img: Image.Image) -> np.ndarray:
    """Preprocess image with adaptive Gaussian thresholding (for low-contrast scans)."""
    img = _upsample_if_small(img, min_dim=1800)
    arr = pil_to_cv2(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    return binary


def _clean_ocr_line(line: str) -> str:
    """Fix common Tesseract word-boundary segmentation errors."""
    # 1. Join spaced email addresses (e.g. "compliance @cadencefinancial.com" or "user @ domain.com")
    line = re.sub(r'(\b[a-zA-Z0-9._%+\-]+)\s*@\s*([a-zA-Z0-9.\-]+)\b', r'\1@\2', line)
    # 2. Fix missing dot before common domain TLDs (e.g. "example net" -> "example.net")
    line = re.sub(r'(@[a-zA-Z0-9\-]+)\s+(com|org|net|edu|gov|io|ai|co)\b', r'\1.\2', line)
    # 3. Join broken IP address octets where a dot was read as space (e.g. "10.63 58.36" -> "10.63.58.36")
    line = re.sub(r'(\b\d{1,3}\.\d{1,3})\s+(\d{1,3}\.\d{1,3}\b)', r'\1.\2', line)
    line = re.sub(r'(\b\d{1,3}\.\d{1,3}\.\d{1,3})\s+(\d{1,3}\b)', r'\1.\2', line)
    return line


class OCRResult(tuple):
    """
    A 2-tuple (text, mean_confidence) backward-compatible with all callers,
    with an additional line_confidences attribute storing [(line_text, mean_line_conf), ...].
    """
    text: str
    confidence: float
    line_confidences: List[Tuple[str, float]]

    def __new__(cls, text: str, confidence: float, line_confidences: Optional[List[Tuple[str, float]]] = None):
        obj = super().__new__(cls, (text, confidence))
        obj.text = text
        obj.confidence = confidence
        obj.line_confidences = line_confidences or []
        return obj


def _reconstruct_text_with_confs(valid: "pd.DataFrame") -> Tuple[str, List[Tuple[str, float]]]:
    """Preserve line structure and collect per-line mean word confidences."""
    if valid.empty:
        return "", []
    lines = []
    line_confs = []
    sorted_df = valid.sort_values(by=["block_num", "par_num", "line_num", "word_num"])
    for _, line_group in sorted_df.groupby(["block_num", "par_num", "line_num"], sort=False):
        words = [w for w in line_group["text"].dropna().astype(str) if w.strip()]
        if words:
            clean_l = _clean_ocr_line(" ".join(words))
            lines.append(clean_l)
            l_conf = float(line_group["conf"].mean())
            line_confs.append((clean_l, l_conf))
    return "\n".join(lines), line_confs


def _reconstruct_text(valid: "pd.DataFrame") -> str:
    text, _ = _reconstruct_text_with_confs(valid)
    return text


def is_tesseract_available() -> bool:
    """Check if Tesseract executable is found on PATH or at standard Windows paths."""
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    return bool(shutil.which(cmd) or (cmd and Path(cmd).is_file()))


def ocr_image(
    img: Image.Image,
    lang: str = "eng",
    min_confidence: float = 40.0,
    config: Optional[dict] = None,
) -> OCRResult:
    """
    Run Tesseract on a (possibly preprocessed) PIL image.

    Returns
    -------
    OCRResult(text, mean_confidence, line_confidences)
        text             : The recognised text preserving line structure (\\n).
        mean_confidence  : Mean per-word confidence from image_to_data (0–100).
        line_confidences : List of (line_text, mean_line_confidence).
    """
    if not is_tesseract_available():
        log.error("Tesseract not found. Please install Tesseract-OCR and add to PATH.")
        return OCRResult("", 0.0, [])

    psm_primary = config.get("ocr_psm", 6) if config else 6
    oem = config.get("ocr_oem", 3) if config else 3

    preprocessed_otsu = preprocess(img)
    pil_otsu = cv2_to_pil(preprocessed_otsu)

    def _run_tesseract(pil_img: Image.Image, psm: int) -> OCRResult:
        tess_config = f"--oem {oem} --psm {psm}"
        try:
            data = pytesseract.image_to_data(
                pil_img,
                lang=lang,
                config=tess_config,
                output_type=Output.DATAFRAME,
            )
            if data is None or data.empty:
                return OCRResult("", 0.0, [])
            valid = data[data["conf"] > 0].copy()
            if valid.empty:
                return OCRResult("", 0.0, [])
            conf = float(valid["conf"].mean())
            text, line_confs = _reconstruct_text_with_confs(valid)
            return OCRResult(text, conf, line_confs)
        except pytesseract.TesseractNotFoundError:
            log.error("Tesseract not found. Please install Tesseract-OCR and add to PATH.")
            return OCRResult("", 0.0, [])
        except Exception as exc:
            log.error("Tesseract execution failed (psm=%d): %s", psm, exc)
            return OCRResult("", 0.0, [])

    # Primary attempt: psm_primary with Otsu
    best = _run_tesseract(pil_otsu, psm=psm_primary)

    # Fallback 1: If below min_confidence or empty, try PSM 4 (single column variable sizes)
    if best.confidence < min_confidence or not best.text.strip():
        res_psm4 = _run_tesseract(pil_otsu, psm=4)
        if res_psm4.confidence > best.confidence or (not best.text.strip() and res_psm4.text.strip()):
            log.info("PSM-4 retry improved confidence: %.1f → %.1f", best.confidence, res_psm4.confidence)
            best = res_psm4

    # Fallback 2: If still below min_confidence, try PSM 11 (sparse text)
    if best.confidence < min_confidence or not best.text.strip():
        res_psm11 = _run_tesseract(pil_otsu, psm=11)
        if res_psm11.confidence > best.confidence or (not best.text.strip() and res_psm11.text.strip()):
            log.info("PSM-11 retry improved confidence: %.1f → %.1f", best.confidence, res_psm11.confidence)
            best = res_psm11

    # Fallback 3: If still below min_confidence, try adaptive thresholding
    if best.confidence < min_confidence or not best.text.strip():
        preprocessed_adapt = preprocess_adaptive(img)
        pil_adapt = cv2_to_pil(preprocessed_adapt)
        res_adapt = _run_tesseract(pil_adapt, psm=psm_primary)
        if res_adapt.confidence > best.confidence or (not best.text.strip() and res_adapt.text.strip()):
            log.info("Adaptive threshold retry improved confidence: %.1f → %.1f", best.confidence, res_adapt.confidence)
            best = res_adapt

    if not best.text.strip():
        log.warning(
            "Tesseract returned no words with positive confidence (image size: %s)",
            img.size,
        )
        return OCRResult("", 0.0, [])

    log.debug("OCR: %d characters, mean_conf=%.1f", len(best.text), best.confidence)
    return best


_OCR_BYTES_CACHE: Dict[str, Tuple[str, float]] = {}


def get_ocr_cache_size() -> int:
    return len(_OCR_BYTES_CACHE)


def clear_ocr_cache() -> None:
    _OCR_BYTES_CACHE.clear()


def ocr_bytes(
    raw_bytes: bytes,
    fmt: str = "PNG",
    lang: str = "eng",
    min_confidence: float = 40.0,
    config: Optional[dict] = None,
) -> Tuple[str, float]:
    """
    Convenience wrapper: accepts raw image bytes, returns (text, mean_confidence).
    Used by docx.py and pptx.py to OCR embedded images, with sha1 caching.
    """
    import hashlib
    sha1 = hashlib.sha1(raw_bytes).hexdigest()
    if sha1 in _OCR_BYTES_CACHE:
        return _OCR_BYTES_CACHE[sha1]

    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except Exception as exc:
        log.error("Cannot open image bytes: %s", exc)
        return "", 0.0
    res = ocr_image(img, lang=lang, min_confidence=min_confidence, config=config)
    _OCR_BYTES_CACHE[sha1] = res
    return res
