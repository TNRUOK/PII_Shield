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
    if shutil.which("tesseract"):
        return
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if Path(c).is_file():
            pytesseract.pytesseract.tesseract_cmd = c
            log.info("Configured Tesseract binary at %s", c)
            return

_setup_tesseract_cmd()


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV numpy array (BGR)."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


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
    # minAreaRect returns angles in [-90, 0); convert to [-45, 45)
    if angle < -45:
        angle = 90 + angle

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
      1. Convert to grayscale
      2. Deskew
      3. Fast non-local means denoising
      4. Adaptive (Gaussian) thresholding
    """
    arr = pil_to_cv2(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    # Denoise — conservative parameters to avoid blurring thin strokes
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    # Adaptive threshold — better than global Otsu on documents with uneven lighting
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2,
    )
    return binary


def ocr_image(
    img: Image.Image,
    lang: str = "eng",
    min_confidence: float = 40.0,
) -> Tuple[str, float]:
    """
    Run Tesseract on a (possibly preprocessed) PIL image.

    Returns
    -------
    (text, mean_confidence)
        text             : The recognised text (empty string if below threshold).
        mean_confidence  : Mean per-word confidence from image_to_data (0–100).
                           Returns 0.0 if no words recognised.

    If mean_confidence < min_confidence the text is still returned — the *caller*
    decides whether to emit a BLOCKED chunk based on the configured threshold.
    This lets unit tests inspect the confidence value independently.
    """
    preprocessed = preprocess(img)
    pil_preprocessed = cv2_to_pil(preprocessed)

    # image_to_data returns one row per word with individual confidence values
    try:
        data = pytesseract.image_to_data(
            pil_preprocessed,
            lang=lang,
            output_type=Output.DATAFRAME,
        )
    except pytesseract.TesseractNotFoundError:
        log.error("Tesseract not found. Please install Tesseract-OCR and add to PATH.")
        return "", 0.0
    except Exception as exc:
        log.error("Tesseract execution failed: %s", exc)
        return "", 0.0

    # Filter out rows where Tesseract didn't assign a confidence (conf == -1)
    valid = data[data["conf"] > 0].copy()

    if valid.empty:
        log.warning("Tesseract returned no words with positive confidence")
        return "", 0.0

    mean_conf = float(valid["conf"].mean())
    words = valid["text"].dropna().astype(str).tolist()
    text = " ".join(w for w in words if w.strip())

    log.debug("OCR: %d words, mean_conf=%.1f", len(words), mean_conf)
    return text, mean_conf


def ocr_bytes(
    raw_bytes: bytes,
    fmt: str = "PNG",
    lang: str = "eng",
    min_confidence: float = 40.0,
) -> Tuple[str, float]:
    """
    Convenience wrapper: accepts raw image bytes, returns (text, mean_confidence).
    Used by docx.py and pptx.py to OCR embedded images.
    """
    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except Exception as exc:
        log.error("Cannot open image bytes: %s", exc)
        return "", 0.0
    return ocr_image(img, lang=lang, min_confidence=min_confidence)
