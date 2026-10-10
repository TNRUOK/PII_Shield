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


def is_tesseract_available() -> bool:
    """Check if Tesseract executable is found on PATH or at standard Windows paths."""
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    return bool(shutil.which(cmd) or (cmd and Path(cmd).is_file()))


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
    if not is_tesseract_available():
        log.error("Tesseract not found. Please install Tesseract-OCR and add to PATH.")
        return "", 0.0

    preprocessed = preprocess(img)
    pil_preprocessed = cv2_to_pil(preprocessed)

    def _run_tesseract(pil_img: Image.Image, psm: int) -> "pd.DataFrame | None":
        """Run tesseract with LSTM engine and given PSM; return dataframe or None."""
        tess_config = f"--oem 3 --psm {psm}"
        try:
            return pytesseract.image_to_data(
                pil_img,
                lang=lang,
                config=tess_config,
                output_type=Output.DATAFRAME,
            )
        except pytesseract.TesseractNotFoundError:
            log.error("Tesseract not found. Please install Tesseract-OCR and add to PATH.")
            return None
        except Exception as exc:
            log.error("Tesseract execution failed (psm=%d): %s", psm, exc)
            return None

    # Primary attempt: PSM 6 — assume uniform block of text (best for documents)
    data = _run_tesseract(pil_preprocessed, psm=6)
    if data is None:
        return "", 0.0

    valid = data[data["conf"] > 0].copy()
    mean_conf = float(valid["conf"].mean()) if not valid.empty else 0.0

    # Fallback: PSM 3 (fully automatic) when primary confidence is very low
    if mean_conf < 50 or valid.empty:
        log.info(
            "OCR PSM-6 low confidence (%.1f). Retrying with PSM-3 (auto page segmentation)...",
            mean_conf,
        )
        data2 = _run_tesseract(pil_preprocessed, psm=3)
        if data2 is not None:
            valid2 = data2[data2["conf"] > 0].copy()
            mean_conf2 = float(valid2["conf"].mean()) if not valid2.empty else 0.0
            if mean_conf2 > mean_conf:
                log.info("PSM-3 retry improved confidence: %.1f → %.1f", mean_conf, mean_conf2)
                valid = valid2
                mean_conf = mean_conf2

    if valid.empty:
        log.warning(
            "Tesseract returned no words with positive confidence (image size: %s)",
            img.size,
        )
        return "", 0.0

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
