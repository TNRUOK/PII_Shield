"""
tests/test_fail_closed.py
-------------------------
Tests for fail-closed behavior across extractors and routing:
  - Non-existent file → BLOCKED chunk
  - Empty (0-byte) file → BLOCKED chunk
  - Corrupt PDF file → BLOCKED chunk
  - Unsupported extension/magic bytes → BLOCKED chunk
  - Extremely blurry/empty image below OCR threshold → BLOCKED chunk
"""
from pathlib import Path
import tempfile
import pytest
import yaml
from PIL import Image

from extract.router import route
from extract.models import Chunk
from extract.ocr import ocr_image

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@pytest.fixture
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_nonexistent_file_blocked(config):
    """Routing a non-existent file path must return a BLOCKED chunk."""
    chunks = route("non_existent_path_12345.pdf", config)
    assert len(chunks) == 1
    assert chunks[0].is_blocked()
    assert "not found" in chunks[0].blocked_reason.lower()


def test_empty_file_blocked(config, tmp_path):
    """An empty 0-byte file must return a BLOCKED chunk."""
    empty_file = tmp_path / "empty.docx"
    empty_file.write_bytes(b"")

    chunks = route(str(empty_file), config)
    assert len(chunks) == 1
    assert chunks[0].is_blocked()
    assert "empty file" in chunks[0].blocked_reason.lower()


def test_corrupt_pdf_blocked(config, tmp_path):
    """A corrupt file claiming to be a PDF must fail closed with a BLOCKED chunk."""
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4\ncorrupted garbage binary data\x00\xff")

    chunks = route(str(bad_pdf), config)
    assert len(chunks) >= 1
    assert any(c.is_blocked() for c in chunks)


def test_unsupported_file_format_blocked(config, tmp_path):
    """An unsupported binary format must fail closed with a BLOCKED chunk."""
    unsupported = tmp_path / "data.xyz"
    unsupported.write_bytes(b"SOME_UNKNOWN_FORMAT_MAGIC_BYTES_123456789")

    chunks = route(str(unsupported), config)
    assert len(chunks) == 1
    assert chunks[0].is_blocked()
    assert "unsupported" in chunks[0].blocked_reason.lower()


def test_blank_image_ocr_blocked(config, tmp_path):
    """A completely blank white image should fail to meet OCR min confidence and be BLOCKED."""
    blank_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    img_path = tmp_path / "blank.png"
    blank_img.save(str(img_path))

    chunks = route(str(img_path), config)
    assert len(chunks) == 1
    assert chunks[0].is_blocked()
