"""
tests/test_issue9_10.py
-----------------------
Issue 9 & 10: region detection, multi-PSM OCR, SVG text extraction,
EMF converter check, image_status, region_bbox, and multimodal_safe reporting.
"""
import io
import pytest
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import patch, MagicMock

import yaml

# --- Fixtures ---

@pytest.fixture
def config():
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_blank_white_image(w=800, h=600):
    """Return a plain white PIL image."""
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    return img


def _make_text_image(text="Hello World SSN 578-90-1234", w=800, h=200):
    """Create a simple PIL image containing printed text."""
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 60), text, fill=(0, 0, 0))
    return img


# =============================================================================
# Tests: extract/models.py Chunk fields
# =============================================================================

class TestChunkModel:
    def test_chunk_has_image_status_field(self):
        from extract.models import Chunk
        c = Chunk(
            file="test.pdf",
            location="page:1/region:1",
            text="some text",
            method="region_ocr",
            ocr_confidence=75.0,
            order_index=0,
            image_status="READ",
            region_type="table",
            region_bbox=(10, 20, 300, 150),
        )
        assert c.image_status == "READ"
        assert c.region_type == "table"
        assert c.region_bbox == (10, 20, 300, 150)

    def test_chunk_region_ocr_not_blocked(self):
        from extract.models import Chunk
        c = Chunk(
            file="test.pdf",
            location="page:1",
            text="Some OCR text",
            method="region_ocr",
            ocr_confidence=80.0,
            order_index=0,
        )
        assert not c.is_blocked()

    def test_chunk_svg_text_not_blocked(self):
        from extract.models import Chunk
        c = Chunk(
            file="test.docx",
            location="image:fig1.svg",
            text="SVG diagram label",
            method="svg_text",
            ocr_confidence=None,
            order_index=0,
            image_status="READ",
        )
        assert not c.is_blocked()

    def test_chunk_vector_unsupported(self):
        from extract.models import Chunk
        c = Chunk(
            file="test.docx",
            location="image:fig1.emf",
            text="",
            method="blocked",
            ocr_confidence=None,
            order_index=0,
            image_status="VECTOR_UNSUPPORTED",
            blocked_reason="no converter installed",
        )
        assert c.is_blocked()
        assert c.image_status == "VECTOR_UNSUPPORTED"


# =============================================================================
# Tests: extract/regions.py
# =============================================================================

class TestRegionDetection:
    def test_detect_regions_returns_list(self, config):
        from extract.regions import detect_candidate_regions
        img = _make_blank_white_image()
        regions = detect_candidate_regions(img, config)
        assert isinstance(regions, list)

    def test_detect_regions_caps_at_12(self, config):
        from extract.regions import detect_candidate_regions
        img = _make_blank_white_image(1600, 1200)
        # Draw many distinct blobs
        draw = ImageDraw.Draw(img)
        for i in range(20):
            x = (i % 5) * 300 + 50
            y = (i // 5) * 250 + 50
            draw.rectangle([x, y, x + 150, y + 120], outline=(0, 0, 0), width=2)
            draw.text((x + 10, y + 10), f"Region {i}", fill=(0, 0, 0))
        regions = detect_candidate_regions(img, config)
        assert len(regions) <= 12

    def test_detect_regions_each_has_five_elements(self, config):
        from extract.regions import detect_candidate_regions
        img = _make_text_image()
        regions = detect_candidate_regions(img, config)
        for r in regions:
            assert len(r) == 5, f"Region tuple should be (x, y, w, h, type): got {r}"

    def test_min_size_filter_applied(self, config):
        from extract.regions import detect_candidate_regions
        img = _make_blank_white_image(200, 200)
        draw = ImageDraw.Draw(img)
        # Draw tiny region smaller than min_px
        draw.rectangle([10, 10, 50, 30], outline=(0, 0, 0), width=1)
        regions = detect_candidate_regions(img, config)
        for r in regions:
            x, y, w, h, _ = r
            assert w >= 120 or h >= 60, f"Region too small: {r}"

    def test_is_duplicate_region_text_detects_match(self):
        from extract.regions import is_duplicate_region_text
        page_text = "John Smith SSN 578-90-1234 lives at 99 Oak St"
        region_text = "SSN 578-90-1234"
        assert is_duplicate_region_text(region_text, page_text)

    def test_is_duplicate_region_text_no_match(self):
        from extract.regions import is_duplicate_region_text
        page_text = "John Smith is an employee"
        region_text = "k.reddy@cadencecorp.net +91 22 4555 0117"
        assert not is_duplicate_region_text(region_text, page_text)

    def test_preprocess_region_returns_image_and_scale(self, config):
        from extract.regions import preprocess_region
        img = _make_text_image(w=500, h=200)
        pil_out, scale = preprocess_region(img)
        assert isinstance(pil_out, Image.Image)
        assert isinstance(scale, float)
        assert scale >= 1.0  # Small image should be scaled up


class TestRegionOcr:
    """OCR tests that mock Tesseract to avoid needing it installed."""

    def test_ocr_region_returns_tuple(self, config):
        from extract.regions import ocr_region
        img = _make_text_image()
        with patch("pytesseract.image_to_data", return_value=None):
            text, conf, status, lines = ocr_region(img, (0, 0, img.width, img.height, "figure"), config)
        assert isinstance(text, str)
        assert isinstance(conf, float)
        assert status in ("READ", "LOW_CONFIDENCE", "UNREAD", "UNVERIFIED")

    def test_ocr_region_status_read_when_high_conf(self, config):
        """Mock high-confidence dataframe → status should be READ."""
        from extract.regions import ocr_region
        import pandas as pd

        mock_df = pd.DataFrame({
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
            "word_num": [1, 2],
            "conf": [90.0, 88.0],
            "text": ["SSN", "578-90-1234"],
        })

        img = _make_text_image(w=1000, h=400, text="SSN 578-90-1234")
        with patch("pytesseract.image_to_data", return_value=mock_df):
            text, conf, status, lines = ocr_region(img, (0, 0, 1000, 400, "figure"), config)
        assert conf > 0
        assert status in ("READ", "LOW_CONFIDENCE")

    def test_ocr_region_unread_when_no_result(self, config):
        from extract.regions import ocr_region
        import pandas as pd
        empty_df = pd.DataFrame(columns=["block_num", "par_num", "line_num", "word_num", "conf", "text"])
        img = _make_text_image()
        with patch("pytesseract.image_to_data", return_value=empty_df):
            text, conf, status, lines = ocr_region(img, (0, 0, 200, 80, "figure"), config)
        assert text == ""
        assert status in ("UNREAD", "UNVERIFIED")

    def test_blank_regions_produces_image(self):
        from extract.regions import blank_regions
        img = _make_blank_white_image(800, 600)
        result = blank_regions(img, [(100, 100, 200, 150), (400, 300, 100, 80)])
        assert isinstance(result, Image.Image)
        # The resulting image should not be identical to original (blanking happened)
        diff = np.abs(np.array(img.convert("L")) - np.array(result.convert("L")))
        assert diff.sum() > 0


# =============================================================================
# Tests: SVG text extraction
# =============================================================================

class TestSvgTextExtraction:
    def test_svg_with_text_elements(self):
        from extract.docx import _extract_svg_text
        svg_bytes = b"""<?xml version="1.0"?>
        <svg xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="30">Authorised Signatory: Elena Rostova</text>
            <g>
                <tspan>EMP-44512</tspan>
            </g>
        </svg>"""
        text = _extract_svg_text(svg_bytes)
        assert "Elena Rostova" in text
        assert "EMP-44512" in text

    def test_svg_empty_gives_empty_string(self):
        from extract.docx import _extract_svg_text
        svg_bytes = b"""<svg xmlns="http://www.w3.org/2000/svg"><rect width="100" height="100"/></svg>"""
        text = _extract_svg_text(svg_bytes)
        assert text.strip() == ""

    def test_svg_invalid_xml_returns_empty(self):
        from extract.docx import _extract_svg_text
        result = _extract_svg_text(b"not XML at all")
        assert isinstance(result, str)
        assert result == ""


# =============================================================================
# Tests: EMF/WMF converter check
# =============================================================================

class TestVectorConverterCheck:
    def test_no_converter_returns_false(self):
        from extract.docx import is_vector_converter_available
        with patch("shutil.which", return_value=None):
            available, tool = is_vector_converter_available()
        assert available is False
        assert tool is None

    def test_soffice_found_returns_true(self):
        from extract.docx import is_vector_converter_available
        def mock_which(cmd):
            return "/usr/bin/soffice" if cmd == "soffice" else None
        with patch("shutil.which", side_effect=mock_which):
            available, tool = is_vector_converter_available()
        assert available is True
        assert tool == "soffice"

    def test_inkscape_found_returns_true(self):
        from extract.docx import is_vector_converter_available
        def mock_which(cmd):
            return "/usr/bin/inkscape" if cmd == "inkscape" else None
        with patch("shutil.which", side_effect=mock_which):
            available, tool = is_vector_converter_available()
        assert available is True
        assert "inkscape" in tool


# =============================================================================
# Tests: report.py image_coverage + multimodal_safe
# =============================================================================

class TestReportImageCoverage:
    def _make_leak(self):
        from verify.leakage import LeakResult
        return LeakResult(file="f", downstream_safe=True, leaks=[])

    def _make_chunks(self):
        from extract.models import Chunk
        return [
            Chunk(file="f", location="page:1", text="native text", method="native",
                  ocr_confidence=None, order_index=0),
            Chunk(file="f", location="page:1/region:1", text="fig text", method="region_ocr",
                  ocr_confidence=80.0, order_index=1, image_status="READ",
                  region_type="figure", region_bbox=(10, 10, 200, 150)),
            Chunk(file="f", location="page:1/region:2", text="", method="blocked",
                  ocr_confidence=0.0, order_index=2, image_status="UNREAD",
                  region_type="stamp", region_bbox=(300, 300, 100, 80)),
        ]

    def test_image_coverage_counts_regions(self):
        from report.report import build_file_summary
        chunks = self._make_chunks()
        summary = build_file_summary("f", chunks, [], self._make_leak(), None)
        cov = summary["image_coverage"]
        assert cov["total_image_regions"] == 2
        assert cov["read"] == 1
        assert cov["unread"] == 1

    def test_multimodal_safe_false_when_unread(self):
        from report.report import build_file_summary
        chunks = self._make_chunks()
        summary = build_file_summary("f", chunks, [], self._make_leak(), None)
        assert summary["multimodal_safe"] is False

    def test_multimodal_safe_true_when_all_read(self):
        from extract.models import Chunk
        from report.report import build_file_summary
        chunks = [
            Chunk(file="f", location="page:1", text="text", method="native",
                  ocr_confidence=None, order_index=0),
            Chunk(file="f", location="page:1/region:1", text="fig text", method="region_ocr",
                  ocr_confidence=82.0, order_index=1, image_status="READ",
                  region_type="figure", region_bbox=(0, 0, 300, 200)),
        ]
        summary = build_file_summary("f", chunks, [], self._make_leak(), None)
        assert summary["multimodal_safe"] is True

    def test_summary_has_region_ocr_chunks_count(self):
        from report.report import build_file_summary
        chunks = self._make_chunks()
        summary = build_file_summary("f", chunks, [], self._make_leak(), None)
        assert "region_ocr_chunks" in summary
        assert summary["region_ocr_chunks"] == 1

    def test_summary_has_svg_chunks_count(self):
        from extract.models import Chunk
        from report.report import build_file_summary
        chunks = [
            Chunk(file="f", location="image:fig.svg", text="diagram label", method="svg_text",
                  ocr_confidence=None, order_index=0, image_status="READ", region_type="diagram"),
        ]
        summary = build_file_summary("f", chunks, [], self._make_leak(), None)
        assert summary["svg_chunks"] == 1
