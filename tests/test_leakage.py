"""
tests/test_leakage.py
---------------------
Tests for verify/leakage.py:
  - Clean redacted text produces downstream_safe=True (0 leaks).
  - Placeholder tokens are not falsely flagged as PII.
  - Injected leftover PII produces downstream_safe=False and lists leaks.
  - Blocked chunks are handled safely without errors.
"""
from pathlib import Path
import pytest
import yaml

from extract.models import Chunk
from verify.leakage import check_leakage, _strip_placeholders

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@pytest.fixture
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_strip_placeholders():
    text = "Contact [PERSON_1] at [EMAIL_1] or call [PHONE_1]."
    stripped = _strip_placeholders(text)
    assert "[PERSON_1]" not in stripped
    assert "[EMAIL_1]" not in stripped
    assert "[PHONE_1]" not in stripped
    assert "Contact" in stripped and "call" in stripped


def test_clean_redacted_text_is_safe(config):
    """A chunk containing only non-PII and placeholder tokens should pass cleanly."""
    chunks = [
        Chunk(
            file="test_doc.txt",
            location="page:1",
            text="The report was approved by [PERSON_1]. You can reach out at [EMAIL_1].",
            method="native",
            ocr_confidence=None,
            order_index=0,
        )
    ]
    result = check_leakage(chunks, config)
    assert result.downstream_safe is True
    assert len(result.leaks) == 0
    assert "downstream_safe=YES" in result.summary()


def test_leaked_ssn_detected(config):
    """A chunk containing unredacted SSN should be flagged as a leak."""
    chunks = [
        Chunk(
            file="test_doc.txt",
            location="page:1",
            text="The participant SSN is 219-45-6789 and was reviewed by [PERSON_1].",
            method="native",
            ocr_confidence=None,
            order_index=0,
        )
    ]
    result = check_leakage(chunks, config)
    assert result.downstream_safe is False
    assert len(result.leaks) > 0
    leak_types = [e.entity_type for e in result.leaks]
    assert "SSN" in leak_types


def test_leaked_email_detected(config):
    """A chunk containing unredacted email should be flagged as a leak."""
    chunks = [
        Chunk(
            file="test_doc.txt",
            location="page:2",
            text="Send replies directly to secret.agent@cadence-corp.com promptly.",
            method="native",
            ocr_confidence=None,
            order_index=1,
        )
    ]
    result = check_leakage(chunks, config)
    assert result.downstream_safe is False
    leak_types = [e.entity_type for e in result.leaks]
    assert "EMAIL" in leak_types


def test_blocked_chunk_handled_safely(config):
    """Blocked chunks should be skipped during leakage check."""
    chunks = [
        Chunk(
            file="test_doc.pdf",
            location="page:3",
            text="",
            method="blocked",
            ocr_confidence=None,
            order_index=2,
            blocked_reason="OCR confidence below threshold",
        )
    ]
    result = check_leakage(chunks, config)
    assert result.downstream_safe is True
    assert len(result.leaks) == 0
