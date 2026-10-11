"""
tests/test_redaction.py
-----------------------
Tests for redact/redactor.py and redact/mapping.py.
"""
import pytest
import tempfile
import os
from extract.models import Chunk
from detect.merge import Entity, detect_chunk
from redact.redactor import redact_chunks
from redact.mapping import PlaceholderMapping


def _make_entity(etype, start, end, raw, file="test.docx", location="body"):
    return Entity(
        entity_type=etype,
        sensitivity="Medium",
        confidence=0.9,
        methods=["rule"],
        file=file,
        location=location,
        start=start,
        end=end,
        raw_value=raw,
        masked_value=raw[0] + "*" * (len(raw) - 2) + raw[-1] if len(raw) > 2 else "*" * len(raw),
        placeholder="",
    )


def _chunk(text, location="body", file="test.docx"):
    return Chunk(file=file, location=location, text=text,
                 method="native", ocr_confidence=None, order_index=0)


@pytest.fixture
def mapping(tmp_path):
    return PlaceholderMapping("test.docx", str(tmp_path))


@pytest.fixture
def config():
    return {
        "active_profile": "balanced",
        "profiles": {"balanced": {"ocr_min_confidence": 40, "ner_confidence_threshold": 0.5, "rule_confidence_threshold": 0.6}},
        "sensitivity_map": {"High": [], "Medium": ["PERSON", "EMAIL"], "Low": []},
    }


class TestRedaction:
    def test_single_entity_replaced(self, mapping, config):
        text = "Email john@example.com here."
        chunk = _chunk(text)
        ent = _make_entity("EMAIL", 6, 22, "john@example.com")
        redacted = redact_chunks([chunk], [ent], mapping, config)
        assert "john@example.com" not in redacted[0].text
        assert "[EMAIL_1]" in redacted[0].text

    def test_multiple_entities_same_chunk(self, mapping, config):
        text = "Name: Alice Smith, Email: alice@x.com"
        chunk = _chunk(text)
        ents = [
            _make_entity("PERSON", 6, 17,  "Alice Smith"),
            _make_entity("EMAIL",  26, 36, "alice@x.com"),
        ]
        redacted = redact_chunks([chunk], ents, mapping, config)
        rt = redacted[0].text
        assert "Alice Smith" not in rt
        assert "alice@x.com" not in rt
        assert "[PERSON_1]" in rt
        assert "[EMAIL_1]" in rt

    def test_same_value_same_placeholder(self, mapping, config):
        """The same raw value in two chunks must get the same placeholder token."""
        text1 = "Contact bob@test.com."
        text2 = "Reply to bob@test.com."
        c1 = _chunk(text1, location="body")
        c2 = _chunk(text2, location="footer")
        ent1 = _make_entity("EMAIL", 8, 20, "bob@test.com", location="body")
        ent2 = _make_entity("EMAIL", 9, 21, "bob@test.com", location="footer")
        redacted = redact_chunks([c1, c2], [ent1, ent2], mapping, config)
        ph1 = next((t for t in redacted[0].text.split() if t.startswith("[EMAIL_")), None)
        ph2 = next((t for t in redacted[1].text.split() if t.startswith("[EMAIL_")), None)
        assert ph1 is not None and ph2 is not None
        assert ph1 == ph2, "Same raw value must get same placeholder"

    def test_different_values_different_placeholders(self, mapping, config):
        text = "a@example.com and b@example.com"
        chunk = _chunk(text)
        ents = [
            _make_entity("EMAIL", 0,  13, "a@example.com"),
            _make_entity("EMAIL", 18, 31, "b@example.com"),
        ]
        redacted = redact_chunks([chunk], ents, mapping, config)
        rt = redacted[0].text
        # Both should be replaced but with different tokens
        assert "[EMAIL_1]" in rt
        assert "[EMAIL_2]" in rt

    def test_table_structure_preserved(self, mapping, config):
        """Pipe separators in table rows must not be disturbed by redaction."""
        text = "Alice Johnson | alice@co.com | 555-999-8888"
        chunk = _chunk(text, location="table:1/r2[headers: Name | Email | Phone]")
        ent = _make_entity("EMAIL", 15, 28, "alice@co.com",
                           location="table:1/r2[headers: Name | Email | Phone]")
        redacted = redact_chunks([chunk], [ent], mapping, config)
        assert "|" in redacted[0].text, "Pipe separators must survive redaction"

    def test_blocked_chunk_passes_through(self, mapping, config):
        blocked = Chunk(file="f.pdf", location="page:1", text="",
                        method="blocked", ocr_confidence=5.0, order_index=0,
                        blocked_reason="Low confidence")
        redacted = redact_chunks([blocked], [], mapping, config)
        assert redacted[0].is_blocked()

    def test_mapping_saved(self, mapping, config, tmp_path):
        text = "SSN: 123-45-6789"
        chunk = _chunk(text)
        ent = _make_entity("SSN", 5, 16, "123-45-6789")
        redact_chunks([chunk], [ent], mapping, config)
        mapping.save()
        assert mapping.mapping_path.exists()
        # Verify file at rest is encrypted — raw PII never appears in plaintext
        raw_content = mapping.mapping_path.read_bytes()
        assert b"123-45-6789" not in raw_content
        # Decrypt via load() using session key
        m2 = PlaceholderMapping("test.docx", str(tmp_path), key=mapping.key)
        m2.load()
        assert m2._placeholder_to_raw.get("[SSN_1]") == "123-45-6789"

    def test_mapping_secure_purge(self, mapping, config, tmp_path):
        """Purge must securely overwrite and delete the mapping file."""
        text = "Account: 9876543210"
        chunk = _chunk(text)
        ent = _make_entity("ACCOUNT", 9, 19, "9876543210")
        redact_chunks([chunk], [ent], mapping, config)
        mapping.save()
        assert mapping.mapping_path.exists()
        res = mapping.purge()
        assert res is True
        assert not mapping.mapping_path.exists()
        assert len(mapping._placeholder_to_raw) == 0

    def test_mapping_separate_key_storage(self, config, tmp_path):
        """Key file stored in separate directory when requested; never next to mapping."""
        key_dir = tmp_path / "keys_vault"
        out_dir = tmp_path / "output_vault"
        m = PlaceholderMapping("doc.docx", str(out_dir), key_dir=str(key_dir), save_key_to_file=True)
        m.get_or_create("EMAIL", "user@test.org")
        m.save()
        assert m.key_path is not None
        assert m.key_path.exists()
        assert m.key_path.parent != m.mapping_path.parent
        # Purge deletes both
        from redact.mapping import purge_mapping
        purge_mapping(m)
        assert not m.mapping_path.exists()
        assert not m.key_path.exists()

