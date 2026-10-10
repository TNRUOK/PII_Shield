"""
tests/test_merge.py
-------------------
Unit tests for detect/merge.py: overlap resolution, span union, de-dup.
"""
import pytest
from detect.rules import RawHit
from detect.merge import merge_hits, _mask


def _hit(etype, start, end, conf, method="rule"):
    return RawHit(
        entity_type=etype,
        start=start,
        end=end,
        text=f"<text_{start}_{end}>",
        confidence=conf,
        sensitivity="Medium",
        method=method,
    )


@pytest.fixture
def config():
    return {
        "active_profile": "balanced",
        "profiles": {"balanced": {"ocr_min_confidence": 40, "ner_confidence_threshold": 0.5, "rule_confidence_threshold": 0.6}},
        "sensitivity_map": {
            "High": ["SSN", "CREDIT_CARD"],
            "Medium": ["EMAIL", "PHONE", "PERSON"],
            "Low": ["IP_ADDRESS"],
        },
    }


class TestMergeHits:
    def test_no_overlap_returns_all(self, config):
        hits = [_hit("EMAIL", 0, 20, 0.9), _hit("PHONE", 30, 50, 0.8)]
        entities = merge_hits(hits, "f.docx", "body", config)
        assert len(entities) == 2

    def test_same_type_overlap_unioned(self, config):
        """Two same-type overlapping hits → one unified entity."""
        hits = [_hit("EMAIL", 0, 20, 0.9), _hit("EMAIL", 15, 30, 0.85)]
        entities = merge_hits(hits, "f.docx", "body", config)
        assert len(entities) == 1
        assert entities[0].start == 0
        assert entities[0].end == 30
        assert entities[0].confidence == 0.9  # max confidence kept

    def test_different_type_overlap_keeps_higher_conf(self, config):
        """Two overlapping hits of different types → keep the higher confidence one."""
        hits = [_hit("PHONE", 0, 15, 0.7), _hit("PERSON", 0, 15, 0.9, method="ner")]
        entities = merge_hits(hits, "f.docx", "body", config)
        assert len(entities) == 1
        assert entities[0].entity_type == "PERSON"
        assert entities[0].confidence == 0.9

    def test_non_overlapping_order(self, config):
        hits = [_hit("SSN", 50, 61, 0.95), _hit("EMAIL", 0, 25, 0.9)]
        entities = merge_hits(hits, "f.docx", "body", config)
        assert len(entities) == 2
        # Should be sorted by start in output
        assert entities[0].start <= entities[1].start

    def test_sensitivity_assigned_from_config(self, config):
        hits = [_hit("SSN", 0, 11, 0.95)]
        entities = merge_hits(hits, "f.docx", "body", config)
        assert entities[0].sensitivity == "High"

    def test_methods_deduplicated(self, config):
        hits = [_hit("EMAIL", 0, 20, 0.9, method="rule"),
                _hit("EMAIL", 0, 20, 0.8, method="ner")]
        entities = merge_hits(hits, "f.docx", "body", config)
        # methods list should not have duplicates
        assert len(entities[0].methods) == len(set(entities[0].methods))

    def test_empty_hits(self, config):
        entities = merge_hits([], "f.docx", "body", config)
        assert entities == []

    def test_single_hit(self, config):
        entities = merge_hits([_hit("IP_ADDRESS", 5, 18, 0.88)], "f.docx", "body", config)
        assert len(entities) == 1
        assert entities[0].entity_type == "IP_ADDRESS"
        assert entities[0].sensitivity == "Low"

    def test_touching_same_type_merged(self, config):
        """Adjacent (touching) same-type hits should merge."""
        hits = [_hit("PERSON", 0, 10, 0.85), _hit("PERSON", 9, 20, 0.80)]
        entities = merge_hits(hits, "f.docx", "body", config)
        assert len(entities) == 1

    def test_different_type_overlap_unions_span(self, config):
        """T5: a lower-confidence hit that overlaps AND extends beyond a higher-confidence hit
        of a different type must still be fully covered — the union span is kept."""
        # hit_a: PERSON at [0,15) with high confidence (rule method)
        # hit_b: EMAIL at [10,25) with lower confidence (ner method)
        # After T5 fix: result must cover [0,25), not [0,15)
        hit_a = _hit("PERSON", 0, 15, 0.90, method="rule")
        hit_b = _hit("EMAIL", 10, 25, 0.70, method="ner")
        entities = merge_hits([hit_a, hit_b], "f.docx", "body", config)
        assert len(entities) == 1
        assert entities[0].start == 0
        assert entities[0].end == 25, f"Expected end=25, got {entities[0].end}"
        assert entities[0].entity_type == "PERSON"  # higher confidence wins type
        # Combined method must contain both original methods
        combined = "+".join(entities[0].methods)
        assert "rule" in combined and "ner" in combined, f"Both methods must be recorded, got: {combined}"
