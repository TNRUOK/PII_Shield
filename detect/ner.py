"""
detect/ner.py
-------------
Layer B: Named Entity Recognition via Microsoft Presidio + spaCy en_core_web_lg.

Presidio wraps spaCy and applies its own set of recognisers on top.
We use it for PERSON (and optionally LOCATION) detection.

All processing is 100% local — no network calls.
"""
from __future__ import annotations

import logging
from typing import List

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

from detect.rules import RawHit

log = logging.getLogger(__name__)

# Module-level singleton — loading spaCy is expensive (~2s); do it once.
_engine: AnalyzerEngine | None = None


def _get_engine() -> AnalyzerEngine:
    global _engine
    if _engine is None:
        log.info("Loading Presidio + spaCy en_core_web_lg (first call only)…")
        # Build the NLP engine backed by spaCy
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        nlp_engine = provider.create_engine()
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        _engine = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
        log.info("Presidio engine ready.")
    return _engine


# Presidio entity type → our canonical type mapping
# As per architecture specification, NER is used for PERSON (and optionally LOCATION)
_TYPE_MAP = {
    "PERSON":   "PERSON",
    "LOCATION": "LOCATION",
}

# Sensitivity for NER-detected types
_SENSITIVITY = {
    "PERSON":       "Medium",
    "LOCATION":     "Low",
    "EMAIL":        "Medium",
    "PHONE":        "Medium",
    "SSN":          "High",
    "CREDIT_CARD":  "High",
    "IBAN":         "High",
    "PASSPORT":     "High",
    "IP_ADDRESS":   "Low",
    "DATE_OF_BIRTH": "Medium",
}


def run_ner(text: str, config: dict) -> List[RawHit]:
    """
    Run Presidio NER on `text` and return RawHit objects.

    Only entity types that are both returned by Presidio and present in
    config["enabled_entities"] are emitted.
    """
    if not text.strip():
        return []

    enabled: set = set(config.get("enabled_entities", []))
    profile_name = config.get("active_profile", "balanced")
    min_conf: float = config["profiles"][profile_name]["ner_confidence_threshold"]

    engine = _get_engine()

    # Presidio entities we care about — map back from their names
    presidio_entities = [k for k, v in _TYPE_MAP.items() if v in enabled]

    try:
        results = engine.analyze(
            text=text,
            entities=presidio_entities if presidio_entities else None,
            language="en",
        )
    except Exception as exc:
        log.error("Presidio analysis error: %s", exc)
        return []

    hits: List[RawHit] = []
    for res in results:
        our_type = _TYPE_MAP.get(res.entity_type, res.entity_type)
        if our_type not in enabled:
            continue
        conf = float(res.score)
        if conf < min_conf:
            continue

        raw_text = text[res.start: res.end]

        hits.append(RawHit(
            entity_type=our_type,
            start=res.start,
            end=res.end,
            text=raw_text,
            confidence=conf,
            sensitivity=_SENSITIVITY.get(our_type, "Medium"),
            method="ner",
        ))

    return hits
