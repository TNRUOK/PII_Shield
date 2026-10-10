"""
extract/models.py
-----------------
Core data contracts for the extraction layer.
A Chunk represents one logical text block from any document format.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class Chunk:
    """
    A normalised unit of extracted text from any supported document format.

    Attributes
    ----------
    file          : Absolute path to the source file.
    location      : Human-readable location descriptor, e.g.
                    "page:3", "slide:2/notes", "table:1/r2c3",
                    "header", "footer", "footnote:1", "metadata", "image:2".
    text          : The extracted text content (empty string if blocked).
    method        : How the text was obtained: "native" | "ocr" | "blocked".
    ocr_confidence: Mean per-word Tesseract confidence (0–100). None if native.
    order_index   : Global reading order within the file (0-based).
    blocked_reason: Reason for blocking (only set when method == "blocked").
    """
    file: str
    location: str
    text: str
    method: Literal["native", "ocr", "blocked"]
    ocr_confidence: Optional[float]
    order_index: int
    blocked_reason: Optional[str] = None
    low_confidence: bool = False

    def is_blocked(self) -> bool:
        return self.method == "blocked"

    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0

    def __repr__(self) -> str:
        snippet = (self.text[:60] + "…") if len(self.text) > 60 else self.text
        return (
            f"Chunk(file={self.file!r}, location={self.location!r}, "
            f"method={self.method!r}, words={self.word_count()}, "
            f"text={snippet!r})"
        )
