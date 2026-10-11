# Ground Truth & Data Generation Anomalies

This document records verified anomalies, physical omissions, and unrecoverable OCR errors identified in the synthetic evaluation datasets (`/synthetic` and `/synthetic_holdout`).

---

## 1. Physical Omission: Footnote Entities in DOCX Files

- **Affected Files**:
  - `synthetic/training_guide.docx`
  - `synthetic_holdout/training_guide.docx`
- **Affected Entities**:
  - `PERSON` at `footnote:1` (e.g. `Dana Okoye` / `Dr. Michael Ryan`)
  - `EMAIL` at `footnote:1` (e.g. `b.okoye@example.net` / `m.ryan@example.net`)
- **Root Cause**:
  In `synthetic/generate.py`, the generator script registers footnote metadata in the ground truth JSON files, but python-docx does not serialize standalone footnotes into the standard document body OpenXML (`word/document.xml`).
- **Impact**:
  The footnote text is physically absent from the extracted text stream of the `.docx` container. No local parser or extractor can read characters that do not exist within the document XML.
- **Classification**: True physical omission in generator script.

---

## 2. Total OCR Drop on Degraded Document Pages

- **Affected Files**:
  - `synthetic/scanned_policy.pdf` (page 2)
  - `synthetic_holdout/scanned_policy.pdf` (page 2)
- **Affected Entities**:
  - `SSN` on page 2 (`655-15-0410` in synthetic, `452-15-0419` in holdout)
- **Root Cause**:
  Page 2 of `scanned_policy.pdf` contains simulated heavy scan degradation (rotation, skew, and pixel noise). The local Tesseract OCR engine drops the line entirely, emitting no text chunk or character candidates for this span.
- **Impact**:
  Produces an unavoidable false negative (FN) under 100% local processing constraints.

---

## 3. Ambiguous OCR Misreads Yielding Valid Alternative Tokens

- **Affected Files**:
  - `synthetic_holdout/scanned_policy.pdf` (page 2)
  - `synthetic/scanned_policy.pdf` (page 3)
- **Affected Entities**:
  - `IP_ADDRESS` on page 2: Ground truth is `10.46.59.64`. Tesseract OCR extracts `19.46.59.64`. Because `19.46.59.64` is also a mathematically valid public IPv4 address, deterministic correction without hallucination or ground-truth leakage is impossible.
  - `PHONE` on page 3: Ground truth is `584-695-9310`. Tesseract extracts `$84-695-9310`. While detectable as low-confidence OCR noise, the raw string differs from ground truth, producing an evaluation mismatch under exact-string matching rules.

---

## 4. Policy on Metrics Matching

Per **Hard Rule 5**, no ground truth files or `verify/metrics.py` matching logic are altered. All reported metrics reflect strict evaluation against the provided datasets.
