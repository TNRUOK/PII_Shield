# PII Shield 🛡️
### 100% Local, Privacy-Preserving Business Document PII Pipeline

> **Client Context**: Cadence Financial Services  
> **Mandate**: Enterprise policy strictly prohibits raw Personally Identifiable Information (PII) from being ingested by external Large Language Models (LLMs) or cloud services. All detection, redaction, and auditing must run 100% locally with zero cloud API dependencies, zero external telemetry, and an auditable fail-closed safety gate.

---

## Key Features

- **100% Local Execution**: Powered by PyMuPDF, python-docx, python-pptx, OpenCV, Tesseract OCR, Presidio Analyzer, and spaCy (`en_core_web_lg`). No cloud OCR, no external model calls.
- **Fail-Closed Security**: Documents or pages that cannot be parsed safely (corrupted files, empty files, or OCR confidence below threshold) emit explicit `BLOCKED` chunks rather than passing unverified text downstream.
- **Multi-Layer Detection Engine**:
  - **Regex Rules & Specialized Validators**: High-precision patterns with mathematical verification:
    - Credit Card: Luhn algorithm check (ISO/IEC 7812)
    - Aadhaar: Verhoeff algorithm check digit
    - IBAN: ISO 7064 mod-97 check
    - SSN: Area number validity (rejects 000, 666, 900–999) & group/serial sanity
    - IPv4: Octet boundaries (0–255)
  - **Local Named Entity Recognition (NER)**: Microsoft Presidio backed by spaCy transformer/statistical models for high-recall identification of person names.
  - **Context-Aware Boost**: Column-header boosting in tables (`"Employee Name"` header boosts cell confidence by 1.2x) and prefix patterns (`"Approved by:"`).
  - **Span Merge & Conflict Resolution**: Greedy resolution of overlapping spans, union of adjacent identical entities, and deterministic deduplication.
- **Deterministic & Reversible Redaction**:
  - Replaces PII with uniform placeholder tokens (`[PERSON_1]`, `[EMAIL_1]`).
  - Identical raw entities receive identical placeholders across document chunks.
  - Offsets replaced right-to-left to prevent coordinate drift.
  - Raw-to-placeholder mappings are saved into an isolated mapping store, strictly excluded from the audit log and downstream LLM payloads.
- **Post-Redaction Leakage Verification**:
  - Automatically re-scans redacted text through the entire detection suite.
  - If any residual PII is detected, the document is flagged `downstream_safe: false` and prevented from reaching any LLM.
- **Zero-Raw-PII Audit Report**:
  - Produces CSV and JSON audit summaries containing masked values (e.g. `J*****h`), entity types, sensitivity levels, and confidence scores. Never discloses raw PII.
- **Interactive Gradio UI**:
  - Four-tab interface for uploading files, inspecting extracted chunks, viewing the masked PII inventory, previewing redacted text with highlighted placeholders, and downloading sanitized outputs.

---

## Directory Layout

```
c:\D Drive\OCR\pii_shield\
├── app/
│   └── gradio_app.py          # Interactive 4-tab Gradio interface
├── config.yaml                # Detection profiles (strict/balanced), thresholds, entities
├── detect/
│   ├── context.py             # Table header boost & labelled prefix rules
│   ├── merge.py               # Span union, greedy overlap resolution, dedup
│   ├── ner.py                 # Presidio + spaCy NER wrapper
│   ├── rules.py               # Compiled regex library for 14+ PII types
│   └── validators.py          # Luhn, Verhoeff, IBAN, SSN, IPv4 validators
├── extract/
│   ├── docx.py                # Paragraphs, tables, headers/footers, footnotes, images
│   ├── models.py              # Chunk dataclass (text, method, confidence, location)
│   ├── ocr.py                 # Deskew, adaptive threshold, Tesseract runner
│   ├── pdf.py                 # PyMuPDF native extractor with 300 DPI OCR fallback
│   ├── pptx.py                # Slide shapes, tables, speaker notes, images
│   └── router.py              # Magic-byte and extension-based dispatch
├── notebooks/
│   └── demo.ipynb             # Interactive end-to-end Jupyter walkthrough
├── redact/
│   ├── mapping.py             # Isolated raw ↔ placeholder mapping store
│   └── redactor.py            # Right-to-left token replacer preserving table structure
├── report/
│   └── report.py              # Zero-raw-PII CSV, JSON, and summary table generator
├── synthetic/
│   ├── generate.py            # Faker-driven generator for PDF, DOCX, PPTX test docs
│   └── ground_truth/          # Annotated JSON ground truth & expected plain text
├── tests/
│   ├── conftest.py            # Shared pytest fixtures
│   ├── test_detection.py      # Rule, NER, and context boost test cases
│   ├── test_extractors.py     # PDF, DOCX, and PPTX extractor tests
│   ├── test_fail_closed.py    # Corrupt, empty, unsupported, and low-conf tests
│   ├── test_leakage.py        # Post-redaction safety gate tests
│   ├── test_merge.py          # Span overlap and conflict resolution tests
│   ├── test_redaction.py      # Consistent token and table preservation tests
│   └── test_validators.py     # Unit tests for mathematical check algorithms
├── verify/
│   ├── leakage.py             # Re-detection safety scanner
│   └── metrics.py             # Precision, Recall, F1, and structure retention
├── requirements.txt           # Python package dependencies
└── README.md                  # This file
```

---

## Installation & Setup

1. **Clone or navigate to the workspace**:
   ```bash
   cd "c:\D Drive\OCR\pii_shield"
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR engine (Windows)**:
   - Automated via winget:
     ```bash
     winget install --id UB-Mannheim.TesseractOCR --silent
     ```
   - Standard installation installs to `C:\Program Files\Tesseract-OCR\tesseract.exe`. The pipeline automatically auto-detects this binary.

4. **Install spaCy local model**:
   ```bash
   python -m spacy download en_core_web_lg
   ```

---

## Quickstart

### 1. Generate Synthetic Test Documents
```bash
python synthetic/generate.py
```
This generates three complex test documents in `synthetic/` along with ground-truth annotations in `synthetic/ground_truth/`:
- `scanned_policy.pdf`: Multi-page scanned document with OCR noise, skewed text, and metadata PII.
- `training_guide.docx`: Rich document containing tables, headers, footers, footnotes, and embedded screenshots.
- `org_pack.pptx`: Presentation with org directory tables, speaker notes, and grouped diagram elements.

### 2. Launch the Gradio Web Application
```bash
python -m app.gradio_app
```
Open `http://localhost:7860` in your browser:
- **Tab 1: Extracted Text**: Inspect extracted chunks with exact document locations (`page:1`, `table:1/r2c3`, `slide:3/notes`).
- **Tab 2: Detected PII**: Review the detected PII inventory with sensitivity ratings, confidence scores, and detection methods. (Raw values are masked).
- **Tab 3: Redacted Text**: Review sanitized content with replaced placeholders highlighted.
- **Tab 4: Audit Report**: Download audit reports (CSV/JSON) and redacted text archives. Check the **Downstream LLM Safe** badge.

### 3. Run Verification & Evaluation Metrics
```bash
python -m verify.metrics
```
Evaluates precision, recall, F1, and structure retention against the generated ground-truth dataset.

### 4. Run Pytest Suite
```bash
pytest -v
```
Executes comprehensive unit and integration tests across validators, extractors, detection rules, redaction engines, leakage verifiers, and fail-closed mechanisms.

---

## Architecture & Detection Pipeline

```
Raw Document (PDF / DOCX / PPTX / Image)
                 │
                 ▼
        [ extract/router.py ]
          (Magic-byte check)
                 │
  ┌──────────────┼──────────────┬──────────────┐
  ▼              ▼              ▼              ▼
[ pdf.py ]   [ docx.py ]    [ pptx.py ]    [ ocr.py ]
(Native/OCR) (DOM & images) (Slides/Notes) (Deskew/Filter)
  │              │              │              │
  └──────────────┴──────┬───────┴──────────────┘
                        │
                        ▼
                 List of [ Chunk ]
     (location, text, method, ocr_conf, blocked)
                        │
                        ▼
       [ detect/merge.py :: detect_chunk ]
  ┌─────────────────────┼─────────────────────┐
  ▼                     ▼                     ▼
[ rules.py ]       [ ner.py ]           [ context.py ]
Regex + Validators Presidio + spaCy      Header / prefix boost
  │                     │                     │
  └─────────────────────┼─────────────────────┘
                        │
                        ▼
             Span Overlap Resolution
                        │
                        ▼
                 List of [ Entity ]
                        │
                        ▼
            [ redact/redactor.py ]
          (Right-to-left replacement)
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 [ Redacted Chunks ]          [ mapping.py Store ]
        │                       (Never sent to LLM)
        ▼
 [ verify/leakage.py ]
  (Re-scan safety gate)
        │
   Downstream Safe?
   ├─ YES ──► Emit sanitized text for LLM ingestion
   └─ NO  ──► Block downstream transfer & log alert
        │
        ▼
 [ report/report.py ]
 (Zero-raw-PII audit report: CSV / JSON)
```

---

## Browser Extension (Use with Real LLMs like ChatGPT, Claude, Gemini)

PII Shield includes a Chromium/Chrome/Edge browser extension that automatically shields your prompts and restores real values in AI replies:

### 1. Start the Local Bridge Server
Run:
```bash
python -m extension_server.server
```
or double-click `run_extension_bridge.bat`. The bridge listens securely on `http://127.0.0.1:8765`.

### 2. Load Extension in Chrome / Edge / Brave
1. Open your browser and navigate to `chrome://extensions` (or `edge://extensions`).
2. Toggle on **Developer mode** (top right switch).
3. Click **"Load unpacked"**.
4. Select the folder: `c:\D Drive\OCR\pii_shield\extension`.

### 3. Usage on ChatGPT / Claude / Gemini
- **In-Place Shielding**: When typing into any prompt box on ChatGPT, Claude, or Gemini, click the floating **"🛡️ Shield PII"** button or press **`Ctrl+Shift+S`**. PII is replaced instantly with tokens (`[PERSON_1]`, `[EMAIL_1]`).
- **In-Place Restoration**: When the LLM replies referencing tokens (e.g., `"[PERSON_1] should contact [EMAIL_1]"`), click the inline **"🔓 Restore PII"** pill to decode the placeholders back to original names and numbers right on the webpage.
- **Extension Popup**: Click the PII Shield icon in your extension bar for a standalone prompt redactor, document sanitizer (drag-and-drop PDF/DOCX/PPTX), and session viewer.

---

## Known Limitations & Edge Cases

1. **Vector Graphics with Embedded Text (EMF / WMF)**:
   - Older Word documents may embed drawings or charts as Windows Metafiles (EMF/WMF). Without native Windows GDI rendering, these images cannot be converted to bitmap raster for OCR. PII Shield safely emits a `BLOCKED` chunk for unsupported formats.
2. **Severely Distorted / Highly Rotated Scans**:
   - The OpenCV deskew filter corrects skew within $\pm 45^\circ$. Inverted or sideways pages ($90^\circ / 180^\circ$) require orientation detection (`tesseract --psm 0`). Pages where mean word confidence drops below the configured threshold (default: 40.0) are marked `BLOCKED`.
3. **Complex Multi-Column Layouts**:
   - In image-only PDFs with dense multi-column layouts, standard horizontal block reading order can occasionally interleave across column gutters. Using PyMuPDF block layout sorting mitigates this for native PDFs.
4. **LOCATION Entity Sensitivity**:
   - Detecting geographic locations has a high false-positive rate on corporate entity names (e.g., "Cadence Financial" or "Dallas Asset Management"). `LOCATION` detection is disabled by default in `config.yaml` to optimize precision.
