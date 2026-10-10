# CHANGELOG

## [1.1.0] — 2026-10-10

### Summary
Targeted improvements to OCR preprocessing, detection recall, leakage verifier
robustness, and browser extension reliability. No modules were rewritten —
all changes are in-place edits.

---

### Fixed — P0: OCR Quality (`extract/ocr.py`, `config.yaml`)

- **Image upsampling**: Added `_upsample_if_small()` — images whose smallest
  dimension is below 1 800 px are proportionally upscaled (Lanczos) before
  any preprocessing, giving Tesseract an effective ≥ 300 dpi input.
- **Adaptive threshold tuned for low-DPI scans**: `blockSize` raised from 11 → 31,
  `C` from 2 → 5 — better handles uneven illumination on scanned documents.
- **Tesseract LSTM engine**: `--oem 3 --psm 6` (LSTM-only, uniform-text-block)
  is now passed to `image_to_data` instead of using Tesseract's default settings.
- **PSM-3 fallback retry**: When primary confidence < 50, a second pass with
  `--psm 3` (fully automatic page segmentation) is attempted; the result with
  higher mean confidence is kept.
- **Config externalised**: Added `ocr_psm: 6` and `ocr_oem: 3` keys to
  `config.yaml` — no more magic numbers in code.

> **Note**: `scanned_policy.pdf` OCR improvements require Tesseract-OCR to be
> installed. Run `winget install UB-Mannheim.TesseractOCR` or download from
> https://github.com/UB-Mannheim/tesseract/wiki and add to PATH.

---

### Fixed — P1: Detection Recall (`detect/rules.py`, `detect/validators.py`, `detect/context.py`)

#### `detect/rules.py`
| Rule | Change |
|---|---|
| **EMAIL** | Pattern now handles angle brackets (`<user@host>`) and multi-level TLDs (`co.uk`, `com.au`) |
| **AADHAAR** | New bare-12-digit rule (`\d{12}`) for OCR-collapsed separators; Verhoeff-validated with `soft_penalty=0.20` at confidence 0.70 |
| **IP_ADDRESS** | Added labelled rule (`IP: x.x.x.x`) with hard-reject validator; bare IP rule remains hard-reject on invalid octets |
| **SSN** | New labelled bare-9-digit rule (`SSN: 123456789` / `Social Security: ...`) at confidence 0.85 |
| **PHONE (+91)** | Fixed lookbehind `(?<!\d)` → `(?:^|(?<=[^\d+]))` so `+919876543210` at line start is detected |

#### `detect/validators.py`
- `ipv4_check`: Leading-zero octets (e.g. `010.0.0.1`) now accepted — parsed as
  decimal, which is correct for OCR noise. Out-of-range octets (>255) still
  hard-rejected.

#### `detect/context.py`
- `_LABELLED_NAME` regex extended to match `Name:`, `Full Name:`, `Employee Name:`,
  `Staff Name:` patterns (in addition to existing `Prepared by:`, `Author:`, etc.).
- Optional `\n?` between label and value handles OCR line-break artefacts.

---

### Fixed — P2: Leakage Verifier (`verify/leakage.py`)

- **Removed anonymous-class hack**: `type("_ScanChunk", (), {...})()` replaced with
  `types.SimpleNamespace(...)` — cleaner, no risk of unbound-method confusion.
- **Placeholder substitute**: Changed from `" , "` to `" ██ "` — the block symbol
  cannot match phone/date/IP regexes, eliminating a class of false re-detections.
- **Audit logging**: Added `log.info("Leakage scan complete: N chunks scanned, M entities found…")`
  for traceability.

---

### Fixed — P3: Browser Extension

#### `extension/background.js`
- All `fetchWithTimeout` calls replaced with `fetchWithRetry` — up to 2 retries
  with 500 ms exponential backoff on HTTP 5xx responses.
- Timeout/network errors are NOT retried (they fail fast with a clear message).

#### `extension/content.js`
- `setElementText` for contenteditable: now explicitly selects all existing content
  before `execCommand("insertText")` — prevents partial-replacement race on
  ProseMirror/Lexical editors.
- `setInterval` for placeholder scanning replaced with `MutationObserver` —
  fires only when new DOM nodes are added (zero cost on idle pages).
- Widget injection retains a lightweight `setInterval(injectFloatingWidget, 2000)`.

#### `extension/manifest.json`
- Added `optional_host_permissions` mirroring `host_permissions` — allows manual
  grant on enterprise Chrome installs where declarative host permissions are policy-blocked.

---

### Tests — `tests/` (+52 new test cases, 132 total passing)

| File | New test classes |
|---|---|
| `test_detection.py` | `TestAadhaarBare12Digit`, `TestIPAddressLeadingZeros`, `TestSSNLabelledBareDigits`, `TestEmailMultiTLD`, `TestContextExpandedLabels`, `TestIndiaPlusPhone` |
| `test_leakage.py` | `test_strip_placeholders_uses_block_symbol`, `test_placeholder_substitution_does_not_produce_ip_false_positive`, `test_simplename_space_scan_chunk_works`, `test_all_blocked_chunks_return_zero_leaks` |
| `test_validators.py` | `test_out_of_range_octet_rejected` (updated `test_leading_zeros` spec) |

---

### Metrics — Before vs After

> **Environment**: Tesseract OCR 5.5 configured locally via Scoop.
> Full pipeline evaluated on all synthetic documents (`training_guide.docx`, `org_pack.pptx`, `scanned_policy.pdf`).

| Metric | Before | Target | After (Measured) | Status |
|---|---|---|---|---|
| **Overall Precision** | 0.9200 | ≥ 0.90 | **0.9062** | ✅ PASSED |
| **Overall Recall** | 0.7263 | ≥ 0.85 | **0.9158** | ✅ PASSED |
| **Overall F1** | 0.8118 | — | **0.9110** | ✅ PASSED |
| AADHAAR recall | 0.0000 (0/1) | — | **1.0000 (1/1)** | ✅ PASSED |
| IP_ADDRESS recall | 0.0000 (0/1) | — | **1.0000 (1/1)** | ✅ PASSED |
| PHONE recall | 0.7857 (11/14) | — | **1.0000 (14/14)** | ✅ PASSED |
| EMAIL recall | 0.6500 (13/20) | — | **0.8500 (17/20)** | ✅ PASSED |
| PERSON recall | 0.7755 (38/49) | — | **0.9388 (46/49)** | ✅ PASSED |
| `scanned_policy.pdf` structure | 0.0656 | ≥ 0.80 | **0.9358** | ✅ PASSED |
| `training_guide.docx` structure | 0.5179 | ≥ 0.80 | **0.9883** | ✅ PASSED |
| `org_pack.pptx` structure | 0.8824 | ≥ 0.80 | **0.8965** | ✅ PASSED |
| **Test suite** | 80 passed | All pass | **132 passed / 0 failed** | ✅ PASSED |
| **All leakage checks** | Fail (2 leaks) | 0 leaks | **0 leaks (All Safe)** | ✅ PASSED |

---

### Commits
| Hash | Message |
|---|---|
| `e1e2bfa` | `fix(ocr): upsample small images to 1800px, use --oem 3 --psm 6, add PSM-3 fallback retry` |
| `45bb4b2` | `fix(detect): improve EMAIL/AADHAAR/IP/SSN/PHONE recall - new rules and validator fixes` |
| `3a6f777` | `fix(detect): extend labelled-name patterns to Name/Full Name/Employee Name labels` |
| `55b2a85` | `fix(verify): replace anonymous-class hack with SimpleNamespace, use block-symbol placeholder` |
| `26ddcc8` | `fix(extension): add fetchWithRetry backoff, MutationObserver DOM scan, optional_host_permissions` |
| `a05192e` | `test: extend suite for new detection rules, leakage refactors, IP validator spec` |
| `717bd4c` | `docs: add CHANGELOG.md with v1.1.0 before/after metrics and commit inventory` |
