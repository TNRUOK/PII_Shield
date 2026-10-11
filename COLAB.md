# Running PII Shield on Google Colab

Instructions for setting up and running PII Shield on a clean Google Colab runtime.

---

### 1. Install System Dependencies (Tesseract OCR)
Run this in a Colab code cell:
```bash
!apt-get update -qq
!apt-get install -y -qq tesseract-ocr tesseract-ocr-eng libtesseract-dev
```

### 2. Install Pinned Python Dependencies
```bash
!pip install -r requirements.txt
```

### 3. Download spaCy English Model
```bash
!python -m spacy download en_core_web_lg
```

### 4. Verify Local Environment
```python
from extract.ocr import is_tesseract_available
print("Tesseract Available:", is_tesseract_available())

import spacy
nlp = spacy.load("en_core_web_lg")
print("spaCy model loaded:", nlp.meta["name"])
```

### 5. Run Verification & Evaluation
```bash
# Run full pytest suite (181 tests)
!pytest -q --tb=short

# Run benchmark evaluation on synthetic dataset
!python evaluate.py --input synthetic

# Run benchmark on hold-out dataset
!python evaluate.py --input synthetic_holdout

# Run 6-stage ablation study
!python evaluate.py --input synthetic --ablation
```

### 6. Launch Local Interface
```python
from app.gradio_app import build_ui
ui = build_ui()
# Binds strictly to 127.0.0.1 with share=False for zero data egress
ui.launch(server_name="127.0.0.1", server_port=7860, inline=True)
```
