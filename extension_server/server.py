"""
extension_server/server.py
--------------------------
Local HTTP bridge for the PII Shield Browser Extension.
Exposes endpoints on 127.0.0.1:8765 for:
  - /health       : check if local engine is running
  - /redact_text  : redact clipboard or prompt text
  - /restore_text : restore placeholders in LLM responses
  - /redact_file  : upload PDF/DOCX/PPTX and return redacted text
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import yaml

# Ensure pii_shield root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from extract.models import Chunk
from extract.router import route
from detect.merge import detect_chunk, Entity
from redact.mapping import PlaceholderMapping
from redact.redactor import redact_chunks
from verify.leakage import check_leakage

log = logging.getLogger("pii_shield_bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="PII Shield Browser Extension Bridge", version="1.0.0")

# Enable CORS for browser extensions and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Extensions make requests from chrome-extension:// origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = _ROOT / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
    _CONFIG = yaml.safe_load(fh)

# In-memory session store for placeholder mappings: session_id -> {placeholder: raw_value}
_SESSION_MAPPINGS: Dict[str, Dict[str, str]] = {}


class RedactTextRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    profile: Optional[str] = "balanced"


class RedactTextResponse(BaseModel):
    session_id: str
    original_length: int
    redacted_text: str
    downstream_safe: bool
    detected_pii: List[Dict[str, str]]  # masked values only


class RestoreTextRequest(BaseModel):
    text: str
    session_id: str


class RestoreTextResponse(BaseModel):
    restored_text: str


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "engine": "PII Shield Local",
        "profile": _CONFIG.get("active_profile", "balanced"),
    }


@app.post("/redact_text", response_model=RedactTextResponse)
def redact_text_endpoint(req: RedactTextRequest):
    if not req.text.strip():
        return RedactTextResponse(
            session_id=req.session_id or str(uuid.uuid4()),
            original_length=0,
            redacted_text="",
            downstream_safe=True,
            detected_pii=[],
        )

    session_id = req.session_id or str(uuid.uuid4())
    config = dict(_CONFIG)
    if req.profile:
        config["active_profile"] = req.profile.lower()

    # Create virtual chunk for text
    chunk = Chunk(
        file="browser_input",
        location="prompt",
        text=req.text,
        method="native",
        ocr_confidence=None,
        order_index=0,
    )

    # 1. Detect PII
    entities: List[Entity] = detect_chunk(chunk, config)

    # 2. Redact PII
    session_store = _SESSION_MAPPINGS.setdefault(session_id, {})
    
    # We use a custom in-memory mapping adapter
    tmp_out = _ROOT / "output" / "extension_sessions"
    mapping = PlaceholderMapping(source_file=f"session_{session_id}.txt", output_dir=str(tmp_out))
    
    # Pre-populate with existing session mappings if any
    for ph, raw in session_store.items():
        etype = ph.strip("[]").rsplit("_", 1)[0]
        mapping._raw_to_placeholder[(etype, raw)] = ph
        mapping._placeholder_to_raw[ph] = raw

    redacted_chunks = redact_chunks([chunk], entities, mapping, config)
    redacted_text = redacted_chunks[0].text if redacted_chunks else req.text

    # Update in-memory session mappings
    session_store.update(mapping._placeholder_to_raw)

    # 3. Leakage safety verification
    leak_res = check_leakage(redacted_chunks, config)

    # Return masked entity list (safe for UI)
    pii_summary = [
        {
            "type": e.entity_type,
            "masked": e.masked_value,
            "placeholder": e.placeholder,
            "sensitivity": e.sensitivity,
        }
        for e in entities
    ]

    return RedactTextResponse(
        session_id=session_id,
        original_length=len(req.text),
        redacted_text=redacted_text,
        downstream_safe=leak_res.downstream_safe,
        detected_pii=pii_summary,
    )


@app.post("/restore_text", response_model=RestoreTextResponse)
def restore_text_endpoint(req: RestoreTextRequest):
    """
    Given an LLM output containing [PERSON_1], [EMAIL_1], etc.,
    replace them back with the raw original values from the session.
    """
    mapping = _SESSION_MAPPINGS.get(req.session_id, {})
    if not mapping:
        return RestoreTextResponse(restored_text=req.text)

    restored = req.text
    # Sort placeholders by length descending to avoid partial token collisions
    for placeholder, raw_val in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
        restored = restored.replace(placeholder, raw_val)

    return RestoreTextResponse(restored_text=restored)


@app.post("/redact_file", response_model=RedactTextResponse)
async def redact_file_endpoint(
    file: UploadFile = File(...),
    profile: Optional[str] = Form("balanced"),
):
    """
    Upload a PDF, DOCX, or PPTX file, extract text, detect PII,
    and return the sanitized text ready to be pasted or sent to an LLM.
    """
    session_id = str(uuid.uuid4())
    upload_dir = _ROOT / "output" / "extension_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    config = dict(_CONFIG)
    if profile:
        config["active_profile"] = profile.lower()

    # Route and extract
    chunks = route(str(file_path), config)

    # Detect
    all_entities = []
    for c in chunks:
        if not c.is_blocked():
            all_entities.extend(detect_chunk(c, config))

    # Redact
    tmp_out = _ROOT / "output" / "extension_sessions"
    mapping = PlaceholderMapping(source_file=str(file_path), output_dir=str(tmp_out))
    redacted_chunks = redact_chunks(chunks, all_entities, mapping, config)

    _SESSION_MAPPINGS[session_id] = dict(mapping._placeholder_to_raw)

    leak_res = check_leakage(redacted_chunks, config)

    # Format text
    clean_lines = []
    for c in redacted_chunks:
        if c.is_blocked():
            clean_lines.append(f"[BLOCKED SECTION: {c.blocked_reason}]")
        else:
            clean_lines.append(c.text)

    combined_text = "\n\n".join(clean_lines)

    pii_summary = [
        {
            "type": e.entity_type,
            "masked": e.masked_value,
            "placeholder": e.placeholder,
            "sensitivity": e.sensitivity,
        }
        for e in all_entities
    ]

    return RedactTextResponse(
        session_id=session_id,
        original_length=len(combined_text),
        redacted_text=combined_text,
        downstream_safe=leak_res.downstream_safe,
        detected_pii=pii_summary,
    )


def start_server(host: str = "127.0.0.1", port: int = 8765):
    print("=" * 65)
    print(f"🛡️  PII Shield Bridge Server running at http://{host}:{port}")
    print("   Endpoints:")
    print(f"     • Health check : http://{host}:{port}/health")
    print(f"     • Redact text  : POST http://{host}:{port}/redact_text")
    print(f"     • Restore text : POST http://{host}:{port}/restore_text")
    print(f"     • Redact file  : POST http://{host}:{port}/redact_file")
    print("=" * 65)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
