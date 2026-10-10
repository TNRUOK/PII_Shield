"""
app/gradio_app.py
-----------------
Gradio UI for PII Shield.

Tabs:
  (a) Extracted Text  — per-chunk view with location labels
  (b) Detected PII    — sortable table (masked values, no raw PII)
  (c) Redacted Text   — per-chunk view with placeholders highlighted
  (d) Report          — per-file summary + downstream-safe badge + download buttons

All processing is local; no cloud calls of any kind.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Any

import gradio as gr
import pandas as pd
import yaml

# Make pii_shield importable when running from the app/ directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from extract.router import route
from extract.models import Chunk
from extract.ocr import is_tesseract_available
from detect.merge import detect_chunk, Entity
from redact.mapping import PlaceholderMapping
from redact.redactor import redact_chunks
from verify.leakage import check_leakage
from report.report import build_file_summary, generate_report, print_summary_table

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = _ROOT / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _chunks_to_text(chunks: List[Chunk]) -> str:
    """Format chunks as readable text with location labels."""
    lines = []
    for c in chunks:
        if c.is_blocked():
            lines.append(f"[BLOCKED @ {c.location}] — {c.blocked_reason}")
        else:
            lines.append(f"--- {c.location} ---")
            lines.append(c.text)
            lines.append("")
    return "\n".join(lines)


def _entities_to_df(entities: List[Entity]) -> pd.DataFrame:
    """Convert entities to a DataFrame safe for display (no raw values)."""
    rows = []
    for e in entities:
        rows.append({
            "Type":        e.entity_type,
            "Masked":      e.masked_value,
            "Sensitivity": e.sensitivity,
            "Confidence":  f"{e.confidence:.2f}",
            "Methods":     " | ".join(e.methods),
            "Location":    e.location,
            "File":        Path(e.file).name,
            "Placeholder": e.placeholder,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Type", "Masked", "Sensitivity", "Confidence",
                 "Methods", "Location", "File", "Placeholder"])


def _highlight_placeholders(text: str) -> str:
    """Wrap placeholder tokens in bold-ish markers for Gradio markdown display."""
    import re
    return re.sub(
        r"(\[[A-Z_]+_\d+\])",
        r"**\1**",
        text
    )


def process_files(
    uploaded_files: List[Any],
    profile_choice: str,
) -> Tuple[str, pd.DataFrame, str, str, str, str]:
    """
    Main pipeline function called by Gradio.

    Returns
    -------
    (extracted_text, pii_df, redacted_text, report_summary,
     csv_path_for_download, json_path_for_download)
    """
    if not uploaded_files:
        return ("No files uploaded.", pd.DataFrame(), "", "No files uploaded.", None, None)

    config = _load_config()
    # Override profile from UI
    config["active_profile"] = profile_choice.lower()

    # Use a temp dir for output
    out_dir = tempfile.mkdtemp(prefix="pii_shield_")
    mapping_dir = os.path.join(out_dir, "mappings")
    report_dir  = os.path.join(out_dir, "reports")

    all_chunks_text: List[str] = []
    all_entities:    List[Entity] = []
    all_redacted:    List[str] = []
    file_summaries:  List[dict] = []

    for uploaded in uploaded_files:
        # Gradio gives us a temp file path (str) for each upload
        file_path = uploaded if isinstance(uploaded, str) else uploaded.name
        fname = Path(file_path).name
        log.info("Processing %s", fname)

        # --- Extract ---
        chunks = route(file_path, config)
        all_chunks_text.append(f"=== {fname} ===\n" + _chunks_to_text(chunks))

        # --- Detect ---
        file_entities: List[Entity] = []
        for chunk in chunks:
            ents = detect_chunk(chunk, config)
            file_entities.extend(ents)

        all_entities.extend(file_entities)

        # --- Redact ---
        mapping = PlaceholderMapping(file_path, mapping_dir)
        redacted = redact_chunks(chunks, file_entities, mapping, config)
        mapping.save()

        all_redacted.append(f"=== {fname} ===\n" + _chunks_to_text(redacted))

        # --- Leakage check ---
        leak = check_leakage(redacted, config, known_values=mapping)

        # --- File summary ---
        summary = build_file_summary(file_path, chunks, file_entities, leak, None)
        file_summaries.append(summary)

    # --- Generate report ---
    report_paths = generate_report(file_summaries, all_entities, report_dir)

    # --- Format outputs ---
    extracted_out = "\n\n".join(all_chunks_text)
    redacted_out  = _highlight_placeholders("\n\n".join(all_redacted))
    pii_df        = _entities_to_df(all_entities)

    # Build report summary text
    tess_status = "✅ Installed" if is_tesseract_available() else "⚠️ Not Installed"
    summary_lines = [
        "## PII Shield — Run Summary\n",
        f"OCR Engine (Tesseract): **{tess_status}**\n",
    ]
    for fs in file_summaries:
        safe = "✅ YES" if fs["leak_check"]["downstream_safe"] else "❌ NO"
        summary_lines.append(
            f"**{Path(fs['file']).name}**  \n"
            f"Chunks: {fs['chunk_count']} (native={fs['native_chunks']}, "
            f"ocr={fs['ocr_chunks']}, blocked={fs['blocked_chunks']})  \n"
            f"PII found: {fs['pii_count']} entities  \n"
            f"Risk level: **{fs['risk_level']}**  \n"
            f"Downstream safe: **{safe}**  \n"
            f"PII by type: {json.dumps(fs['pii_by_type'], indent=0)}  \n"
        )

    report_md = "\n---\n".join(summary_lines)

    return (
        extracted_out,
        pii_df,
        redacted_out,
        report_md,
        report_paths["csv"],
        report_paths["json"],
    )


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    tess_status = "✅ Installed" if is_tesseract_available() else "⚠️ Not Installed"
    with gr.Blocks(
        title="PII Shield — Local Privacy Pipeline",
        theme=gr.themes.Soft(),
        css="""
        .gr-box { border-radius: 8px; }
        h1 { color: #1a237e; }
        """,
    ) as demo:
        gr.Markdown(
            f"# 🛡️ PII Shield\n"
            f"**Local, privacy-preserving PII detection and redaction.** "
            f"OCR Engine (Tesseract): **{tess_status}**. "
            f"No data leaves your machine. No LLM calls for detection.\n"
            f"> ⚠️ Upload only **synthetic or non-sensitive test documents**."
        )

        with gr.Row():
            file_input = gr.File(
                label="Upload Documents (PDF, DOCX, PPTX, Images)",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".tiff"],
            )
            with gr.Column():
                profile_radio = gr.Radio(
                    choices=["Balanced", "Strict"],
                    value="Balanced",
                    label="Detection Profile",
                    info="Balanced = recall-biased. Strict = higher confidence thresholds.",
                )
                run_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")

        with gr.Tabs():
            with gr.Tab("📄 Extracted Text"):
                extracted_out = gr.Textbox(
                    label="Extracted text with location labels",
                    lines=25, max_lines=50,
                    show_copy_button=True,
                )

            with gr.Tab("🔍 Detected PII"):
                pii_table = gr.Dataframe(
                    label="Detected PII (masked values — no raw data shown)",
                    interactive=False,
                    wrap=True,
                )

            with gr.Tab("✏️ Redacted Text"):
                redacted_out = gr.Markdown(
                    label="Redacted text (placeholders in **bold**)",
                )

            with gr.Tab("📊 Report & Download"):
                report_md = gr.Markdown(label="Run summary")
                with gr.Row():
                    dl_csv  = gr.File(label="⬇️ Download Report (CSV)")
                    dl_json = gr.File(label="⬇️ Download Report (JSON)")

        run_btn.click(
            fn=process_files,
            inputs=[file_input, profile_radio],
            outputs=[extracted_out, pii_table, redacted_out, report_md, dl_csv, dl_json],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
