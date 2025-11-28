# Streamlit entrypoint for the ATS Resume Analyzer
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import pathlib
import sys
import time
import uuid
from datetime import datetime
from typing import Any

# System path setup
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st
from streamlit.components.v1 import html as st_html

from ats.evidence import compute_evidence_ranking
from ats.llm import score_fit_dimensions, suggest_resume_improvements
from ats.nlp_backends import (
    extract_jd_keywords_batch,
    present_missing_vs_resume,
)
from ats.parse import parse_job_description, parse_resume
from ats.pdf_utils import extract_text_auto, extract_text_from_pdf
from ats.prompts.prompts import build_make_edits_prompt
from ats.scoring import compute_score, compute_score_advanced

# Logging configuration (file and console)
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("ats_app.log"), logging.StreamHandler()],
    )

logger = logging.getLogger(__name__)

# Configuration constants
MAX_BULLETS_DEFAULT = 5
PROMPT_TEXTAREA_HEIGHT = 520
MAX_PDF_PAGES_DEFAULT = 5
MAX_FILE_SIZE_MB = 10

# Load .env early for LLM configuration
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


USE_SEMANTIC_DEFAULT = os.getenv("USE_SEMANTIC", "1").lower() not in (
    "0",
    "false",
    "no",
)

st.set_page_config(page_title="ATS Resume Analyzer", page_icon="🧭", layout="wide")

st.title("🧭 ATS Resume Analyzer — v0.2")

# Session-state caches
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}
if "resume_text_cache" not in st.session_state:
    st.session_state.resume_text_cache = None
if "last_resume_hash" not in st.session_state:
    st.session_state.last_resume_hash = None

# Sidebar controls
with st.sidebar:
    st.markdown("### Settings")
    enable_ocr = st.toggle("Enable OCR for PDFs", value=True)
    ocr_lang = st.text_input("OCR language (Tesseract)", value="eng")
    max_pages = st.number_input(
        "Max PDF pages to read", value=MAX_PDF_PAGES_DEFAULT, min_value=1, max_value=50
    )

    st.divider()
    st.markdown("### LLM")

    llm_enable = st.toggle(
        "Use Claude to write bullets",
        value=os.getenv("LLM_ENABLE", "1").lower() not in {"0", "", "false"},
    )

    AVAILABLE_MODELS = ["anthropic/claude-3-5-haiku-20241022"]
    default_model = os.getenv("LLM_MODEL", "anthropic/claude-3-5-haiku-20241022")
    if default_model not in AVAILABLE_MODELS:
        default_model = "anthropic/claude-3-5-haiku-20241022"

    llm_model = st.selectbox(
        "LLM model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(default_model),
        help="Pick a model available to your Anthropic workspace.",
    )

st.markdown(
    "Upload **Resume (PDF)** and add a **Job Description** (paste or upload). Then click **Analyze**."
)

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
with col2:
    jd_text = st.text_area("Paste Job Description (JD)", height=220, placeholder="Paste JD here…")
    jd_file = st.file_uploader(
        "…or upload JD (PDF / DOCX / TXT / MD)", type=["pdf", "docx", "txt", "md"]
    )

analyze = st.button("Analyze", type="primary", use_container_width=True)


# Helper serialization utilities
def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def _csv_keywords_bytes(present: list[str], missing: list[str]) -> bytes:
    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["Keyword", "Status"])
    for kw in present:
        w.writerow([kw, "Present"])
    for kw in missing:
        w.writerow([kw, "Missing"])
    return output.getvalue().encode("utf-8")


# Main analysis workflow
if analyze:
    # Basic file size validation
    if not resume_file:
        st.error("Please upload a **Resume PDF**.")
        st.stop()

    if resume_file.size and resume_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(
            f"❌ Resume file too large ({resume_file.size / (1024*1024):.1f}MB). Max: {MAX_FILE_SIZE_MB}MB"
        )
        st.stop()

    if jd_file and jd_file.size and jd_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"❌ JD file too large. Max: {MAX_FILE_SIZE_MB}MB")
        st.stop()

    # Prefer uploaded JD files over pasted text
    if jd_file is not None:
        jd_text = extract_text_auto(
            jd_file.read(),
            filename=jd_file.name,
            enable_ocr=enable_ocr,
            ocr_lang=ocr_lang,
            max_pages=max_pages,
        )
    if not jd_text or not jd_text.strip():
        st.error("Please provide a Job Description (paste or upload).")
        st.stop()

    # Cache resume text by file hash
    rbytes = resume_file.getvalue()
    resume_hash = hashlib.md5(rbytes).hexdigest()
    logger.info(f"Analysis started: resume={resume_file.name}, jd_len={len(jd_text)}")

    if st.session_state.last_resume_hash != resume_hash:
        st.session_state.resume_text_cache = extract_text_from_pdf(
            rbytes,
            enable_ocr=enable_ocr,
            ocr_lang=ocr_lang,
            max_pages=max_pages,
        )
        st.session_state.last_resume_hash = resume_hash

    resume_text = st.session_state.resume_text_cache
    resume_parsed = parse_resume(resume_text)
    jd_parsed = parse_job_description(jd_text)

    # Analysis cache key combines resume hash, JD text, and model selection
    analysis_key = hashlib.md5(
        f"{resume_hash}::{jd_text}::{llm_model}::{llm_enable}::{USE_SEMANTIC_DEFAULT}".encode()
    ).hexdigest()

    progress = st.progress(0, "Starting analysis…")

    if analysis_key not in st.session_state.analysis_cache:
        # Step 1: Extract JD keywords (batch API) and intersect with resume
        progress.progress(20, "Extracting keywords from JD…")
        jd_batch = extract_jd_keywords_batch([jd_text])
        jd_cats = (
            jd_batch[0]
            if jd_batch
            else {
                "tools": [],
                "methods": [],
                "competencies": [],
                "industry": [],
            }
        )

        present_pm, missing_pm = present_missing_vs_resume(jd_cats, resume_text)

        # Step 2: Compute ATS scoring (advanced + simple)
        progress.progress(50, "Computing ATS score…")
        adv = compute_score_advanced(
            jd_text=jd_text,
            resume_text=resume_text,
            resume_parsed=resume_parsed,
            use_semantic=USE_SEMANTIC_DEFAULT,
        )
        adv_d = adv.to_dict()

        simple = compute_score(
            jd_text=jd_text,
            resume_text=resume_text,
        )
        simple_d = simple.to_dict()

        # Step 3: Semantic evidence ranking: identify bullets closest to the JD
        progress.progress(65, "Ranking evidence bullets…")
        evidence = compute_evidence_ranking(
            jd_text=jd_text,
            resume_text=resume_text,
            top_k=5,
            bottom_k=5,
        )

        # Step 4: Multi-dimensional fit scores
        progress.progress(80, "Scoring multi-dimensional fit…")
        dim_scores = score_fit_dimensions(
            jd_text=jd_text,
            resume_text=resume_text,
            present_keywords=adv_d["present_keywords"],
            missing_keywords=adv_d["missing_keywords"],
            model=llm_model,
            enable_llm=llm_enable,
        )

        # Step 5: LLM suggestions (with fallback)
        progress.progress(90, "Generating tailored suggestions…")
        try:
            sugg = suggest_resume_improvements(
                jd_text=jd_text,
                resume_text=resume_text,
                missing_keywords=adv_d["missing_keywords"],
                present_keywords=adv_d["present_keywords"],
                enable_llm=llm_enable,
                model=llm_model,
                max_bullets=MAX_BULLETS_DEFAULT,
            ).to_dict()
        except Exception as e:
            st.error(f"⚠️ LLM suggestion failed: {e}")
            st.info("Using keyword-only recommendations instead.")
            sugg = {
                "missing_keywords": list(adv_d["missing_keywords"]),
                "tailored_bullets": [],
                "notes": "LLM unavailable",
            }

        # Cache everything needed for UI render
        st.session_state.analysis_cache[analysis_key] = {
            "adv": adv,
            "adv_d": adv_d,
            "simple_d": simple_d,
            "jd_cats": jd_cats,
            "present_pm": present_pm,
            "missing_pm": missing_pm,
            "evidence": evidence,
            "dim_scores": dim_scores,
            "sugg": sugg,
        }

        progress.progress(100, "Done!")
        time.sleep(0.3)
        progress.empty()

    # Reuse cached results
    results = st.session_state.analysis_cache[analysis_key]
    adv = results["adv"]
    adv_d = results["adv_d"]
    simple_d = results["simple_d"]
    jd_cats = results["jd_cats"]
    present_pm = results["present_pm"]
    missing_pm = results["missing_pm"]
    evidence = results["evidence"]
    dim_scores = results["dim_scores"]
    sugg = results["sugg"]

    logger.info(
        "Score: %.1f (advanced), Present=%d, Missing=%d",
        adv.composite,
        len(adv.present_keywords),
        len(adv.missing_keywords),
    )

    # ---- Multi-dimensional fit section ----
    st.subheader("Multi-dimensional fit (0–10)")
    ds = dim_scores.to_dict()
    st.write(
        f"Tools: {ds['tools_score']}  |  "
        f"Methods: {ds['methods_score']}  |  "
        f"Domain: {ds['domain_score']}  |  "
        f"Seniority: {ds['seniority_score']}  |  "
        f"Communication: {ds['communication_score']}  |  "
        f"Overall: {ds['overall_score']:.1f}"
    )

    # ---- Simple vs Advanced ATS fit ----
    st.subheader("ATS Fit Score")
    st.write(
        f"Simple score: {simple_d['composite']:.1f} / 100 "
        f"(keyword coverage ≈ {simple_d['keyword_coverage'] * 100:.0f}%, "
        f"TF-IDF cosine ≈ {simple_d['tfidf_cosine']:.2f})"
    )
    st.write(f"Advanced score: {adv_d['composite']:.1f} / 100")

    st.subheader("ATS Keywords (present / missing)")
    c1, c2 = st.columns(2)
    c1.markdown("**Present (in resume)**")
    c1.write(", ".join(present_pm) or "—")
    c2.markdown("**Missing (from resume)**")
    c2.write(", ".join(missing_pm) or "—")

    with st.expander("JD keyword groups (for tailoring)", expanded=False):
        colA, colB = st.columns(2)
        colA.markdown("**Tools/Technologies**")
        colA.write(", ".join(jd_cats["tools"]) or "—")
        colA.markdown("**Methods/Capabilities**")
        colA.write(", ".join(jd_cats["methods"]) or "—")
        colB.markdown("**Competencies**")
        colB.write(", ".join(jd_cats["competencies"]) or "—")
        colB.markdown("**Industry/Domain**")
        colB.write(", ".join(jd_cats["industry"]) or "—")

    st.subheader("Evidence ranking: how your bullets match this JD")

    col_top, col_bottom = st.columns(2)

    with col_top:
        st.markdown("**Most relevant bullets (keep / emphasise):**")
        top_items = evidence.get("top", []) or []
        if not top_items:
            st.write("—")
        else:
            for item in top_items:
                score_pct = item.get("score", 0.0) * 100.0
                text = item.get("text", "")
                st.write(f"✅ {score_pct:.0f}% &nbsp; {text}")

    with col_bottom:
        st.markdown("**Least relevant bullets (candidates to rewrite/remove):**")
        bottom_items = evidence.get("bottom", []) or []
        if not bottom_items:
            st.write("—")
        else:
            # Display weakest matches first
            for item in reversed(bottom_items):
                score_pct = item.get("score", 0.0) * 100.0
                text = item.get("text", "")
                st.write(f"⚠️ {score_pct:.0f}% &nbsp; {text}")

    st.subheader("Tailored Resume Bullets (Claude)")
    st.markdown("**Missing keywords (final):** " + (", ".join(sugg["missing_keywords"]) or "—"))
    for b in sugg["tailored_bullets"]:
        st.write(f"• {b}")
    st.caption(f"Notes: {sugg.get('notes', '') or '—'}")

    st.subheader("Make-Edits Prompt (copy into your LLM)")
    make_edits_prompt = build_make_edits_prompt(
        jd_text=jd_text,  # Keep JD text for context; avoid inlining raw content
        present_keywords=adv.present_keywords,
        missing_keywords=adv.missing_keywords,
        tailored_bullets=(sugg.get("tailored_bullets") or None),
        resume_text=None,  # Do not embed raw resume text
        jd_keywords_categorized=jd_cats,
        industry_hint=None,
        role_title=jd_parsed.get("title"),
        deadline=None,
        role_type=None,
        seniority=jd_parsed.get("seniority"),
        focus=None,
        extras=None,
        include_raw=False,
    )

    st.text_area(
        "Prompt",
        value=make_edits_prompt,
        label_visibility="collapsed",
        height=PROMPT_TEXTAREA_HEIGHT,
    )

    # Copy-to-clipboard button with JS handler
    _btn_id = f"copy_{uuid.uuid4().hex}"

    st_html(
        f"""
    <div>
    <button id="{_btn_id}" style="
        width:100%; padding:0.6rem; margin-top:0.5rem; border-radius:0.5rem;
        background:#262730; color:#fff; border:1px solid #4a4a4a; cursor:pointer;">
        Copy prompt to clipboard
    </button>
    </div>
    <script>
    (function() {{
    const text = {json.dumps(make_edits_prompt)};
    const btn = document.getElementById("{_btn_id}");
    if (!btn) return;
    btn.addEventListener("click", async () => {{
        try {{
        await navigator.clipboard.writeText(text);
        btn.textContent = "Copied!";
        setTimeout(() => (btn.textContent = "Copy prompt to clipboard"), 1500);
        }} catch (e) {{
        console.error(e);
        alert("Copy failed. Please allow clipboard permissions or copy manually.");
        }}
    }});
    }})();
    </script>
    """,
        height=70,
    )

    st.divider()
    st.subheader("📥 Export Results")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download Results (JSON)",
            data=_json_bytes(adv_d),
            file_name=f"ats_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download Keywords (CSV)",
            data=_csv_keywords_bytes(list(adv.present_keywords), list(adv.missing_keywords)),
            file_name=f"keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.success("Done.")
