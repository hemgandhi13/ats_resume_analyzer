# ATS Resume Analyzer (Beginner-Friendly)

Upload a PDF resume and paste a Job Description (JD) to:

- Extract & normalize text (PDF → text)
- Parse basics (name, email, phone, skills)
- Score fit with **keyword coverage** + **TF-IDF cosine** (0–100)
- Get **tailored resume bullets** + **missing keywords** (LLM optional)
- Export results to **CSV/JSON**

> This project is built step-by-step. Start with environment setup below. Code files (Streamlit UI, CLI, etc.) will be added next.

---

## Tech Stack

- **Python 3.11+**
- **uv** (fast env + dependencies using `pyproject.toml`)
- **Streamlit** (web UI)
- **Typer** (CLI)
- **PyMuPDF** (PDF text extraction)
- **pandas**, **scikit-learn** (TF-IDF + cosine)
- **rapidfuzz** (fast keyword matching)
- **pydantic**, **pydantic-settings** (typed models & config)
- **LiteLLM** (optional: Claude/GPT/Gemini via one wrapper)

---

## Prerequisites

- Python 3.11 or newer
- `uv` installed:
  ```bash
  pipx install uv || pip install uv
  ```
