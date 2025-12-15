<div align="center">

# ATS Resume Analyzer

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

**Live demo (Streamlit, lightweight mode – no semantic embeddings):**  
**[▶️ Open the ATS Resume Analyzer on Streamlit Cloud](https://atsresumeanalyzer-wqf8cfyyqbg9wpha7rfhze.streamlit.app/)]**

</div>

---

## Table of Contents

[ATS Resume Analyzer](#ats-resume-analyzer-1)

1. [Why this project exists](#1-why-this-project-exists)
2. [What the app does](#2-what-the-app-does)
3. [How it works – high level](#3-how-it-works--high-level)
4. [Architecture and module layout](#4-architecture-and-module-layout)  
   • [Workflow Pipeline](#workflow-pipeline)

5. [Tech stack](#5-tech-stack)

6. [Scoring logic](#6-scoring-logic)  
   • [6.1 Simple score](#61-simple-score)  
   • [6.2 Advanced score](#62-advanced-score)

7. [LLM integration (conceptual)](#7-llm-integration-conceptual)

8. [Getting started locally (uv)](#8-getting-started-locally-uv)  
   • [8.1 Prerequisites](#81-prerequisites)  
   • [8.2 Clone and install base dependencies](#82-clone-and-install-base-dependencies)  
   • [Run the app](#run-the-app)  
   • [8.3 (Optional) Install semantic dependency group locally](#83-optional-install-semantic-dependency-group-locally)  
   • [8.4 Enable LLM features (optional)](#84-enable-llm-features-optional)

9. [Run via Docker (lightweight, no semantic group)](#9-run-via-docker-lightweight-no-semantic-group)

10. [Streamlit Cloud deployment (no semantic group)](#10-streamlit-cloud-deployment-no-semantic-group)  
    • [10.1 Why the semantic group is not installed online](#101-why-the-semantic-group-is-not-installed-online)  
    • [10.2 How the Streamlit demo is configured](#102-how-the-streamlit-demo-is-configured)

11. [Design & infrastructure trade-offs](#11-design--infrastructure-trade-offs)  
    • [11.1 Semantic embeddings vs hosting constraints](#111-semantic-embeddings-vs-hosting-constraints)  
    • [11.2 LLM dependence vs robustness](#112-llm-dependence-vs-robustness)  
    • [11.3 Containerisation vs managed hosting](#113-containerisation-vs-managed-hosting)

12. [What this project demonstrates (Recruiters Friendly Section)](#12-what-this-project-demonstrates-recruiters-friendly-section)

13. [Roadmap / future work / Limitations and Assumptions](#13-roadmap--future-work)

---

# ATS Resume Analyzer

An end-to-end ATS assistant that:

- Parses PDF / DOCX resumes and raw job descriptions
- Extracts structured skills and JD keyword groups
- Scores ATS fit using classic NLP + engineered features
- Optionally uses LLMs for deeper “fit” feedback and rewrite suggestions

> Live demo (Streamlit, **lightweight mode – no semantic embeddings**):  
> **[▶️ Open the ATS Resume Analyzer on Streamlit Cloud](https://atsresumeanalyzer-wqf8cfyyqbg9wpha7rfhze.streamlit.app/)**

---

## 1. Why this project exists

Most graduates and early-career candidates run into the same problems:

- Every job description uses slightly different language for **the same skills**.
- ATS screeners are unforgiving about **keyword coverage**.
- Manually tailoring a resume for every role is **slow, repetitive and error-prone**.
- Existing “ATS score” tools often behave like **black boxes**: you see a number, but not the reasoning.

I wanted a project that would:

1. Help me tailor my own applications more systematically.
2. Demonstrate that I can design and ship a **full data/ML/LLM workflow**, not just a Jupyter notebook.
3. Show that I understand **infrastructure trade-offs**: heavy models, managed hosting limits, and containerisation.

---

## 2. What the app does

At a user level, the ATS Resume Analyzer lets you:

- Upload a **resume** (PDF / DOCX / TXT).
- Paste a **job description**.
- See:

  - A **simple ATS score** and a more nuanced **advanced score** (0–100).
  - **Present vs missing JD keywords** broken down by type.
  - (When semantic mode is enabled) an **evidence ranking** of your strongest and weakest bullets.
  - (When LLM is enabled) a **multi-dimensional fit breakdown** and **rewrite suggestions**.

- Generate a **“make-edits” prompt** you can paste into any LLM to continue iterating on the resume.

The goal is to support a realistic recruiter workflow: “What does this JD care about? Where does my resume already prove it? What should I rewrite next?”

---

## 3. How it works – high level

1. **Inputs**

   - Resume file upload (PDF / DOCX / TXT).
   - Free-text job description pasted into a text area.

2. **Parsing**

   - `pdf_utils.py` extracts and normalises text from uploaded files.
   - `parse.py` splits content into sections and experience bullets, and identifies raw skills.

3. **JD keyword extraction**

   - `nlp_backends.py` extracts keywords and categorises them into:
     - Tools / technologies
     - Methods / capabilities
     - Soft skills / competencies
     - Domain / industry terms
   - Can run in:
     - **Heuristic mode** (no LLM; rule-based only).
     - **Auto mode** (LLM + heuristics, with safe fallback).

4. **Scoring**

   - `scoring.py` computes:
     - A **simple score** based on TF-IDF similarity + keyword coverage.
     - An **advanced score** that also considers section weighting and “must-have” keywords.

5. **Evidence ranking (optional semantic layer)**

   - `evidence.py` calls `_embed(...)` from `nlp_backends.py` to embed JD + bullets when the **semantic dependency group** is installed.
   - Cosine similarity identifies **top-K bullets to keep/emphasise** and **bottom-K to rewrite/remove**.

6. **LLM suggestions (optional)**

   - `llm.py` uses LiteLLM to talk to a configured provider (e.g. Anthropic Claude).
   - `prompts.py` constructs structured prompts for:
     - Multi-dimensional fit scoring.
     - Bullet rewrite suggestions.
     - A copy-paste “make-edits” block for your own LLM usage.

7. **UI**

   - `app.py` wires everything together in a Streamlit app:
     - Inputs, scores, present/missing keywords.
     - JD keyword groups.
     - Evidence ranking (if available).
     - LLM-powered explanations and suggestions (if enabled).

---

## 4. Architecture and module layout
```
This is a small but fully modular Python package with a `src/` layout.
ats_resume_analyzer/
├── src/ats/
│ ├── app.py # 🎨 Streamlit UI & orchestration
│ ├── pdf_utils.py # 📄 PDF/DOCX reading & text extraction
│ ├── parse.py # ✂️ Resume/JD parsing & section detection
│ ├── nlp_backends.py # 🧠 Keyword extraction & embeddings
│ ├── scoring.py # 📊 Simple & advanced ATS scoring
│ ├── evidence.py # 🔍 Semantic bullet ranking
│ ├── llm.py # 🤖 LiteLLM wrapper for AI providers
│ └── prompts/
│ └── prompts.py # 💬 Prompt templates
├── config/ # 📋 Stopwords, skill lists, normalization
├── tests/ # ✅ Unit tests (in progress)
├── Dockerfile # 🐳 Container definition
├── pyproject.toml # 📦 Dependencies (uv)
├── uv.lock # 🔒 Locked dependency versions
└── README.md # 📖 This file
```
### Workflow Pipeline

![ATS Resume Analyzer Architecture](./assets/architecture_diagram.png)
| Step | Module | Description |
|------|--------|-------------|
| **1. Input** | `app.py` | Resume upload + JD text area |
| **2. Parsing** | `pdf_utils.py`, `parse.py` | Extract & normalize text; identify sections |
| **3. Keyword Extraction** | `nlp_backends.py` | Categorize into Tools, Methods, Competencies, Domain |
| **4. Scoring** | `scoring.py` | Simple (TF-IDF + coverage) & Advanced (weighted) scores |
| **5. Evidence Ranking** | `evidence.py` | Semantic similarity for top/bottom bullets _(optional)_ |
| **6. LLM Suggestions** | `llm.py`, `prompts.py` | Multi-dimensional fit + rewrite suggestions _(optional)_ |
| **7. Output** | `app.py` | Streamlit dashboard with scores, keywords, suggestions |

**Project structure (core parts)**

- `src/ats/app.py`  
  Streamlit UI and orchestration.

- `src/ats/pdf_utils.py`  
  PDF / DOCX reading, text extraction, basic cleaning.

- `src/ats/parse.py`

  - Resume section detection (experience, skills, education, etc.).
  - Bullet extraction and normalisation.
  - Job description parsing and raw skill extraction.

- `src/ats/nlp_backends.py`

  - JD keyword extraction (heuristic + optional LLM-assisted).
  - Embedding utilities (`_embed`) using sentence-transformers (when installed).
  - Configuration via environment variables (e.g. `NLP_EXTRACTOR`, `EMBEDDING_MODEL`).

- `src/ats/scoring.py`

  - Simple ATS scoring.
  - Advanced scoring combining multiple signals.

- `src/ats/evidence.py`

  - Takes JD + resume bullets.
  - Uses embeddings (when available) to rank which bullets most/least strongly support the JD.

- `src/ats/llm.py`

  - LiteLLM wrapper for calling LLMs in a controlled way.
  - Central place for handling model names, timeouts, and error handling.

- `src/ats/prompts.py`

  - Prompt templates for JD keyword grouping, fit scoring, and bullet rewrites.

- `config/`

  - Stopwords, normalisation maps, canonical skill lists, and other text resources.

- `tests/`
  - Space reserved for unit tests (parsing, scoring, prompts, etc.).

This separation makes it easy to:

- Swap Streamlit for another frontend (e.g. FastAPI + React).
- Reuse parsing and scoring functions in a CLI or batch job.
- Extend or replace the LLM backend independently of the UI.

---

## 5. Tech stack

**Language & packaging**

- Python 3.11
- `pyproject.toml` with `hatchling` build backend
- Dependency management via **`uv`**

**Core libraries**

| Category            | Technologies                                                                |
| ------------------- | --------------------------------------------------------------------------- |
| **Language**        | Python 3.11+                                                                |
| **Package Manager** | [uv](https://github.com/astral-sh/uv) (fast, modern)                        |
| **Build System**    | Hatchling                                                                   |
| **UI Framework**    | Streamlit                                                                   |
| **PDF/DOCX**        | PyMuPDF, python-docx, Pillow, pytesseract                                   |
| **Data & ML**       | pandas, numpy, scikit-learn, rapidfuzz                                      |
| **Config**          | pydantic, pydantic-settings, PyYAML, python-dotenv                          |
| **LLM Integration** | LiteLLM (as the routing layer ), instructor(for schema-constrained outputs) |

- **LLM integration:**
  - `litellm` as the routing layer
  - `instructor` for schema-constrained outputs

**Optional semantic dependency group**

To keep the **default environment lightweight**, the heavy semantic libraries are placed in a _separate_ dependency group in `pyproject.toml`, for example:

| Package                 | Purpose             |
| ----------------------- | ------------------- |
| `torch`                 | PyTorch backend     |
| `sentence-transformers` | Semantic embeddings |
| `transformers`          | Hugging Face models |
| `huggingface_hub`       | Model downloads     |

This design is intentional:

- Locally (and in suitable Docker images), you can install this group and get **full semantic evidence ranking**.
- On constrained platforms like free Streamlit Cloud, these dependencies are **not installed by default** to avoid huge CUDA/NVIDIA wheel downloads and image size explosions.
- The README and UI explicitly document this behaviour so it’s clear which features are active in each environment.

Details of the different environments (local, Docker, Streamlit Cloud) and how the semantic group behaves there are covered in the deployment section that follows.

---

## 6. Scoring logic

The app exposes two complementary scores: **simple** and **advanced**.

### 6.1 Simple score

The simple score is designed to be:

- Fast
- Deterministic
- Easy to explain

It combines:

- **TF-IDF cosine similarity** between JD and resume text.
- **Keyword coverage** (what fraction of JD keywords appear in the resume).
- A soft **penalty** when critical JD terms are missing.

This gives an immediate sense of “rough ATS alignment” that is comparable across JDs.

### 6.2 Advanced score

The advanced score builds on the simple components and adds more structure:

- **Section-aware weighting**

  - Skills embedded inside experience bullets count more than those only listed in a skills section.

- **Penalty for missing must-have skills**

  - Keywords that appear frequently and prominently in the JD weigh more heavily.

- **Normalisation to a 0–100 scale**
  - Makes it easier to interpret at a glance and compare across roles.

The advanced score is what the UI highlights as the primary “fit” indicator.

---

## 7. LLM integration (conceptual)

The app treats LLMs as a **copilot**, not a hard dependency.

**Where LLMs are used (when enabled):**

- **JD keyword grouping & enrichment**

  - More nuanced grouping of tools vs methods vs behaviours.
  - Suggestions for related skills that recruiters might accept as equivalents.

- **Multi-dimensional fit scoring**

  - A structured “review” of the resume against the JD along dimensions like:
    - Tools / technologies
    - Methods / analytical capabilities
    - Domain knowledge
    - Seniority / ownership
    - Communication / stakeholder management

- **Rewrite suggestions**
  - Candidate bullets that integrate missing keywords while preserving realistic tone.

**How it’s controlled:**

- `LLM_ENABLE` – master switch (`0` / `1`).
- `LLM_MODEL` – e.g. `claude-3-5-haiku-20241022` (configurable).
- `ANTHROPIC_API_KEY` (or other provider keys) – read from environment.
- `NLP_EXTRACTOR` – controls whether JD keyword extraction uses LLMs (`auto`) or stays heuristic (`heuristic`).

If keys are missing or disabled, the app:

- Still parses and scores using classical NLP.
- Surfaces deterministic outputs only.
- Avoids hard crashes by falling back to heuristic behaviour where possible.

---

## 8. Getting started locally (uv)

Local development is based on [`uv`](https://github.com/astral-sh/uv), which manages a `.venv` from `pyproject.toml`.

### 8.1 Prerequisites

- Python **3.11** installed on your machine
- `uv` installed globally:

```bash
pip install uv
```

### 8.2 Clone and install base dependencies

```bash
git clone https://github.com/hemgandhi13/ats_resume_analyzer.git
cd ats_resume_analyzer

# Create .venv and install base dependencies (no semantic, no dev groups)
uv sync
```

This installs the core stack: Streamlit UI, parsing, scoring, config, and LLM plumbing (without forcing GPU/CUDA-heavy packages).

### Run the app

```bash
uv run streamlit run src/ats/app.py
```

Open: http://localhost:8501

You’ll be running in lightweight mode:

- Parsing + keyword extraction + simple/advanced scoring.
- Heuristic JD keyword extraction by default.
- No semantic embeddings unless you explicitly install them.
- LLM features off unless you enable them via environment variables.

---

### 8.3 (Optional) Install semantic dependency group locally

If you want the full evidence ranking experience on your own machine and you’re comfortable installing heavier packages:

Install the `semantic` group (as defined in `pyproject.toml`):

```bash
uv sync --group semantic
```

Restart the app:

```bash
uv run streamlit run src/ats/app.py
```

Now `_embed(...)` will load a sentence-transformers model, and the **Evidence ranking** panel will show your most and least relevant bullets.

On managed platforms like free Streamlit Cloud, this group is intentionally not installed to avoid large CUDA/NVIDIA-backed wheels. The deployment section explains how the hosted demo is configured and why.

### 8.4 Enable LLM features (optional)

By default, the app runs purely on classic NLP / heuristics so it works even without any API keys.

To turn on the LLM-enhanced features locally (multi-dimensional fit panel + rewrite suggestions):

1. Make sure base dependencies are installed:

   ```bash
   uv sync
   ```

2. Set the LLM environment variables before running Streamlit.

**PowerShell:**

```bash
$env:LLM_ENABLE = "1"
$env:LLM_MODEL = "claude-3-5-haiku-20241022"
$env:ANTHROPIC_API_KEY = "sk-ant-..."  # your real key
```

**CMD:**

```bash
set LLM_ENABLE=1
set LLM_MODEL=claude-3-5-haiku-20241022
set ANTHROPIC_API_KEY=sk-ant-...
```

**macOS / Linux (bash/zsh):**

```bash
export LLM_ENABLE=1
export LLM_MODEL=claude-3-5-haiku-20241022
export ANTHROPIC_API_KEY=sk-ant-...
```

3. Run the app via `uv`:

```bash
uv run streamlit run src/ats/app.py
```

With this enabled you’ll see:

- The multi-dimensional fit panel populated by the LLM.
- Tailored resume bullet suggestions that integrate missing keywords.

If keys are missing or the LLM call fails, the app is designed to fall back to deterministic behaviour where possible.

### 9. Run via Docker (lightweight, no semantic group)

Some environments don’t make it easy (or necessary) to install the full semantic stack (`torch`, `sentence-transformers`, etc.).  
For those cases the repo includes a `Dockerfile` that:

- Uses `python:3.11-slim`.
- Installs `uv` and the base dependency set (no `semantic` group).
- Creates a `.venv` inside the image.
- Runs the Streamlit app in the container.
- Exposes port `8501` and adds a simple HTTP healthcheck.

**Build the image:**

```bash
docker build -t ats-resume-analyzer .
```

**Run in lightweight mode (no LLM, no semantic embeddings):**

```bash
docker run --rm -p 8501:8501 \
  -e LLM_ENABLE=0 \
  -e USE_SEMANTIC=0 \
  -e NLP_EXTRACTOR=heuristic \
  ats-resume-analyzer
```

Then open:

```text
http://localhost:8501
```

This mode deliberately skips:

- The semantic dependency group (`torch`, `sentence-transformers`, `transformers`, `huggingface_hub`).
- External LLM calls.

…but still showcases:

- Parsing and JD keyword extraction.
- Simple and advanced ATS scoring.
- The full Streamlit UX and workflow.

This is the mode I use for quick demos and portfolio screenshots.

---

### 10. Streamlit Cloud deployment (no semantic group)

The public demo linked at the top of this README is deployed on Streamlit Community Cloud using the same base environment as above.

#### 10.1 Why the semantic group is not installed online

The semantic dependency group includes:

- `torch`
- `sentence-transformers`
- `transformers`
- `huggingface_hub`

Installing these on a managed free-tier platform:

- Pulls very large wheels (often including CUDA/NVIDIA-related binaries even for CPU-only usage).
- Can hit timeouts, memory limits, or long cold-starts.

To keep the hosted demo reliable and fast, the Streamlit Cloud deployment:

- Installs only the base dependencies from `pyproject.toml` (no `semantic` group).
- Runs with `USE_SEMANTIC=0` so the evidence-ranking panel politely degrades when embeddings are not available.

The README and UI explicitly call this out so it’s clear that:

- The local / Docker version can run with full semantic evidence ranking if you install the `semantic` group, while
- The online demo focuses on parsing + keyword coverage + classical scoring.

#### 10.2 How the Streamlit demo is configured

At a high level the deployed app is configured with environment variables like:

```text
NLP_EXTRACTOR=heuristic
LLM_ENABLE=0
USE_SEMANTIC=0
```

This keeps the demo independent of:

- Heavy GPU-oriented libraries.
- External API keys.

If you fork the repo and deploy your own Streamlit app, you can choose to:

- Keep this lightweight configuration, or
- Add your own keys (`ANTHROPIC_API_KEY`, etc.) and enable LLM features, understanding that semantic features may still be best kept for local/Docker runs.

### 11. Design & infrastructure trade-offs

Some trade-offs in this project are deliberate and documented.

#### 11.1 Semantic embeddings vs hosting constraints

Full evidence ranking requires the `semantic` group (`sentence-transformers` + `torch`):

- This is great for local analysis, but heavy for free cloud environments.

The solution:

- Semantic dependencies live in a separate `uv` dependency group.
- Local and Docker environments can opt in via:

  ```bash
  uv sync --group semantic
  ```

- The Streamlit Cloud demo runs with the base group only, clearly labelled as **“no semantic embeddings”**.

#### 11.2 LLM dependence vs robustness

LLMs are used for enrichment, not as a hard dependency:

- JD keyword grouping.
- Multi-dimensional fit interpretation.
- Rewrite suggestions.

When `LLM_ENABLE=0` or keys are missing:

- The app still parses and scores via classic NLP + heuristics.
- JD extraction can run purely in heuristic mode.
- The UI remains usable without API keys or network calls.

#### 11.3 Containerisation vs managed hosting

The repo includes:

- A `Dockerfile` and `.dockerignore` suitable for CI/CD pipelines and internal deployment.
- A `pyproject.toml` + `uv` setup that plays nicely with managed platforms like Streamlit Cloud.

This mirrors a realistic pattern:

- Containers for reproducibility, internal tools, and MLOps stories.
- Managed hosting for frictionless demos and sharing with non-technical stakeholders.

---

### 12. What this project demonstrates (Recruiters Friendly Section)

This project is intentionally more than a single Jupyter notebook. It shows that I can:

- **Design an end-to-end ATS / ML / LLM workflow:**

  - Ingestion of PDF/DOCX/TXT resumes and raw JDs.
  - Parsing, structured extraction, and feature engineering.
  - Scoring, ranking, and LLM-assisted guidance.
  - A usable UI that ties the flow together.

- **Build a modular Python package with clear boundaries:**

  - `pdf_utils` / `parse` / `scoring` / `evidence` / `llm` / `prompts` / `app`.
  - Easy to plug into other frontends (FastAPI, CLI, etc.).

- **Balance classical ML/NLP with modern LLMs:**

  - Use TF–IDF, keyword coverage, and fuzzy matching as a solid baseline.
  - Layer LLMs on top for interpretation and rewriting, not as a crutch.

- **Handle real-world infrastructure constraints:**

  - Heavy semantic dependencies and their impact on build times and hosting.
  - API keys / authentication and safe fallbacks.
  - Different deployment targets: local dev, Docker, Streamlit Cloud.

- **Communicate design decisions and trade-offs clearly:**
  - Why certain features are disabled in the public demo.
  - How to enable them locally.
  - What would be required to take this towards production.

If you’d like to discuss how I’d evolve this into a multi-tenant, production-grade service (e.g. FastAPI backend + streaming UI + Redis/Postgres + CI pipelines), I’m happy to walk through the design.

### 13. Roadmap / future work

Planned extensions for the next iterations:

- **Supervised shortlist model**  
  Train a small classifier to predict shortlist likelihood using the engineered features (coverage, scores, evidence ranking, JD complexity).

- **API backend**  
  Factor parsing and scoring into a FastAPI (or similar) backend so Streamlit becomes just one of several clients (web app, Slack bot, CLI).

- **Tests & CI**  
  Expand unit and integration tests (parsing, scoring, prompts, evidence ranking) and add CI workflows to run linting and tests on each push/PR.

- **Lexical fallback for evidence ranking**  
  Add a non-semantic lexical ranking so the evidence panel stays useful even when embeddings are disabled or unavailable.

- **Visualisations & reports**  
  Add richer visual components and exportable reports, e.g.:
  - Skill coverage radar charts.
  - Time-series of score improvements as you edit a resume.
  - PDF/HTML “candidate report” for recruiters.

### Limitations & assumptions

- Best for English-language JDs and resumes.
- Parsing works best on text-based PDFs/DOCX; scanned images rely on OCR and may be noisy.
- Scores are heuristic / proxy ATS metrics, not an exact replica of any specific vendor.
- LLM suggestions are non-deterministic and depend on the configured model + API key.
- Online demo runs in “lightweight mode” (no semantic embeddings, LLM disabled by default).

---

### License

This project is licensed under the MIT License – see `LICENSE` for details.

---

### Author

Built by **Hem Gandhi** – feel free to reach out on LinkedIn:  
[linkedin.com/in/hem-gandhi-92757b195](https://www.linkedin.com/in/hem-gandhi-92757b195/)
