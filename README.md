<div align="center">

# 🎯 ATS Resume Analyzer

### AI-Powered Resume Optimization & Job Description Matching

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**[🚀 Live Demo](https://your-streamlit-url.streamlit.app)** • **[📖 Documentation](#documentation)** • **[🐛 Report Bug](https://github.com/hemgandhi13/ats_resume_analyzer/issues)** • **[✨ Request Feature](https://github.com/hemgandhi13/ats_resume_analyzer/issues)**

</div>

---

## 📋 Table of Contents

- [Why This Project Exists](#-why-this-project-exists)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Architecture](#-architecture-and-module-layout)
- [Tech Stack](#-tech-stack)
- [Scoring Logic](#-scoring-logic)
- [Getting Started](#-getting-started-locally)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
  - [Docker Deployment](#-docker-deployment)
- [LLM Integration](#-llm-integration)
- [Streamlit Cloud Deployment](#-streamlit-cloud-deployment)
- [Design Decisions](#-design--infrastructure-trade-offs)
- [What This Demonstrates](#-what-this-project-demonstrates)
- [Roadmap](#-roadmap--future-work)
- [Contributing](#-contributing)
- [License](#-license)

---
## 💡 Why This Project Exists

Most graduates and early-career candidates face the same challenges:

- 🔄 **Keyword mismatch**: Every JD uses slightly different language for the same skills
- 🚫 **Unforgiving ATS**: Missing a few keywords = instant rejection
- ⏰ **Manual tailoring is slow**: Hours spent customizing resumes for each application
- 🎰 **Black-box scoring**: Existing tools give you a number without explaining *why*

**This project solves that by:**

1. 🎯 Helping candidates tailor applications systematically and transparently
2. 🏗️ Demonstrating a **full-stack ML/LLM workflow** from parsing → scoring → UI
3. ⚖️ Showing infrastructure trade-offs: heavy models, managed hosting, containerization

> 💼 **Built for:** Job seekers who want data-driven resume optimization  
> 🎓 **Built by:** A data science graduate showcasing end-to-end ML engineering skills

---
## ✨ Features

### 📄 Core Functionality
- **Multi-format parsing**: Upload PDF, DOCX, or TXT resumes
- **Smart JD analysis**: Paste any job description for instant keyword extraction
- **Dual scoring system**: Simple (TF-IDF + keywords) and Advanced (weighted, section-aware)
- **Missing keywords identification**: See exactly what your resume lacks

### 🤖 AI-Powered Enhancements *(optional)*
- **Multi-dimensional fit**: Get LLM analysis across Tools, Methods, Domain, Seniority, Communication
- **Evidence ranking**: Semantic similarity shows your strongest/weakest bullets *(requires semantic group)*
- **Rewrite suggestions**: AI-generated tailored bullets incorporating missing keywords
- **Make-edits prompt**: Export a ready-to-use prompt for any LLM

### 🎛️ Flexible Configuration
- **Heuristic mode**: Works without API keys using classical NLP
- **LLM mode**: Enhanced analysis with Claude/GPT (configurable)
- **Semantic mode**: Full embedding-based evidence ranking (optional install)

<details>
<summary>📸 Click to see screenshots</summary>

*Add your Streamlit app screenshots here*

</details>

---
## 🔧 How It Works






### Workflow Pipeline

| Step | Module | Description |
|------|--------|-------------|
| **1. Input** | `app.py` | Resume upload + JD text area |
| **2. Parsing** | `pdf_utils.py`, `parse.py` | Extract & normalize text; identify sections |
| **3. Keyword Extraction** | `nlp_backends.py` | Categorize into Tools, Methods, Competencies, Domain |
| **4. Scoring** | `scoring.py` | Simple (TF-IDF + coverage) & Advanced (weighted) scores |
| **5. Evidence Ranking** | `evidence.py` | Semantic similarity for top/bottom bullets *(optional)* |
| **6. LLM Suggestions** | `llm.py`, `prompts.py` | Multi-dimensional fit + rewrite suggestions *(optional)* |
| **7. Output** | `app.py` | Streamlit dashboard with scores, keywords, suggestions |

<details>
<summary>🔍 Detailed breakdown of each step</summary>

### 1️⃣ **Inputs**
- Resume file upload (PDF / DOCX / TXT)
- Free-text job description pasted into a text area

### 2️⃣ **Parsing**
- `pdf_utils.py` extracts and normalizes text from uploaded files
- `parse.py` splits content into sections and experience bullets, identifies raw skills

### 3️⃣ **JD Keyword Extraction**
- `nlp_backends.py` extracts keywords and categorizes them:
  - 🛠️ Tools / technologies
  - 📊 Methods / capabilities
  - 💬 Soft skills / competencies
  - 🏢 Domain / industry terms
- Modes: **Heuristic** (no LLM) or **Auto** (LLM + heuristics with fallback)

### 4️⃣ **Scoring**
- `scoring.py` computes:
  - **Simple score**: TF-IDF similarity + keyword coverage
  - **Advanced score**: Section weighting + must-have keyword penalties

### 5️⃣ **Evidence Ranking** *(optional)*
- `evidence.py` embeds JD + bullets using `sentence-transformers`
- Cosine similarity identifies top-K bullets to keep and bottom-K to rewrite

### 6️⃣ **LLM Suggestions** *(optional)*
- `llm.py` uses LiteLLM to call configured providers (Claude/GPT)
- `prompts.py` constructs prompts for:
  - Multi-dimensional fit scoring
  - Bullet rewrite suggestions
  - Copy-paste "make-edits" prompt

### 7️⃣ **UI**
- `app.py` Streamlit interface showing scores, keywords, evidence ranking, and LLM suggestions

</details>

---
## 🏗️ Architecture and Module Layout

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


### Design Principles

✅ **Modular**: Each module has a single responsibility  
✅ **Swappable**: Easy to replace Streamlit with FastAPI/CLI  
✅ **Testable**: Clear boundaries for unit testing  
✅ **Extensible**: Add new backends, scorers, or LLM providers independently

---
## 🛠️ Tech Stack

### Core Technologies

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11+ |
| **Package Manager** | [uv](https://github.com/astral-sh/uv) (fast, modern) |
| **Build System** | Hatchling |
| **UI Framework** | Streamlit |
| **PDF/DOCX** | PyMuPDF, python-docx, Pillow, pytesseract |
| **Data & ML** | pandas, numpy, scikit-learn, rapidfuzz |
| **Config** | pydantic, pydantic-settings, PyYAML, python-dotenv |
| **LLM Integration** | LiteLLM, instructor |

### Optional Semantic Group

> ⚠️ **Heavy dependencies** (800+ MB) — not installed by default in hosted demo

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch backend |
| `sentence-transformers` | Semantic embeddings |
| `transformers` | Hugging Face models |
| `huggingface_hub` | Model downloads |

**Why separate?**
- Keeps base install lightweight (~100 MB vs 1+ GB)
- Avoids CUDA wheel timeouts on free hosting
- Still fully functional without semantic features

---
## 📊 Scoring Logic

### 🎯 Simple Score (0-100)

Fast, deterministic, and explainable.

**Formula:**
simple_score = (0.5 × TF-IDF_cosine) + (0.5 × keyword_coverage) - missing_penalty


**Components:**
- **TF-IDF cosine similarity**: How close resume vocabulary matches JD vocabulary
- **Keyword coverage**: % of JD keywords present in resume
- **Missing penalty**: Soft penalty when critical JD terms are absent

**Example:**
Simple score: 42.5 / 100
├─ Keyword coverage: 35% (7/20 keywords found)
├─ TF-IDF cosine: 0.58
└─ Missing penalty: -8 (critical terms: "SQL", "Python" absent)


---

### 🎯 Advanced Score (0-100)

Adds structure and context awareness.

**Enhancements over simple:**
- **Section-aware weighting**: Skills in experience bullets > skills section
- **Must-have detection**: JD keywords appearing frequently weigh more
- **Normalized scale**: Consistent 0-100 range for cross-JD comparison

**When to use which:**
- **Simple**: Quick gut-check, comparing many resumes to one JD
- **Advanced**: Tailoring a specific resume, understanding nuanced fit

---
## 🚀 Getting Started Locally

### Prerequisites

- ✅ Python **3.11+** installed
- ✅ [`uv`](https://github.com/astral-sh/uv) package manager:
pip install uv


---

### Installation
1. Clone the repository
git clone https://github.com/hemgandhi13/ats_resume_analyzer.git
cd ats_resume_analyzer

2. Install base dependencies (no semantic, no dev)
uv sync

3. (Optional) Install semantic group for evidence ranking
uv sync --group semantic


---

### Running Locally

#### 🔹 **Lightweight Mode** (no LLM, no semantic)

uv run streamlit run src/ats/app.py


Open: http://localhost:8501

**What's available:**
- ✅ Resume/JD parsing
- ✅ Simple & advanced scoring
- ✅ Heuristic keyword extraction
- ✅ Present/missing keywords
- ❌ No semantic evidence ranking
- ❌ No LLM suggestions

---

#### 🔹 **Full Mode** (with LLM + semantic)

1. **Create `.env` file:**

.env
LLM_ENABLE=1
LLM_MODEL=claude-3-haiku-latest
ANTHROPIC_API_KEY=sk-ant-your-key-here
USE_SEMANTIC=1
NLP_EXTRACTOR=auto

2. **Install semantic group:**
uv sync --group semantic


3. **Run:**
uv run streamlit run src/ats/app.py


**What's available:**
- ✅ All lightweight features
- ✅ Semantic evidence ranking (top/bottom bullets)
- ✅ Multi-dimensional LLM fit analysis
- ✅ AI-powered rewrite suggestions

---
## 🐳 Docker Deployment

### Build the Image

docker build -t ats-resume-analyzer .


---

### Run (Lightweight Mode)

docker run --rm -p 8501:8501
-e LLM_ENABLE=0
-e USE_SEMANTIC=0
-e NLP_EXTRACTOR=heuristic
ats-resume-analyzer

Open: http://localhost:8501

---

### Run (with LLM via `.env`)

docker run --rm -p 8501:8501 --env-file .env ats-resume-analyzer


**Your `.env` should contain:**
ANTHROPIC_API_KEY=sk-ant-...
LLM_ENABLE=1
LLM_MODEL=claude-3-haiku-latest
USE_SEMANTIC=0 # still no semantic in lightweight Docker
NLP_EXTRACTOR=auto


---

### Health Check

The Dockerfile includes a built-in health check:

docker ps # Check STATUS column shows "healthy"

---
## 🤖 LLM Integration

### Philosophy

LLMs are a **copilot**, not a hard dependency.

- ✅ **With LLM**: Enriched analysis, rewrite suggestions, multi-dimensional scoring
- ✅ **Without LLM**: Still fully functional with classical NLP

---

### Configuration

| Variable | Values | Description |
|----------|--------|-------------|
| `LLM_ENABLE` | `0` / `1` | Master switch for LLM features |
| `LLM_MODEL` | `claude-3-haiku-latest`, `gpt-4o-mini`, etc. | Model to use via LiteLLM |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Provider API key |
| `NLP_EXTRACTOR` | `heuristic` / `auto` | JD keyword extraction mode |

---

### Where LLMs Are Used

#### 1. **JD Keyword Grouping** *(when `NLP_EXTRACTOR=auto`)*
- More nuanced categorization (Tools vs Methods vs Soft Skills)
- Suggests related/equivalent skills recruiters might accept

#### 2. **Multi-Dimensional Fit Scoring**
- Structured review across 5 dimensions:
  - 🛠️ Tools / Technologies
  - 📊 Methods / Capabilities
  - 🏢 Domain Knowledge
  - 👔 Seniority / Ownership
  - 💬 Communication / Stakeholder Management

#### 3. **Rewrite Suggestions**
- AI-generated bullets integrating missing keywords
- Preserves realistic tone and avoids obvious "keyword stuffing"

---

### Fallback Behavior

If LLM calls fail (missing key, timeout, etc.):
- ✅ App continues with heuristic-only extraction
- ✅ Scoring still works (TF-IDF + classical NLP)
- ✅ No hard crashes

---
## ☁️ Streamlit Cloud Deployment

### Current Live Demo

🔗 **[Live App](https://your-url.streamlit.app)** *(lightweight mode)*

**Configuration:**
- `USE_SEMANTIC=0` (no semantic embeddings)
- `LLM_ENABLE=0` (no API calls)
- `NLP_EXTRACTOR=heuristic` (rule-based only)

---

### Why No Semantic Group on Streamlit Cloud?

The semantic dependency group (`torch`, `sentence-transformers`, etc.) includes:

- 📦 **800+ MB** of wheels (including CUDA/NVIDIA binaries)
- ⏱️ **10-15 minute** cold starts
- 💾 **High memory** usage on free tier

**Solution:**
- Deploy with **base dependencies only**
- Keep semantic features for **local/Docker** use
- Clearly document this trade-off in UI

---

### How to Deploy Your Own

1. **Fork this repo**

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Click "New app"** → Select your fork

4. **Advanced settings → Secrets:**
Optional: Enable LLM features
ANTHROPIC_API_KEY = "sk-ant-your-key"
LLM_ENABLE = "1"
LLM_MODEL = "claude-3-haiku-latest"

Keep semantic disabled for fast deploys
USE_SEMANTIC = "0"
NLP_EXTRACTOR = "heuristic"

5. **Deploy**

---
## ⚖️ Design & Infrastructure Trade-offs

### 1. Semantic Embeddings vs Hosting Constraints

**Challenge:** Full evidence ranking requires `torch` + `sentence-transformers` (1+ GB)

**Solution:**
- 📦 Separate `[dependency-groups.semantic]` in `pyproject.toml`
- 💻 Install locally: `uv sync --group semantic`
- ☁️ Skip on Streamlit Cloud (fast, lightweight demo)
- 🐳 Choose per Docker build

---

### 2. LLM Dependence vs Robustness

**Challenge:** Don't want the app to break when API keys are missing

**Solution:**
- 🎛️ LLMs are **optional enhancements**, not requirements
- ✅ Core parsing/scoring works with classical NLP only
- 🔀 Graceful fallback when `LLM_ENABLE=0` or keys missing
- 🧪 Easy to test in "no-network" mode

---

### 3. Containerization vs Managed Hosting

**Challenge:** Different deployment targets have different constraints

**Solution:**
- 🐳 **Dockerfile** for reproducibility, CI/CD, internal tools
- ☁️ **Streamlit Cloud** for frictionless demos and sharing
- 📋 **Clear documentation** of what works where

This mirrors real-world patterns: containers for production, managed hosting for prototypes.

---
## 🎓 What This Project Demonstrates

For recruiters and hiring managers, this project shows I can:

### 1. **Design End-to-End ML/LLM Workflows**
- ✅ Ingestion (PDF/DOCX parsing)
- ✅ Feature engineering (TF-IDF, keyword coverage, embeddings)
- ✅ Scoring & ranking
- ✅ LLM integration for interpretability
- ✅ User-facing UI

---

### 2. **Build Modular, Production-Ready Code**
- 📦 Clear separation: `parse` → `score` → `evidence` → `llm` → `app`
- 🔌 Easy to swap Streamlit for FastAPI/CLI
- ✅ Testable components (unit tests in progress)

---

### 3. **Balance Classical ML with Modern LLMs**
- 🏗️ Solid baseline: TF-IDF, keyword matching, fuzzy search
- 🤖 LLMs as enhancement: interpretation, rewriting, not a crutch

---

### 4. **Handle Real-World Infrastructure Constraints**
- ⚖️ Heavy dependencies (torch, CUDA) and deployment trade-offs
- 🔐 API keys, authentication, safe fallbacks
- 🌍 Multiple targets: local dev, Docker, managed cloud

---

### 5. **Communicate Technical Decisions Clearly**
- 📖 Transparent about what's enabled where (semantic, LLM, etc.)
- 🛠️ Instructions for enabling features locally
- 🚀 Roadmap to production-grade system

---

### Next-Level Discussion

I'm happy to discuss:
- 🏢 Multi-tenant, production architecture (FastAPI + Postgres + Redis + CI)
- 📊 Training a supervised shortlist classifier on engineered features
- 🔄 Streaming UI with WebSocket updates
- 🧪 Comprehensive test suite + CI/CD pipeline

---
## 🗺️ Roadmap / Future Work

### Planned Extensions

- [ ] **Supervised Shortlist Model**  
  Train a classifier to predict "shortlist likelihood" using engineered features

- [ ] **API Backend**  
  Factor parsing/scoring into FastAPI so Streamlit becomes one of many clients (web app, Slack bot, CLI)

- [ ] **Comprehensive Tests & CI**  
  Expand unit/integration tests, add GitHub Actions for linting + tests on each PR

- [ ] **Lexical Fallback for Evidence Ranking**  
  Non-semantic lexical ranking when embeddings unavailable

- [ ] **Enhanced Visualizations**  
  - Skill coverage radar charts
  - Time-series of score improvements
  - Exportable PDF/HTML candidate reports

---

*These are intentionally future work to keep the README honest about what's implemented today, while signaling a clear path forward.*

---
## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Areas for Contribution
- 🧪 Unit tests for parsing, scoring, prompts
- 🌍 Multi-language support
- 📊 New scoring metrics
- 🎨 UI improvements

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📬 Contact

**Hem Gandhi**  
📧 Email: [your-email@example.com](mailto:your-email@example.com)  
💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
🐙 GitHub: [@hemgandhi13](https://github.com/hemgandhi13)

**Project Link:** [https://github.com/hemgandhi13/ats_resume_analyzer](https://github.com/hemgandhi13/ats_resume_analyzer)

---

<div align="center">

### ⭐ If you found this project helpful, please star it!

**[⬆ Back to Top](#-ats-resume-analyzer)**

</div>
