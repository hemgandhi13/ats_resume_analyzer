from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

# Central JD extractor from nlp_backends to keep vocabulary consistent
from ats.nlp_backends import _CFG, extract_jd_keywords
from ats.pdf_utils import normalize, tokenize_words

# ============================================================
# Regex patterns used for resume parsing
# ============================================================

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# International-friendly phone heuristic (AU-biased but permissive)
PHONE_RE = re.compile(
    r"""
    (?:\+?\d{1,3}[\s\-]?)?         # optional country code (+61)
    (?:\(?\d{1,4}\)?[\s\-]?)?      # optional area / trunk code (03), (0x)
    \d{3,4}[\s\-]?\d{3,4}          # local number chunks 3-4 + 3-4
    """,
    re.VERBOSE,
)

# Month names (Jan, January, etc.)
MONTH_RE = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

# Common date ranges: "Jan 2020 - Dec 2023", "2020-Present", "07/2020 - 08/2021"
DATE_RANGE_RE = re.compile(
    rf"""
    (?:
        (?:{MONTH_RE}\s+\d{{4}}|\d{{2}}/\d{{4}}|\d{{4}})
        \s*(?:-|–|to)\s*
        (?:{MONTH_RE}\s+\d{{4}}|\d{{2}}/\d{{4}}|\d{{4}}|Present|present|Current|current)
    )
    """,
    re.VERBOSE,
)

# ============================================================
# Skills vocabulary (kept lowercase for matching; display keeps original case)
# ============================================================

# Prefer canonical skills from config when available
BASIC_SKILLS: Set[str] = (
    set(_CFG.canonical_skills)
    if _CFG.canonical_skills
    else {
        # Languages / core
        "python",
        "java",
        "javascript",
        "typescript",
        "r",
        "sql",
        "c",
        "c++",
        "scala",
        "go",
        # DS / ML / NLP
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "keras",
        "xgboost",
        "lightgbm",
        "catboost",
        "statsmodels",
        "spacy",
        "nltk",
        "nlp",
        "computer vision",
        "recommendation systems",
        "tf-idf",
        "tfidf",
        "cosine similarity",
        "word2vec",
        "transformers",
        # Data viz
        "matplotlib",
        "seaborn",
        "plotly",
        "altair",
        "power bi",
        "tableau",
        "looker",
        "grafana",
        # Data tooling / ETL
        "airflow",
        "dbt",
        "great expectations",
        "kafka",
        "spark",
        # Databases
        "postgresql",
        "mysql",
        "sqlite",
        "mongodb",
        "dynamodb",
        "redshift",
        "snowflake",
        "bigquery",
        # Web / apps
        "streamlit",
        "fastapi",
        "flask",
        "dash",
        # DevOps / MLOps
        "docker",
        "kubernetes",
        "mlflow",
        "dvc",
        "jenkins",
        "github actions",
        # Cloud
        "aws",
        "azure",
        "gcp",
        "s3",
        "ec2",
        "lambda",
        "sagemaker",
        "vertex ai",
        # OS / tools
        "linux",
        "bash",
        "git",
        # Analytics / math
        "statistics",
        "hypothesis testing",
        "feature engineering",
        "time series",
        "a/b testing",
        # Excel
        "excel",
        "vlookup",
        "pivot tables",
    }
)

# Categorised skills used for resume parsing, quick matching, and UI display.
# For JD keyword extraction, call extract_jd_keywords() in nlp_backends instead.
SKILLS_BY_CATEGORY: Dict[str, Set[str]] = {
    "programming": {
        "python",
        "java",
        "javascript",
        "typescript",
        "r",
        "sql",
        "c",
        "c++",
        "scala",
        "go",
    },
    "ml_frameworks": {
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "keras",
        "xgboost",
        "lightgbm",
        "catboost",
    },
    "nlp_cv": {"spacy", "nltk", "transformers", "word2vec", "nlp", "computer vision"},
    "data_tools": {"pandas", "numpy", "matplotlib", "seaborn", "plotly", "altair"},
    "databases": {
        "postgresql",
        "mysql",
        "sqlite",
        "mongodb",
        "snowflake",
        "redshift",
        "bigquery",
        "dynamodb",
    },
    "viz_bi": {"power bi", "tableau", "looker", "grafana"},
    "cloud": {"aws", "azure", "gcp", "s3", "ec2", "lambda", "sagemaker", "vertex ai"},
    "mlops": {"docker", "kubernetes", "mlflow", "dvc", "jenkins", "github actions"},
    "etl": {"airflow", "dbt", "kafka", "spark"},
    "analytics": {
        "statistics",
        "hypothesis testing",
        "feature engineering",
        "time series",
        "a/b testing",
    },
    "apps": {"streamlit", "fastapi", "flask", "dash"},
    "excel": {"excel", "vlookup", "pivot tables"},
    "similarity": {"tf-idf", "tfidf", "cosine similarity"},
}


# ============================================================
# Basic parsers (email / phone / name)
# ============================================================


def find_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None


def find_phone(text: str) -> str | None:
    # Prefer lines with phone markers; fallback to any matching line
    markers = ("phone", "mobile", "tel", "contact", "+")
    cand = None
    for line in text.splitlines():
        if any(k in line.lower() for k in markers):
            m = PHONE_RE.search(line)
            if m:
                return m.group(0)
        if not cand:
            m = PHONE_RE.search(line)
            if m:
                cand = m.group(0)
    return cand


def heuristic_name(resume_text: str) -> str | None:
    """
    Guess a name from the first few visible lines that:
    - aren't email/phone lines
    - have 2–4 alphabetic tokens
    """
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    for ln in lines[:12]:
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln):
            continue
        parts = ln.split()
        if 2 <= len(parts) <= 4 and all(p[:1].isalpha() for p in parts):
            bad = {"resume", "curriculum vitae", "profile", "summary"}
            if normalize(ln) in bad:
                continue
            return ln
    return None


# ============================================================
# Links (LinkedIn, GitHub, generic portfolio)
# ============================================================

LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_/]+", re.IGNORECASE
)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9\-_/]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)


def extract_links(text: str) -> Dict[str, Optional[str]]:
    links: Dict[str, Optional[str]] = {
        "linkedin": None,
        "github": None,
        "portfolio": None,
    }

    m = LINKEDIN_RE.search(text)
    if m:
        url = m.group(0)
        if not url.lower().startswith("http"):
            url = "https://" + url
        links["linkedin"] = url

    m = GITHUB_RE.search(text)
    if m:
        url = m.group(0)
        if not url.lower().startswith("http"):
            url = "https://" + url
        links["github"] = url

    # Use first non-LinkedIn/GitHub URL as a portfolio link
    for url in URL_RE.findall(text):
        u = url.strip().strip(").,]")
        low = u.lower()
        if "linkedin.com" not in low and "github.com" not in low:
            links["portfolio"] = u
            break

    return links


# ============================================================
# Location (simple AU-first heuristic + a few global cities)
# ============================================================

AUS_LOCATIONS = {
    "melbourne",
    "sydney",
    "brisbane",
    "perth",
    "adelaide",
    "canberra",
    "hobart",
    "darwin",
    "vic",
    "nsw",
    "qld",
    "wa",
    "sa",
    "act",
    "tas",
    "nt",
    "victoria",
    "new south wales",
    "queensland",
    "western australia",
    "south australia",
    "australian capital territory",
    "tasmania",
    "northern territory",
}

GLOBAL_LOCATIONS = {
    "mumbai",
    "delhi",
    "new delhi",
    "bengaluru",
    "bangalore",
    "hyderabad",
    "pune",
    "ahmedabad",
    "kolkata",
    "chennai",
    "auckland",
    "wellington",
    "london",
    "new york",
    "san francisco",
    "singapore",
}


def extract_location(text: str) -> Optional[str]:
    # Focus on the header portion of the resume
    for ln in text.splitlines()[:20]:
        low = ln.lower()
        if any(loc in low for loc in AUS_LOCATIONS | GLOBAL_LOCATIONS):
            return ln.strip()
    return None


# ============================================================
# Summary / Objective
# ============================================================


def extract_summary(text: str) -> Optional[str]:
    """
    Extract 3–5 lines following a header like: Summary / Objective / Profile / About
    """
    headers = {"summary", "objective", "profile", "about"}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if any(h in line.lower() for h in headers):
            # Capture subsequent non-empty lines until blank or five lines
            buf: List[str] = []
            for j in range(i + 1, min(i + 6, len(lines))):
                s = lines[j].strip()
                if not s:
                    break
                buf.append(s)
            if buf:
                return " ".join(buf)
    return None


# ============================================================
# Skills (flat + categorised)
# ============================================================


def extract_skills(text: str, skills_vocab: Set[str] | None = None) -> Set[str]:
    """
    Detect skills from a vocabulary.
    - Single-word terms match as tokens.
    - Multi-word terms match as substring on normalized text.
    """
    vocab = {normalize(x) for x in (skills_vocab or BASIC_SKILLS)}
    norm_text = normalize(text)
    wordset = set(tokenize_words(norm_text))
    hits: Set[str] = set()
    for term in vocab:
        if " " in term:
            if term in norm_text:
                hits.add(term)
        else:
            if term in wordset:
                hits.add(term)
    return hits


def extract_skills_categorized(text: str) -> Dict[str, Set[str]]:
    """
    Group skills by category using SKILLS_BY_CATEGORY.
    """
    res: Dict[str, Set[str]] = {}
    norm_text = normalize(text)
    wordset = set(tokenize_words(norm_text))
    for cat, terms in SKILLS_BY_CATEGORY.items():
        hits: Set[str] = set()
        for term in terms:
            nt = normalize(term)
            if " " in nt:
                if nt in norm_text:
                    hits.add(term)
            else:
                if nt in wordset:
                    hits.add(term)
        if hits:
            res[cat] = hits
    return res


# ============================================================
# Education (degree, institution, year/dates)
# ============================================================

DEGREE_WORDS = [
    # long forms
    "bachelor",
    "master",
    "doctor",
    "phd",
    "mba",
    "btech",
    "mtech",
    # short forms
    "b.s",
    "b.sc",
    "bsc",
    "b.e",
    "be",
    "b.eng",
    "beng",
    "m.s",
    "m.sc",
    "msc",
    "m.e",
    "me",
    "m.eng",
    "meng",
]
FIELD_WORDS = [
    "data science",
    "computer science",
    "software engineering",
    "information technology",
    "statistics",
    "mathematics",
    "business",
    "finance",
    "economics",
    "electrical engineering",
]


def extract_education(text: str) -> List[Dict[str, str]]:
    """
    Extract education lines; heuristic:
    - find lines mentioning degree words
    - look +/- 2 lines for institution keywords and years/dates
    """
    uni_keywords = {"university", "institute", "college", "school of", "faculty of"}
    lines = text.splitlines()
    out: List[Dict[str, str]] = []

    def has_degree(line: str) -> bool:
        low = line.lower()
        return any(dw in low for dw in DEGREE_WORDS)

    for i, line in enumerate(lines):
        if not has_degree(line):
            continue

        entry: Dict[str, str] = {"degree": line.strip()}
        # Capture field if present on the same line
        low = line.lower()
        for fw in FIELD_WORDS:
            if fw in low and "field" not in entry:
                entry["field"] = fw.title()

        # Scan nearby lines for institution and date/year
        for j in range(max(0, i - 2), min(len(lines), i + 3)):
            near = lines[j].strip()
            if not near:
                continue
            low = near.lower()

            if "institution" not in entry and any(k in low for k in uni_keywords):
                entry["institution"] = near

            if "dates" not in entry:
                m = DATE_RANGE_RE.search(near)
                if m:
                    entry["dates"] = m.group(0)
                else:
                    y = re.search(r"\b(19|20)\d{2}\b", near)
                    if y:
                        entry["year"] = y.group(0)

        # Avoid duplicate entries
        if entry not in out:
            out.append(entry)

    return out


# ============================================================
# Experience (title, company, dates) — simple heuristic
# ============================================================

TITLE_HINTS = {
    "analyst",
    "scientist",
    "engineer",
    "developer",
    "manager",
    "consultant",
    "specialist",
    "coordinator",
    "lead",
    "head",
    "intern",
    "associate",
    "architect",
}


def extract_experience(text: str) -> List[Dict[str, str]]:
    """
    Extract basic experience items:
    - A line containing a title hint becomes the 'title'
    - Nearby lines that look like company (short proper-case) or date range become 'company'/'dates'
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: List[Dict[str, str]] = []

    def looks_like_company(s: str) -> bool:
        s = s.strip()
        if not s or s.isupper():  # resume headers often ALLCAPS; skip
            return False
        words = s.split()
        return 1 <= len(words) <= 6 and any(w[0:1].isupper() for w in words)

    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(h in low for h in TITLE_HINTS):
            item: Dict[str, str] = {"title": ln.strip()}
            # Scan nearby lines for company and dates
            for j in range(max(0, i - 2), min(len(lines), i + 4)):
                s = lines[j].strip()
                if not s:
                    continue
                if "dates" not in item:
                    m = DATE_RANGE_RE.search(s)
                    if m:
                        item["dates"] = m.group(0)
                if "company" not in item and j != i and looks_like_company(s):
                    item["company"] = s
            if item not in out:
                out.append(item)

    return out


# ============================================================
# Aggregate parsers
# ============================================================


def parse_basics(resume_text: str) -> Dict[str, Optional[str]]:
    return {
        "name": heuristic_name(resume_text),
        "email": find_email(resume_text),
        "phone": find_phone(resume_text),
    }


def parse_resume(resume_text: str) -> Dict:
    """
    Comprehensive parse for UI / JSON export.
    Note: sets are converted to sorted lists for JSON-friendliness.
    """
    basics = parse_basics(resume_text)
    links = extract_links(resume_text)
    location = extract_location(resume_text)
    summary = extract_summary(resume_text)
    skills_flat = sorted(extract_skills(resume_text, BASIC_SKILLS))
    skills_cat_raw = extract_skills_categorized(resume_text)
    skills_cat = {k: sorted(v) for k, v in skills_cat_raw.items()}
    education = extract_education(resume_text)
    experience = extract_experience(resume_text)

    return {
        "basics": basics,
        "links": links,
        "location": location,
        "summary": summary,
        "skills": skills_flat,
        "skills_categorized": skills_cat,
        "education": education,
        "experience": experience,
    }


# ============================================================
# JD helpers
# ============================================================


def extract_jd_title(jd_text: str) -> Optional[str]:
    """
    Heuristic: first non-empty line that doesn't look like a section header.
    """
    headers = {
        "about",
        "summary",
        "responsibilities",
        "requirements",
        "role",
        "what you will do",
    }
    for ln in (x.strip() for x in jd_text.splitlines()):
        if not ln:
            continue
        if normalize(ln) in headers or len(ln) > 120:
            continue
        # Favor lines with title keywords
        if any(
            k in ln.lower()
            for k in (
                "data",
                "engineer",
                "scientist",
                "analyst",
                "ml",
                "ai",
                "developer",
            )
        ):
            return ln
        # Otherwise take the first plausible option
        if 2 <= len(ln.split()) <= 8:
            return ln
    return None


def extract_bullets(text: str) -> List[str]:
    """
    Pull bullet-like lines (•, -, *) or lines under 'Responsibilities/Requirements'.
    """
    lines = text.splitlines()
    out: List[str] = []
    capture = False
    section_keys = {
        "responsibilities",
        "requirements",
        "what you will do",
        "what you'll do",
        "about you",
    }
    for ln in lines:
        low = ln.strip().lower()
        if any(k in low for k in section_keys):
            capture = True
            continue
        if capture and not ln.strip():
            capture = False
        if capture:
            out.append(ln.strip())
        elif ln.strip().startswith(("-", "•", "*")):
            out.append(ln.strip(" •*\t"))
    # Deduplicate while preserving order
    uniq = []
    seen = set()
    for b in out:
        bb = normalize(b)
        if bb and bb not in seen:
            uniq.append(b)
            seen.add(bb)
    return uniq[:50]


def extract_jd_skills(jd_text: str) -> Set[str]:
    """
    Skills present in JD using same vocabulary as resumes.
    """
    return extract_skills(jd_text, BASIC_SKILLS)


SENIORITY_HINTS = {
    "intern": {"intern", "trainee"},
    "graduate": {"graduate", "entry level", "junior"},
    "mid": {"mid", "intermediate"},
    "senior": {"senior", "lead"},
    "principal": {"principal", "staff"},
}


def extract_seniority(jd_text: str) -> Optional[str]:
    low = jd_text.lower()
    for level, words in SENIORITY_HINTS.items():
        if any(w in low for w in words):
            return level
    return None


def parse_job_description(jd_text: str) -> Dict:
    """
    Compact JD struct you can pass to scoring/LLM/UI.
    Uses nlp_backends.extract_jd_keywords for robust JD keyword grouping.
    """
    title = extract_jd_title(jd_text)
    bullets = extract_bullets(jd_text)
    seniority = extract_seniority(jd_text)

    # Resume-style quick skill scan (BASIC_SKILLS vocab) — useful for baseline UI
    skills_quick = sorted(extract_jd_skills(jd_text))
    skills_cat_quick = {k: sorted(v) for k, v in extract_skills_categorized(jd_text).items()}

    # Canonical JD keyword groups (LLM/embedding-powered in nlp_backends)
    jd_keywords = extract_jd_keywords(jd_text)

    # A flat union (tools+methods) is handy for some UIs
    jd_flat = sorted(set(jd_keywords.get("tools", [])) | set(jd_keywords.get("methods", [])))

    return {
        "title": title,
        "seniority": seniority,
        "skills": skills_quick,  # baseline vocab hits
        "skills_categorized": skills_cat_quick,  # baseline categories
        "jd_keywords": jd_keywords,  # robust grouped keywords
        "jd_keywords_flat": jd_flat,  # tools ∪ methods
        "bullets": bullets,
        "text": jd_text,
    }
