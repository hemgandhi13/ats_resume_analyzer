# NLP backends: config-driven keyword and embedding utilities
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Optional dependencies (all fail-safe)
try:
    import litellm  # type: ignore
except Exception:
    litellm = None  # type: ignore

try:
    import numpy as np  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore

try:
    import spacy  # type: ignore
except Exception:
    spacy = None  # type: ignore

try:
    from nltk.stem import WordNetLemmatizer  # type: ignore
except Exception:
    WordNetLemmatizer = None  # type: ignore

try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

from pydantic import BaseModel, ValidationError  # type: ignore

from ats.pdf_utils import normalize, tokenize_words

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer
else:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None  # type: ignore
        np = None  # type: ignore

# Logging / environment
_LOG = logging.getLogger(__name__)
_LOG.setLevel(os.getenv("NLP_LOG_LEVEL", "WARNING").upper())

# Config / thresholds
_LLM_MODEL_DEFAULT = os.getenv(
    "NLP_LLM_MODEL", os.getenv("LLM_MODEL", "anthropic/claude-3-haiku-20240307")
)
print(f"🔍 DEBUG: Model being used: {_LLM_MODEL_DEFAULT}")
_LLM_MAX_TOKENS = int(os.getenv("NLP_LLM_MAX_TOKENS", "1200"))
_CHUNK_SIZE = int(os.getenv("NLP_CHUNK_SIZE", "3500"))  # characters
_CHUNK_OVERLAP = int(os.getenv("NLP_CHUNK_OVERLAP", "200"))  # characters
_SNAP_THRESHOLD = float(os.getenv("SNAP_THRESHOLD", "0.75"))

_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# External config directory
_CFG_DIR = Path(os.getenv("ATS_CONFIG_DIR", "config"))
_CFG_STOPWORDS = _CFG_DIR / "stopwords.txt"
_CFG_SYNONYMS = _CFG_DIR / "synonyms.json"
_CFG_PHRASES = _CFG_DIR / "phrases.txt"
_CFG_CANON_SKILLS = _CFG_DIR / "canonical_skills.txt"

# Redis-backed cache (optional)
_REDIS_URL = os.getenv("REDIS_URL", "")
_CACHE_TTL = int(os.getenv("NLP_CACHE_TTL_SECONDS", "86400"))  # 1 day


# Pydantic schema
class JDKeywordsModel(BaseModel):
    tools_technologies: List[str] = []
    methods_capabilities: List[str] = []
    competencies: List[str] = []
    industry_domain: List[str] = []


# Config loading (no hard-coded lists)
class _Config:
    def __init__(self):
        self.stopwords: List[str] = self._read_lines(_CFG_STOPWORDS)
        self.phrases: List[str] = self._read_lines(_CFG_PHRASES)
        self.canonical_skills: List[str] = self._read_lines(_CFG_CANON_SKILLS)
        self.synonyms: Dict[str, str] = self._read_json(_CFG_SYNONYMS)

        # Normalize configuration inputs
        self.stopwords = [normalize(x) for x in self.stopwords if x.strip()]
        self.phrases = [normalize(x) for x in self.phrases if x.strip()]
        self.canonical_skills = [normalize(x) for x in self.canonical_skills if x.strip()]
        self.synonyms = {normalize(k): normalize(v) for k, v in self.synonyms.items() if k and v}

        self.version = self._fingerprint()

    def _read_lines(self, p: Path) -> List[str]:
        try:
            if p.exists():
                return [
                    ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()
                ]
        except Exception as e:
            _LOG.warning("Config read failed for %s: %s", p, e)
        return []

    def _read_json(self, p: Path) -> Dict[str, str]:
        try:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            _LOG.warning("Config JSON read failed for %s: %s", p, e)
        return {}

    def _fingerprint(self) -> str:
        s = json.dumps(
            {
                "stop": self.stopwords[:1000],
                "phr": self.phrases[:1000],
                "canon": self.canonical_skills[:10000],
                "syn": self.synonyms,
            },
            sort_keys=True,
        )
        return hashlib.sha1(s.encode("utf-8")).hexdigest()


_CFG = _Config()


# Lemmatization (spaCy → NLTK → fallback)
class _Lemmatizer:
    def __init__(self):
        self._spacy_nlp = None
        if spacy is not None:
            try:
                # Use env override if a larger spaCy model is configured
                model = os.getenv("SPACY_MODEL", "en_core_web_sm")
                self._spacy_nlp = spacy.load(model, disable=["ner", "parser", "textcat"])
            except Exception as e:
                _LOG.info("spaCy load failed (%s) – will try NLTK.", e)
                self._spacy_nlp = None
        self._wnl = WordNetLemmatizer() if WordNetLemmatizer is not None else None

    def lemma(self, s: str) -> str:
        t = normalize(s)
        if not t:
            return t
        # Try spaCy
        if self._spacy_nlp is not None:
            try:
                doc = self._spacy_nlp(t)
                return " ".join([w.lemma_.lower() for w in doc])
            except Exception:
                pass
        # Fallback to NLTK WordNet (token-wise)
        if self._wnl is not None:
            toks = tokenize_words(t)
            lem = [self._wnl.lemmatize(w) for w in toks]
            return " ".join(lem)
        # Last resort: identity
        return t


_LEM = _Lemmatizer()


def _canon(term: str) -> str:
    t = normalize(term)
    t = _LEM.lemma(t)
    # Synonyms from external file
    return _CFG.synonyms.get(t, t)


# Embeddings (thread-safe)
_EMB_LOCK = threading.Lock()
_EMB_MODEL: Optional[SentenceTransformer] = None


def _get_emb_model() -> Optional[SentenceTransformer]:
    global _EMB_MODEL
    if SentenceTransformer is None:
        return None
    if _EMB_MODEL is None:
        with _EMB_LOCK:
            if _EMB_MODEL is None:
                _EMB_MODEL = SentenceTransformer(_EMBEDDING_MODEL)
    return _EMB_MODEL


def _embed(texts: List[str]) -> Optional["np.ndarray"]:
    m = _get_emb_model()
    if m is None or np is None or not texts:
        return None
    try:
        vecs = m.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=float)
    except Exception as e:
        _LOG.warning("Embedding encode failed: %s", e)
        return None


def _snap_to_canonical(cands: Iterable[str], threshold: float = _SNAP_THRESHOLD) -> List[str]:
    cands = [normalize(x) for x in cands if x]
    canon = [normalize(x) for x in _CFG.canonical_skills]  # external list
    if not cands or not canon or np is None:
        return _dedupe_keep_order(cands)
    E_cand = _embed(cands)
    E_canon = _embed(canon)
    if E_cand is None or E_canon is None:
        return _dedupe_keep_order(cands)
    out = []
    for i, v in enumerate(E_cand):
        sims = E_canon @ v
        j = int(sims.argmax())
        if float(sims[j]) >= threshold:
            out.append(canon[j])
        else:
            out.append(cands[i])
    return _dedupe_keep_order(out)


# Caching (Redis or in-process LRU)
class _Cache:
    def __init__(self):
        self.r = None
        if _REDIS_URL and redis is not None:
            try:
                self.r = redis.from_url(_REDIS_URL)
            except Exception as e:
                _LOG.warning("Redis connect failed (%s). Falling back to memory cache.", e)
                self.r = None
        self.mem: Dict[str, Tuple[float, str]] = {}

    def get(self, key: str) -> Optional[Dict]:
        try:
            if self.r is not None:
                s = self.r.get(key)
                return json.loads(s) if s else None
            # Enforce in-memory TTL
            item = self.mem.get(key)
            if not item:
                return None
            ts, val = item
            if time.time() - ts > _CACHE_TTL:
                self.mem.pop(key, None)
                return None
            return json.loads(val)
        except Exception:
            return None

    def set(self, key: str, value: Dict):
        s = json.dumps(value)
        try:
            if self.r is not None:
                self.r.setex(key, _CACHE_TTL, s)
            else:
                self.mem[key] = (time.time(), s)
        except Exception:
            pass


_CACHE = _Cache()


def _hash_key(*parts: str) -> str:
    m = hashlib.sha1()
    for p in parts:
        m.update((p or "").encode("utf-8"))
    return "jdkw:" + m.hexdigest()


# Utilities
def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        n = normalize(x)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _clean_terms(xs: Iterable[str], stopwords: Iterable[str], max_words: int = 6) -> List[str]:
    # Allow phrases up to max_words (configurable via env)
    limit = int(os.getenv("MAX_KEYWORD_WORDS", str(max_words)))
    stop = set(normalize(s) for s in stopwords)
    out = []
    for x in xs:
        n = _canon(x)
        if not n or n in stop:
            continue
        if len(n.split()) > limit:
            continue
        out.append(n)
    return _dedupe_keep_order(out)


def _chunk_text(s: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    s = s or ""
    if len(s) <= size:
        return [s]
    chunks, i = [], 0
    while i < len(s):
        j = min(len(s), i + size)
        chunk = s[i:j]
        chunks.append(chunk)
        if j == len(s):
            break
        i = j - overlap if j - overlap > i else j
    return chunks


# LLM extraction (JSON-safe)
_LLM_SYS = (
    "You extract ATS-relevant keywords from job descriptions. "
    "Return STRICT JSON with keys: tools_technologies, methods_capabilities, "
    "competencies, industry_domain. Each value is an array of short, lowercase phrases "
    "(<=6 words). No filler. Prefer concrete tools (databricks, power bi, sql, python), "
    "methods (etl, data migration, validation), competencies (stakeholder communication), and industry terms."
)

_LLM_USER_TMPL = """Job Description:
{jd}

Return JSON ONLY:
{{
  "tools_technologies": [],
  "methods_capabilities": [],
  "competencies": [],
  "industry_domain": []
}}"""


def _llm_extract_chunk(model: str, jd_chunk: str) -> Optional[JDKeywordsModel]:
    if litellm is None:
        return None
    msgs = [
        {"role": "system", "content": _LLM_SYS},
        {"role": "user", "content": _LLM_USER_TMPL.format(jd=jd_chunk)},
    ]
    kwargs = {"temperature": 0, "max_tokens": _LLM_MAX_TOKENS}

    # Try JSON mode where supported (OpenAI/compatible)
    try:
        resp = litellm.completion(
            model=model,
            messages=msgs,
            response_format={"type": "json_object"},  # ignored by providers that don't support it
            **kwargs,  # type: ignore
        )
        content = resp["choices"][0]["message"]["content"]  # type: ignore
    except Exception as e:
        _LOG.info("JSON mode failed (%s); retrying without.", e)
        # Fallback without response_format
        resp = litellm.completion(model=model, messages=msgs, **kwargs)  # type: ignore
        content = resp["choices"][0]["message"]["content"]  # type: ignore

    try:
        s = content.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s).strip()
        data = JDKeywordsModel.model_validate_json(s)
        return data
    except (ValidationError, json.JSONDecodeError) as e:
        _LOG.warning("LLM JSON parse failed: %s; content (truncated): %s", e, content[:240])
        return None


def _merge_models(models: List[JDKeywordsModel]) -> JDKeywordsModel:
    tools, methods, comps, industry = [], [], [], []
    for m in models:
        tools.extend(m.tools_technologies)
        methods.extend(m.methods_capabilities)
        comps.extend(m.competencies)
        industry.extend(m.industry_domain)
    return JDKeywordsModel(
        tools_technologies=_dedupe_keep_order(tools),
        methods_capabilities=_dedupe_keep_order(methods),
        competencies=_dedupe_keep_order(comps),
        industry_domain=_dedupe_keep_order(industry),
    )


def _llm_extract(model: str, jd_text: str) -> Optional[JDKeywordsModel]:
    chunks = _chunk_text(jd_text, _CHUNK_SIZE, _CHUNK_OVERLAP)
    pieces: List[JDKeywordsModel] = []
    for ch in chunks:
        m = _llm_extract_chunk(model, ch)
        if m is not None:
            pieces.append(m)
    if not pieces:
        return None
    return _merge_models(pieces)


# Heuristic fallback (structure-only, no hard-coded knowledge)
def _heuristic_extract(jd_text: str) -> JDKeywordsModel:
    # Lightweight fallback: nouns/phrases via simple regex and token heuristics
    low = jd_text.lower()
    toks = tokenize_words(low)

    # Keep phrases from external config only (no hard-coded phrases)
    phrases = [p for p in _CFG.phrases if p in low]

    # Capture noun-like tokens and tool-like strings
    rough = set()
    for t in toks:
        if len(t) <= 2:
            continue
        if re.search(r"[a-z]+[0-9]+|[0-9]+[a-z]+", t):  # e.g., s3, ec2
            rough.add(t)
        if t.isalpha():
            rough.add(t)

    # Put everything in tools by default; cleaned later
    tools = list(rough) + phrases
    return JDKeywordsModel(
        tools_technologies=_dedupe_keep_order(tools),
        methods_capabilities=[],
        competencies=[],
        industry_domain=[],
    )


# Public API
def extract_jd_keywords(jd_text: str) -> Dict[str, List[str]]:
    """
    Robust JD keyword extraction with:
      - Caching (Redis/in-memory)
      - LLM JSON-mode + smart chunking (optional)
      - No hard-coded lists (configs drive stopwords/synonyms/phrases/canon skills)
      - Lemmatization (spaCy/NLTK) + embedding snapping
    Returns dict with keys: tools, methods, competencies, industry
    """
    model = _LLM_MODEL_DEFAULT
    prefer = os.getenv("NLP_EXTRACTOR", "auto").lower()

    cache_key = _hash_key(
        jd_text, model if prefer in {"llm", "auto"} else "heuristic", _CFG.version
    )
    cached = _CACHE.get(cache_key)
    if cached:
        return cached

    data: Optional[JDKeywordsModel] = None
    if prefer in {"llm", "auto"} and litellm is not None:
        data = _llm_extract(model=model, jd_text=jd_text)

    if data is None:
        data = _heuristic_extract(jd_text)

    # Clean and canonicalize
    tools = _clean_terms(
        data.tools_technologies,
        _CFG.stopwords,
        max_words=int(os.getenv("MAX_KEYWORD_WORDS", "6")),
    )
    methods = _clean_terms(
        data.methods_capabilities,
        _CFG.stopwords,
        max_words=int(os.getenv("MAX_KEYWORD_WORDS", "6")),
    )
    comps = _clean_terms(
        data.competencies,
        _CFG.stopwords,
        max_words=int(os.getenv("MAX_KEYWORD_WORDS", "6")),
    )
    industry = _clean_terms(
        data.industry_domain,
        _CFG.stopwords,
        max_words=int(os.getenv("MAX_KEYWORD_WORDS", "6")),
    )

    # Snap to canonical (externally defined)
    tools = _snap_to_canonical(tools, threshold=_SNAP_THRESHOLD)
    methods = _snap_to_canonical(methods, threshold=_SNAP_THRESHOLD)

    result = {
        "tools": tools,
        "methods": methods,
        "competencies": comps,
        "industry": industry,
    }
    _CACHE.set(cache_key, result)
    return result


def extract_jd_keywords_batch(jds: List[str]) -> List[Dict[str, List[str]]]:
    """
    Efficient batch; still cached per JD using the same extractor.
    """
    out = []
    for jd in jds:
        out.append(extract_jd_keywords(jd))
    return out


def present_missing_vs_resume(
    jd_kw: Dict[str, List[str]], resume_text: str
) -> Tuple[List[str], List[str]]:
    """
    Intersect JD keywords with resume content using normalized text; keeps phrases as substrings.
    """
    norm = normalize(resume_text)
    tokset = set(tokenize_words(norm))

    def _has(term: str) -> bool:
        t = normalize(term)
        if " " in t:
            return t in norm
        return t in tokset

    jd_all = _dedupe_keep_order(
        [
            *jd_kw.get("tools", []),
            *jd_kw.get("methods", []),
            *jd_kw.get("competencies", []),
            *jd_kw.get("industry", []),
        ]
    )
    present: List[str] = []
    missing: List[str] = []
    for t in jd_all:
        (present if _has(t) else missing).append(t)
    return present, missing
