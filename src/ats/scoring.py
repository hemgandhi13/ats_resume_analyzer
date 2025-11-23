from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Pull shared config + JD extractor + embedding from nlp_backends
# (_CFG exposes stopwords/synonyms loaded from your config files)
from ats.nlp_backends import _canon, _embed, extract_jd_keywords

from .parse import (
    BASIC_SKILLS,
    extract_skills,
    normalize,
    tokenize_words,
)


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class MatchDetail:
    skill: str
    matched_via: str  # "exact" | "fuzzy"
    context: str  # "resume"
    snippet: str  # short text where it was found (if available)
    weight: float  # weight used for coverage


@dataclass
class AdvancedScore:
    keyword_coverage: float  # unweighted, 0..1
    weighted_keyword_coverage: float  # 0..1
    tfidf_cosine: float  # 0..1
    semantic_cosine: Optional[float]  # 0..1 or None if unavailable
    recency_factor: float  # 0.7..1.0 small boost for recent exp
    composite: float  # 0..100

    # explainability
    present_keywords: Tuple[str, ...]
    missing_keywords: Tuple[str, ...]
    jd_skill_count: int
    resume_skill_count: int
    details: Tuple[MatchDetail, ...]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["keyword_coverage"] = round(self.keyword_coverage, 4)
        d["weighted_keyword_coverage"] = round(self.weighted_keyword_coverage, 4)
        d["tfidf_cosine"] = round(self.tfidf_cosine, 4)
        if self.semantic_cosine is not None:
            d["semantic_cosine"] = round(self.semantic_cosine, 4)
        d["recency_factor"] = round(self.recency_factor, 3)
        d["composite"] = round(float(self.composite), 1)
        # marshal MatchDetail dataclasses
        d["details"] = [asdict(x) for x in self.details]
        return d


# -----------------------------
# Helpers
# -----------------------------
_MUST_HINTS = {"must", "required", "mandatory", "min", "minimum", "at least"}
_NICE_HINTS = {"nice to have", "preferred", "bonus", "plus", "good to have"}


def _lines(text: str) -> List[str]:
    return [ln.rstrip() for ln in (text or "").splitlines()]


def _clip(s: str, n: int = 160) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return (s[: n - 1] + "…") if len(s) > n else s


# -----------------------------
# Similarities (TF-IDF / Embedding cosine)
# -----------------------------
def _tfidf_cosine(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(min_df=1, stop_words="english", ngram_range=(1, 2))
        X = vec.fit_transform([a, b])
        sim = cosine_similarity(X[0:1], X[1:2])[0, 0]
        if np.isnan(sim):
            return 0.0
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        try:
            vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
            X = vec.fit_transform([a, b])
            sim = cosine_similarity(X[0:1], X[1:2])[0, 0]
            if np.isnan(sim):
                return 0.0
            return float(max(0.0, min(1.0, sim)))
        except Exception:
            return 0.0


def _semantic_cosine(a: str, b: str) -> Optional[float]:
    """
    Semantic similarity using the shared _embed() from nlp_backends.
    Returns None if embeddings are unavailable or fail.
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return None
    try:
        embs = _embed([a, b])
        if not embs or len(embs) != 2:
            return None
        va = np.array(embs[0], dtype=float)
        vb = np.array(embs[1], dtype=float)
        num = float(np.dot(va, vb))
        den = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if den == 0.0:
            return 0.0
        sim = num / den  # -1..1
        return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))
    except Exception:
        return None


# -----------------------------
# Weighting / fuzzy matching / recency
# -----------------------------
def _mandatory_score_hint(line: str) -> float:
    low = line.lower()
    if any(h in low for h in _MUST_HINTS):
        return 2.0
    if any(h in low for h in _NICE_HINTS):
        return 0.6
    return 1.0


def _fuzzy_present(term: str, text: str, threshold: int = 86) -> Tuple[bool, str]:
    """
    Use rapidfuzz for approximate matching; returns (is_present, best_snippet).
    """
    t = normalize(term)
    best = 0
    best_line = ""
    for ln in _lines(text):
        score = fuzz.token_set_ratio(t, normalize(ln))
        if score > best:
            best = score
            best_line = ln
    return (best >= threshold, best_line)


def _experience_recency_years(
    experience: Optional[Sequence[Dict[str, str]]],
) -> Optional[int]:
    """
    Return years since most recent end year in parsed experience.
    - If any role is 'Present/Current' -> return 0
    - Otherwise find the latest 4-digit year (19xx/20xx) and compute recency.
    - If nothing usable is found -> None (neutral default used by _recency_factor)
    """
    if not experience:
        return None

    now_year = datetime.now().year
    best_end: Optional[int] = None

    for item in experience:
        if not isinstance(item, dict):
            continue
        dates = str(item.get("dates") or item.get("year") or "").strip()
        if not dates:
            continue

        if re.search(r"\b(present|current)\b", dates, re.IGNORECASE):
            return 0

        years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", dates)]
        if years:
            end_year = max(years)
            best_end = end_year if best_end is None else max(best_end, end_year)

    if best_end is None:
        return None

    return max(0, now_year - best_end)


def _recency_factor(experience: Optional[List[Dict[str, str]]]) -> float:
    """
    1.0 if currently active; otherwise decays mildly with years since latest end.
    """
    yrs = _experience_recency_years(experience or [])
    if yrs is None:
        return 0.9
    if yrs == 0:
        return 1.0
    if yrs <= 2:
        return 0.96
    if yrs <= 4:
        return 0.92
    if yrs <= 6:
        return 0.88
    return 0.84


# -----------------------------
# Public APIs
# -----------------------------
def compute_score(
    jd_text: str,
    resume_text: str,
    vocab: Set[str] = BASIC_SKILLS,
    *,
    jd_skills: Optional[Iterable[str]] = None,
    w_cov: float = 0.65,
    w_sim: float = 0.35,
):
    """
    Legacy simple scorer kept for backwards compatibility.
    """
    # Compute JD and resume hits
    jd_skills_set: Optional[Set[str]] = None
    if jd_skills is not None:
        jd_skills_set = {normalize(s) for s in jd_skills}
    jd_hits = jd_skills_set if jd_skills_set is not None else extract_skills(jd_text, vocab)
    res_hits = extract_skills(resume_text, vocab)

    # Coverage
    cov = (len(jd_hits & res_hits) / max(1, len(jd_hits))) if jd_hits else 0.0
    # TF-IDF
    sim = _tfidf_cosine(jd_text, resume_text)
    # Composite
    total_w = (w_cov or 0.0) + (w_sim or 0.0)
    if total_w <= 0:
        w_cov, w_sim = 0.65, 0.35
        total_w = 1.0
    w_cov, w_sim = w_cov / total_w, w_sim / total_w
    composite = (w_cov * cov + w_sim * sim) * 100.0

    @dataclass
    class _Simple:
        keyword_coverage: float
        tfidf_cosine: float
        composite: float
        missing_keywords: Tuple[str, ...]
        present_keywords: Tuple[str, ...]
        jd_skill_count: int
        resume_skill_count: int

        def to_dict(self):  # match earlier API
            return {
                "keyword_coverage": round(self.keyword_coverage, 4),
                "tfidf_cosine": round(self.tfidf_cosine, 4),
                "composite": round(self.composite, 1),
                "missing_keywords": self.missing_keywords,
                "present_keywords": self.present_keywords,
                "jd_skill_count": self.jd_skill_count,
                "resume_skill_count": self.resume_skill_count,
            }

    missing = tuple(sorted(jd_hits - res_hits))
    present = tuple(sorted(jd_hits & res_hits))
    return _Simple(
        keyword_coverage=float(cov),
        tfidf_cosine=float(sim),
        composite=float(round(composite, 1)),
        missing_keywords=missing,
        present_keywords=present,
        jd_skill_count=len(jd_hits),
        resume_skill_count=len(res_hits),
    )


def compute_score_advanced(
    *,
    jd_text: str,
    resume_text: str,
    resume_parsed: Optional[Dict] = None,  # output of parse_resume(...) if available
    base_vocab: Set[str] = BASIC_SKILLS,
    use_semantic: bool = True,
    fuzzy_threshold: int = 86,
    weights: Dict[str, float] | None = None,
) -> AdvancedScore:
    """
    Advanced scoring with:
      - single-source JD keyword groups (nlp_backends.extract_jd_keywords)
      - weighted keyword coverage (must-have > nice-to-have + frequency)
      - fuzzy matching (rapidfuzz)
      - optional semantic similarity (embeddings via _embed)
      - small recency factor from parsed experience
    """
    W = {"coverage": 0.60, "tfidf": 0.25, "semantic": 0.15}
    if weights:
        W.update(weights)

    # 1) Allowed vocabulary for THIS JD:
    jd_kw = extract_jd_keywords(
        jd_text
    )  # {"tools": [], "methods": [], "competencies": [], "industry": []}
    allowed: Set[str] = {
        *(normalize(x) for x in base_vocab),
        *(normalize(x) for x in jd_kw.get("tools", [])),
        *(normalize(x) for x in jd_kw.get("methods", [])),
        *(normalize(x) for x in jd_kw.get("competencies", [])),
        *(normalize(x) for x in jd_kw.get("industry", [])),
    }

    # Synonyms from config (external JSON), optional

    # 2) Determine JD skills set (canonicalised)
    jd_hits_raw = extract_skills(jd_text, allowed)

    # map synonyms; stable-dedup
    def _canonicalize_list(terms: Iterable[str]) -> List[str]:
        out, seen = [], set()
        for t in terms:
            n = _canon(t)  # Handles normalize + lemma + synonym
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    jd_hits: Set[str] = set(_canonicalize_list(jd_hits_raw))

    # 3) Scan resume for hits (exact + fuzzy) against canonical JD skills
    present: Set[str] = set()
    details: List[MatchDetail] = []
    norm_resume = normalize(resume_text)

    for sk in sorted(jd_hits):
        # exact token / substring
        exact = False
        if " " in sk:
            if sk in norm_resume:
                exact = True
        else:
            if sk in set(tokenize_words(norm_resume)):
                exact = True

        if exact:
            present.add(sk)
            details.append(
                MatchDetail(
                    skill=sk,
                    matched_via="exact",
                    context="resume",
                    snippet="",
                    weight=1.0,
                )
            )
            continue

        # fuzzy fallback
        ok, snippet = _fuzzy_present(sk, resume_text, threshold=fuzzy_threshold)
        if ok:
            present.add(sk)
            details.append(
                MatchDetail(
                    skill=sk,
                    matched_via="fuzzy",
                    context="resume",
                    snippet=_clip(snippet),
                    weight=1.0,
                )
            )

    # 4) Weighted coverage (must-have vs nice-to-have + frequency)
    jd_lines = _lines(jd_text)
    term_weights: Dict[str, float] = {}
    base_weight = 1.0
    for term in jd_hits:
        w = base_weight
        occurrences = 0
        for ln in jd_lines:
            low = ln.lower()
            # quick presence heuristic
            if (term in normalize(low)) or (
                fuzz.token_set_ratio(term, normalize(low)) >= fuzzy_threshold
            ):
                occurrences += 1
                w *= _mandatory_score_hint(low)
        # frequency bonus (capped)
        w += min(0.5, max(0, occurrences - 1) * 0.15)
        term_weights[term] = max(0.4, min(3.0, w))  # clamp

    total_weight = sum(term_weights.values()) or 1.0
    covered_weight = sum(term_weights[t] for t in present)
    weighted_cov = covered_weight / total_weight

    # 5) Unweighted coverage
    cov = (len(present) / max(1, len(jd_hits))) if jd_hits else 0.0

    # 6) Similarities
    tfidf = _tfidf_cosine(jd_text, resume_text)
    sem = _semantic_cosine(jd_text, resume_text) if use_semantic else None

    # Re-normalise contributions if semantic not available
    c_cov, c_tfidf, c_sem = W["coverage"], W["tfidf"], W["semantic"]
    if sem is None:
        s = c_cov + c_tfidf
        c_cov, c_tfidf = c_cov / s, c_tfidf / s
        c_sem = 0.0

    # 7) Recency factor (small multiplier)
    recency = _recency_factor((resume_parsed or {}).get("experience"))

    # 8) Composite score
    composite = c_cov * weighted_cov + c_tfidf * tfidf + c_sem * (sem or 0.0)
    composite = composite * recency * 100.0

    missing = tuple(sorted(jd_hits - present))
    present_sorted = tuple(sorted(present))

    # attach weights/snippets to details
    final_details: List[MatchDetail] = []
    for d in details:
        w = term_weights.get(d.skill, 1.0)
        final_details.append(
            MatchDetail(
                skill=d.skill,
                matched_via=d.matched_via,
                context=d.context,
                snippet=d.snippet,
                weight=float(round(w, 3)),
            )
        )

    return AdvancedScore(
        keyword_coverage=float(cov),
        weighted_keyword_coverage=float(weighted_cov),
        tfidf_cosine=float(tfidf),
        semantic_cosine=None if sem is None else float(sem),
        recency_factor=float(round(recency, 3)),
        composite=float(round(composite, 1)),
        present_keywords=present_sorted,
        missing_keywords=missing,
        jd_skill_count=len(jd_hits),
        resume_skill_count=len(extract_skills(resume_text, allowed)),
        details=tuple(final_details),
    )
