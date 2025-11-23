# src/ats/llm.py
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Sequence, Set

# Config-driven NLP backend (no hard-coded lists)
from .nlp_backends import _CFG, _canon

# Keep lightweight parsing utilities
from .parse import extract_skills
from .pdf_utils import normalize

# Load .env early (optional)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# LiteLLM is optional (lets you swap Claude / GPT / Gemini by model name)
try:
    import litellm  # type: ignore
except Exception:  # pragma: no cover
    litellm = None  # type: ignore


_MODEL_ALIASES = {
    "claude-3-5-sonnet-latest": "anthropic/claude-3-5-sonnet-20241022",
    "claude-3-sonnet-latest": "anthropic/claude-3-sonnet-20240229",
    "claude-3-haiku-latest": "anthropic/claude-3-haiku-20240307",
}
_FALLBACK_ANTHROPIC = [
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-5-haiku-20241022",
    "anthropic/claude-3-sonnet-20240229",
    "anthropic/claude-3-haiku-20240307",
]


def _resolve_model(name: str) -> str:
    return _MODEL_ALIASES.get(name, name)


@dataclass
class Suggestions:
    missing_keywords: List[str]
    tailored_bullets: List[str]
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["missing_keywords"] = sorted(d["missing_keywords"])
        d["tailored_bullets"] = [b.strip() for b in d["tailored_bullets"] if b.strip()]
        return d


def _json_only(s: str) -> dict:
    """
    Try to parse JSON; if the model returns prose + JSON, extract the JSON block.
    """
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def _have_any_llm_key() -> bool:
    return any(
        os.environ.get(k)
        for k in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "AZURE_OPENAI_API_KEY",
        )
    )


def _fallback_bullets(
    jd_text: str,
    resume_text: str,
    missing: Sequence[str],
    present: Sequence[str],
    max_bullets: int = 5,
) -> List[str]:
    """
    Deterministic, no-LLM bullet generator (keeps demo working without keys).
    """
    missing_norm = [m for m in (normalize(x) for x in missing) if m]
    present_norm = [p for p in (normalize(x) for x in present) if p]

    templates = [
        "Implemented {kw} workflows to automate repetitive tasks, cutting manual effort by ~30%.",
        "Built analytics in {kw} and documented a reusable pattern to speed up similar projects.",
        "Integrated {kw} into the existing stack (leveraging {present}) to improve reliability and visibility.",
        "Created dashboards/alerts around {kw} to surface KPIs and reduce time-to-insight.",
        "Up-skilled teammates with short how-to notes on {kw}, raising adoption across the team.",
    ]

    bullets: List[str] = []
    for i, kw in enumerate(missing_norm[:max_bullets]):
        t = templates[i % len(templates)]
        bullets.append(
            t.format(kw=_canon(kw), present=", ".join(present_norm[:3]) or "current tools")
        )

    if not bullets:
        generic = [
            "Quantified impact (time saved, defects reduced); added metrics directly into resume bullets.",
            "Refactored code for readability and tests; introduced pre-commit (ruff/black) to keep quality high.",
            "Shipped a small Streamlit demo to showcase results with downloads (CSV/JSON) for stakeholders.",
            "Used TF-IDF similarity to align achievements with JD keywords; addressed gaps explicitly.",
            "Wrote clear runbooks so others could reproduce results end-to-end.",
        ]
        bullets.extend(generic[:max_bullets])

    return bullets


def _combine_vocab(vocab: Optional[Iterable[str]]) -> Set[str]:
    """
    Combine caller-provided vocab with config canonical skills.
    Everything normalized; empty-safe.
    """
    cfg = set(_CFG.canonical_skills or [])
    user = set(vocab or [])
    return {normalize(x) for x in (cfg | user) if x}


# Minimal extra soft terms you explicitly called out; bulk comes from _CFG.stopwords
_EXTRA_EXCLUDE: Set[str] = {
    "must",
    "required",
    "requirement",
    "preferred",
    "nice to have",
}


def _clean_terms(seq: Sequence[str]) -> List[str]:
    """
    Canonicalise via synonyms map, drop stopwords/junk from config (+ tiny extra),
    and dedupe while preserving order.
    """
    stop = set(_CFG.stopwords or []) | _EXTRA_EXCLUDE
    out: List[str] = []
    seen: Set[str] = set()
    for t in seq or []:
        nt = _canon(normalize(t))
        if not nt or nt in stop:
            continue
        if nt in seen:
            continue
        seen.add(nt)
        out.append(nt)
    return out


def suggest_resume_improvements(
    *,
    jd_text: str,
    resume_text: str,
    missing_keywords: Optional[Sequence[str]] = None,
    present_keywords: Optional[Sequence[str]] = None,
    vocab: Optional[Sequence[str]] = None,
    model: Optional[str] = None,
    enable_llm: Optional[bool] = None,
    max_bullets: int = 5,
) -> Suggestions:
    """
    Produce missing keywords + 3–5 tailored resume bullets.

    Alignment:
      - Stopwords come from _CFG.stopwords (centralised) + a tiny extra soft set.
      - Canonicalisation via _canon (synonyms -> canonical terms).
      - Vocab = _CFG.canonical_skills ∪ (vocab arg).
      - Multi-model fallback via LiteLLM; deterministic fallback if disabled/unavailable.
    """
    # ---------- derive & clean inputs ----------
    vocab_set = _combine_vocab(vocab)

    if missing_keywords is None or present_keywords is None:
        jd_hits = extract_skills(jd_text, vocab_set)
        res_hits = extract_skills(resume_text, vocab_set)
        missing_set = jd_hits - res_hits
        present_set = jd_hits & res_hits
        missing_keywords = sorted(missing_set)
        present_keywords = sorted(present_set)

    missing_keywords = _clean_terms(missing_keywords or [])
    present_keywords = _clean_terms(present_keywords or [])

    # Decide whether to call an LLM
    if enable_llm is None:
        enable_llm = os.environ.get("LLM_ENABLE", "0").strip() not in {
            "",
            "0",
            "false",
            "False",
        }

    have_key = _have_any_llm_key()
    model_name = model or os.environ.get("LLM_MODEL") or "claude-3-haiku-latest"
    target = _resolve_model(model_name)

    # If LLM disabled/unavailable → deterministic fallback
    if not enable_llm or litellm is None or not have_key:
        return Suggestions(
            missing_keywords=list(missing_keywords),
            tailored_bullets=_fallback_bullets(
                jd_text, resume_text, missing_keywords, present_keywords, max_bullets
            ),
            notes="fallback",
        )

    # ---------- build strict prompt ----------
    sys_prompt = (
        "You are an expert resume editor. "
        "Return ONLY compact JSON. No markdown. No comments. Keys: "
        '{"missing_keywords": [], "tailored_bullets": [], "notes": ""}. '
        "Bullets must be first-person-omitted, action-led, quant-or-impact oriented, "
        "≤24 words each, no secrets, no hallucinations."
    )

    user_prompt = f"""
Resume (truncated OK):
{resume_text[:6000]}

Job Description (truncated OK):
{jd_text[:6000]}

Known present keywords: {list(present_keywords or [])}
Known missing keywords: {list(missing_keywords or [])}

TASK:
1) Confirm/adjust the missing_keywords (only terms truly present in the JD; lowercase; avoid soft words).
2) Write {max_bullets} tailored resume bullets integrating missing keywords naturally (no stuffing).
3) Add a 1-line 'notes' field with the single best advice (≤18 words).

OUTPUT STRICT JSON ONLY:
{{"missing_keywords": ["..."], "tailored_bullets": ["..."], "notes": "..."}}
""".strip()

    # ---------- try primary + Anthropic fallbacks ----------
    candidates = [target]
    if target.startswith("anthropic/"):
        candidates += [m for m in _FALLBACK_ANTHROPIC if m != target]

    last_err: Optional[Exception] = None

    for m in candidates:
        try:
            resp = litellm.completion(  # type: ignore
                model=m,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "500")),
            )
            content = resp["choices"][0]["message"]["content"]  # type: ignore
            data = _json_only(content)

            mk = data.get("missing_keywords") if isinstance(data, dict) else None
            tb = data.get("tailored_bullets") if isinstance(data, dict) else None
            notes = (
                data.get("notes") if isinstance(data, dict) else None
            ) or "Mirror the JD’s must-haves precisely."

            if not isinstance(mk, list) or not isinstance(tb, list):
                raise ValueError("LLM returned invalid schema")

            mk_clean = _clean_terms(mk)
            tb_clean = [t.strip() for t in tb if isinstance(t, str) and t.strip()][:max_bullets]

            if not tb_clean:
                tb_clean = _fallback_bullets(
                    jd_text,
                    resume_text,
                    missing_keywords,
                    present_keywords,
                    max_bullets,
                )

            return Suggestions(
                missing_keywords=list(sorted(set(mk_clean))),
                tailored_bullets=tb_clean,
                notes=notes,
            )

        except Exception as e:
            last_err = e
            continue

    # ---------- hard fallback if all models fail ----------
    return Suggestions(
        missing_keywords=list(missing_keywords),
        tailored_bullets=_fallback_bullets(
            jd_text, resume_text, missing_keywords, present_keywords, max_bullets
        ),
        notes=f"fallback ({type(last_err).__name__ if last_err else 'unknown'})",
    )
