from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .nlp_backends import _embed
from .parse import normalize


@dataclass
class RankedBullet:
    text: str
    score: float  # cosine similarity, bounded in [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "score": float(self.score)}


def _split_resume_into_bullets(resume_text: str) -> List[str]:
    """
    Heuristic splitter: turn raw resume text into bullet-like lines.

    - splits on newlines and inline "•"
    - strips bullet markers
    - keeps lines of reasonable length
    - de-duplicates by normalised text
    """
    raw: List[str] = []

    for line in resume_text.splitlines():
        ln = line.strip()
        if not ln:
            continue

        # Split lines that contain multiple bullet separators, e.g., "• foo • bar"
        if "•" in ln[1:]:
            parts = [p.strip(" •\t-–*") for p in ln.split("•")]
            for p in parts:
                if p:
                    raw.append(p)
        else:
            # Standard single bullet or sentence line
            raw.append(ln.lstrip("• \t-–*"))

    # Drop lines that are too short or too long to be useful
    candidates = [r for r in raw if 20 <= len(r) <= 350]

    # Remove duplicates while preserving order
    seen: set[str] = set()
    bullets: List[str] = []
    for b in candidates:
        nb = normalize(b)
        if not nb or nb in seen:
            continue
        seen.add(nb)
        bullets.append(b)

    return bullets


def compute_evidence_ranking(
    *,
    jd_text: str,
    resume_text: str,
    top_k: int = 5,
    bottom_k: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Rank resume bullets by semantic similarity to the JD.

    Returns a dict with:
      - all:   all bullets sorted by similarity (desc)
      - top:   top_k strongest bullets
      - bottom: bottom_k weakest bullets
    """
    jd_text = (jd_text or "").strip()
    if not jd_text:
        return {"all": [], "top": [], "bottom": []}

    bullets = _split_resume_into_bullets(resume_text or "")
    if not bullets:
        return {"all": [], "top": [], "bottom": []}

    try:
        # Treat first embedding as the JD; remaining vectors are bullets
        embs = _embed([jd_text, *bullets])
    except Exception:
        # On embedding failure, return zero scores without raising
        return {
            "all": [{"text": b, "score": 0.0} for b in bullets],
            "top": [],
            "bottom": [],
        }

    arr = np.asarray(embs, dtype="float32")
    if arr.ndim != 2 or arr.shape[0] != len(bullets) + 1:
        return {
            "all": [{"text": b, "score": 0.0} for b in bullets],
            "top": [],
            "bottom": [],
        }

    jd_vec = arr[0:1, :]
    bullet_vecs = arr[1:, :]

    # Compute cosine similarity
    jd_norm = jd_vec / (np.linalg.norm(jd_vec, axis=1, keepdims=True) + 1e-8)
    b_norm = bullet_vecs / (np.linalg.norm(bullet_vecs, axis=1, keepdims=True) + 1e-8)
    sims = (jd_norm @ b_norm.T)[0]  # similarity per bullet
    sims = np.clip(sims, 0.0, 1.0)

    ranked: List[RankedBullet] = [
        RankedBullet(text=b, score=float(s)) for b, s in zip(bullets, sims, strict=False)
    ]
    ranked.sort(key=lambda rb: rb.score, reverse=True)

    top_k = max(0, min(top_k, len(ranked)))
    bottom_k = max(0, min(bottom_k, len(ranked)))

    top = ranked[:top_k]
    bottom = ranked[-bottom_k:] if bottom_k else []

    return {
        "all": [rb.to_dict() for rb in ranked],
        "top": [rb.to_dict() for rb in top],
        "bottom": [rb.to_dict() for rb in bottom],
    }
