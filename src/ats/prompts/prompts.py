# Prompt builders for ATS Resume Analyzer

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def _join_csv(items: Iterable[str]) -> str:
    items = [x.strip() for x in (items or []) if x and str(x).strip()]
    return ", ".join(items)


def _section(title: str, body: str) -> str:
    return f"{title}\n{body}\n"


def build_make_edits_prompt(
    *,
    jd_text: str,  # kept for compatibility; inlined only when include_raw=True
    present_keywords: Iterable[str],
    missing_keywords: Iterable[str],
    tailored_bullets: Optional[Iterable[str]] = None,
    resume_text: Optional[str] = None,  # used only when include_raw=True
    jd_keywords_categorized: Optional[Dict[str, List[str]]] = None,
    industry_hint: Optional[str] = None,
    role_title: Optional[str] = None,
    deadline: Optional[str] = None,
    role_type: Optional[str] = None,
    seniority: Optional[str] = None,
    focus: Optional[str] = None,
    extras: Optional[str] = None,
    include_raw: bool = False,  # default avoids inlining raw resume/JD
) -> str:
    """
    Build a clean 'Make Edits' prompt without embedding raw resume/JD content.
    The LLM should have the resume attached separately by the user in their chat UI.
    Set include_raw=True ONLY if you explicitly want to paste raw texts (not recommended).
    """

    tools = (jd_keywords_categorized or {}).get("tools", [])
    methods = (jd_keywords_categorized or {}).get("methods", [])
    comps = (jd_keywords_categorized or {}).get("competencies", [])
    industry = (jd_keywords_categorized or {}).get("industry", [])

    present_csv = _join_csv(present_keywords)
    missing_csv = _join_csv(missing_keywords)
    tools_csv = _join_csv(tools)
    methods_csv = _join_csv(methods)
    comps_csv = _join_csv(comps)
    industry_csv = _join_csv(industry)

    bullets = [b.strip() for b in (tailored_bullets or []) if b and str(b).strip()]
    bullets_block = (
        "\n".join(f"• {b}" for b in bullets) if bullets else "• (model to propose 4–6 bullets)"
    )

    header = (
        "You are my Job Application Co-Pilot with 40+ years’ experience in Australian Data/AI/Analytics hiring.\n"
        "Your job: transform my master résumé + content library into a JD-matched, ATS-clean, paste-ready two-page résumé plus a cover letter."
    )

    rules = (
        "Operating Rules\n"
        "• Timezone: Australia/Melbourne. Spelling: British English. Use en dashes (–) for date ranges.\n"
        "• Output format: Word-friendly plain text (headings + bullets). No tables, text boxes, images, or emojis.\n"
        "• Privacy/NDA: Use only what I share or what’s already public; no fabrication.\n"
        "• Tailoring knobs you may change: summary wording, bullet phrasing, section ordering/emphasis to mirror the JD. Do not invent tools I haven’t used.\n"
        "• Quant: Prefer numbers (%, x/y, time/cost deltas, throughput, P95 latencies)."
    )

    context = (
        "My Constants\n"
        "• Location: Melbourne; open Australia-wide; hybrid/onsite/remote.\n"
        "• Work rights: See résumé attachment.\n"
        "• Salary: A$100k+ (adjust if not applicable)."
    )

    inputs_meta = (
        "Inputs (this run)\n"
        f"• Role title (hint): {role_title or '—'}\n"
        f"• Role type: {role_type or '—'} | Seniority: {seniority or '—'}\n"
        f"• Focus: {focus or '—'}\n"
        f"• Deadline (AEST/AEDT): {deadline or '—'}\n"
        f"• Industry hint: {industry_hint or '—'}\n"
        f"• Extras/constraints: {extras or '—'}"
    )

    jd_keys = (
        "JD Keyword Guide (grouped)\n"
        f"• Tools/Technologies: {tools_csv or '—'}\n"
        f"• Methods/Capabilities: {methods_csv or '—'}\n"
        f"• Competencies: {comps_csv or '—'}\n"
        f"• Industry/Domain: {industry_csv or '—'}"
    )

    coverage = (
        "Coverage Snapshot\n"
        f"• Already present (from my résumé): {present_csv or '—'}\n"
        f"• Missing or weak (prioritise naturally, no stuffing): {missing_csv or '—'}"
    )

    usage_notes = (
        "How to Use the Keywords\n"
        "• Only include keywords that align with demonstrable evidence in the attached résumé/projects.\n"
        "• Prefer noun phrases and exact technical terms over vague adjectives.\n"
        "• Integrate keywords in context with measurable impact; avoid filler words."
    )

    deliverables = (
        "Deliverables (return exactly in this order; paste-ready)\n"
        "1) JD Debrief (5 bullets) + Top-12 ATS Keywords (exact phrases; mirror JD wording where sensible)\n"
        "2) Profile Summary (3–4 lines; mirror JD title + critical tools; cite strongest evidence)\n"
        "3) Skills (Tools/Technologies) — ordered; tag each (Expert/Working/Foundational); do NOT list Learning items\n"
        "4) Methodologies (Capabilities) — ordered to JD emphasis (e.g., ETL, data modelling, QA, validation, dashboards)\n"
        "5) Coverage Notes (5–8 bullets) — map JD → my evidence; for weaker items add ‘Interview navigation’ + ‘mitigation’ line\n"
        "6) Selected Projects (2–3) — 3–5 impact-first bullets each with [Relevance: <JD requirement>]; Include/Omit + rationale\n"
        "7) Experience — rewrite relevant roles (3–5 bullets each) mirroring JD verbs/outcomes with quant; add [Relevance: <req>]\n"
        "8) Cover Letter (180–220 words; British date) — use the provided template fields; concise and JD-mirrored\n"
        "9) Referees — résumé line + template for named referees (only if requested)\n"
        "10) Optional: Recruiter/HR outreach (≤550 chars) + 48-hour follow-up email\n"
        "11) Optional: Interview prep — 8 JD-based Qs with one-line answer angles; 2 mini-cases; 5-bullet rapid-prep plan for Learning items"
    )

    edit_tasks = (
        "Edit Tasks (strict)\n"
        "• Use ONLY evidence from the attached résumé/projects; no new tools or duties.\n"
        "• Prioritise the grouped JD keywords; prefer multi-word phrases (e.g., “end-to-end ETL”, “Databricks notebooks”).\n"
        "• Convert responsibilities → impact with numbers and outcomes.\n"
        "• If evidence is weak: suggest 1-line mitigation (e.g., ‘ship a small demo before interview’)."
    )

    # Optional raw inlining (off by default)
    raw_block = ""
    if include_raw:
        raw_block = (
            "\nDO NOT SUMMARISE BELOW; USE ONLY FOR REFERENCE IF PROVIDED:\n"
            f"— JD (raw/truncated OK) —\n{jd_text[:4000]}\n"
            f"— Résumé (raw/truncated OK) —\n{(resume_text or '')[:4000]}\n"
        )

    starter_bullets = _section("Starter tailored bullets (optional)", bullets_block)

    prompt = "\n".join(
        [
            header,
            _section("", rules).rstrip(),
            _section("", context).rstrip(),
            _section("", inputs_meta).rstrip(),
            _section("", jd_keys).rstrip(),
            _section("", coverage).rstrip(),
            _section("", usage_notes).rstrip(),
            _section("", deliverables).rstrip(),
            _section("", edit_tasks).rstrip(),
            starter_bullets.rstrip(),
            raw_block.rstrip(),  # populated only when include_raw=True
            "Return the deliverables in plain text, ready to paste into Word.",
        ]
    )

    return prompt
