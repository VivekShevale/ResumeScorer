"""
agents/experience_scoring_agent.py
-------------------------------------
Agent 3: Experience Scoring Agent (OPTIMISED — zero LLM calls)

Company tier is now pre-computed by the resume_parser_agent in a single
batched LLM call. This agent uses pure math + cosine similarity only.

Scoring strategy (3 components):

  Component 1 — Years of Experience  (pure math, no LLM)
      Parse start/end dates from each experience item → compute duration
      Compare total_years vs years_experience_required

  Component 2 — Company Tier  (pre-computed by parser, NO LLM here)
      tier value stored in experience[i].company_tier:
        Tier 1 → 10.0 (FAANG/Top MNC)
        Tier 2 → 6.5  (mid-size/known startups)
        Tier 3 → 3.5  (small/unknown)

  Component 3 — Role Relevance  (cosine similarity, no LLM)
      Cosine similarity between job titles/descriptions and job input.

Final score formula:
  score = clamp(0.40*years_score + 0.35*tier_score + 0.25*relevance_score, 0, 10)
"""

from __future__ import annotations
import re
from datetime import datetime
from typing import List, Optional, Tuple

from models.state import ResumeGraphState, ExperienceScore
from utils.logger import get_logger, log_agent_start, log_agent_end
from utils.validators import clamp_score

logger = get_logger("agents.experience_scoring")


# ─── Component 1: Years of Experience ────────────────────────────────────────

def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = date_str.strip()
    if s.lower() in ("present", "current", "now", "ongoing"):
        return datetime.now()
    formats = [
        "%Y-%m-%d", "%Y-%m", "%Y",
        "%b %Y", "%B %Y",
        "%m/%Y", "%m-%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # Try extracting year only
    m = re.search(r"\b(20\d{2}|19\d{2})\b", s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y")
        except ValueError:
            pass
    return None


def _compute_duration_years(start_str: Optional[str], end_str: Optional[str], is_current: bool) -> float:
    start = _parse_date(start_str)
    end = datetime.now() if is_current else _parse_date(end_str)
    if not start or not end:
        return 0.0
    if end < start:
        return 0.0
    delta = (end - start).days / 365.25
    return round(max(0.0, delta), 2)


def _total_experience_years(experience: List[dict]) -> Tuple[float, List[dict]]:
    details = []
    total = 0.0
    for exp in experience:
        years = _compute_duration_years(
            exp.get("start_date"),
            exp.get("end_date"),
            bool(exp.get("is_current", False))
        )
        company = exp.get("company") or "Unknown"
        tier = exp.get("company_tier", 3)
        if tier not in (1, 2, 3):
            tier = 3
        details.append({"company": company, "years": years, "tier": tier,
                         "position": exp.get("position", ""),
                         "description": exp.get("description", "")})
        total += years
    return round(total, 2), details


def _years_score(total_years: float, required_years: float, is_internship: bool = False) -> float:
    if is_internship:
        # For internships, even 0 years is OK; treat as bonus
        if total_years == 0:
            return 5.0
        return clamp_score(min(total_years / max(required_years, 0.5), 1.0) * 10)
    if required_years <= 0:
        # No requirement: score based on total years (5y = 10)
        return clamp_score(min(total_years / 5.0, 1.0) * 10)
    ratio = total_years / required_years
    if ratio >= 1.0:
        # Bonus for extra experience, capped at 10
        return clamp_score(min(7.0 + (ratio - 1.0) * 3.0, 10.0))
    return clamp_score(ratio * 7.0)


# ─── Component 2: Company Tier (pre-computed — NO LLM) ───────────────────────

TIER_SCORE_MAP = {1: 10.0, 2: 6.5, 3: 3.5}


def _company_tier_score(exp_details: List[dict]) -> Tuple[float, List[dict]]:
    """
    Reads company_tier from each experience entry (set by resume_parser_agent).
    Weights by years_at_company so longer stints matter more.
    No LLM call needed here.
    """
    if not exp_details:
        return 0.0, []
    total_weighted = 0.0
    total_years = 0.0
    tier_results = []
    for exp in exp_details:
        tier = exp.get("tier", 3)
        if tier not in (1, 2, 3):
            tier = 3
        years = exp.get("years", 0.0)
        score = TIER_SCORE_MAP[tier]
        weight = max(years, 0.5)  # floor 0.5 to give short stints some weight
        total_weighted += score * weight
        total_years += weight
        tier_results.append({"company": exp["company"], "tier": tier, "years": years})
    if total_years == 0:
        return 3.5, tier_results
    return clamp_score(total_weighted / total_years), tier_results


# ─── Component 3: Role Relevance ──────────────────────────────────────────────

def _relevance_score(exp_details: List[dict], job_title: str, job_desc: str) -> float:
    if not exp_details:
        return 0.0
    resume_text = " ".join(
        f"{e.get('position', '')} {e.get('description', '')}" for e in exp_details
    ).strip().lower()
    job_text = f"{job_title} {job_desc}".strip().lower()
    if not resume_text or not job_text:
        return 3.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform([resume_text, job_text])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        score = clamp_score(sim ** 0.5 * 10)
        logger.info(f"[ExperienceAgent] Role relevance cosine={sim:.3f} → {score:.2f}")
        return score
    except Exception as e:
        logger.warning(f"[ExperienceAgent] Relevance score failed: {e}")
        return 3.0


# ─── Final score calculation ──────────────────────────────────────────────────

def _calculate_experience_score(
    years_score: float,
    tier_score: float,
    relevance_score: float,
    is_internship: bool = False,
) -> float:
    if is_internship:
        # For internships, relevance + tier matter more than years
        raw = 0.25 * years_score + 0.35 * tier_score + 0.40 * relevance_score
    else:
        raw = 0.40 * years_score + 0.35 * tier_score + 0.25 * relevance_score
    return clamp_score(raw)


# ─── Main agent function ──────────────────────────────────────────────────────

def experience_scoring_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 3 — Experience Scoring (pure math, zero LLM calls)

    Company tiers are read directly from parsed_resume (pre-computed by Agent 1).
    Input state keys:  parsed_resume, job_input
    Output state keys: experience_score
    """
    agent_name = "ExperienceScoringAgent"
    log_agent_start(logger, agent_name, {
        "has_parsed_resume": bool(state.get("parsed_resume")),
        "has_job_input": bool(state.get("job_input")),
    })

    errors = list(state.get("errors") or [])
    parsed_resume = state.get("parsed_resume") or {}
    job_input = state.get("job_input") or {}

    experience: List[dict] = parsed_resume.get("experience") or []
    job_title:  str        = job_input.get("job_title", "")
    job_desc:   str        = job_input.get("job_description", "")
    req_years:  float      = float(job_input.get("years_experience_required") or 0)
    is_internship: bool    = str(job_input.get("opportunity_type", "job")).lower() == "internship"

    logger.info(f"[{agent_name}] experience_items={len(experience)} | req_years={req_years} | internship={is_internship}")

    if not experience:
        base_score = 5.0 if is_internship else 1.0
        reason = (
            "No work experience found — expected for internship applicants."
            if is_internship else
            "No work experience found in resume."
        )
        logger.info(f"[{agent_name}] No experience entries → score={base_score}")
        exp_score = ExperienceScore(
            score=base_score, total_years=0.0, company_tier_avg=None, reasoning=reason,
        )
        return {**state, "experience_score": exp_score.model_dump(),
                "errors": errors, "current_step": agent_name}

    # Component 1: Years
    total_years, exp_details = _total_experience_years(experience)
    y_score = _years_score(total_years, req_years, is_internship=is_internship)
    logger.info(f"[{agent_name}] total_years={total_years} | years_score={y_score:.2f}")

    # Component 2: Company Tier (from pre-computed parser data — NO LLM)
    tier_score_val, tier_details = _company_tier_score(exp_details)
    tier_avg = (
        sum(t["tier"] for t in tier_details) / len(tier_details)
        if tier_details else None
    )
    logger.info(f"[{agent_name}] tier_score={tier_score_val:.2f} [pre-computed, no LLM]")

    # Component 3: Role Relevance
    rel_score = _relevance_score(exp_details, job_title, job_desc)

    # Final score
    final_score = _calculate_experience_score(y_score, tier_score_val, rel_score, is_internship=is_internship)

    tier_str = " | ".join(
        f"{t['company']}→T{t['tier']}({t['years']:.1f}y)" for t in tier_details
    )
    reasoning = (
        f"Total experience: {total_years:.1f} years (required: {req_years:.1f}). "
        f"Years score: {y_score:.2f}/10. "
        f"Company tiers: {tier_str or 'N/A'}. "
        f"Tier score: {tier_score_val:.2f}/10. "
        f"Role relevance: {rel_score:.2f}/10. "
        f"Weights: {'25% years + 35% tier + 40% relevance (internship)' if is_internship else '40% years + 35% tier + 25% relevance'}."
    )

    exp_score = ExperienceScore(
        score=final_score,
        total_years=total_years,
        company_tier_avg=tier_avg,
        reasoning=reasoning,
    )

    log_agent_end(
        logger, agent_name,
        f"score={final_score:.2f}/10 | years={y_score:.2f} tier={tier_score_val:.2f} rel={rel_score:.2f}"
    )

    return {
        **state,
        "experience_score": exp_score.model_dump(),
        "errors": errors,
        "current_step": agent_name,
    }
