"""
agents/achievement_scoring_agent.py
-------------------------------------
Agent 5: Achievement Scoring Agent (OPTIMISED — zero LLM calls)

Achievement quality score is now pre-computed by the resume_parser_agent in a
single batched LLM call. This agent uses pure math only.

Scoring strategy (4 components):

  Component 1 — Achievement Count & Diversity  (rule-based)
      Counts total achievements, certifications, projects.

  Component 2 — Achievement Quality  (pre-computed by parser, NO LLM here)
      achievement_quality_score stored in parsed_resume — 0 to 10.

  Component 3 — Certification Quality  (rule-based)
      Known high-value certs → higher score.

  Component 4 — Projects as Achievements  (rule-based)
      Number and quality of projects.

Final score formula:
  score = clamp(0.30*count_score + 0.40*quality_score + 0.15*cert_score + 0.15*project_score, 0, 10)
"""

from __future__ import annotations
from typing import List, Tuple

from models.state import ResumeGraphState, AchievementScore
from utils.logger import get_logger, log_agent_start, log_agent_end
from utils.validators import clamp_score

logger = get_logger("agents.achievement_scoring")


# ─── Component 1: Count & Diversity ───────────────────────────────────────────

def _count_score(
    achievements: List[dict],
    certifications: List[str],
    projects: List[dict],
) -> Tuple[float, int]:
    total = len(achievements) + len(certifications) + len(projects)
    if total == 0:
        return 0.0, 0
    # Scoring: diminishing returns
    if total >= 10:
        score = 10.0
    elif total >= 7:
        score = 8.0
    elif total >= 5:
        score = 7.0
    elif total >= 3:
        score = 6.0
    elif total >= 2:
        score = 5.0
    else:
        score = 3.5
    # Diversity bonus: if all 3 categories have at least 1 item
    if achievements and certifications and projects:
        score = min(score + 1.0, 10.0)
    return clamp_score(score), total


# ─── Component 2: Achievement Quality (pre-computed — NO LLM) ────────────────

def _quality_score_from_precomputed(achievement_quality_score: float | None) -> float:
    """
    Uses the achievement_quality_score set by resume_parser_agent.
    Falls back to 3.0 if not available.
    """
    if achievement_quality_score is None:
        logger.info("[AchievementAgent] No pre-computed quality score → fallback 3.0")
        return 3.0
    score = max(0.0, min(10.0, float(achievement_quality_score)))
    logger.info(f"[AchievementAgent] Pre-computed quality score: {score:.2f} [no LLM needed]")
    return score


# ─── Component 3: Certification Quality ───────────────────────────────────────

HIGH_VALUE_CERTS = [
    "aws", "azure", "gcp", "google cloud", "google professional",
    "tensorflow", "pytorch", "kubernetes", "cka", "ckad",
    "cisco", "ccna", "ccnp", "pmp", "scrum", "agile",
    "meta", "microsoft certified", "oracle", "salesforce",
    "databricks", "snowflake", "tableau", "power bi",
    "cfa", "cpa", "frm", "cissp", "ceh",
]

MEDIUM_VALUE_CERTS = [
    "coursera", "udemy", "edx", "nptel", "udacity",
    "hackerrank", "leetcode", "datacamp",
]


def _certification_quality_score(certifications: List[str]) -> float:
    if not certifications:
        return 0.0
    best = 0.0
    total_score = 0.0
    for cert in certifications:
        c = cert.lower()
        score = 3.0  # base
        for hv in HIGH_VALUE_CERTS:
            if hv in c:
                score = 8.5
                break
        else:
            for mv in MEDIUM_VALUE_CERTS:
                if mv in c:
                    score = 5.5
                    break
        total_score += score
        if score > best:
            best = score
    # Mix best + average to reward breadth
    avg = total_score / len(certifications)
    return clamp_score(0.6 * best + 0.4 * avg)


# ─── Component 4: Project Score ───────────────────────────────────────────────

def _project_score(projects: List[dict]) -> float:
    if not projects:
        return 0.0
    count = len(projects)
    # Base: number of projects
    if count >= 5:
        base = 8.0
    elif count >= 3:
        base = 7.0
    elif count >= 2:
        base = 5.5
    else:
        base = 4.0
    # Bonus: projects with technologies listed (shows technical depth)
    tech_count = sum(1 for p in projects if p.get("technologies") and len(p.get("technologies", [])) > 0)
    tech_bonus = min(tech_count * 0.5, 2.0)
    # Bonus: projects with descriptions
    desc_count = sum(1 for p in projects if p.get("description") and len(p.get("description", "")) > 30)
    desc_bonus = min(desc_count * 0.3, 1.0)
    return clamp_score(base + tech_bonus + desc_bonus)


# ─── Final score calculation ───────────────────────────────────────────────────

def _calculate_achievement_score(
    count_sc: float,
    quality_sc: float,
    cert_sc: float,
    proj_sc: float,
) -> float:
    raw = (0.30 * count_sc + 0.40 * quality_sc +
           0.15 * cert_sc  + 0.15 * proj_sc)
    return clamp_score(raw)


# ─── Main agent function ──────────────────────────────────────────────────────

def achievement_scoring_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 5 — Achievement Scoring (pure math, zero LLM calls)

    Achievement quality score is read from parsed_resume (pre-computed by Agent 1).
    Input state keys:  parsed_resume
    Output state keys: achievement_score
    """
    agent_name = "AchievementScoringAgent"
    log_agent_start(logger, agent_name, {
        "has_parsed_resume": bool(state.get("parsed_resume")),
    })

    errors        = list(state.get("errors") or [])
    parsed_resume = state.get("parsed_resume") or {}

    achievements:   List[dict] = parsed_resume.get("achievements")   or []
    certifications: List[str]  = parsed_resume.get("certifications") or []
    projects:       List[dict] = parsed_resume.get("projects")       or []
    # Pre-computed by resume_parser_agent — no extra LLM call needed
    achievement_quality_score  = parsed_resume.get("achievement_quality_score")

    logger.info(
        f"[{agent_name}] achievements={len(achievements)} "
        f"certs={len(certifications)} projects={len(projects)} "
        f"pre_quality={achievement_quality_score}"
    )

    if not achievements and not certifications and not projects:
        logger.warning(f"[{agent_name}] Nothing to score → 1.0")
        ach_score = AchievementScore(
            score=1.0, achievement_count=0,
            reasoning="No achievements, certifications, or projects found in resume.",
        )
        return {**state, "achievement_score": ach_score.model_dump(),
                "errors": errors, "current_step": agent_name}

    # Component 1: Count
    count_sc, total_count = _count_score(achievements, certifications, projects)

    # Component 2: Quality (from pre-computed parser data — NO LLM)
    quality_sc = _quality_score_from_precomputed(achievement_quality_score)

    # Component 3: Cert Quality
    cert_sc = _certification_quality_score(certifications)

    # Component 4: Projects
    proj_sc = _project_score(projects)

    # Final
    final_score = _calculate_achievement_score(count_sc, quality_sc, cert_sc, proj_sc)

    reasoning = (
        f"Achievements: {len(achievements)} | Certifications: {len(certifications)} | "
        f"Projects: {len(projects)} | Total items: {total_count}. "
        f"Count score: {count_sc:.2f}/10. "
        f"Quality score: {quality_sc:.2f}/10 [pre-computed, no LLM]. "
        f"Cert score: {cert_sc:.2f}/10. "
        f"Project score: {proj_sc:.2f}/10. "
        f"Weights: 30% count + 40% quality + 15% certs + 15% projects."
    )

    ach_score = AchievementScore(
        score=final_score,
        achievement_count=total_count,
        reasoning=reasoning,
    )

    log_agent_end(
        logger, agent_name,
        f"score={final_score:.2f}/10 | count={count_sc:.2f} | quality={quality_sc:.2f} "
        f"| cert={cert_sc:.2f} | proj={proj_sc:.2f}"
    )

    return {
        **state,
        "achievement_score": ach_score.model_dump(),
        "errors": errors,
        "current_step": agent_name,
    }
