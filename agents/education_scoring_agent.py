"""
agents/education_scoring_agent.py
-----------------------------------
Agent 4: Education Scoring Agent (OPTIMISED — zero LLM calls)

Institution tier is now pre-computed by the resume_parser_agent in a single
batched LLM call. This agent uses pure math only.

Scoring strategy (4 components):

  Component 1 — Degree Level  (rule-based, no LLM)
      Map degree to a level score:
        PhD / Doctorate        → 10.0
        M.Tech / M.S. / MBA    → 8.5
        B.Tech / B.E. / B.S.   → 7.0
        Diploma / Associate    → 5.0
        High School / 12th     → 3.0
        Unknown                → 4.0
      Compare against education_required from job input.
      If candidate meets/exceeds requirement → full component score.
      If below requirement → penalise proportionally.

  Component 2 — Institution Tier  (pre-computed by parser, NO LLM here)
      tier value stored in education[i].institution_tier:
        Tier 1 → 10.0 (IITs, top global)
        Tier 2 → 6.5  (good private/state colleges)
        Tier 3 → 3.5  (unknown/local)

  Component 3 — Field Relevance  (cosine similarity, no LLM)
      Cosine similarity between candidate's field/degree text
      and job_title + job_description.
      Score [0, 10].

  Component 4 — GPA / Academic Performance  (rule-based)
      Normalise GPA to 0-10 scale.
      No GPA → neutral score (5.0)

Final score formula:
  score = clamp(0.35*degree + 0.30*tier + 0.20*relevance + 0.15*gpa, 0, 10)
"""

from __future__ import annotations
import re
from typing import List, Optional, Tuple

from models.state import ResumeGraphState, EducationScore
from utils.logger import get_logger, log_agent_start, log_agent_end
from utils.validators import clamp_score

logger = get_logger("agents.education_scoring")


# ─── Component 1: Degree Level ────────────────────────────────────────────────

DEGREE_LEVELS = [
    (["phd", "ph.d", "doctorate", "doctor of philosophy"],         5, 10.0),
    (["m.tech", "mtech", "m.e.", "me ", "master of technology"],   4, 8.5),
    (["m.s.", "ms ", "m.sc", "msc", "master of science"],          4, 8.5),
    (["mba", "master of business"],                                 4, 8.5),
    (["master"],                                                    4, 8.5),
    (["b.tech", "btech", "b.e.", "be ", "bachelor of technology"], 3, 7.0),
    (["b.s.", "bs ", "b.sc", "bsc", "bachelor of science"],        3, 7.0),
    (["b.a.", "ba ", "bachelor of arts"],                           3, 6.5),
    (["bachelor"],                                                  3, 7.0),
    (["diploma", "associate"],                                      2, 5.0),
    (["high school", "hsc", "12th", "10+2", "higher secondary"],   1, 3.0),
]

REQUIREMENT_LEVEL_MAP = {
    "phd": 5, "doctorate": 5,
    "master": 4, "mtech": 4, "msc": 4, "mba": 4, "ms": 4,
    "bachelor": 3, "btech": 3, "bsc": 3, "be": 3, "graduate": 3,
    "diploma": 2,
    "high school": 1, "hsc": 1, "12th": 1,
}


def _degree_to_level(degree_str: Optional[str]) -> Tuple[int, float, str]:
    if not degree_str:
        return 2, 4.0, "Unknown"
    d = degree_str.lower().strip()
    for keywords, level, score in DEGREE_LEVELS:
        if any(kw in d for kw in keywords):
            return level, score, degree_str
    branch_signals = [
        "computer", "software", "electronics", "mechanical", "civil",
        "chemical", "biotechnology", "information", "science", "engineering",
        "technology", "mathematics", "physics", "statistics",
    ]
    if any(sig in d for sig in branch_signals):
        return 3, 7.0, degree_str
    return 2, 4.0, degree_str


def _requirement_level(education_required: Optional[str]) -> int:
    if not education_required:
        return 0
    r = education_required.lower()
    for key, level in REQUIREMENT_LEVEL_MAP.items():
        if key in r:
            return level
    return 0


def _degree_score(
    education_list: List[dict],
    education_required: Optional[str],
) -> Tuple[float, str, str]:
    if not education_list:
        return 2.0, "None", "No education entries found."
    best_level, best_score, best_degree = -1, 0.0, "Unknown"
    for edu in education_list:
        deg = edu.get("degree", "")
        level, score, canonical = _degree_to_level(deg)
        if level > best_level:
            best_level, best_score, best_degree = level, score, canonical
    req_level = _requirement_level(education_required)
    if req_level == 0:
        return clamp_score(best_score), best_degree, "No degree requirement specified."
    elif best_level >= req_level:
        return clamp_score(best_score), best_degree, f"Meets/exceeds requirement ({education_required})."
    else:
        penalty = (req_level - best_level) * 1.5
        final_score = max(1.0, best_score - penalty)
        return clamp_score(final_score), best_degree, f"Below required level ({education_required}). Penalty applied."


# ─── Component 2: Institution Tier (pre-computed — NO LLM) ───────────────────

TIER_SCORE_MAP = {1: 10.0, 2: 6.5, 3: 3.5}


def _institution_tier_score(education_list: List[dict]) -> Tuple[float, str, List[dict]]:
    """
    Reads institution_tier from each education entry (set by resume_parser_agent).
    No LLM call needed here.
    """
    if not education_list:
        return 0.0, "Unknown", []
    details = []
    best_tier_score = 0.0
    best_tier_label = "Unknown"
    for edu in education_list:
        inst = edu.get("institution") or ""
        degree = edu.get("degree", "")
        if not inst:
            continue
        tier = edu.get("institution_tier", 2)
        if tier not in (1, 2, 3):
            tier = 2
        score = TIER_SCORE_MAP[tier]
        details.append({"institution": inst, "degree": degree, "tier": tier, "score": score})
        if score > best_tier_score:
            best_tier_score = score
            best_tier_label = f"Tier {tier}"
    if not details:
        return 3.5, "Tier 3 (unclassified)", details
    return clamp_score(best_tier_score), best_tier_label, details


# ─── Component 3: Field Relevance ─────────────────────────────────────────────

def _field_relevance_score(
    education_list: List[dict],
    job_title: str,
    job_description: str,
) -> float:
    if not education_list:
        return 3.0
    edu_text = " ".join(
        f"{e.get('degree','')} {e.get('field','')} {e.get('institution','')}"
        for e in education_list
    ).strip().lower()
    job_text = f"{job_title} {job_description}".strip().lower()
    CS_KEYWORDS = [
        "computer science", "artificial intelligence", "machine learning",
        "ai", "ml", "aiml", "ai-ml", "data science", "information technology",
        "software engineering", "computer engineering", "it", "cse",
    ]
    bonus = 0.0
    for kw in CS_KEYWORDS:
        if kw in edu_text:
            bonus = 3.0
            break
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        if not edu_text or not job_text:
            return clamp_score(bonus + 3.0)
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform([edu_text, job_text])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        boosted = sim ** 0.6
        cosine_score = boosted * 10
        score = clamp_score(max(cosine_score, cosine_score * 0.5 + bonus * 1.2))
        logger.info(f"[EducationAgent] Field relevance: cosine={sim:.3f} boosted={boosted:.3f} bonus={bonus} → {score:.2f}")
        return score
    except Exception as e:
        logger.warning(f"[EducationAgent] Field relevance failed: {e}")
        return clamp_score(bonus + 3.0)


# ─── Component 4: GPA ─────────────────────────────────────────────────────────

def _parse_gpa(gpa_str: Optional[str]) -> Optional[Tuple[float, float]]:
    if not gpa_str or not gpa_str.strip():
        return None
    s = gpa_str.strip().replace(",", ".")
    frac = re.search(r"([\d.]+)\s*/\s*([\d.]+)", s)
    if frac:
        return float(frac.group(1)), float(frac.group(2))
    pct = re.search(r"([\d.]+)\s*%", s)
    if pct:
        return float(pct.group(1)), 100.0
    plain = re.search(r"([\d.]+)", s)
    if plain:
        val = float(plain.group(1))
        if val > 10:
            return val, 100.0
        elif val > 4:
            return val, 10.0
        else:
            return val, 4.0
    return None


def _gpa_score(education_list: List[dict]) -> float:
    best = 0.0
    found_any = False
    for edu in education_list:
        result = _parse_gpa(edu.get("gpa"))
        if result is None:
            continue
        found_any = True
        val, scale = result
        normalised = (val / scale) * 10
        if normalised > best:
            best = normalised
    if not found_any:
        logger.info("[EducationAgent] No GPA found → neutral 5.0")
        return 5.0
    score = clamp_score(best)
    logger.info(f"[EducationAgent] Best GPA normalised → {score:.2f}/10")
    return score


# ─── Final score calculation ───────────────────────────────────────────────────

def _calculate_education_score(
    degree_score: float,
    tier_score: float,
    relevance_score: float,
    gpa_score: float,
) -> float:
    raw = (0.35 * degree_score + 0.30 * tier_score +
           0.20 * relevance_score + 0.15 * gpa_score)
    return clamp_score(raw)


# ─── Main agent function ───────────────────────────────────────────────────────

def education_scoring_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 4 — Education Scoring (pure math, zero LLM calls)

    Institution tiers are read directly from parsed_resume (pre-computed by Agent 1).
    Input state keys:  parsed_resume, job_input
    Output state keys: education_score
    """
    agent_name = "EducationScoringAgent"
    log_agent_start(logger, agent_name, {
        "has_parsed_resume": bool(state.get("parsed_resume")),
        "has_job_input":     bool(state.get("job_input")),
    })

    errors        = list(state.get("errors") or [])
    parsed_resume = state.get("parsed_resume") or {}
    job_input     = state.get("job_input")     or {}

    education:          List[dict] = parsed_resume.get("education") or []
    job_title:          str        = job_input.get("job_title", "")
    job_desc:           str        = job_input.get("job_description", "")
    education_required: str        = job_input.get("education_required", "") or ""

    logger.info(f"[{agent_name}] education_items={len(education)} | required='{education_required}'")

    if not education:
        logger.warning(f"[{agent_name}] No education entries → score=2.0")
        edu_score = EducationScore(
            score=2.0, highest_degree=None, institution_tier=None,
            reasoning="No education entries found in resume.",
        )
        return {**state, "education_score": edu_score.model_dump(),
                "errors": errors, "current_step": agent_name}

    # Component 1: Degree Level
    deg_score, highest_degree, deg_note = _degree_score(education, education_required)
    logger.info(f"[{agent_name}] degree='{highest_degree}' score={deg_score:.2f} | {deg_note}")

    # Component 2: Institution Tier (from pre-computed parser data — NO LLM)
    tier_score, tier_label, tier_details = _institution_tier_score(education)
    logger.info(f"[{agent_name}] tier_label='{tier_label}' score={tier_score:.2f} [pre-computed, no LLM]")

    # Component 3: Field Relevance
    rel_score = _field_relevance_score(education, job_title, job_desc)

    # Component 4: GPA
    gpa_sc = _gpa_score(education)

    # Final score
    final_score = _calculate_education_score(deg_score, tier_score, rel_score, gpa_sc)

    tier_str = " | ".join(
        f"{t['institution']}→T{t['tier']}" for t in tier_details
    ) or "N/A"
    reasoning = (
        f"Highest degree: {highest_degree}. {deg_note} "
        f"Degree score: {deg_score:.2f}/10. "
        f"Institutions: {tier_str}. "
        f"Tier score: {tier_score:.2f}/10. "
        f"Field relevance: {rel_score:.2f}/10. "
        f"GPA score: {gpa_sc:.2f}/10. "
        f"Weights: 35% degree + 30% tier + 20% relevance + 15% GPA."
    )

    edu_score = EducationScore(
        score=final_score,
        highest_degree=highest_degree,
        institution_tier=tier_label,
        reasoning=reasoning,
    )

    log_agent_end(
        logger, agent_name,
        f"score={final_score:.2f}/10 | deg={deg_score:.2f} tier={tier_score:.2f} "
        f"rel={rel_score:.2f} gpa={gpa_sc:.2f}"
    )

    return {
        **state,
        "education_score": edu_score.model_dump(),
        "errors": errors,
        "current_step": agent_name,
    }
