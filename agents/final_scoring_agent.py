"""
agents/final_scoring_agent.py
------------------------------
Agent 8 (Final): Final Scoring Agent — pure math, ZERO LLM calls

Replaces the old llm_scoring_agent. The semantic/holistic fit that was
previously computed by an LLM is now approximated by a cosine similarity
between the resume's skills+experience text and the job description.
This saves one LLM call while still capturing semantic fit.

Weighted Regression → Final Score 0-100:

  Weights for full-time job roles:
    skill_score:       25%   (most direct signal)
    experience_score:  20%   (years + company quality)
    project_score:     15%   (NEW — projects + skill usage)
    education_score:   13%   (degree + institution)
    achievement_score: 12%   (competitive programming, certs, hackathons)
    social_score:      10%   (GitHub, LeetCode, CF activity)
    semantic_score:    05%   (cosine fit of resume vs job description)

  Weights for internships:
    skill_score:       30%
    project_score:     20%   (projects matter most for freshers)
    experience_score:  08%
    education_score:   15%
    achievement_score: 15%
    social_score:      07%
    semantic_score:    05%

  Formula:
    weighted_sum = Σ (weight_i × score_i)     [each score is 0-10]
    total_score  = clamp(weighted_sum × 10, 0, 100)

  Missing scores (agent failed) → substituted with 3.0 (below-average neutral)

Score thresholds for label:
  90-100 → Exceptional Match 🏆
  75-89  → Strong Match ✅
  60-74  → Good Match 👍
  45-59  → Moderate Match ⚠️
  30-44  → Weak Match 📉
  0-29   → Poor Match ❌
"""

from __future__ import annotations
from typing import Optional

from models.state import ResumeGraphState, FinalScore, SemanticScore
from utils.logger import get_logger, log_agent_start, log_agent_end
from utils.validators import clamp_score

logger = get_logger("agents.final_scoring")

# ─── Weights ──────────────────────────────────────────────────────────────────

WEIGHTS_JOB = {
    "skill":       0.25,
    "experience":  0.20,
    "project":     0.15,
    "education":   0.13,
    "achievement": 0.12,
    "social":      0.10,
    "semantic":    0.05,
}

WEIGHTS_INTERNSHIP = {
    "skill":       0.30,
    "project":     0.20,
    "experience":  0.08,
    "education":   0.15,
    "achievement": 0.15,
    "social":      0.07,
    "semantic":    0.05,
}

SCORE_LABELS = [
    (90, "Exceptional Match 🏆"),
    (75, "Strong Match ✅"),
    (60, "Good Match 👍"),
    (45, "Moderate Match ⚠️"),
    (30, "Weak Match 📉"),
    (0,  "Poor Match ❌"),
]


def _score_label(score: float) -> str:
    for threshold, label in SCORE_LABELS:
        if score >= threshold:
            return label
    return "Poor Match ❌"


# ─── Semantic score via cosine (no LLM) ──────────────────────────────────────

def _cosine_semantic_score(parsed_resume: dict, job_input: dict) -> SemanticScore:
    """
    Approximate semantic fit using TF-IDF cosine similarity between:
      - Resume text (skills + experience descriptions + professional summary)
      - Job text (title + description + required skills)
    Returns SemanticScore with score [0, 10].
    """
    # Build resume representation
    resume_parts = []
    summary = parsed_resume.get("professional_summary") or ""
    if summary:
        resume_parts.append(summary)
    skills = parsed_resume.get("skills") or []
    if skills:
        resume_parts.append(" ".join(skills))
    for exp in (parsed_resume.get("experience") or []):
        pos = exp.get("position") or ""
        desc = exp.get("description") or ""
        if pos:
            resume_parts.append(pos)
        if desc:
            resume_parts.append(desc[:500])
    for proj in (parsed_resume.get("projects") or []):
        desc = proj.get("description") or ""
        techs = " ".join(proj.get("technologies") or [])
        if desc:
            resume_parts.append(desc[:300])
        if techs:
            resume_parts.append(techs)

    # Build job representation
    job_parts = [
        job_input.get("job_title", ""),
        job_input.get("job_description", ""),
        " ".join(job_input.get("skills_required") or []),
        job_input.get("job_role", ""),
    ]

    resume_text = " ".join(filter(None, resume_parts)).strip()
    job_text    = " ".join(filter(None, job_parts)).strip()

    if not resume_text or not job_text:
        return SemanticScore(score=5.0, reasoning="Insufficient text for semantic comparison.")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        tfidf = vectorizer.fit_transform([resume_text, job_text])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        # Apply gentle power boost since short texts underestimate similarity
        boosted = sim ** 0.6
        score = clamp_score(boosted * 10)
        reasoning = f"TF-IDF cosine similarity={sim:.3f} (boosted={boosted:.3f}) → {score:.2f}/10"
        logger.info(f"[FinalScoringAgent] Semantic cosine: {reasoning}")
        return SemanticScore(score=score, reasoning=reasoning)
    except Exception as e:
        logger.warning(f"[FinalScoringAgent] Cosine semantic failed: {e} → default 5.0")
        return SemanticScore(score=5.0, reasoning=f"Cosine similarity failed: {e}")


# ─── Weighted regression ──────────────────────────────────────────────────────

def _weighted_regression(
    skill_s:       float,
    experience_s:  float,
    education_s:   float,
    achievement_s: float,
    social_s:      float,
    project_s:     float,
    semantic_s:    float,
    weights:       dict,
) -> tuple[float, dict]:
    """
    Compute final weighted score 0-100 and per-component breakdown.
    Returns (total_score, breakdown_dict).
    """
    scores = {
        "skill":       skill_s,
        "experience":  experience_s,
        "education":   education_s,
        "achievement": achievement_s,
        "social":      social_s,
        "project":     project_s,
        "semantic":    semantic_s,
    }

    weighted_sum = sum(weights[k] * scores[k] for k in weights)
    total = clamp_score(weighted_sum * 10, min_val=0.0, max_val=100.0)

    breakdown = {
        k: {
            "raw_score":             round(scores[k], 2),
            "weight":                weights[k],
            "weighted_contribution": round(weights[k] * scores[k] * 10, 2),
        }
        for k in weights
    }
    breakdown["total"] = round(total, 2)
    breakdown["label"] = _score_label(total)

    return round(total, 2), breakdown


# ─── Main agent function ──────────────────────────────────────────────────────

def final_scoring_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 8 — Final Scoring (pure math formula, ZERO LLM calls)

    Aggregates all component scores using a weighted regression.
    Semantic fit is computed via TF-IDF cosine (not LLM).

    Input state keys:  all previous scores + parsed_resume + job_input
    Output state keys: semantic_score, final_score
    """
    agent_name = "FinalScoringAgent"
    log_agent_start(logger, agent_name, {
        "has_skill_score":       bool(state.get("skill_score")),
        "has_experience_score":  bool(state.get("experience_score")),
        "has_education_score":   bool(state.get("education_score")),
        "has_achievement_score": bool(state.get("achievement_score")),
        "has_social_score":      bool(state.get("social_score")),
        "has_project_score":     bool(state.get("project_score")),
    })

    errors        = list(state.get("errors") or [])
    parsed_resume = state.get("parsed_resume") or {}
    job_input     = state.get("job_input")     or {}

    is_internship = str(job_input.get("opportunity_type", "job")).lower() == "internship"
    weights = WEIGHTS_INTERNSHIP if is_internship else WEIGHTS_JOB
    opp_label = "internship" if is_internship else "job"

    # Pull component scores — default to 3.0 if agent failed/missing
    FALLBACK = 3.0
    skill_s  = float((state.get("skill_score")       or {}).get("score") or FALLBACK)
    exp_s    = float((state.get("experience_score")  or {}).get("score") or FALLBACK)
    edu_s    = float((state.get("education_score")   or {}).get("score") or FALLBACK)
    ach_s    = float((state.get("achievement_score") or {}).get("score") or FALLBACK)
    soc_s    = float((state.get("social_score")      or {}).get("score") or FALLBACK)
    proj_s   = float((state.get("project_score")     or {}).get("score") or FALLBACK)

    logger.info(
        f"[{agent_name}] opportunity={opp_label} | "
        f"skill={skill_s:.2f} exp={exp_s:.2f} edu={edu_s:.2f} "
        f"ach={ach_s:.2f} soc={soc_s:.2f} proj={proj_s:.2f}"
    )

    # Cosine semantic score (no LLM)
    sem_score_obj = _cosine_semantic_score(parsed_resume, job_input)
    sem_s = sem_score_obj.score
    logger.info(f"[{agent_name}] Semantic (cosine) score={sem_s:.2f}")

    # Weighted regression
    total_score, breakdown = _weighted_regression(
        skill_s, exp_s, edu_s, ach_s, soc_s, proj_s, sem_s,
        weights=weights,
    )
    breakdown["opportunity_type"] = opp_label

    label = _score_label(total_score)
    logger.info(f"[{agent_name}] Final score={total_score:.2f}/100 | {label}")

    final_score_obj = FinalScore(
        total_score=total_score,
        breakdown=breakdown,
    )

    log_agent_end(
        logger, agent_name,
        f"FINAL={total_score:.2f}/100 | label='{label}' | "
        f"opportunity={opp_label} | semantic={sem_s:.2f} | "
        f"weights: sk={weights['skill']} ex={weights['experience']} "
        f"proj={weights['project']} edu={weights['education']} "
        f"ach={weights['achievement']} soc={weights['social']} sem={weights['semantic']}"
    )

    return {
        **state,
        "semantic_score": sem_score_obj.model_dump(),
        "final_score":    final_score_obj.model_dump(),
        "errors":         errors,
        "current_step":   agent_name,
    }
