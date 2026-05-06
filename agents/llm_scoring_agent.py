"""
agents/llm_scoring_agent.py
-----------------------------
Agent 7: LLM Scoring Agent + Final Weighted Regression

Two sub-components:

  Sub-component A — Semantic Similarity Score  (LLM, temperature=0)
      The LLM reads:
        - Candidate's professional summary + skills + experience descriptions
        - Job title + job description + required skills
      It rates semantic fit holistically: 0-10
      This captures things the rule-based agents miss:
        - "The candidate built distributed systems" ↔ "We need backend scale experience"
        - Tone of experience matching seniority level
      Anti-hallucination: output must be {"score": float, "reasoning": str}
      Score validated strictly as float in [0, 10].

  Sub-component B — Weighted Regression → Final Score 0-100
      Weights (tuned for software/tech roles):
        skill_score:       25%   (most direct signal)
        experience_score:  22%   (years + quality)
        education_score:   15%   (degree + institution)
        achievement_score: 13%   (competitive programming, certs, hackathons)
        social_score:      10%   (GitHub, LeetCode, CF activity)
        semantic_score:    15%   (holistic LLM fit assessment)

      Formula:
        weighted_sum = Σ (weight_i × score_i)     [each score is 0-10]
        total_score  = clamp(weighted_sum × 10, 0, 100)

      Missing scores (agent failed) → substituted with 3.0 (below average neutral)

Score thresholds for label:
  90-100 → Exceptional Match
  75-89  → Strong Match
  60-74  → Good Match
  45-59  → Moderate Match
  30-44  → Weak Match
  0-29   → Poor Match
"""

from __future__ import annotations
import os
from typing import Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from models.state import ResumeGraphState, SemanticScore, FinalScore
from utils.logger import get_logger, log_agent_start, log_agent_end, log_agent_error
from utils.validators import safe_parse_json, clamp_score

load_dotenv()
logger = get_logger("agents.llm_scoring")


# ─── LLM setup ────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env file")
    return ChatGroq(api_key=api_key, model=model, temperature=0, max_tokens=512)


# ─── Weights ──────────────────────────────────────────────────────────────────

# Weights for full-time job roles
WEIGHTS_JOB = {
    "skill":       0.25,
    "experience":  0.22,
    "semantic":    0.15,
    "education":   0.15,
    "achievement": 0.13,
    "social":      0.10,
}

# Weights for internships — experience matters less, skills/projects/achievements more
WEIGHTS_INTERNSHIP = {
    "skill":       0.30,   # skills are the primary signal
    "experience":  0.10,   # prior exp is a bonus, not a requirement
    "semantic":    0.20,   # holistic project+skills fit matters most
    "education":   0.15,   # still relevant (ongoing degree)
    "achievement": 0.15,   # hackathons, competitive coding signal potential
    "social":      0.10,   # GitHub/LeetCode activity
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


# ─── Sub-component A: Semantic Similarity (LLM) ───────────────────────────────

SEMANTIC_SYSTEM_PROMPT_JOB = """You are an expert technical recruiter. Rate how well a candidate's profile matches a job requirement.

Evaluate holistically based on:
1. Alignment between candidate's experience/projects and job requirements
2. Technology stack overlap and transferability
3. Seniority level match (junior/mid/senior signals)
4. Domain relevance (e.g., AI/ML, fintech, edtech, enterprise)
5. Overall professional narrative fit
6. Achievements and competitive programming that signal ability

STRICT RULES:
1. Output ONLY valid JSON — no explanation, no markdown, no preamble.
2. Return EXACTLY: {"score": <float 0-10>, "reasoning": "<one concise sentence>"}
3. Score must be a float between 0.0 and 10.0.
4. Base score ONLY on the text provided. Do not assume or infer beyond what is written.
5. Be calibrated: 5.0 = average fit, 8+ = strong fit, 3- = poor fit."""

SEMANTIC_SYSTEM_PROMPT_INTERNSHIP = """You are an expert technical recruiter evaluating candidates for an INTERNSHIP position.

CRITICAL CONTEXT — THIS IS AN INTERNSHIP:
- Interns are STUDENTS or FRESH GRADUATES. Zero or minimal work experience is COMPLETELY NORMAL.
- DO NOT penalise for: lack of work experience, missing seniority signals, no professional history.
- DO NOT use phrases like "lacking seniority cues" or "lacking experience depth" as negatives.
- JUDGE PRIMARILY ON: skill match + project relevance + hackathon/competition wins + learning trajectory.

Scoring guidance for internships:
- Strong skill match + relevant projects → 7.5 to 9.0
- Decent skill match + some relevant projects → 6.0 to 7.5
- Partial skill match, genuinely missing core required tools → 4.0 to 6.0
- Skills mostly absent from requirements → 2.0 to 4.0

STRICT RULES:
1. Output ONLY valid JSON — no explanation, no markdown, no preamble.
2. Return EXACTLY: {"score": <float 0-10>, "reasoning": "<one concise sentence focusing on skill/project fit>"}
3. Score must be a float between 0.0 and 10.0.
4. Do not mention seniority, work history depth, or years of experience as negatives."""


def _semantic_score(parsed_resume: dict, job_input: dict, is_internship: bool = False) -> SemanticScore:
    """
    LLM rates the holistic semantic fit between resume and job.
    Returns SemanticScore. Defaults to 5.0 on failure.
    """
    try:
        # Build candidate summary (concise — avoid token overflow)
        pi       = parsed_resume.get("personal_info") or {}
        summary  = parsed_resume.get("professional_summary") or ""
        skills   = parsed_resume.get("skills") or []
        exps     = parsed_resume.get("experience") or []
        projects = parsed_resume.get("projects") or []
        achievements = parsed_resume.get("achievements") or []

        exp_text = "\n".join(
            f"  - {e.get('position','')} at {e.get('company','')} "
            f"({e.get('start_date','')}–{e.get('end_date','')}): {(e.get('description','') or '')[:200]}"
            for e in exps[:4]
        )
        proj_text = "\n".join(
            f"  - {p.get('name','')}: {(p.get('description','') or '')[:150]}"
            for p in projects[:4]
        )
        ach_text = "\n".join(
            f"  - {a.get('title','')}"
            for a in achievements[:3]
        )

        candidate_text = (
            f"Profession: {pi.get('profession','')}\n"
            f"Summary: {summary[:400]}\n"
            f"Skills: {', '.join(skills[:30])}\n"
            f"Experience:\n{exp_text}\n"
            f"Projects:\n{proj_text}\n"
            f"Achievements:\n{ach_text}"
        ).strip()

        job_text = (
            f"Opportunity Type: {'INTERNSHIP' if is_internship else 'FULL-TIME JOB'}\n"
            f"Job Title: {job_input.get('job_title','')}\n"
            f"Job Role: {job_input.get('job_role','')}\n"
            f"Required Skills: {', '.join(job_input.get('skills_required',[])[:20])}\n"
            f"Description: {(job_input.get('job_description','') or '')[:600]}"
        ).strip()

        prompt = f"CANDIDATE PROFILE:\n{candidate_text}\n\nJOB REQUIREMENT:\n{job_text}"

        system_prompt = SEMANTIC_SYSTEM_PROMPT_INTERNSHIP if is_internship else SEMANTIC_SYSTEM_PROMPT_JOB
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ])
        raw = response.content
        logger.debug(f"[LLMScoringAgent] Semantic raw: {raw[:300]!r}")

        parsed = safe_parse_json(raw, context="semantic_score")
        if not parsed or not isinstance(parsed, dict):
            logger.warning("[LLMScoringAgent] Invalid semantic JSON → default 5.0")
            return SemanticScore(score=5.0, reasoning="LLM output could not be parsed.")

        score = parsed.get("score")
        reasoning = parsed.get("reasoning", "")

        try:
            score = clamp_score(float(score))
        except (TypeError, ValueError):
            logger.warning(f"[LLMScoringAgent] Invalid score value {score!r} → 5.0")
            score = 5.0

        logger.info(f"[LLMScoringAgent] Semantic score={score:.2f} | {reasoning}")
        return SemanticScore(score=score, reasoning=str(reasoning)[:500])

    except Exception as e:
        logger.warning(f"[LLMScoringAgent] Semantic scoring failed: {e}")
        return SemanticScore(score=5.0, reasoning=f"Scoring failed: {e}")


# ─── Sub-component B: Weighted Regression ────────────────────────────────────

def _weighted_regression(
    skill_s:       float,
    experience_s:  float,
    education_s:   float,
    achievement_s: float,
    social_s:      float,
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

def llm_scoring_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 7 — LLM Semantic Scoring + Final Weighted Regression

    Input state keys:  all previous scores + parsed_resume + job_input
    Output state keys: semantic_score, final_score
    """
    agent_name = "LLMScoringAgent"
    log_agent_start(logger, agent_name, {
        "has_skill_score":       bool(state.get("skill_score")),
        "has_experience_score":  bool(state.get("experience_score")),
        "has_education_score":   bool(state.get("education_score")),
        "has_achievement_score": bool(state.get("achievement_score")),
        "has_social_score":      bool(state.get("social_score")),
    })

    errors        = list(state.get("errors") or [])
    parsed_resume = state.get("parsed_resume") or {}
    job_input     = state.get("job_input")     or {}

    is_internship = str(job_input.get("opportunity_type", "job")).lower() == "internship"
    weights = WEIGHTS_INTERNSHIP if is_internship else WEIGHTS_JOB
    opp_label = "internship" if is_internship else "job"

    # Pull scores — default to 3.0 if agent failed/missing
    FALLBACK = 3.0
    skill_s  = float((state.get("skill_score")       or {}).get("score") or FALLBACK)
    exp_s    = float((state.get("experience_score")  or {}).get("score") or FALLBACK)
    edu_s    = float((state.get("education_score")   or {}).get("score") or FALLBACK)
    ach_s    = float((state.get("achievement_score") or {}).get("score") or FALLBACK)
    soc_s    = float((state.get("social_score")      or {}).get("score") or FALLBACK)

    logger.info(
        f"[{agent_name}] opportunity={opp_label} | "
        f"skill={skill_s:.2f} exp={exp_s:.2f} edu={edu_s:.2f} "
        f"ach={ach_s:.2f} soc={soc_s:.2f}"
    )

    # ── Sub-component A: Semantic score ──
    sem_score_obj = _semantic_score(parsed_resume, job_input, is_internship)
    sem_s = sem_score_obj.score
    logger.info(f"[{agent_name}] Semantic score={sem_s:.2f} | opportunity={opp_label}")

    # ── Sub-component B: Weighted regression ──
    total_score, breakdown = _weighted_regression(
        skill_s, exp_s, edu_s, ach_s, soc_s, sem_s,
        weights=weights,
    )
    breakdown["opportunity_type"] = opp_label

    label = _score_label(total_score)
    logger.info(f"[{agent_name}] Final score={total_score:.2f}/100 | {label}")

    # Build FinalScore object
    final_score_obj = FinalScore(
        total_score=total_score,
        breakdown=breakdown,
    )

    log_agent_end(
        logger, agent_name,
        f"FINAL={total_score:.2f}/100 | label='{label}' | "
        f"opportunity={opp_label} | semantic={sem_s:.2f} | "
        f"weights: sk={weights['skill']} ex={weights['experience']} "
        f"edu={weights['education']} ach={weights['achievement']} "
        f"soc={weights['social']} sem={weights['semantic']}"
    )

    return {
        **state,
        "semantic_score": sem_score_obj.model_dump(),
        "final_score":    final_score_obj.model_dump(),
        "errors":         errors,
        "current_step":   agent_name,
    }