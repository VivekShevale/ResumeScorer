"""
agents/project_scoring_agent.py
---------------------------------
Agent 6b: Project Scoring Agent

Evaluates the candidate's projects in two dimensions:

  Dimension 1 — Project Quality  (LLM, temperature=0)
      LLM rates each project holistically:
        - Complexity (simple CRUD vs distributed system)
        - Impact/scale (personal tool vs production system)
        - Technical depth (generic tutorials vs original work)
        - Domain relevance to the job role
      Returns per-project scores (0-10) + overall quality score.

  Dimension 2 — Skill Match in Projects  (keyword + cosine, no LLM)
      For each project, compare project.technologies against required_skills.
      Measures: how many required skills appear in actual project work.
      This is stronger than listing a skill — it means the candidate
      has *used* the skill in a real project context.

      Formula:
        tech_match_ratio = required_techs_in_projects / total_required_skills
        tech_match_score = clamp(tech_match_ratio * 10, 0, 10)

Final score formula:
  project_score = clamp(0.55 * quality_score + 0.45 * tech_match_score, 0, 10)

Anti-hallucination:
  - LLM output validated: scores must be floats in [0, 10]
  - Temperature = 0
  - LLM only rates projects given to it — no inference beyond text
"""

from __future__ import annotations
import os
import re
from typing import List, Tuple, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from models.state import ResumeGraphState, ProjectScore
from utils.logger import get_logger, log_agent_start, log_agent_end, log_agent_error
from utils.validators import safe_parse_json, clamp_score

load_dotenv()
logger = get_logger("agents.project_scoring")


# ─── LLM setup ────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env file")
    return ChatGroq(api_key=api_key, model=model, temperature=0, max_tokens=1024)


# ─── Dimension 1: LLM Project Quality ────────────────────────────────────────

PROJECT_QUALITY_PROMPT = """You are a senior software engineer evaluating candidate projects for a job application.

SCORING CRITERIA per project (0-10):
  9-10: Production-grade / open-source tool / complex distributed system / significant impact
  7-8:  Well-built project with clear technical depth, multiple integrations, deployed
  5-6:  Solid personal project with a reasonable tech stack, some complexity
  3-4:  Basic tutorial-level CRUD app or generic to-do list / blog
  1-2:  Trivial project with minimal effort
  0:    No meaningful project content

RULES:
1. Output ONLY valid JSON — no explanation, no markdown, no preamble.
2. Rate each project by its zero-indexed position in the list.
3. Also give an "overall" score (average with quality bias, not just mean).
4. Never invent project details not in the provided text.

Return EXACTLY this JSON:
{
  "project_scores": [<score_for_project_0>, <score_for_project_1>, ...],
  "overall": <float 0-10>,
  "reasoning": "<one sentence summary>"
}"""


def _llm_project_quality(projects: List[dict], job_title: str, job_description: str) -> Tuple[float, List[float], str]:
    """
    Returns (overall_score, per_project_scores, reasoning).
    Falls back to rule-based if LLM fails.
    """
    if not projects:
        return 0.0, [], "No projects found."

    # Build project list for LLM
    project_texts = []
    for i, p in enumerate(projects[:10]):  # cap at 10
        name = p.get("name") or f"Project {i+1}"
        desc = p.get("description") or ""
        techs = ", ".join(p.get("technologies") or [])
        ptype = p.get("type") or ""
        text = f"Project {i+1}: {name}"
        if ptype:
            text += f" [{ptype}]"
        if techs:
            text += f"\n  Technologies: {techs}"
        if desc:
            text += f"\n  Description: {desc[:300]}"
        project_texts.append(text)

    prompt = (
        f"Job Role: {job_title}\n"
        f"Job Description (excerpt): {job_description[:500]}\n\n"
        f"Projects to evaluate:\n" + "\n\n".join(project_texts)
    )

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=PROJECT_QUALITY_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = response.content
        logger.debug(f"[ProjectScoringAgent] LLM raw: {raw[:300]!r}")

        parsed = safe_parse_json(raw, context="project_quality")
        if not parsed or not isinstance(parsed, dict):
            logger.warning("[ProjectScoringAgent] Invalid LLM JSON → rule-based fallback")
            return _rule_based_quality(projects)

        overall = parsed.get("overall")
        scores_raw = parsed.get("project_scores", [])
        reasoning = str(parsed.get("reasoning", ""))[:300]

        # Validate overall
        try:
            overall = clamp_score(float(overall))
        except (TypeError, ValueError):
            overall = None

        # Validate per-project scores
        per_scores = []
        for s in scores_raw:
            try:
                per_scores.append(clamp_score(float(s)))
            except (TypeError, ValueError):
                per_scores.append(3.0)

        if overall is None:
            overall = (sum(per_scores) / len(per_scores)) if per_scores else 3.0

        logger.info(f"[ProjectScoringAgent] LLM quality: overall={overall:.2f} | {reasoning}")
        return clamp_score(overall), per_scores, reasoning

    except Exception as e:
        logger.warning(f"[ProjectScoringAgent] LLM failed: {e} → rule-based fallback")
        return _rule_based_quality(projects)


def _rule_based_quality(projects: List[dict]) -> Tuple[float, List[float], str]:
    """Simple rule-based fallback when LLM fails."""
    if not projects:
        return 0.0, [], "No projects."
    scores = []
    for p in projects:
        score = 3.0
        if p.get("description") and len(p.get("description", "")) > 100:
            score += 1.5
        if p.get("technologies") and len(p.get("technologies", [])) >= 3:
            score += 1.5
        if p.get("type") and p["type"].lower() in ("web app", "mobile app", "ml", "api", "fullstack"):
            score += 1.0
        scores.append(clamp_score(score))
    overall = sum(scores) / len(scores) if scores else 0.0
    return clamp_score(overall), scores, "Rule-based fallback (LLM unavailable)."


# ─── Dimension 2: Skill Match in Projects ────────────────────────────────────

def _normalize(skill: str) -> str:
    """Normalize skill for comparison."""
    s = skill.lower().strip()
    s = re.sub(r"[\s\-_\.]+", "", s)
    return s


def _tech_match_score(projects: List[dict], required_skills: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Checks how many required skills appear in actual project technologies.
    Uses keyword matching + normalization.

    Returns (score, skills_used_in_projects, required_skills_missing_from_projects).
    """
    if not required_skills:
        return 5.0, [], []

    # Collect all technologies used across all projects
    all_project_techs = []
    for p in projects:
        for tech in (p.get("technologies") or []):
            if tech and tech.strip():
                all_project_techs.append(tech.strip())

    if not all_project_techs:
        return 0.0, [], required_skills[:]

    # Normalize
    project_tech_norms = {_normalize(t): t for t in all_project_techs}
    required_norm = [(s, _normalize(s)) for s in required_skills]

    matched_in_projects = []
    missing_in_projects = []

    for req_orig, req_norm in required_norm:
        # Direct match
        if req_norm in project_tech_norms:
            matched_in_projects.append(req_orig)
            continue
        # Partial: required skill is substring of a project tech or vice versa
        found = False
        for pnorm in project_tech_norms:
            if req_norm in pnorm or pnorm in req_norm:
                matched_in_projects.append(req_orig)
                found = True
                break
        if not found:
            missing_in_projects.append(req_orig)

    match_ratio = len(matched_in_projects) / len(required_skills)
    score = clamp_score(match_ratio * 10)

    logger.info(
        f"[ProjectScoringAgent] Tech match: {len(matched_in_projects)}/{len(required_skills)} "
        f"required skills found in projects → {score:.2f}"
    )
    return score, matched_in_projects, missing_in_projects


# ─── Final score calculation ──────────────────────────────────────────────────

def _calculate_project_score(quality_score: float, tech_match_score: float) -> float:
    """
    55% project quality (depth, complexity, relevance)
    45% skill usage in projects (are required skills actually used?)
    """
    raw = 0.55 * quality_score + 0.45 * tech_match_score
    return clamp_score(raw)


# ─── Main agent function ──────────────────────────────────────────────────────

def project_scoring_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 6b — Project Scoring

    Evaluates project quality (LLM) and how many required skills
    are actually used in projects (keyword match — no extra LLM call).

    Input state keys:  parsed_resume, job_input
    Output state keys: project_score
    """
    agent_name = "ProjectScoringAgent"
    log_agent_start(logger, agent_name, {
        "has_parsed_resume": bool(state.get("parsed_resume")),
        "has_job_input": bool(state.get("job_input")),
    })

    errors        = list(state.get("errors") or [])
    parsed_resume = state.get("parsed_resume") or {}
    job_input     = state.get("job_input")     or {}

    projects:        List[dict] = parsed_resume.get("projects")       or []
    required_skills: List[str]  = job_input.get("skills_required")    or []
    job_title:       str        = job_input.get("job_title", "")
    job_description: str        = job_input.get("job_description", "")

    logger.info(
        f"[{agent_name}] projects={len(projects)} | "
        f"required_skills={len(required_skills)}"
    )

    # Edge case: no projects
    if not projects:
        logger.warning(f"[{agent_name}] No projects found → score=1.0")
        proj_score = ProjectScore(
            score=1.0,
            project_count=0,
            quality_score=0.0,
            tech_match_score=0.0,
            skills_used_in_projects=[],
            required_skills_missing_from_projects=required_skills[:],
            reasoning="No projects found in resume.",
        )
        return {**state, "project_score": proj_score.model_dump(),
                "errors": errors, "current_step": agent_name}

    # Dimension 1: LLM Project Quality
    quality_sc, per_scores, quality_reasoning = _llm_project_quality(
        projects, job_title, job_description
    )
    logger.info(f"[{agent_name}] Quality score: {quality_sc:.2f}/10")

    # Dimension 2: Skill Match in Projects (no LLM)
    tech_sc, skills_used, skills_missing_from_projects = _tech_match_score(
        projects, required_skills
    )
    logger.info(f"[{agent_name}] Tech match score: {tech_sc:.2f}/10")

    # Final
    final_score = _calculate_project_score(quality_sc, tech_sc)

    reasoning = (
        f"Projects: {len(projects)} | Quality score: {quality_sc:.2f}/10. {quality_reasoning} "
        f"Required skills found in project tech stacks: {len(skills_used)}/{len(required_skills)}. "
        f"Tech match score: {tech_sc:.2f}/10. "
        f"Skills used in projects: {', '.join(skills_used[:10]) or 'none'}. "
        f"Weights: 55% quality + 45% tech match."
    )

    proj_score = ProjectScore(
        score=final_score,
        project_count=len(projects),
        quality_score=quality_sc,
        tech_match_score=tech_sc,
        skills_used_in_projects=skills_used,
        required_skills_missing_from_projects=skills_missing_from_projects,
        reasoning=reasoning,
    )

    log_agent_end(
        logger, agent_name,
        f"score={final_score:.2f}/10 | quality={quality_sc:.2f} tech_match={tech_sc:.2f} "
        f"skills_in_projects={len(skills_used)}/{len(required_skills)}"
    )

    return {
        **state,
        "project_score": proj_score.model_dump(),
        "errors": errors,
        "current_step": agent_name,
    }
