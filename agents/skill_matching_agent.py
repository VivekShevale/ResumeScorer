"""
agents/skill_matching_agent.py
--------------------------------
Agent 2: Skill Matching Agent

Scoring strategy (hybrid — 3 layers):
  Layer 1 — Exact / normalized match  (fast, no LLM)
      Normalize both sides: lowercase, strip spaces/dots/dashes
      Direct set intersection → gives base matched/missing lists

  Layer 2 — Cosine similarity on TF-IDF vectors  (no LLM)
      sklearn TfidfVectorizer (char n-grams) + cosine_similarity
      Handles partial matches like "ReactJS" <-> "React.js"

  Layer 3 — LLM semantic expansion  (Groq, temperature=0)
      LLM only CONFIRMS matches from the provided lists — never invents new ones.
      Constrained output validated against input before accepting.

Final score formula:
  score = clamp(0.5*base + 0.3*cosine + 0.2*semantic_bonus, 0, 10)
  where base = exact_matched/total_required  [0-1]
        cosine = cosine_similarity            [0-1]
        semantic_bonus = semantic/total_req   [0-1]

Anti-hallucination:
  - LLM output validated: every skill name must be subset of input lists
  - Temperature = 0
  - Score always clamped [0, 10]
"""

from __future__ import annotations
import os
import re
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from models.state import ResumeGraphState, SkillScore
from utils.logger import get_logger, log_agent_start, log_agent_end, log_agent_error
from utils.validators import safe_parse_json, clamp_score

load_dotenv()
logger = get_logger("agents.skill_matching")


# ─── LLM setup ────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env file")
    return ChatGroq(api_key=api_key, model=model, temperature=0, max_tokens=1024)


# ─── Normalizer ───────────────────────────────────────────────────────────────

def _normalize(skill: str) -> str:
    """'React.js' -> 'reactjs', 'Node JS' -> 'nodejs', 'C++' -> 'c++'"""
    s = skill.lower().strip()
    s = re.sub(r"[\s\-_\.]+", "", s)
    return s


def _normalize_list(skills: List[str]) -> List[Tuple[str, str]]:
    """Returns list of (original, normalized) tuples."""
    return [(s, _normalize(s)) for s in skills if s and s.strip()]


# ─── Layer 1: Exact / Normalized match ────────────────────────────────────────

def _exact_match(
    resume_skills: List[str],
    required_skills: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
        matched          - required skills found in resume
        missing          - required skills NOT found in resume
        unmatched_resume - resume skills that didn't match any required skill
    """
    resume_norm = {norm: orig for orig, norm in _normalize_list(resume_skills)}
    required_norm = _normalize_list(required_skills)

    matched = []
    missing = []

    for req_orig, req_norm in required_norm:
        if req_norm in resume_norm:
            matched.append(req_orig)
        else:
            missing.append(req_orig)

    required_norm_set = {n for _, n in required_norm}
    unmatched_resume = [
        orig for orig, norm in _normalize_list(resume_skills)
        if norm not in required_norm_set
    ]

    return matched, missing, unmatched_resume


# ─── Layer 2: TF-IDF Cosine Similarity ────────────────────────────────────────

def _cosine_skill_score(resume_skills: List[str], required_skills: List[str]) -> float:
    """Cosine similarity between resume skills bag and required skills bag. Returns [0, 1]."""
    if not resume_skills or not required_skills:
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        resume_doc = " ".join(_normalize(s) for s in resume_skills)
        required_doc = " ".join(_normalize(s) for s in required_skills)

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        tfidf = vectorizer.fit_transform([resume_doc, required_doc])
        score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning(f"Cosine similarity failed: {e}")
        return 0.0


# ─── Layer 3: LLM Semantic expansion ──────────────────────────────────────────

SEMANTIC_SYSTEM_PROMPT = """You are a technical skills matcher. Your ONLY job is to find which "unmatched resume skills" are semantically equivalent to "missing required skills".

STRICT RULES:
1. Output ONLY valid JSON — no explanation, no markdown, no preamble.
2. You can ONLY use skill names EXACTLY as they appear in the provided lists. Never invent new names.
3. A match means truly equivalent skills (e.g., "Express.js" and "ExpressJS", "Postgres" and "PostgreSQL").
4. Do NOT match loosely related skills (e.g., "Python" does NOT match "Machine Learning").
5. If no semantic matches exist, return {"semantic_matches": []}.

Return ONLY this JSON:
{
  "semantic_matches": [
    {"required": "<exact name from missing required list>", "resume": "<exact name from unmatched resume list>"}
  ]
}"""


def _llm_semantic_match(unmatched_resume: List[str], missing_required: List[str]) -> List[str]:
    """
    Returns list of required skills that are semantically covered by resume skills.
    Anti-hallucination: validates all returned names are strict subsets of input lists.
    """
    if not unmatched_resume or not missing_required:
        return []

    unmatched_sample = unmatched_resume[:30]
    missing_sample = missing_required[:30]

    try:
        llm = _get_llm()
        prompt = (
            f"Unmatched resume skills: {unmatched_sample}\n\n"
            f"Missing required skills: {missing_sample}\n\n"
            "Find semantic equivalences only."
        )
        response = llm.invoke([
            SystemMessage(content=SEMANTIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = response.content
        logger.debug(f"[SkillMatchingAgent] LLM semantic raw: {raw[:300]!r}")

        parsed = safe_parse_json(raw, context="skill_semantic")
        if not parsed or not isinstance(parsed, dict):
            return []

        matches = parsed.get("semantic_matches", [])
        if not isinstance(matches, list):
            return []

        # Anti-hallucination: both sides must exist in provided input lists
        valid_required = {_normalize(s): s for s in missing_sample}
        valid_resume   = {_normalize(s): s for s in unmatched_sample}

        semantic_matched = []
        for m in matches:
            if not isinstance(m, dict):
                continue
            req = m.get("required", "")
            res = m.get("resume", "")
            if _normalize(req) in valid_required and _normalize(res) in valid_resume:
                semantic_matched.append(valid_required[_normalize(req)])
            else:
                logger.warning(
                    f"[SkillMatchingAgent] Rejected hallucinated pair: "
                    f"required={req!r} resume={res!r}"
                )
        return semantic_matched

    except Exception as e:
        logger.warning(f"[SkillMatchingAgent] LLM semantic match failed: {e}")
        return []


# ─── Score calculation ────────────────────────────────────────────────────────

def _calculate_score(
    total_required: int,
    exact_matched: int,
    semantic_matched: int,
    cosine: float,
    total_resume_skills: int = 0,
) -> float:
    """
    Scoring rules:
      - All required skills matched → minimum 9.0
      - All required matched + extra relevant resume skills → up to 10.0
      - Partial match → weighted formula (50% exact, 30% cosine, 20% semantic)

    Guarantees:
      full match (all required covered)       → score >= 9.0
      full match + more relevant extras       → score up to 10.0
      partial match                           → proportional 0–8.9
    """
    if total_required == 0:
        return clamp_score(5.0 + cosine * 5.0)

    total_matched = exact_matched + semantic_matched
    match_ratio = total_matched / total_required  # 0.0 – 1.0+ (can exceed 1.0 if semantic overlaps)
    match_ratio = min(match_ratio, 1.0)

    # ── Full match path ──
    if total_matched >= total_required:
        base_score = 9.0
        # Bonus for extra skills beyond required (up to +1.0 → score 10.0)
        extra_skills = max(0, total_resume_skills - total_required)
        extra_bonus = min(extra_skills / max(total_required, 1) * 1.0, 1.0)
        # Also reward cosine similarity (more similar = wider/deeper coverage)
        cosine_bonus = cosine * 0.5
        score = base_score + min(extra_bonus + cosine_bonus, 1.0)
        return clamp_score(score)

    # ── Partial match path ──
    base           = exact_matched / total_required
    semantic_bonus = semantic_matched / total_required
    raw = (0.5 * base + 0.3 * cosine + 0.2 * semantic_bonus) * 10
    # Cap partial matches at 8.9 to maintain the 9.0 guarantee for full match
    return clamp_score(min(raw, 8.9))


# ─── Main agent function ──────────────────────────────────────────────────────

def skill_matching_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 2 — Skill Matching

    Scoring strategy (3 layers):
      Layer 1 — Exact/normalized keyword match (no LLM)
      Layer 2 — TF-IDF cosine similarity (no LLM)
      Layer 3 — LLM semantic expansion (ONLY for skills still unmatched after L1+L2)
                → Already-matched skills are NOT sent to LLM, minimising token cost.

    Score guarantees:
      All required skills matched             → score >= 9.0
      All required matched + extra relevant   → score up to 10.0

    Input state keys:  parsed_resume, job_input
    Output state keys: skill_score
    """
    agent_name = "SkillMatchingAgent"
    log_agent_start(logger, agent_name, {
        "has_parsed_resume": bool(state.get("parsed_resume")),
        "has_job_input": bool(state.get("job_input")),
    })

    errors = list(state.get("errors") or [])
    parsed_resume  = state.get("parsed_resume") or {}
    job_input      = state.get("job_input")     or {}

    resume_skills: List[str]   = parsed_resume.get("skills")      or []
    required_skills: List[str] = job_input.get("skills_required") or []

    logger.info(
        f"[{agent_name}] resume_skills={len(resume_skills)} | "
        f"required_skills={len(required_skills)}"
    )

    # Edge case: nothing on either side
    if not resume_skills and not required_skills:
        logger.warning(f"[{agent_name}] No skills found anywhere — score=3.0")
        skill_score = SkillScore(
            score=3.0, matched_skills=[], missing_skills=[],
            reasoning="No skills found in resume or job requirements.",
        )
        return {**state, "skill_score": skill_score.model_dump(),
                "errors": errors, "current_step": agent_name}

    # ── Layer 1: Exact / Normalized match ──
    exact_matched, missing_after_exact, unmatched_resume = _exact_match(
        resume_skills, required_skills
    )
    logger.info(
        f"[{agent_name}] L1 exact: matched={len(exact_matched)} "
        f"missing={len(missing_after_exact)} unmatched_resume={len(unmatched_resume)}"
    )

    # ── Layer 2: Cosine similarity (full skill sets) ──
    cosine = _cosine_skill_score(resume_skills, required_skills)
    logger.info(f"[{agent_name}] L2 cosine={cosine:.3f}")

    # ── Layer 3: LLM Semantic match ──
    # OPTIMISATION: Only pass UNMATCHED skills to LLM.
    # Already-matched skills (exact_matched) are stripped from both sides.
    # This minimises tokens and avoids redundant LLM work.
    semantic_matched = []
    if missing_after_exact and unmatched_resume:
        logger.info(
            f"[{agent_name}] L3 LLM semantic: "
            f"sending {len(unmatched_resume)} unmatched resume skills vs "
            f"{len(missing_after_exact)} still-missing required skills (matched skills excluded)"
        )
        semantic_matched = _llm_semantic_match(unmatched_resume, missing_after_exact)
        logger.info(f"[{agent_name}] L3 semantic extra={len(semantic_matched)}")
    else:
        logger.info(f"[{agent_name}] L3 LLM skipped — no unmatched skills remaining")

    # ── Compute final missing after all layers ──
    semantic_norm = {_normalize(s) for s in semantic_matched}
    final_missing = [s for s in missing_after_exact if _normalize(s) not in semantic_norm]

    # ── Score (with full-match guarantee) ──
    score = _calculate_score(
        total_required=len(required_skills),
        exact_matched=len(exact_matched),
        semantic_matched=len(semantic_matched),
        cosine=cosine,
        total_resume_skills=len(resume_skills),
    )

    total_matched = len(exact_matched) + len(semantic_matched)
    full_match = total_matched >= len(required_skills)
    reasoning = (
        f"Resume skills: {len(resume_skills)} | Required: {len(required_skills)} | "
        f"Exact matches: {len(exact_matched)} | Semantic matches: {len(semantic_matched)} | "
        f"Total matched: {total_matched} | Cosine similarity: {cosine:.2f} | "
        f"Still missing: {len(final_missing)} | "
        f"{'FULL MATCH → score guaranteed ≥9.0' if full_match else 'Partial match'}"
    )

    skill_score = SkillScore(
        score=score,
        matched_skills=exact_matched + semantic_matched,
        missing_skills=final_missing,
        reasoning=reasoning,
    )

    log_agent_end(
        logger, agent_name,
        f"score={score:.2f}/10 | matched={total_matched}/{len(required_skills)} "
        f"| missing={len(final_missing)} | cosine={cosine:.3f} | full_match={full_match}"
    )

    return {
        **state,
        "skill_score": skill_score.model_dump(),
        "errors": errors,
        "current_step": agent_name,
    }