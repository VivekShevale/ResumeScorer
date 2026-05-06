"""
agents/achievement_scoring_agent.py
-------------------------------------
Agent 5: Achievement Scoring Agent

Scoring strategy (4 components):

  Component 1 — Achievement Count & Diversity  (rule-based)
      Counts total achievements from:
        - resume.achievements list
        - resume.certifications list
        - resume.projects list (each counts as a minor achievement)
      Diversity bonus: achievements across multiple platforms/types

  Component 2 — Achievement Quality  (LLM scoring, temperature=0)
      For each achievement, LLM rates it 1-10:
        - Competitive programming rank/rating (LeetCode, Codeforces, CodeChef)
        - Hackathon wins / placements
        - Publications / patents
        - Open source contributions
        - Awards / recognitions
        - Certifications from reputed orgs (Google, AWS, Microsoft, etc.)
      Anti-hallucination: LLM only scores from provided text, returns {"score": float}
      Validated: must be float 0-10

  Component 3 — Certification Quality  (rule-based)
      Known high-value certs: AWS, GCP, Azure, Google, Meta, CFA, etc. → higher score
      Generic/unknown → lower score

  Component 4 — Projects as Achievements  (rule-based + count)
      Number of projects with descriptions → signals hands-on work
      Bonus for projects with tech stack listed

Final score formula:
  score = clamp(0.30*count_score + 0.40*quality_score + 0.15*cert_score + 0.15*project_score, 0, 10)

Anti-hallucination:
  - LLM scores validated as float in [0, 10]
  - Temperature = 0
  - LLM only rates achievements given to it — no inference beyond text
  - All scores clamped [0, 10]
"""

from __future__ import annotations
import os
import re
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from models.state import ResumeGraphState, AchievementScore
from utils.logger import get_logger, log_agent_start, log_agent_end, log_agent_error
from utils.validators import safe_parse_json, clamp_score

load_dotenv()
logger = get_logger("agents.achievement_scoring")


# ─── LLM setup ────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env file")
    return ChatGroq(api_key=api_key, model=model, temperature=0, max_tokens=512)


# ─── Component 1: Count & Diversity ───────────────────────────────────────────

def _count_score(
    achievements: List[dict],
    certifications: List[str],
    projects: List[dict],
) -> Tuple[float, int]:
    """
    Score based on sheer number of achievements + certifications + projects.
    Returns (score, total_count).

    Scale:
      0 items  → 0.0
      1-2      → 3.0
      3-4      → 5.0
      5-7      → 7.0
      8-10     → 8.5
      11+      → 10.0
    Diversity bonus (+0.5) if achievements span multiple platforms.
    """
    ach_count  = len([a for a in achievements if a.get("title") or a.get("description")])
    cert_count = len([c for c in certifications if c and c.strip()])
    proj_count = len([p for p in projects if p.get("name")])

    # Projects contribute less (they are already scored partially in experience)
    total = ach_count + cert_count + int(proj_count * 0.5)

    if total == 0:   base = 0.0
    elif total <= 2: base = 3.0
    elif total <= 4: base = 5.0
    elif total <= 7: base = 7.0
    elif total <= 10:base = 8.5
    else:            base = 10.0

    # Diversity: multiple platforms
    platforms = {a.get("platform", "").lower() for a in achievements if a.get("platform")}
    diversity_bonus = 0.5 if len(platforms) >= 2 else 0.0

    score = clamp_score(base + diversity_bonus)
    logger.info(
        f"[AchievementAgent] count: ach={ach_count} cert={cert_count} proj={proj_count} "
        f"→ total={total} score={score:.2f}"
    )
    return score, ach_count + cert_count + proj_count


# ─── Component 2: Achievement Quality (LLM) ───────────────────────────────────

QUALITY_SYSTEM_PROMPT = """You are an achievement evaluator for tech resumes. Score the given list of achievements.

For each achievement, assign a score 0-10 based on:
- Competitive programming: LeetCode rating/rank, Codeforces/CodeChef rating, contest ranks (higher = better)
- Hackathons: Winner > Runner-up > Participation
- Publications / Patents: High value
- Open source: Stars, contributions to major projects
- Awards: Company/national/international recognition
- Speaking: Conferences, tech talks

RULES:
1. Output ONLY valid JSON — no explanation, no preamble, no markdown.
2. Return exactly: {"scores": [<float>, <float>, ...], "overall": <float>}
3. scores array must have SAME LENGTH as input achievements list.
4. Each score must be a float between 0 and 10.
5. overall is the weighted average (emphasise top achievements more).
6. Base scores ONLY on what is written. Do not infer or assume."""


def _llm_quality_score(achievements: List[dict], certifications: List[str]) -> float:
    """
    LLM rates the quality of achievements collectively.
    Returns overall quality score [0, 10].
    Anti-hallucination: validates output structure strictly.
    """
    if not achievements and not certifications:
        return 0.0

    # Build a clean text list for LLM
    items = []
    for a in achievements:
        title = a.get("title", "")
        desc  = a.get("description", "")
        plat  = a.get("platform", "")
        text  = " — ".join(filter(None, [title, desc, plat]))
        if text.strip():
            items.append(text.strip())

    for c in certifications:
        if c and c.strip():
            items.append(f"Certification: {c.strip()}")

    if not items:
        return 0.0

    try:
        llm = _get_llm()
        prompt = (
            f"Achievements list ({len(items)} items):\n" +
            "\n".join(f"{i+1}. {item}" for i, item in enumerate(items[:20]))  # cap at 20
        )
        response = llm.invoke([
            SystemMessage(content=QUALITY_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = response.content
        logger.debug(f"[AchievementAgent] LLM quality raw: {raw[:300]!r}")

        parsed = safe_parse_json(raw, context="achievement_quality")
        if not parsed or not isinstance(parsed, dict):
            logger.warning("[AchievementAgent] Invalid quality JSON → default 3.0")
            return 3.0

        # Validate scores array length
        scores = parsed.get("scores", [])
        overall = parsed.get("overall")

        # Validate individual scores
        if isinstance(scores, list) and len(scores) == len(items[:20]):
            valid_scores = []
            for s in scores:
                try:
                    v = float(s)
                    valid_scores.append(clamp_score(v))
                except (TypeError, ValueError):
                    valid_scores.append(3.0)
        else:
            valid_scores = []

        # Validate overall
        if overall is not None:
            try:
                result = clamp_score(float(overall))
                logger.info(f"[AchievementAgent] LLM quality overall={result:.2f}")
                return result
            except (TypeError, ValueError):
                pass

        # Fallback: average of individual scores
        if valid_scores:
            result = clamp_score(sum(valid_scores) / len(valid_scores))
            logger.info(f"[AchievementAgent] Quality from avg of scores={result:.2f}")
            return result

        return 3.0

    except Exception as e:
        logger.warning(f"[AchievementAgent] LLM quality scoring failed: {e}")
        return 3.0


# ─── Component 3: Certification Quality ──────────────────────────────────────

# High-value certification keywords → score
CERT_QUALITY_MAP = [
    (["aws certified", "amazon web services", "aws solutions", "aws developer", "aws sysops"], 9.5),
    (["google cloud", "gcp certified", "associate cloud engineer", "professional cloud"], 9.5),
    (["azure certified", "microsoft certified", "az-900", "az-104", "az-204", "dp-900"], 9.0),
    (["google professional", "tensorflow certificate", "google associate"], 8.5),
    (["certified kubernetes", "cka", "ckad", "cks"], 9.0),
    (["meta certified", "facebook certified"], 8.0),
    (["oracle certified", "java certified", "ocp", "oca"], 8.0),
    (["comptia", "security+", "network+", "a+"], 7.5),
    (["cisco", "ccna", "ccnp", "ccie"], 8.5),
    (["pmp", "project management professional"], 8.0),
    (["cfa", "chartered financial analyst"], 9.0),
    (["coursera", "udemy", "edx", "linkedin learning"], 4.0),
    (["nptel", "swayam"], 5.0),
]


def _certification_quality_score(certifications: List[str]) -> float:
    """
    Rule-based scoring of certifications.
    Returns [0, 10] — average of best certs found.
    """
    if not certifications:
        return 0.0

    cert_scores = []
    for cert in certifications:
        if not cert or not cert.strip():
            continue
        cl = cert.lower()
        matched = False
        for keywords, score in CERT_QUALITY_MAP:
            if any(kw in cl for kw in keywords):
                cert_scores.append(score)
                matched = True
                break
        if not matched:
            cert_scores.append(5.0)  # unknown cert → neutral

    if not cert_scores:
        return 0.0

    # Weight towards top certs
    cert_scores.sort(reverse=True)
    if len(cert_scores) == 1:
        return clamp_score(cert_scores[0])
    # Weighted: best cert 60%, rest averaged 40%
    rest_avg = sum(cert_scores[1:]) / len(cert_scores[1:])
    score = 0.60 * cert_scores[0] + 0.40 * rest_avg
    logger.info(f"[AchievementAgent] Cert quality → {score:.2f} ({len(cert_scores)} certs)")
    return clamp_score(score)


# ─── Component 4: Projects Score ─────────────────────────────────────────────

def _project_score(projects: List[dict]) -> float:
    """
    Score based on project count and description richness.
    Returns [0, 10].
    """
    if not projects:
        return 0.0

    scored = []
    for p in projects:
        name = p.get("name", "")
        desc = p.get("description", "")
        tech = p.get("technologies", [])

        if not name:
            continue

        base = 5.0  # has a name
        if desc and len(desc) > 30:
            base += 2.0   # has meaningful description
        if tech and len(tech) >= 2:
            base += 1.5   # has tech stack
        if len(desc or "") > 100:
            base += 1.5   # detailed description

        scored.append(min(10.0, base))

    if not scored:
        return 0.0

    # More projects = better, but diminishing returns
    count_bonus = min(2.0, len(scored) * 0.4)
    avg = sum(scored) / len(scored)
    result = clamp_score(avg * 0.8 + count_bonus)
    logger.info(f"[AchievementAgent] Project score → {result:.2f} ({len(scored)} projects)")
    return result


# ─── Final score calculation ──────────────────────────────────────────────────

def _calculate_achievement_score(
    count_score:   float,
    quality_score: float,
    cert_score:    float,
    project_score: float,
) -> float:
    """
    Weights:
      30% count + diversity
      40% achievement quality (LLM)
      15% certification quality
      15% project score
    """
    raw = (0.30 * count_score + 0.40 * quality_score +
           0.15 * cert_score  + 0.15 * project_score)
    return clamp_score(raw)


# ─── Main agent function ──────────────────────────────────────────────────────

def achievement_scoring_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 5 — Achievement Scoring

    Input state keys:  parsed_resume, job_input
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

    logger.info(
        f"[{agent_name}] achievements={len(achievements)} "
        f"certs={len(certifications)} projects={len(projects)}"
    )

    # ── Edge case: nothing to score ──
    if not achievements and not certifications and not projects:
        logger.warning(f"[{agent_name}] Nothing to score → 1.0")
        ach_score = AchievementScore(
            score=1.0,
            achievement_count=0,
            reasoning="No achievements, certifications, or projects found in resume.",
        )
        return {
            **state,
            "achievement_score": ach_score.model_dump(),
            "errors": errors,
            "current_step": agent_name,
        }

    # ── Component 1: Count ──
    count_sc, total_count = _count_score(achievements, certifications, projects)

    # ── Component 2: LLM Quality ──
    quality_sc = _llm_quality_score(achievements, certifications)

    # ── Component 3: Cert Quality ──
    cert_sc = _certification_quality_score(certifications)

    # ── Component 4: Projects ──
    proj_sc = _project_score(projects)

    # ── Final ──
    final_score = _calculate_achievement_score(count_sc, quality_sc, cert_sc, proj_sc)

    reasoning = (
        f"Achievements: {len(achievements)} | Certifications: {len(certifications)} | "
        f"Projects: {len(projects)} | Total items: {total_count}. "
        f"Count score: {count_sc:.2f}/10. "
        f"Quality score: {quality_sc:.2f}/10. "
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