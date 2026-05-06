"""
agents/education_scoring_agent.py
-----------------------------------
Agent 4: Education Scoring Agent

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

  Component 2 — Institution Tier  (LLM classification, temperature=0)
      Classify institution into:
        Tier 1 (10.0): IITs, IISc, NITs (top), BITS Pilani, top global unis
                       (MIT, Stanford, CMU, Oxford, Cambridge, etc.)
        Tier 2 (6.5):  State universities, decent private colleges, known institutes
        Tier 3 (3.5):  Unknown/local colleges, unaccredited institutions
      Anti-hallucination: output must be exactly {"tier": 1|2|3}

  Component 3 — Field Relevance  (cosine similarity, no LLM)
      Cosine similarity between candidate's field/degree text
      and job_title + job_description.
      Score [0, 10].

  Component 4 — GPA / Academic Performance  (rule-based)
      Normalise GPA to 0-10 scale:
        10-point scale (e.g. 8.7/10) → direct mapping
        4-point scale  (e.g. 3.5/4)  → multiply by 2.5
        Percentage     (e.g. 85%)    → divide by 10
      No GPA → neutral score (5.0)

Final score formula:
  score = clamp(0.35*degree + 0.30*tier + 0.20*relevance + 0.15*gpa, 0, 10)

Anti-hallucination:
  - Institution tier: JSON {"tier": int 1/2/3} only — strictly validated
  - Temperature = 0
  - GPA: regex extracted, never trusted from LLM
  - All scores clamped [0, 10]
"""

from __future__ import annotations
import os
import re
from typing import List, Optional, Tuple
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from models.state import ResumeGraphState, EducationScore
from utils.logger import get_logger, log_agent_start, log_agent_end, log_agent_error
from utils.validators import safe_parse_json, clamp_score

load_dotenv()
logger = get_logger("agents.education_scoring")


# ─── LLM setup ────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env file")
    return ChatGroq(api_key=api_key, model=model, temperature=0, max_tokens=256)


# ─── Component 1: Degree Level ────────────────────────────────────────────────

# Ordered from highest to lowest — first match wins
DEGREE_LEVELS = [
    (["phd", "ph.d", "doctorate", "doctor of"], 5, 10.0),
    (["m.tech", "mtech", "m.e.", "m.e ", "m.s", "msc", "m.sc", "master", "mba", "m.b.a", "pgdm", "pg diploma"], 4, 8.5),
    (
        # Standard bachelor degree forms
        ["b.tech", "btech", "b.e.", "b.e ", " be ", "b.s", "bsc", "b.sc", "bachelor", "b.a", "bca", "bcom", "b.com",
         # Indian 4-year engineering programs often listed as just the branch
         "computer science", "cse", "it ", "information technology",
         "electronics", "mechanical", "civil", "electrical",
         "aiml", "ai-ml", "ai & ml", "ai and ml",
         "data science", "artificial intelligence",
         # Common shorthand forms without "B.Tech" prefix
         "engineering and technology", "institute of technology",
        ],
        3, 7.0
    ),
    (["diploma", "associate", "polytechnic"], 2, 5.0),
    (["higher secondary", "hsc", "12th", "class xii", "intermediate", "senior secondary", "std 12", "std. 12"], 1, 3.0),
    (["ssc", "10th", "class x", "matriculation", "secondary school", "std 10", "std. 10"], 0, 1.5),
]

REQUIREMENT_LEVEL_MAP = {
    "phd": 5, "doctorate": 5,
    "master": 4, "mtech": 4, "msc": 4, "mba": 4, "ms": 4,
    "bachelor": 3, "btech": 3, "bsc": 3, "be": 3, "graduate": 3,
    "diploma": 2,
    "high school": 1, "hsc": 1, "12th": 1,
}


def _degree_to_level(degree_str: Optional[str]) -> Tuple[int, float, str]:
    """
    Returns (level_int, score, canonical_name).
    level_int: 0-5 (higher = more qualified)

    Handles Indian resume formats where degree may be written as:
      "CSE (AIML)", "Computer Science Engineering", "B.E", "B.Tech CS",
      "Higher Secondary", "HSC 12th", etc.
    """
    if not degree_str:
        return 2, 4.0, "Unknown"

    d = degree_str.lower().strip()

    # First pass: exact keyword matching
    for keywords, level, score in DEGREE_LEVELS:
        if any(kw in d for kw in keywords):
            return level, score, degree_str

    # Second pass: if the string looks like an engineering/CS branch name
    # without explicit degree prefix, assume it's a bachelor-level degree
    branch_signals = [
        "computer", "software", "electronics", "mechanical", "civil",
        "chemical", "biotechnology", "information", "science", "engineering",
        "technology", "mathematics", "physics", "statistics",
    ]
    if any(sig in d for sig in branch_signals):
        return 3, 7.0, degree_str

    return 2, 4.0, degree_str  # default: treated like diploma


def _requirement_level(education_required: Optional[str]) -> int:
    """Parse the education_required string into a level integer."""
    if not education_required:
        return 0  # no requirement

    r = education_required.lower()
    for key, level in REQUIREMENT_LEVEL_MAP.items():
        if key in r:
            return level
    return 0


def _degree_score(
    education_list: List[dict],
    education_required: Optional[str],
) -> Tuple[float, str, str]:
    """
    Returns (score, highest_degree_str, reasoning_fragment).
    Picks the highest degree from all education entries.
    Compares against requirement — penalises if below.
    """
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
        # No requirement — use raw degree score
        final_score = best_score
        note = f"No degree requirement specified."
    elif best_level >= req_level:
        # Meets or exceeds requirement
        final_score = best_score
        note = f"Meets/exceeds requirement ({education_required})."
    else:
        # Below requirement — penalise
        penalty = (req_level - best_level) * 1.5
        final_score = max(1.0, best_score - penalty)
        note = f"Below required level ({education_required}). Penalty applied."

    return clamp_score(final_score), best_degree, note


# ─── Component 2: Institution Tier ────────────────────────────────────────────

INSTITUTION_TIER_PROMPT = """You are an academic institution classifier for the tech/engineering industry.
Classify the given institution into one of three tiers.

Tier 1 (score 10): IITs, IISc, NITs (top 10), BITS Pilani, IIIT Hyderabad, top global universities
  (MIT, Stanford, CMU, Oxford, Cambridge, UCB, ETH Zurich, NUS, etc.)

Tier 2 (score 6.5): Good private engineering colleges, known state universities, deemed universities,
  IIIT branches (all), NIT lower-ranked, VIT, Manipal, SRM, NMIMS, LJ Institute of Technology, 
  New LJ Institute of Engineering and Technology, LJ University, Nirma University, DAIICT, PDEU,
  Charotar University (CHARUSAT), Silver Oak University, Parul University, GLS University,
  any GTU-affiliated college known for engineering, any NAAC A or A+ accredited college,
  any college with established placement record in tech.

Tier 3 (score 3.5): Unknown/local colleges, unaccredited institutions, very small regional colleges.

RULES:
1. Output ONLY valid JSON — no explanation, no markdown, no preamble.
2. Return EXACTLY: {"tier": 1} or {"tier": 2} or {"tier": 3}
3. When uncertain about an Indian engineering college, DEFAULT TO TIER 2.
4. High schools / secondary schools / 12th standard schools → always Tier 3.
5. Any college with "Institute of Technology" or "Engineering" in the name → at least Tier 2."""


def _classify_institution_tier(institution: str) -> Tuple[int, float]:
    """
    Returns (tier, score).
    Anti-hallucination: accepts only {"tier": 1|2|3}.
    Defaults to tier 2 on failure.
    """
    if not institution or institution.strip().lower() in ("unknown", ""):
        return 3, 3.5

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=INSTITUTION_TIER_PROMPT),
            HumanMessage(content=f"Institution: {institution.strip()}"),
        ])
        raw = response.content
        logger.debug(f"[EducationAgent] Tier raw for '{institution}': {raw!r}")

        parsed = safe_parse_json(raw, context=f"edu_tier:{institution}")
        if not parsed or not isinstance(parsed, dict):
            logger.warning(f"[EducationAgent] Invalid JSON for '{institution}' → default Tier 2")
            return 2, 6.5

        tier = parsed.get("tier")
        if tier not in (1, 2, 3):
            logger.warning(f"[EducationAgent] Invalid tier {tier!r} for '{institution}' → default Tier 2")
            return 2, 6.5

        score_map = {1: 10.0, 2: 6.5, 3: 3.5}
        logger.info(f"[EducationAgent] '{institution}' → Tier {tier}")
        return int(tier), score_map[int(tier)]

    except Exception as e:
        logger.warning(f"[EducationAgent] Institution tier failed for '{institution}': {e}")
        return 2, 6.5


def _institution_tier_score(education_list: List[dict]) -> Tuple[float, str, List[dict]]:
    """
    Classify the highest-degree institution.
    Returns (score, tier_label, tier_details_list).
    """
    if not education_list:
        return 0.0, "Unknown", []

    details = []
    best_tier_score = 0.0
    best_tier_label = "Unknown"

    for edu in education_list:
        inst = edu.get("institution") or edu.get("institute") or ""
        degree = edu.get("degree", "")
        if not inst:
            continue

        tier, score = _classify_institution_tier(inst)
        details.append({
            "institution": inst,
            "degree":      degree,
            "tier":        tier,
            "score":       score,
        })

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
    """
    Scores how relevant the candidate's education field is to the job.
    Uses TF-IDF cosine similarity + direct keyword bonus for exact field matches.
    Returns score [0, 10].
    """
    if not education_list:
        return 3.0

    # Build education text
    edu_text = " ".join(
        f"{e.get('degree','')} {e.get('field','')} {e.get('institution','')}"
        for e in education_list
    ).strip().lower()

    job_text = f"{job_title} {job_description}".strip().lower()

    # ── Direct keyword bonus ──
    # If degree field directly mentions key terms from the job, give a strong base
    CS_KEYWORDS = [
        "computer science", "artificial intelligence", "machine learning",
        "ai", "ml", "aiml", "ai-ml", "data science", "information technology",
        "software engineering", "computer engineering", "it", "cse",
    ]
    bonus = 0.0
    for kw in CS_KEYWORDS:
        if kw in edu_text:
            bonus = 3.0   # strong base for any CS/AI degree
            break

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if not edu_text or not job_text:
            return clamp_score(bonus + 3.0)

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform([edu_text, job_text])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        # Apply power boost for short texts
        boosted = sim ** 0.6
        cosine_score = boosted * 10
        # Final: take max of (cosine alone) or (cosine + keyword bonus), capped at 10
        score = clamp_score(max(cosine_score, cosine_score * 0.5 + bonus * 1.2))
        logger.info(f"[EducationAgent] Field relevance: cosine={sim:.3f} boosted={boosted:.3f} bonus={bonus} → {score:.2f}")
        return score

    except Exception as e:
        logger.warning(f"[EducationAgent] Field relevance failed: {e}")
        return clamp_score(bonus + 3.0)


# ─── Component 4: GPA ─────────────────────────────────────────────────────────

def _parse_gpa(gpa_str: Optional[str]) -> Optional[Tuple[float, float]]:
    """
    Extract numeric GPA value from various formats:
      "8.7"   → 8.7  (10-point)
      "3.5/4" → 3.5  (4-point)
      "85%"   → 85.0
      "8.7/10"→ 8.7
    Returns raw float or None.
    """
    if not gpa_str or not gpa_str.strip():
        return None

    # Remove common noise
    s = gpa_str.strip().replace(",", ".")

    # Format: X/Y
    frac = re.search(r"([\d.]+)\s*/\s*([\d.]+)", s)
    if frac:
        num, denom = float(frac.group(1)), float(frac.group(2))
        return num, denom  # return both for scale detection

    # Percentage
    pct = re.search(r"([\d.]+)\s*%", s)
    if pct:
        return float(pct.group(1)), 100.0

    # Plain number
    plain = re.search(r"([\d.]+)", s)
    if plain:
        val = float(plain.group(1))
        # Heuristic: if > 10, treat as percentage
        if val > 10:
            return val, 100.0
        elif val > 4:
            return val, 10.0   # assume 10-point scale
        else:
            return val, 4.0    # assume 4-point scale

    return None


def _gpa_score(education_list: List[dict]) -> float:
    """
    Normalise best GPA across all education entries to [0, 10].
    Returns 5.0 (neutral) if no GPA found.
    """
    best = 0.0
    found_any = False

    for edu in education_list:
        result = _parse_gpa(edu.get("gpa"))
        if result is None:
            continue
        found_any = True
        val, scale = result
        # Normalise to 0-10
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
    degree_score:    float,
    tier_score:      float,
    relevance_score: float,
    gpa_score:       float,
) -> float:
    """
    Weights:
      35% degree level (+ requirement match)
      30% institution tier
      20% field relevance
      15% GPA / academic performance
    """
    raw = (0.35 * degree_score + 0.30 * tier_score +
           0.20 * relevance_score + 0.15 * gpa_score)
    return clamp_score(raw)


# ─── Main agent function ───────────────────────────────────────────────────────

def education_scoring_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 4 — Education Scoring

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

    # ── Edge case: no education ──
    if not education:
        logger.warning(f"[{agent_name}] No education entries → score=2.0")
        edu_score = EducationScore(
            score=2.0,
            highest_degree=None,
            institution_tier=None,
            reasoning="No education entries found in resume.",
        )
        return {
            **state,
            "education_score": edu_score.model_dump(),
            "errors": errors,
            "current_step": agent_name,
        }

    # ── Component 1: Degree Level ──
    deg_score, highest_degree, deg_note = _degree_score(education, education_required)
    logger.info(f"[{agent_name}] degree='{highest_degree}' score={deg_score:.2f} | {deg_note}")

    # ── Component 2: Institution Tier ──
    tier_score, tier_label, tier_details = _institution_tier_score(education)
    logger.info(f"[{agent_name}] tier_label='{tier_label}' score={tier_score:.2f}")

    # ── Component 3: Field Relevance ──
    rel_score = _field_relevance_score(education, job_title, job_desc)

    # ── Component 4: GPA ──
    gpa_sc = _gpa_score(education)

    # ── Final score ──
    final_score = _calculate_education_score(deg_score, tier_score, rel_score, gpa_sc)

    # ── Reasoning ──
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
        f"score={final_score:.2f}/10 | degree={deg_score:.2f} | "
        f"tier={tier_score:.2f} | rel={rel_score:.2f} | gpa={gpa_sc:.2f}"
    )

    return {
        **state,
        "education_score": edu_score.model_dump(),
        "errors": errors,
        "current_step": agent_name,
    }