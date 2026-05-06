"""
Agent 3: Experience Scoring Agent
 
Scoring strategy (3 components):
 
  Component 1 — Years of Experience  (pure math, no LLM)
      Parse start/end dates from each experience item → compute duration
      Compare total_years vs years_experience_required
      Score formula:
        if total >= required  → full marks (10) scaled slightly above
        if total < required   → (total / required) * 10
      Cap at 10.
 
  Component 2 — Company Tier  (LLM classification, temperature=0)
      For each company name, LLM classifies into:
        Tier 1: FAANG/Top MNC (Google, Microsoft, Amazon, etc.)
        Tier 2: Mid-size tech / well-known startups / good companies
        Tier 3: Small/unknown/local companies
      Score: weighted average of tier scores → Tier1=10, Tier2=6.5, Tier3=3.5
      Anti-hallucination: output must be exactly {"tier": 1|2|3} — validated strictly
 
  Component 3 — Role Relevance  (cosine similarity, no LLM)
      Compute cosine similarity between:
        - all job titles/descriptions from resume experience
        - job_title + job_description from job input
      Score mapped to [0, 10]
 
Final score formula:
  score = clamp(0.40*years_score + 0.35*tier_score + 0.25*relevance_score, 0, 10)
 
Anti-hallucination:
  - Company tier: JSON with single field {"tier": int}, validated 1/2/3 only
  - Temperature = 0
  - Date parsing: regex-based with multiple format support, never trusts LLM for dates
  - All scores clamped [0, 10]
"""

from __future__ import annotations
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from models.state import ResumeGraphState, ExperienceScore
from utils.logger import get_logger, log_agent_start, log_agent_end, log_agent_error
from utils.validators import safe_parse_json, clamp_score

load_dotenv()
logger = get_logger("agents.experience_scoring")

# -- [ LLM Setup ] --

def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env file")
    return ChatGroq(api_key=api_key, model=model, temperature=0, max_tokens=256)

# [ Component 1: Years of Experience ]

def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse various date formats from resumes:
      "2023-06", "2023", "Present", "Jun 2023", "June 2023", "2023-06-01"
    Returns datetime or None.
    """
    if not date_str:
        return None
    
    s = date_str.strip()

    if s.lower() in ("present", "current", "now", "ongoing"):
        return datetime.now()
    
    # Try formats in order
    formats = [
        "%Y-%m",        # 2023-06
        "%Y-%m-%d",     # 2023-06-01
        "%Y",           # 2023
        "%b %Y",        # Jun 2023
        "%B %Y",        # June 2023
        "%m/%Y",        # 06/2023
        "%m-%Y",        # 06-2023
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    # Fallback: extract just a 4-digit year
    match = re.search(r"\b(19|20)\d{2}\b", s)
    if match:
        try:
            return datetime.strptime(match.group(), "%Y")
        except ValueError:
            pass

    return None

def _compute_years(start: Optional[str], end: Optional[str]) -> float:
    """
    Compute years between start_date and end_date strings.
    Returns 0.0 if dates cannot be parsed.
    """
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)

    if not start_dt:
        return 0.0
    if not end_dt:
        end_dt = datetime.now()

    delta = end_dt - start_dt
    years = delta.days / 365.25
    return max(0.0, round(years, 2))

def _total_experience_years(experience: List[dict]) -> Tuple[float, List[dict]]:
    """
    Compute total non-overlapping years of experience.
    Returns (total_years, list of {company, position, years} dicts)
    """
    intervals = []
    details = []

    for exp in experience:
        start = exp.get("start_date")
        end = exp.get("end_date")
        years = _compute_years(start, end)

        start_dt = _parse_date(start)
        end_dt = _parse_date(end) or datetime.now()

        if start_dt and years > 0:
            intervals.append((start_dt, end_dt))

        details.append({
            "company": exp.get("company", "Unknown"),
            "position": exp.get("position", "Unknown"),
            "years": years,
            "start": start,
            "end": end
        })

    # Merge overlapping intervals to avoid double-counting
    if not intervals:
        return 0.0, details
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    total = sum((e - s).days / 365.25 for s, e in merged)
    return round(max(0.0, total), 2), details

def _years_score(total_years: float, required_years: float, is_internship: bool = False) -> float:
    """
    Score based on years of experience vs required years.
    Internship mode: 0 years is perfectly fine — candidates are students.
    Job mode: scored on ratio vs requirement.
    """
    if is_internship:
        # For internships, prior experience is a bonus not a requirement.
        # Projects and skills matter far more. Give a generous base.
        if total_years >= 1.0:  return 9.0   # has prior internship/job — great
        if total_years >= 0.5:  return 7.5   # some part-time/freelance experience
        if total_years >= 0.1:  return 6.0   # any work exposure
        return 5.0                            # pure fresher — totally fine for internship

    # ── Job mode ──
    if required_years <= 0:
        if total_years >= 5:   return 9.0
        if total_years >= 3:   return 7.5
        if total_years >= 1:   return 6.0
        if total_years >= 0.5: return 4.0
        return 2.0

    ratio = total_years / required_years
    if ratio >= 1.5:   return 10.0
    if ratio >= 1.0:   return 8.5 + (ratio - 1.0) * 3.0
    if ratio >= 0.75:  return 6.5 + (ratio - 0.75) * 8.0
    if ratio >= 0.5:   return 4.0 + (ratio - 0.5)  * 10.0
    return clamp_score(ratio * 8.0)

# [ Component 2: Company Tier Classification ]

TIER_SYSTEM_PROMPT = """You are a company tier classifier for tech industry. Classify the given company name into one of three tiers.
 
Tier 1 (score: 10): FAANG/MAANG, top MNCs, globally recognised tech giants
  Examples: Google, Microsoft, Amazon, Apple, Meta, Netflix, OpenAI, Nvidia, Goldman Sachs, McKinsey, Infosys (large MNC division), Tata Consultancy Services, Wipro, Cognizant, Accenture (large scale)
 
Tier 2 (score: 6.5): Well-known tech companies, funded startups (Series B+), reputed mid-size companies
  Examples: Razorpay, Zepto, Zomato, Swiggy, Paytm, Flipkart, Ola, CRED, Meesho, Freshworks, Postman, any company with 500+ employees that is well known
 
Tier 3 (score: 3.5): Small companies, local businesses, unknown startups, agencies, freelance
 
RULES:
1. Output ONLY valid JSON — no explanation, no preamble.
2. Return EXACTLY: {"tier": 1} or {"tier": 2} or {"tier": 3}
3. When uncertain, default to Tier 2 for tech companies, Tier 3 for others.
4. Never return any other keys or values."""

def _classify_company_tier(company_name: str) -> int:
    """
    Classify a company into tier 1/2/3 using LLM.
    Returns int (1, 2, or 3). Defaults to 3 on any failure.
    Anti-hallucination: validates output is exactly 1, 2, or 3.
    """
    if not company_name or company_name.strip().lower() in ("unknown", ""):
        return 3
    
    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=TIER_SYSTEM_PROMPT),
            HumanMessage(content=f"Company: {company_name.strip()}"),
        ])
        raw = response.content
        logger.debug(f"[ExperienceAgent] Tier raw for '{company_name}': {raw!r}")

        parsed = safe_parse_json(raw, context=f"company_tier: {company_name}")
        if not parsed or not isinstance(parsed, dict):
            logger.warning(f"[ExperienceAgent] Invalid tier JSON for '{company_name}' → default 3")
            return 3
        
        tier = parsed.get("tier")

        # Strict validation - must be exactly 1, 2, or 3
        if tier not in (1, 2, 3):
            logger.warning(f"[ExperienceAgent] Invalid tier value {tier!r} for '{company_name}' → default 3")
            return 3
        
        return int(tier)
    
    except Exception as e:
        logger.warning(f"[ExperienceAgent] Tier classification failed for '{company_name}': {e}")
        return 3
    

TIER_SCORE_MAP = {1: 10.0, 2: 6.5, 3: 3.5}


def _company_tier_score(experience: List[dict]) -> Tuple[float, List[dict]]:
    """
    Classify each company and compute weighted average tier score.
    Weights by years_at_company so longer stints matter more.
    Returns (tier_score, list of {company, tier, years} dicts)
    """
    if not experience:
        return 0.0, []
    
    tier_details = []
    total_weight = 0.0
    weighted_sum = 0.0

    for exp in experience:
        company = exp.get("company", "Unknown")
        years = exp.get("years", 1.0)
        if years <= 0:
            years = 0.5 # give minimum weight even if dates missing

        tier = _classify_company_tier(company)
        score = TIER_SCORE_MAP[tier]
        weight = years

        weighted_sum += score * weight
        total_weight += weight

        tier_details.append({
            "company": company,
            "tier": tier,
            "score": score,
            "years": years,
        })
        logger.info(f"[ExperienceAgent] '{company}' → Tier {tier} ({score}) × {years:.1f}y")

    # FIXED: return AFTER the loop, not inside it
    avg_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    return clamp_score(avg_score), tier_details
    
# [ Component 3: Role Relevance (Cosine Similarity) ]

def _relevance_score(experience: List[dict], job_title: str, job_description: str) -> float:
    """
    Cosine similarity between candidate's experience text
    and the target job title + description.
    Returns score [0, 10].
    Boosted: if candidate has 0-1 year experience, relevance is weighted more
    leniently since projects matter as much as job titles.
    """
    if not experience: 
        return 0.0
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Aggregate all experience text — include position, company, description
        exp_text = " ".join(
            f"{e.get('position','')} {e.get('company','')} {e.get('description','')}"
            for e in experience
        ).strip()

        job_text = f"{job_title} {job_description}".strip()

        if not exp_text or not job_text:
            return 0.0
        
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform([exp_text, job_text])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

        # Boost: cosine similarity is harshly low for short exp descriptions.
        # Apply a square-root boost so 0.2 sim → 4.5 instead of 2.0
        boosted_sim = sim ** 0.6   # gentler curve than raw linear
        score = clamp_score(boosted_sim * 10)
        logger.info(f"[ExperienceAgent] Role relevance cosine={sim:.3f} boosted={boosted_sim:.3f} → score={score:.2f}")
        return score
    
    except Exception as e:
        logger.warning(f"[ExperienceAgent] Relevance cosine failed: {e}")
        return 0.0
    

# -- [ Final Score Calculation ] --

def _calculate_experience_score(
        years_score: float,
        tier_score: float,
        relevance_score: float,
        is_internship: bool = False,
) -> float:
    """
    Weighted combination:
      Job mode:        40% years | 35% company tier | 25% relevance
      Internship mode: 35% years | 25% company tier | 40% relevance
      For internships, role relevance matters more than company prestige.
      Interns are expected to come from small companies or none at all.
    """
    if is_internship:
        raw = 0.35 * years_score + 0.25 * tier_score + 0.40 * relevance_score
    else:
        raw = 0.40 * years_score + 0.35 * tier_score + 0.25 * relevance_score
    return clamp_score(raw)

# -- [ Main Agent Function ] --

def experience_scoring_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 3 — Experience Scoring
 
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
    job_input =  state.get("job_input") or {}

    experience: List[dict] = parsed_resume.get("experience") or []
    job_title:  str        = job_input.get("job_title", "")
    job_desc:   str        = job_input.get("job_description", "")
    req_years:  float      = float(job_input.get("years_experience_required") or 0)
    is_internship: bool    = str(job_input.get("opportunity_type", "job")).lower() == "internship"

    logger.info(f"[{agent_name}] experience_items={len(experience)} | req_years={req_years} | internship={is_internship}")

    # -- Edge case: no experience --
    if not experience:
        base_score = 5.0 if is_internship else 1.0
        reason = (
            "No work experience found — expected for internship applicants."
            if is_internship else
            "No work experience found in resume."
        )
        logger.info(f"[{agent_name}] No experience entries → score={base_score} (internship={is_internship})")
        exp_score = ExperienceScore(
            score=base_score,
            total_years=0.0,
            company_tier_avg=None,
            reasoning=reason,
        )
        return {
            **state,
            "experience_score": exp_score.model_dump(),
            "errors": errors,
            "current_step": agent_name,
        }
    
    # Component 1: Years
    total_years, exp_details = _total_experience_years(experience)
    y_score = _years_score(total_years, req_years, is_internship=is_internship)
    logger.info(f"[{agent_name}] total_years={total_years} | years_score={y_score:.2f} | internship={is_internship}")

    # Component 2: Company Tier (uses LLM per company)
    tier_score_val, tier_details = _company_tier_score(exp_details)
    tier_avg = (
        sum(t["tier"] for t in tier_details) / len(tier_details)
        if tier_details else None
    )
    logger.info(f"[{agent_name}] tier_score={tier_score_val:.2f} | tier_avg={tier_avg}")

    # Component 3: Role Relevance
    rel_score = _relevance_score(exp_details, job_title, job_desc)

    # Final score
    final_score = _calculate_experience_score(y_score, tier_score_val, rel_score, is_internship=is_internship)

    # Build detailed reasoning
    tier_str = " | ".join(
        f"{t['company']}→T{t['tier']}({t['years']:.1f}y)"
        for t in tier_details
    )
    reasoning = (
        f"Total experience: {total_years:.1f} years (required: {req_years:.1f}). "
        f"Years score: {y_score:.2f}/10. "
        f"Company tiers: {tier_str}. "
        f"Tier score: {tier_score_val:.2f}/10. "
        f"Role relevance: {rel_score:.2f}/10. "
        f"Weights: 40% years + 35% tier + 25% relevance."
    )
    
    exp_score = ExperienceScore(
        score=final_score,
        total_years=total_years,
        company_tier_avg=round(tier_avg, 2) if tier_avg else None,
        reasoning=reasoning,
    )

    log_agent_end(
        logger, agent_name,
        f"score={final_score:.2f}/10 | years={total_years:.1f} | "
        f"y_score={y_score:.2f} | tier={tier_score_val:.2f} | rel={rel_score:.2f}"
    )

    return {
        **state,
        "experience_score": exp_score.model_dump(),
        "errors": errors,
        "current_step": agent_name,
    }