"""
agents/resume_parser_agent.py
------------------------------
Agent 1: Resume Parser

Pipeline:
  file_path → extract_text() → LLM (Groq) → validated ParsedResume JSON

Anti-hallucination measures:
  - Strict JSON-only system prompt
  - Temperature = 0
  - Schema enforcement via Pydantic
  - Regex fallback extraction for common fields
  - All LLM output sanitized through validators
"""

from __future__ import annotations
import os
import re
import json
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from models.state import (
    ResumeGraphState, ParsedResume, PersonalInfo,
    ExperienceItem, EducationItem, ProjectItem, AchievementItem
)
from utils.logger import get_logger, log_agent_start, log_agent_end, log_agent_error
from utils.file_parser import extract_text
from utils.validators import safe_parse_json, sanitize_string, sanitize_list, validate_email, validate_url

load_dotenv()
logger = get_logger("agents.resume_parser")

# ─── LLM setup ───────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env file")
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0,       # deterministic — reduces hallucination
        max_tokens=4096,
    )


# ─── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise resume parser. Your ONLY job is to extract information from resume text and return it as valid JSON.

STRICT RULES:
1. Output ONLY valid JSON — no explanation, no markdown, no preamble.
2. If a field is not present in the resume, use null for strings and [] for arrays.
3. NEVER invent or guess information. Only extract what is explicitly stated.
4. For dates, use format "YYYY-MM" or "YYYY" or "Present". Never invent dates.
5. For GPA, extract the exact value shown. Never calculate or estimate.
6. Skills must be exact technologies/tools/languages mentioned in the resume.

IMPORTANT — SOCIAL PROFILE EXTRACTION:
- At the end of the resume text you may see a section "[EXTRACTED_LINKS]" — these are hyperlinks 
  extracted from the PDF. Use them to fill github, linkedin, leetcode, codeforces, codechef, website fields.
- Also look for patterns like "@VivekShevale" (GitHub handle) or "/ShevaleVivek" (LinkedIn handle) 
  in the header/contact section — infer full URLs from these patterns.
- For GitHub: "@username" near top of resume → "https://github.com/username"
- For LinkedIn: "/username" or "linkedin.com/in/username" → full LinkedIn URL
- For LeetCode: text like "vvkshvl – LeetCode Profile" or "leetcode.com/u/username" → full URL
- Always prefer explicit URLs from [EXTRACTED_LINKS] over inferred ones.

Return this exact JSON structure:
{
  "personal_info": {
    "full_name": null,
    "email": null,
    "phone": null,
    "location": null,
    "linkedin": null,
    "github": null,
    "leetcode": null,
    "codechef": null,
    "codeforces": null,
    "website": null,
    "profession": null
  },
  "professional_summary": null,
  "skills": [],
  "experience": [
    {
      "company": null,
      "position": null,
      "start_date": null,
      "end_date": null,
      "description": null,
      "is_current": false
    }
  ],
  "education": [
    {
      "institution": null,
      "degree": null,
      "field": null,
      "graduation_date": null,
      "gpa": null
    }
  ],
  "projects": [
    {
      "name": null,
      "type": null,
      "description": null,
      "technologies": []
    }
  ],
  "achievements": [
    {
      "title": null,
      "description": null,
      "platform": null
    }
  ],
  "certifications": []
}"""


# ─── Regex fallback extractors ────────────────────────────────────────────────

def _regex_extract_email(text: str) -> str | None:
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group() if match else None


def _regex_extract_phone(text: str) -> str | None:
    match = re.search(r"[\+\(]?[\d\s\-\(\)]{7,15}", text)
    return match.group().strip() if match else None


def _regex_extract_linkedin(text: str) -> str | None:
    # Full URL
    match = re.search(r"linkedin\.com/in/[\w\-]+", text, re.IGNORECASE)
    if match:
        return f"https://{match.group()}"
    # /username pattern at start of line (common in modern resume templates)
    match = re.search(r"(?:^|[\s|])/([a-zA-Z][a-zA-Z0-9\-_]{2,})", text, re.MULTILINE)
    if match:
        return f"https://linkedin.com/in/{match.group(1)}"
    return None


def _regex_extract_github(text: str) -> str | None:
    # Full URL — most reliable
    match = re.search(r"github\.com/([a-zA-Z0-9][a-zA-Z0-9\-]{1,38})(?:[/\s]|$)", text, re.IGNORECASE)
    if match:
        return f"https://github.com/{match.group(1)}"
    # @username pattern — ONLY if it appears near GitHub context words
    # and is NOT part of an email address (no dot-domain after it)
    match = re.search(
        r"(?:github|git)[^\n]{0,30}@([a-zA-Z][a-zA-Z0-9\-_]{2,38})(?!\.[a-zA-Z])",
        text, re.IGNORECASE
    )
    if match:
        return f"https://github.com/{match.group(1)}"
    # Standalone @username only if it appears on a line by itself or near social context
    # Strictly exclude if the @ is inside an email (has chars before @ with no space)
    match = re.search(
        r"(?:^|[\s|])@([a-zA-Z][a-zA-Z0-9\-_]{2,38})(?!\.[a-zA-Z])(?=\s|$|[|/])",
        text, re.MULTILINE
    )
    if match:
        return f"https://github.com/{match.group(1)}"
    return None


def _regex_extract_leetcode(text: str) -> str | None:
    # Full URL forms
    match = re.search(r"leetcode\.com/(?:u/)?[\w\-]+", text, re.IGNORECASE)
    if match:
        return f"https://{match.group()}"
    # "username – LeetCode Profile" or "(username – LeetCode" patterns
    match = re.search(r"\(?([\w\-]+)\s*[–\-]\s*LeetCode", text, re.IGNORECASE)
    if match:
        return f"https://leetcode.com/u/{match.group(1)}"
    return None


def _regex_extract_codeforces(text: str) -> str | None:
    match = re.search(r"codeforces\.com/(?:profile/)?[\w\-]+", text, re.IGNORECASE)
    return f"https://{match.group()}" if match else None


# ─── Post-processing & validation ─────────────────────────────────────────────

def _validate_and_fix(parsed_dict: dict, raw_text: str) -> ParsedResume:
    """
    Takes raw LLM JSON dict → validates with Pydantic → fixes using regex fallbacks.
    This ensures we never silently accept hallucinated or missing values.
    """
    pi = parsed_dict.get("personal_info") or {}

    # Regex fallbacks for critical contact fields
    email = validate_email(pi.get("email") or "") or _regex_extract_email(raw_text)
    phone = pi.get("phone") or _regex_extract_phone(raw_text)
    linkedin = validate_url(pi.get("linkedin") or "") or _regex_extract_linkedin(raw_text)

    # GitHub: validate it's not an email domain (e.g. github.com/gmail from @gmail.com)
    github_raw = validate_url(pi.get("github") or "") or _regex_extract_github(raw_text)
    # Reject obviously wrong GitHub URLs derived from email domains
    email_domain = email.split("@")[-1].split(".")[0].lower() if email and "@" in email else ""
    if github_raw and email_domain and github_raw.rstrip("/").lower().endswith(f"/{email_domain}"):
        logger.warning(f"[ResumeParser] Rejected likely fake GitHub URL '{github_raw}' (matches email domain)")
        github_raw = None
    github = github_raw

    leetcode = validate_url(pi.get("leetcode") or "") or _regex_extract_leetcode(raw_text)
    codeforces = validate_url(pi.get("codeforces") or "") or _regex_extract_codeforces(raw_text)

    personal_info = PersonalInfo(
        full_name=sanitize_string(pi.get("full_name"), 100),
        email=email,
        phone=sanitize_string(phone, 30) if phone else None,
        location=sanitize_string(pi.get("location"), 100),
        linkedin=linkedin,
        github=github,
        leetcode=leetcode,
        codechef=validate_url(pi.get("codechef") or ""),
        codeforces=codeforces,
        website=validate_url(pi.get("website") or ""),
        profession=sanitize_string(pi.get("profession"), 100),
    )

    # Skills: deduplicate and sanitize
    raw_skills = sanitize_list(parsed_dict.get("skills"), 100)
    skills = list(dict.fromkeys(s for s in raw_skills if len(s) > 1))

    # Experience
    experience = []
    for exp in (parsed_dict.get("experience") or []):
        if not isinstance(exp, dict):
            continue
        experience.append(ExperienceItem(
            company=sanitize_string(exp.get("company"), 200),
            position=sanitize_string(exp.get("position"), 200),
            start_date=sanitize_string(exp.get("start_date"), 20),
            end_date=sanitize_string(exp.get("end_date"), 20),
            description=sanitize_string(exp.get("description"), 1000),
            is_current=bool(exp.get("is_current", False)),
        ))

    # Education
    education = []
    for edu in (parsed_dict.get("education") or []):
        if not isinstance(edu, dict):
            continue
        education.append(EducationItem(
            institution=sanitize_string(edu.get("institution"), 200),
            degree=sanitize_string(edu.get("degree"), 100),
            field=sanitize_string(edu.get("field"), 100),
            graduation_date=sanitize_string(edu.get("graduation_date"), 20),
            gpa=sanitize_string(edu.get("gpa"), 10),
        ))

    # Projects
    projects = []
    for proj in (parsed_dict.get("projects") or []):
        if not isinstance(proj, dict):
            continue
        projects.append(ProjectItem(
            name=sanitize_string(proj.get("name"), 200),
            type=sanitize_string(proj.get("type"), 100),
            description=sanitize_string(proj.get("description"), 1000),
            technologies=sanitize_list(proj.get("technologies"), 30),
        ))

    # Achievements
    achievements = []
    for ach in (parsed_dict.get("achievements") or []):
        if not isinstance(ach, dict):
            continue
        achievements.append(AchievementItem(
            title=sanitize_string(ach.get("title"), 200),
            description=sanitize_string(ach.get("description"), 500),
            platform=sanitize_string(ach.get("platform"), 100),
        ))

    # Certifications
    certifications = sanitize_list(parsed_dict.get("certifications"), 30)

    return ParsedResume(
        personal_info=personal_info,
        professional_summary=sanitize_string(parsed_dict.get("professional_summary"), 2000),
        skills=skills,
        experience=experience,
        education=education,
        projects=projects,
        achievements=achievements,
        certifications=certifications,
        raw_text=raw_text,
    )


# ─── Main agent function ──────────────────────────────────────────────────────

def resume_parser_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 1 — Resume Parser

    Input state keys:  resume_file_path
    Output state keys: raw_text, parsed_resume, errors
    """
    agent_name = "ResumeParserAgent"
    log_agent_start(logger, agent_name, {"resume_file_path": state.get("resume_file_path", "")})

    errors = list(state.get("errors") or [])

    # ── Step 1: Extract text from file ──
    file_path = state.get("resume_file_path", "")
    if not file_path:
        msg = "No resume_file_path provided in state"
        log_agent_error(logger, agent_name, msg)
        errors.append(msg)
        return {**state, "errors": errors, "current_step": agent_name}

    raw_text, extract_error = extract_text(file_path)

    if extract_error or not raw_text.strip():
        msg = f"Text extraction failed: {extract_error or 'Empty text'}"
        log_agent_error(logger, agent_name, msg)
        errors.append(msg)
        return {**state, "raw_text": raw_text, "errors": errors, "current_step": agent_name}

    logger.info(f"[{agent_name}] Text extracted | length={len(raw_text)} chars")

    # ── Step 2: Call LLM to parse resume ──
    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Parse this resume:\n\n{raw_text[:12000]}")  # 12k char limit
        ]
        logger.info(f"[{agent_name}] Calling Groq LLM...")
        response = llm.invoke(messages)
        llm_output = response.content
        logger.debug(f"[{agent_name}] LLM raw output preview: {llm_output[:300]!r}")

    except Exception as e:
        msg = f"LLM call failed: {e}"
        log_agent_error(logger, agent_name, msg)
        errors.append(msg)
        return {**state, "raw_text": raw_text, "errors": errors, "current_step": agent_name}

    # ── Step 3: Parse JSON from LLM output ──
    parsed_dict = safe_parse_json(llm_output, context=agent_name)
    if not parsed_dict or not isinstance(parsed_dict, dict):
        msg = f"LLM returned invalid JSON. Output: {llm_output[:500]}"
        log_agent_error(logger, agent_name, msg)
        errors.append(msg)
        # Still try regex extraction as fallback
        parsed_dict = {}

    # ── Step 4: Validate, sanitize, fix ──
    try:
        parsed_resume = _validate_and_fix(parsed_dict, raw_text)
    except Exception as e:
        msg = f"Validation failed: {e}"
        log_agent_error(logger, agent_name, msg)
        errors.append(msg)
        return {**state, "raw_text": raw_text, "errors": errors, "current_step": agent_name}

    log_agent_end(
        logger, agent_name,
        f"name={parsed_resume.personal_info.full_name!r} | "
        f"skills={len(parsed_resume.skills)} | "
        f"experience={len(parsed_resume.experience)} | "
        f"education={len(parsed_resume.education)}"
    )

    return {
        **state,
        "raw_text": raw_text,
        "parsed_resume": parsed_resume.model_dump(),
        "errors": errors,
        "current_step": agent_name,
    }