"""
models/state.py
---------------
Shared LangGraph state + all Pydantic models used across agents.
Every field is Optional so the graph can be partially populated at each step.
"""

from __future__ import annotations
from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# Sub-models for parsed resume sections
# ─────────────────────────────────────────────

class PersonalInfo(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    leetcode: Optional[str] = None
    codechef: Optional[str] = None
    codeforces: Optional[str] = None
    website: Optional[str] = None
    profession: Optional[str] = None


class ExperienceItem(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    is_current: bool = False
    # Pre-computed by resume_parser_agent (batched LLM call) — no extra LLM needed downstream
    company_tier: int = 3  # 1=FAANG, 2=mid, 3=small


class EducationItem(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[str] = None
    # Pre-computed by resume_parser_agent (batched LLM call) — no extra LLM needed downstream
    institution_tier: int = 2  # 1=IIT/top-global, 2=good-college, 3=unknown


class ProjectItem(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    technologies: Optional[List[str]] = Field(default_factory=list)


class AchievementItem(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    platform: Optional[str] = None  # e.g. "LeetCode", "Codeforces", "Hackathon"


class ParsedResume(BaseModel):
    personal_info: Optional[PersonalInfo] = None
    professional_summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    achievements: List[AchievementItem] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    # Pre-computed by resume_parser_agent — overall achievement quality 0-10
    achievement_quality_score: Optional[float] = None

    @field_validator("skills", mode="before")
    @classmethod
    def clean_skills(cls, v):
        if isinstance(v, list):
            return [s.strip() for s in v if isinstance(s, str) and s.strip()]
        return []


# ─────────────────────────────────────────────
# Job Input model (structured user input)
# ─────────────────────────────────────────────

class JobInput(BaseModel):
    job_title: str
    job_description: str
    job_role: str
    education_required: Optional[str] = None
    years_experience_required: Optional[float] = None
    skills_required: List[str] = Field(default_factory=list)
    opportunity_type: Optional[str] = "job"  # "job" or "internship"

    @field_validator("skills_required", mode="before")
    @classmethod
    def parse_skills(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


# ─────────────────────────────────────────────
# Score models
# ─────────────────────────────────────────────

class SkillScore(BaseModel):
    score: float = Field(ge=0, le=10)
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class ExperienceScore(BaseModel):
    score: float = Field(ge=0, le=10)
    total_years: float = 0.0
    company_tier_avg: Optional[float] = None
    reasoning: Optional[str] = None


class EducationScore(BaseModel):
    score: float = Field(ge=0, le=10)
    highest_degree: Optional[str] = None
    institution_tier: Optional[str] = None
    reasoning: Optional[str] = None


class AchievementScore(BaseModel):
    score: float = Field(ge=0, le=10)
    achievement_count: int = 0
    reasoning: Optional[str] = None


class SocialScore(BaseModel):
    score: float = Field(ge=0, le=10)
    github_data: Optional[dict] = None
    leetcode_data: Optional[dict] = None
    codeforces_data: Optional[dict] = None
    reasoning: Optional[str] = None


class ProjectScore(BaseModel):
    score: float = Field(ge=0, le=10)
    project_count: int = 0
    quality_score: float = 0.0
    tech_match_score: float = 0.0
    skills_used_in_projects: List[str] = Field(default_factory=list)
    required_skills_missing_from_projects: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class SemanticScore(BaseModel):
    score: float = Field(ge=0, le=10)
    reasoning: Optional[str] = None


class FinalScore(BaseModel):
    total_score: float = Field(ge=0, le=100)
    skill_score: Optional[SkillScore] = None
    experience_score: Optional[ExperienceScore] = None
    education_score: Optional[EducationScore] = None
    achievement_score: Optional[AchievementScore] = None
    social_score: Optional[SocialScore] = None
    project_score: Optional[ProjectScore] = None
    semantic_score: Optional[SemanticScore] = None
    breakdown: Optional[dict] = None


# ─────────────────────────────────────────────
# LangGraph shared state (TypedDict)
# ─────────────────────────────────────────────

class ResumeGraphState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────
    resume_file_path: str          # path to uploaded file
    job_input: dict                # raw JobInput dict

    # ── Agent 1 output ──────────────────────
    raw_text: str                  # extracted text from resume
    parsed_resume: dict            # ParsedResume as dict (includes pre-computed tiers)

    # ── Agent 2 output ──────────────────────
    skill_score: dict

    # ── Agent 3 output ──────────────────────
    experience_score: dict

    # ── Agent 4 output ──────────────────────
    education_score: dict

    # ── Agent 5 output ──────────────────────
    achievement_score: dict

    # ── Agent 6 output ──────────────────────
    social_score: dict

    # ── Agent 7 output (NEW) ────────────────
    project_score: dict

    # ── Agent 8 output (was Agent 7) ────────
    semantic_score: dict
    final_score: dict

    # ── Error tracking ───────────────────────
    errors: List[str]
    current_step: str
