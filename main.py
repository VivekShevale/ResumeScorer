"""
main.py — v2.0.0: All 8 agents + Job/Internship mode
"""
from __future__ import annotations
import os, uuid, shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()
from graph.pipeline import run_pipeline
from utils.logger import get_logger
logger = get_logger("main")

app = FastAPI(title="Resume Scorer API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
UPLOAD_DIR    = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
FRONTEND_PATH = Path("frontend/index.html")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=FRONTEND_PATH.read_text(encoding="utf-8"))


@app.post("/api/score-resume")
async def score_resume(
    resume:                    UploadFile      = File(...),
    job_title:                 str             = Form(...),
    job_description:           str             = Form(...),
    job_role:                  str             = Form(...),
    opportunity_type:          str             = Form(default="job"),   # "job" | "internship"
    education_required:        Optional[str]   = Form(default=""),
    years_experience_required: Optional[float] = Form(default=0),
    skills_required:           Optional[str]   = Form(default=""),
):
    ext = Path(resume.filename).suffix.lower()
    if ext not in {".pdf", ".docx", ".doc"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    fp = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    try:
        with open(fp, "wb") as f: shutil.copyfileobj(resume.file, f)
        logger.info(f"Uploaded | {resume.filename} → {fp.name} | type={opportunity_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    # Normalise opportunity type
    opp_type = "internship" if "intern" in opportunity_type.lower() else "job"

    job_input = {
        "job_title":                 job_title,
        "job_description":           job_description,
        "job_role":                  job_role,
        "opportunity_type":          opp_type,
        "education_required":        education_required,
        "years_experience_required": years_experience_required if opp_type == "job" else 0,
        "skills_required":           [s.strip() for s in skills_required.split(",") if s.strip()],
    }

    try:
        result = run_pipeline(str(fp), job_input)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
    finally:
        try: os.remove(fp)
        except Exception: pass

    fs = result.get("final_score") or {}
    response = {
        "status":           "success" if not result.get("errors") else "partial",
        "current_step":     result.get("current_step"),
        "errors":           result.get("errors", []),
        "parsed_resume":    result.get("parsed_resume"),
        "raw_text_length":  len(result.get("raw_text", "")),
        "skill_score":      result.get("skill_score"),
        "experience_score": result.get("experience_score"),
        "education_score":  result.get("education_score"),
        "achievement_score":result.get("achievement_score"),
        "social_score":     result.get("social_score"),
        "project_score":    result.get("project_score"),
        "semantic_score":   result.get("semantic_score"),
        "final_score":      result.get("final_score"),
        "opportunity_type": opp_type,
        "job_input":        job_input,
    }
    logger.info(
        f"DONE | type={opp_type} | total={fs.get('total_score','?')}/100 | "
        f"label={fs.get('breakdown',{}).get('label','?')} | status={response['status']}"
    )
    return JSONResponse(content=response)


@app.post("/api/parse-resume")   # backward compat
async def _compat(
    resume: UploadFile = File(...), job_title: str = Form(...),
    job_description: str = Form(...), job_role: str = Form(...),
    opportunity_type: str = Form(default="job"),
    education_required: Optional[str] = Form(default=""),
    years_experience_required: Optional[float] = Form(default=0),
    skills_required: Optional[str] = Form(default=""),
):
    return await score_resume(
        resume=resume, job_title=job_title, job_description=job_description,
        job_role=job_role, opportunity_type=opportunity_type,
        education_required=education_required,
        years_experience_required=years_experience_required,
        skills_required=skills_required,
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok", "version": "2.0.0",
        "agents": 8, "modes": ["job", "internship"],
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Resume Scorer v2.0.0 — Job & Internship modes active")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)