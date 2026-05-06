# Resume Scorer - LangGraph Agentic System

## Project Structure
```
resume_scorer/
├── agents/
│   ├── __init__.py
│   ├── resume_parser_agent.py       # Agent 1: Parse resume → JSON
│   ├── skill_matching_agent.py      # Agent 2: Match skills with JD
│   ├── experience_scoring_agent.py  # Agent 3: Score experience
│   ├── education_scoring_agent.py   # Agent 4: Score education
│   ├── achievement_scoring_agent.py # Agent 5: Score achievements
│   ├── social_agent.py              # Agent 6: GitHub/LeetCode/Codeforces
│   └── llm_scoring_agent.py         # Agent 7: Final LLM scoring
├── graph/
│   ├── __init__.py
│   └── pipeline.py                  # LangGraph pipeline wiring
├── models/
│   ├── __init__.py
│   └── state.py                     # Shared graph state / Pydantic models
├── utils/
│   ├── __init__.py
│   ├── file_parser.py               # PDF/DOCX text extractor
│   ├── logger.py                    # Structured logging
│   └── validators.py                # Pydantic validators
├── logs/                            # Auto-created log files
├── uploads/                         # Temp resume uploads
├── frontend/
│   └── index.html                   # Frontend UI
├── main.py                          # FastAPI entry point
├── .env                             # GROQ_API_KEY goes here
└── requirements.txt
```

## Setup (Windows)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```
python main.py
# Open: http://localhost:8000
```

## Step Progress
- [x] Step 1: Project Structure + Resume Parser Agent (Agent 1)
- [ ] Step 2: Skill Matching Agent (Agent 2)
- [ ] Step 3: Experience Scoring Agent (Agent 3)
- [ ] Step 4: Education Scoring Agent (Agent 4)
- [ ] Step 5: Achievement Scoring Agent (Agent 5)
- [ ] Step 6: Social Agent (Agent 6)
- [ ] Step 7: LLM Scoring Agent + Regression (Agent 7)