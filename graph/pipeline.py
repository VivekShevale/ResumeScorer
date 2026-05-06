"""
graph/pipeline.py — Step 7 (FINAL)
Full pipeline:
resume_parser → skill_matching → experience_scoring → education_scoring
             → achievement_scoring → social_scoring → llm_scoring
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from models.state import ResumeGraphState
from agents.resume_parser_agent        import resume_parser_agent
from agents.skill_matching_agent       import skill_matching_agent
from agents.experience_scoring_agent   import experience_scoring_agent
from agents.education_scoring_agent    import education_scoring_agent
from agents.achievement_scoring_agent  import achievement_scoring_agent
from agents.social_agent               import social_agent
from agents.llm_scoring_agent          import llm_scoring_agent
from utils.logger import get_logger

logger = get_logger("graph.pipeline")


def build_graph_final() -> StateGraph:
    graph = StateGraph(ResumeGraphState)

    graph.add_node("resume_parser",       resume_parser_agent)
    graph.add_node("skill_matching",      skill_matching_agent)
    graph.add_node("experience_scoring",  experience_scoring_agent)
    graph.add_node("education_scoring",   education_scoring_agent)
    graph.add_node("achievement_scoring", achievement_scoring_agent)
    graph.add_node("social_scoring",      social_agent)
    graph.add_node("llm_scoring",         llm_scoring_agent)

    graph.set_entry_point("resume_parser")
    graph.add_edge("resume_parser",       "skill_matching")
    graph.add_edge("skill_matching",      "experience_scoring")
    graph.add_edge("experience_scoring",  "education_scoring")
    graph.add_edge("education_scoring",   "achievement_scoring")
    graph.add_edge("achievement_scoring", "social_scoring")
    graph.add_edge("social_scoring",      "llm_scoring")
    graph.add_edge("llm_scoring",         END)

    return graph.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph_final()
        logger.info("Pipeline compiled (FINAL: all 7 agents)")
    return _graph


def run_pipeline(resume_file_path: str, job_input: dict) -> ResumeGraphState:
    graph = get_graph()
    initial_state: ResumeGraphState = {
        "resume_file_path": resume_file_path,
        "job_input":        job_input,
        "errors":           [],
        "current_step":     "init",
    }
    logger.info(f"Pipeline started | file={resume_file_path}")
    result = graph.invoke(initial_state)
    final  = result.get("final_score") or {}
    logger.info(
        f"Pipeline COMPLETE | "
        f"final_score={final.get('total_score','?')}/100 | "
        f"label={final.get('breakdown',{}).get('label','?')} | "
        f"errors={result.get('errors')}"
    )
    return result