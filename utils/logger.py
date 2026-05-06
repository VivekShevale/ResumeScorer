import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# -- ensure logs / directory exists --
LOG_DIR = Path(__file__).parent.parent /"logs"
LOG_DIR.mkdir(exist_ok=True)

log_filename = LOG_DIR / f"resume_scorer_{datetime.now().strftime('%Y-%m-%d')}.log"

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger that writes to both console and a daily log file.
 
    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Agent started", extra={"agent": "resume_parser"})
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger # already configured
    
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -- File handler --
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # -- Console handler --
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

def log_agent_start(logger: logging.Logger, agent_name: str, inputs: dict):
    logger.info(f"[{agent_name}] ▶ STARTED | inputs_keys={list(inputs.keys())}")

def log_agent_end(logger: logging.Logger, agent_name: str, output_summary: str):
    logger.info(f"[{agent_name}] ✔ COMPLETED | {output_summary}")

def log_agent_error(logger: logging.Logger, agent_name: str, error: str):
    logger.error(f"[{agent_name}] ✖ ERROR | {error}")