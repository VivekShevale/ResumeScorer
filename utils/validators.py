from __future__ import annotations
import re
import json
from typing import Any
from utils.logger import get_logger

logger = get_logger(__name__)

def clamp_score(value: Any, min_val: float = 0.0, max_val: float = 10.0) -> float:
    """Clamp a score to [min_val, max_val]. Returns 0.0 on invalid input."""
    try:
        f = float(value)
        return max(min_val, min(max_val, f))
    except (TypeError, ValueError):
        logger.warning(f"Invalid score value: {value!r}, defaulting to 0.0")
        return 0.0
    
def safe_parse_json(text: str, context: str = "") -> dict | list | None:
    """
    Safely parse JSON from LLM response text.
    Handles markdown code fences (```json ... ```) automatically.
    Returns None if parsing fails.
    """
    if not text:
        return None
    

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object/array in text
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                logger.debug(f"JSON extracted via regex | context={context}")
                return result
            except json.JSONDecodeError:
                continue

    logger.warning(f"JSON parse failed | context={context} | text_preview={text[:200]!r}")
    return None

def sanitize_string(value: Any, max_length: int = 500) -> str:
    """Sanitize a string value from LLM output."""
    if value is None:
        return ""
    s = str(value).strip()
    # Remove control characters
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    return s[:max_length]

def sanitize_list(value: Any, max_items: int = 50) -> list:
    """Ensure a value is a list of strings."""
    if isinstance(value, list):
        return [sanitize_string(i) for i in value[:max_items] if i]
    if isinstance(value, str):
        return [sanitize_string(value)] if value.strip() else []
    return []

def validate_email(email: str) -> str | None:
    """Returns email if valid, else None."""
    if not email:
        return None
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return email.strip() if re.match(pattern, email.strip()) else None
 
 
def validate_url(url: str) -> str | None:
    """Returns URL if it looks valid, else None."""
    if not url:
        return None
    url = url.strip()
    if url.startswith(("http://", "https://", "www.")):
        return url
    return None
 
 
def extract_years_from_text(text: str) -> float:
    """
    Attempt to extract years of experience from a text snippet.
    e.g. "5 years", "3+ years", "2.5 years" → float
    """
    if not text:
        return 0.0
    match = re.search(r"(\d+\.?\d*)\s*\+?\s*(?:years?|yrs?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0