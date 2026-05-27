"""
agents/social_agent.py
-----------------------
Agent 6: Social Profile Agent

Fetches live data from 3 public APIs (no auth needed for basic data):
  - GitHub  → repos, stars, followers, contributions (public API)
  - LeetCode → problems solved, ranking, acceptance rate (unofficial public GraphQL)
  - Codeforces → rating, rank, contest history (official public API)

Scoring strategy per platform:

  GitHub (weight: 35%) — RELAXED THRESHOLDS
    Signals: public_repos, total_stars, followers
    Score formula:
      repos_score    = clamp(repos / 10, 0, 1) * 10     (10 repos = full score)
      stars_score    = clamp(stars / 20, 0, 1) * 10     (20 stars = full score)
      followers_score= clamp(followers / 30, 0, 1) * 10 (30 followers = full score)
      final = 0.50*repos + 0.30*stars + 0.20*followers
      floor = 3.0 if any repos exist

  LeetCode (weight: 35%) — RELAXED THRESHOLDS
    Signals: total_solved, ranking, easy/medium/hard breakdown
    Score formula:
      solved_score  = clamp(total_solved / 100, 0, 1) * 10  (100 solved = full)
      hard_score    = clamp(hard_solved / 20, 0, 1) * 10    (20 hard = full)
      rank_score    = clamp(1 - ranking/200000, 0, 1) * 10  (rank < 200k = good)
      final = 0.50*solved + 0.30*rank + 0.20*hard
      floor = 3.0 if any problems solved

  Codeforces (weight: 30%)
    Signals: rating, max_rating, rank title
    Score formula (based on rating bands):
      < 1200  → 2.0 (newbie)
      1200-1399 → 4.0 (pupil)
      1400-1599 → 5.5 (specialist)
      1600-1899 → 7.0 (expert)
      1900-2099 → 8.0 (candidate master)
      2100-2299 → 9.0 (master)
      2300+     → 10.0 (grandmaster+)

  Platform not found / URL not present → score 0.0 (not penalised in final)

Final social score:
  Only platforms with available URLs are scored.
  Weighted average of available platforms.
  If no platforms available → score 4.0 (neutral, no social data)

Anti-hallucination:
  All data comes from live APIs — no LLM involved in this agent.
  Scores derived purely from numeric API values.
  All API failures handled gracefully → score 0 for that platform.
"""

from __future__ import annotations
import re
import httpx
from typing import Optional, Tuple
from utils.logger import get_logger, log_agent_start, log_agent_end, log_agent_error
from utils.validators import clamp_score
from models.state import ResumeGraphState, SocialScore

logger = get_logger("agents.social")

# ── HTTP client settings ───────────────────────────────────────────────────────
TIMEOUT   = 10.0   # seconds per request
HEADERS   = {"User-Agent": "ResumeScorer/1.0 (educational project)"}


# ─── URL parsers ──────────────────────────────────────────────────────────────

def _extract_github_username(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"github\.com/([a-zA-Z0-9\-_]+)/?$", url)
    return m.group(1) if m else None


def _extract_leetcode_username(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"leetcode\.com/(?:u/)?([a-zA-Z0-9\-_]+)/?$", url)
    return m.group(1) if m else None


def _extract_codeforces_username(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"codeforces\.com/(?:profile/)?([a-zA-Z0-9\-_\.]+)/?$", url)
    return m.group(1) if m else None


# ─── GitHub ───────────────────────────────────────────────────────────────────

def _fetch_github(username: str) -> Tuple[Optional[dict], float]:
    """
    Fetch GitHub user data via public API.
    Returns (data_dict, score).
    """
    try:
        with httpx.Client(timeout=TIMEOUT, headers=HEADERS) as client:
            # User profile
            resp = client.get(f"https://api.github.com/users/{username}")
            if resp.status_code == 404:
                logger.warning(f"[SocialAgent] GitHub user '{username}' not found")
                return None, 0.0
            if resp.status_code != 200:
                logger.warning(f"[SocialAgent] GitHub API status {resp.status_code}")
                return None, 0.0

            user = resp.json()

            # Repos for star count
            repos_resp = client.get(
                f"https://api.github.com/users/{username}/repos",
                params={"per_page": 100, "sort": "updated"}
            )
            repos = repos_resp.json() if repos_resp.status_code == 200 else []
            total_stars = sum(r.get("stargazers_count", 0) for r in repos if isinstance(r, dict))
            total_forks = sum(r.get("forks_count", 0) for r in repos if isinstance(r, dict))

            data = {
                "username":    username,
                "public_repos": user.get("public_repos", 0),
                "followers":    user.get("followers", 0),
                "following":    user.get("following", 0),
                "total_stars":  total_stars,
                "total_forks":  total_forks,
                "bio":          user.get("bio"),
                "company":      user.get("company"),
                "blog":         user.get("blog"),
                "profile_url":  user.get("html_url"),
            }

            # Score — relaxed thresholds so good candidates aren't penalised
            # 10 repos = max repos score; 20 stars = max stars; 30 followers = max
            repos_s    = clamp_score(min(data["public_repos"] / 10, 1.0) * 10)
            stars_s    = clamp_score(min(total_stars / 20, 1.0) * 10)
            followers_s= clamp_score(min(data["followers"] / 30, 1.0) * 10)
            score      = clamp_score(0.50 * repos_s + 0.30 * stars_s + 0.20 * followers_s)
            # Floor: any profile with repos gets minimum 3.0
            if data["public_repos"] > 0:
                score = max(score, 3.0)

            data["score_breakdown"] = {
                "repos_score":     round(repos_s, 2),
                "stars_score":     round(stars_s, 2),
                "followers_score": round(followers_s, 2),
            }

            logger.info(
                f"[SocialAgent] GitHub '{username}': repos={data['public_repos']} "
                f"stars={total_stars} followers={data['followers']} → score={score:.2f}"
            )
            return data, score

    except httpx.TimeoutException:
        logger.warning(f"[SocialAgent] GitHub timeout for '{username}'")
        return None, 0.0
    except Exception as e:
        logger.warning(f"[SocialAgent] GitHub error for '{username}': {e}")
        return None, 0.0


# ─── LeetCode ─────────────────────────────────────────────────────────────────

LEETCODE_GRAPHQL = "https://leetcode.com/graphql"
LEETCODE_QUERY   = """
query getUserProfile($username: String!) {
  matchedUser(username: $username) {
    username
    profile { ranking }
    submitStats {
      acSubmissionNum {
        difficulty
        count
      }
    }
  }
}
"""


def _fetch_leetcode(username: str) -> Tuple[Optional[dict], float]:
    """
    Fetch LeetCode stats via public GraphQL API.
    Returns (data_dict, score).
    """
    try:
        with httpx.Client(timeout=TIMEOUT, headers={**HEADERS, "Content-Type": "application/json"}) as client:
            resp = client.post(
                LEETCODE_GRAPHQL,
                json={"query": LEETCODE_QUERY, "variables": {"username": username}},
            )
            if resp.status_code != 200:
                logger.warning(f"[SocialAgent] LeetCode API status {resp.status_code}")
                return None, 0.0

            body = resp.json()
            user = body.get("data", {}).get("matchedUser")
            if not user:
                logger.warning(f"[SocialAgent] LeetCode user '{username}' not found")
                return None, 0.0

            ranking   = user.get("profile", {}).get("ranking", 999999) or 999999
            sub_stats = user.get("submitStats", {}).get("acSubmissionNum", [])

            counts = {"All": 0, "Easy": 0, "Medium": 0, "Hard": 0}
            for item in sub_stats:
                diff  = item.get("difficulty", "All")
                count = item.get("count", 0)
                if diff in counts:
                    counts[diff] = count

            total_solved  = counts["All"]
            easy_solved   = counts["Easy"]
            medium_solved = counts["Medium"]
            hard_solved   = counts["Hard"]

            data = {
                "username":      username,
                "ranking":       ranking,
                "total_solved":  total_solved,
                "easy_solved":   easy_solved,
                "medium_solved": medium_solved,
                "hard_solved":   hard_solved,
                "profile_url":   f"https://leetcode.com/{username}",
            }

            # Score — relaxed thresholds
            # 100 solved = good (was 300); 20 hard = good (was 50); rank 200k = good (was 500k)
            solved_s = clamp_score(min(total_solved / 100, 1.0) * 10)
            hard_s   = clamp_score(min(hard_solved / 20, 1.0) * 10)
            rank_s   = clamp_score(max(0, 1 - ranking / 200_000) * 10)
            score    = clamp_score(0.50 * solved_s + 0.30 * rank_s + 0.20 * hard_s)
            # Floor: any profile with solved problems gets minimum 3.0
            if total_solved > 0:
                score = max(score, 3.0)

            data["score_breakdown"] = {
                "solved_score": round(solved_s, 2),
                "rank_score":   round(rank_s,   2),
                "hard_score":   round(hard_s,   2),
            }

            logger.info(
                f"[SocialAgent] LeetCode '{username}': solved={total_solved} "
                f"hard={hard_solved} rank={ranking} → score={score:.2f}"
            )
            return data, score

    except httpx.TimeoutException:
        logger.warning(f"[SocialAgent] LeetCode timeout for '{username}'")
        return None, 0.0
    except Exception as e:
        logger.warning(f"[SocialAgent] LeetCode error for '{username}': {e}")
        return None, 0.0


# ─── Codeforces ───────────────────────────────────────────────────────────────

CF_RATING_BANDS = [
    (2300, 10.0, "Grandmaster+"),
    (2100, 9.0,  "Master"),
    (1900, 8.0,  "Candidate Master"),
    (1600, 7.0,  "Expert"),
    (1400, 5.5,  "Specialist"),
    (1200, 4.0,  "Pupil"),
    (0,    2.0,  "Newbie"),
]


def _cf_rating_to_score(rating: int) -> Tuple[float, str]:
    for threshold, score, title in CF_RATING_BANDS:
        if rating >= threshold:
            return score, title
    return 2.0, "Newbie"


def _fetch_codeforces(username: str) -> Tuple[Optional[dict], float]:
    """
    Fetch Codeforces user stats via official public API.
    Returns (data_dict, score).
    """
    try:
        with httpx.Client(timeout=TIMEOUT, headers=HEADERS) as client:
            resp = client.get(
                f"https://codeforces.com/api/user.info?handles={username}"
            )
            if resp.status_code != 200:
                logger.warning(f"[SocialAgent] CF API status {resp.status_code}")
                return None, 0.0

            body = resp.json()
            if body.get("status") != "OK":
                logger.warning(f"[SocialAgent] CF user '{username}' not found: {body.get('comment')}")
                return None, 0.0

            user       = body["result"][0]
            rating     = user.get("rating", 0) or 0
            max_rating = user.get("maxRating", 0) or 0
            rank       = user.get("rank", "unrated")
            max_rank   = user.get("maxRank", "unrated")

            # Use max_rating for scoring (best performance)
            score_val, title = _cf_rating_to_score(max(rating, max_rating))

            data = {
                "username":    username,
                "rating":      rating,
                "max_rating":  max_rating,
                "rank":        rank,
                "max_rank":    max_rank,
                "title":       title,
                "contribution":user.get("contribution", 0),
                "profile_url": f"https://codeforces.com/profile/{username}",
            }

            logger.info(
                f"[SocialAgent] Codeforces '{username}': rating={rating} "
                f"max={max_rating} rank='{rank}' → score={score_val:.2f}"
            )
            return data, score_val

    except httpx.TimeoutException:
        logger.warning(f"[SocialAgent] Codeforces timeout for '{username}'")
        return None, 0.0
    except Exception as e:
        logger.warning(f"[SocialAgent] Codeforces error for '{username}': {e}")
        return None, 0.0


# ─── Final score calculation ──────────────────────────────────────────────────

PLATFORM_WEIGHTS = {
    "github":     0.35,
    "leetcode":   0.35,
    "codeforces": 0.30,
}


def _calculate_social_score(
    gh_score: Optional[float],
    lc_score: Optional[float],
    cf_score: Optional[float],
) -> float:
    """
    Weighted average of available platform scores.
    Missing platforms are excluded (weight redistributed).
    If nothing available → neutral 4.0.
    """
    available = {}
    if gh_score is not None: available["github"]     = gh_score
    if lc_score is not None: available["leetcode"]   = lc_score
    if cf_score is not None: available["codeforces"] = cf_score

    if not available:
        return 5.0

    total_weight = sum(PLATFORM_WEIGHTS[p] for p in available)
    weighted_sum = sum(PLATFORM_WEIGHTS[p] * s for p, s in available.items())
    return clamp_score(weighted_sum / total_weight)


# ─── Main agent function ──────────────────────────────────────────────────────

def social_agent(state: ResumeGraphState) -> ResumeGraphState:
    """
    LangGraph node: Agent 6 — Social Profile Scoring

    Input state keys:  parsed_resume
    Output state keys: social_score
    """
    agent_name = "SocialAgent"
    log_agent_start(logger, agent_name, {
        "has_parsed_resume": bool(state.get("parsed_resume")),
    })

    errors        = list(state.get("errors") or [])
    parsed_resume = state.get("parsed_resume") or {}
    personal_info = parsed_resume.get("personal_info") or {}

    github_url     = personal_info.get("github")
    leetcode_url   = personal_info.get("leetcode")
    codeforces_url = personal_info.get("codeforces")

    logger.info(
        f"[{agent_name}] github={bool(github_url)} "
        f"leetcode={bool(leetcode_url)} codeforces={bool(codeforces_url)}"
    )

    github_data     = None
    leetcode_data   = None
    codeforces_data = None
    gh_score        = None
    lc_score        = None
    cf_score        = None

    # ── GitHub ──
    gh_user = _extract_github_username(github_url)
    if gh_user:
        github_data, gh_score = _fetch_github(gh_user)
    else:
        logger.info(f"[{agent_name}] No GitHub URL found — skipping")

    # ── LeetCode ──
    lc_user = _extract_leetcode_username(leetcode_url)
    if lc_user:
        leetcode_data, lc_score = _fetch_leetcode(lc_user)
    else:
        logger.info(f"[{agent_name}] No LeetCode URL found — skipping")

    # ── Codeforces ──
    cf_user = _extract_codeforces_username(codeforces_url)
    if cf_user:
        codeforces_data, cf_score = _fetch_codeforces(cf_user)
    else:
        logger.info(f"[{agent_name}] No Codeforces URL found — skipping")

    # ── Final score ──
    final_score = _calculate_social_score(gh_score, lc_score, cf_score)

    # ── Reasoning ──
    parts = []
    if gh_score  is not None: parts.append(f"GitHub score: {gh_score:.2f}/10")
    if lc_score  is not None: parts.append(f"LeetCode score: {lc_score:.2f}/10")
    if cf_score  is not None: parts.append(f"Codeforces score: {cf_score:.2f}/10")
    if not parts:             parts.append("No social profiles found in resume")

    if github_data:
        parts.append(
            f"GitHub: {github_data.get('public_repos',0)} repos, "
            f"{github_data.get('total_stars',0)} stars, "
            f"{github_data.get('followers',0)} followers"
        )
    if leetcode_data:
        parts.append(
            f"LeetCode: {leetcode_data.get('total_solved',0)} solved "
            f"(H:{leetcode_data.get('hard_solved',0)}), rank #{leetcode_data.get('ranking','?')}"
        )
    if codeforces_data:
        parts.append(
            f"Codeforces: rating {codeforces_data.get('rating',0)} "
            f"(max {codeforces_data.get('max_rating',0)}, {codeforces_data.get('rank','?')})"
        )

    reasoning = " | ".join(parts)

    social_score = SocialScore(
        score=final_score,
        github_data=github_data,
        leetcode_data=leetcode_data,
        codeforces_data=codeforces_data,
        reasoning=reasoning,
    )

    log_agent_end(
        logger, agent_name,
        f"score={final_score:.2f}/10 | gh={gh_score} lc={lc_score} cf={cf_score}"
    )

    return {
        **state,
        "social_score": social_score.model_dump(),
        "errors": errors,
        "current_step": agent_name,
    }