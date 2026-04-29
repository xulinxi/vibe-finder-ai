"""
Claude API integration — three specialized prompts.

Each function uses a distinct system prompt that constrains Claude to
domain-specific behavior measurably different from baseline:

    parse_user_input      → constrained-vocabulary preference parser (returns JSON)
    critique_recommendations → quality-assessor agent (returns structured verdict)
    generate_explanation  → music-recommendation explainer (conversational tone)

A baseline_parse_user_input is also exposed so the test harness can prove
that the specialized prompt produces measurably different output than a
generic parse prompt — required for the Specialization stretch (+2).

All functions return None / fallbacks on failure rather than raising, so
the agent pipeline never crashes when the API is unavailable.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from src.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    CLAUDE_TIMEOUT,
    VALID_GENRES,
    VALID_MOODS,
    VALID_SCORING_MODES,
)
from src.logger_config import get_logger

logger = get_logger("llm")

_client = None


def _get_client():
    """Lazy-init the Anthropic client. Returns None if unavailable/unconfigured."""
    global _client
    if _client is not None:
        return _client
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=CLAUDE_TIMEOUT)
        return _client
    except ImportError:
        logger.warning("anthropic package not installed; Claude features disabled")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to initialize Anthropic client: %s", e)
        return None


def _extract_json(text: str) -> Optional[Dict]:
    """Pull a JSON object out of Claude's response, even if wrapped in markdown."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = match.group(0) if match else text.strip()
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None


def _call_claude(system: str, user: str, max_tokens: int = CLAUDE_MAX_TOKENS) -> Optional[str]:
    """One-shot Claude call. Returns response text or None on any failure."""
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if not resp.content:
            return None
        return resp.content[0].text
    except Exception as e:  # noqa: BLE001
        logger.warning("Claude API call failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────
# 1. PARSE — specialized constrained-vocabulary parser
# ──────────────────────────────────────────────────────────────────────

_PARSE_SYSTEM = f"""You are a music preference parser for a content-based recommender.
Convert the user's natural-language description into a strictly typed JSON object.

CONSTRAINTS — use ONLY values from these lists when filling string fields:
  genre: {json.dumps(VALID_GENRES)}
  mood: {json.dumps(VALID_MOODS)}
  scoring_mode: {json.dumps(VALID_SCORING_MODES)}

Output JSON schema (every key required):
{{
  "genre": "<one of the genres above>",
  "mood": "<one of the moods above>",
  "energy": <float 0.0-1.0>,
  "valence": <float 0.0-1.0>,
  "danceability": <float 0.0-1.0>,
  "likes_acoustic": <boolean>,
  "preferred_decade": "<like '2020s' or null>",
  "preferred_mood_tags": [<list of short descriptor strings, or null>],
  "preferred_artists": [<list of artist names mentioned in the user's text, or null>],
  "scoring_mode": "<one of the scoring modes above>"
}}

Rules:
  - Pick the CLOSEST allowed value, never invent new genres or moods.
  - "preferred_artists" must contain only names actually present in the user input.
  - Energy/valence/danceability are physical features: energy=intensity,
    valence=emotional positivity, danceability=groove. Calibrate against
    typical pop ≈ 0.8 / lofi ≈ 0.4 / classical ≈ 0.2.
  - Choose scoring_mode based on the user's emphasis: "mood-first" for
    emotion-heavy queries, "energy-focused" for workouts, "genre-first"
    when genre is paramount, "full-feature" for nuanced multi-attribute
    queries, "balanced" otherwise.

Return ONLY the JSON object — no commentary, no markdown fences."""


def parse_user_input(raw_text: str) -> Optional[Dict]:
    """Parse NL preferences using the specialized prompt. Returns dict or None."""
    response = _call_claude(_PARSE_SYSTEM, raw_text, max_tokens=512)
    if response is None:
        logger.info("Parse failed (Claude unavailable); caller should fall back")
        return None

    parsed = _extract_json(response)
    if parsed is None:
        logger.warning("Parse failed (no JSON in response): %r", response[:200])
        return None

    logger.info("Parsed input → genre=%s mood=%s energy=%.2f mode=%s",
                parsed.get("genre"), parsed.get("mood"),
                parsed.get("energy", 0.0), parsed.get("scoring_mode"))
    return parsed


# ──────────────────────────────────────────────────────────────────────
# Baseline parse (NO specialization) — used by test harness to prove the
# specialized prompt measurably differs from a vanilla extraction prompt.
# ──────────────────────────────────────────────────────────────────────

_BASELINE_PARSE_SYSTEM = (
    "You are a helpful assistant. Extract the music preferences from the user's "
    "message and return them as a JSON object. Use whatever fields and values "
    "seem appropriate."
)


def baseline_parse_user_input(raw_text: str) -> Optional[Dict]:
    """Unconstrained baseline parse — used ONLY for the specialization comparison."""
    response = _call_claude(_BASELINE_PARSE_SYSTEM, raw_text, max_tokens=512)
    if response is None:
        return None
    return _extract_json(response)


def parse_specialization_diff(parsed: Optional[Dict], baseline: Optional[Dict]) -> Dict:
    """Quantify how much the specialized prompt differs from baseline.

    Returns a dict of comparison metrics suitable for the test harness.
    """
    metrics = {
        "specialized_valid_genre": False,
        "baseline_valid_genre": False,
        "specialized_has_mode": False,
        "baseline_has_mode": False,
        "specialized_field_count": 0,
        "baseline_field_count": 0,
    }
    if parsed:
        metrics["specialized_valid_genre"] = parsed.get("genre") in VALID_GENRES
        metrics["specialized_has_mode"] = parsed.get("scoring_mode") in VALID_SCORING_MODES
        metrics["specialized_field_count"] = len(parsed)
    if baseline:
        metrics["baseline_valid_genre"] = baseline.get("genre") in VALID_GENRES
        metrics["baseline_has_mode"] = baseline.get("scoring_mode") in VALID_SCORING_MODES
        metrics["baseline_field_count"] = len(baseline)
    return metrics


# ──────────────────────────────────────────────────────────────────────
# 2. CRITIQUE — specialized self-assessment agent
# ──────────────────────────────────────────────────────────────────────

_CRITIQUE_SYSTEM = """You are a music-recommendation quality assessor.
Evaluate a list of recommendations against the user's stated preferences and
return a strictly typed JSON verdict.

Output schema (every key required):
{
  "quality_score": <float 0.0-1.0>,
  "issues": [<list of short specific problem strings>],
  "should_retry": <boolean — true ONLY if quality_score < 0.4 AND a different scoring mode would obviously help>,
  "suggested_adjustments": {"mode": "<scoring mode name>", "reason": "<short reason>"} or null
}

Assess these dimensions:
  1. Genre/mood coherence with what the user asked for
  2. Diversity — are all picks the same artist or genre when they shouldn't be?
  3. Score distribution — are top scores reasonable, or does the catalog
     simply not contain a good match?
  4. Feature alignment — do energy/valence/danceability matches make sense?

Be honest: if the catalog has no good match, give a low score and say so.
If the recs are solid, give a high score. Never request a retry just to
nudge things; only retry when a *different mode* would clearly help.

Return ONLY the JSON object."""


def critique_recommendations(
    recommendations: List[Tuple[Dict, float, str]],
    user_prefs: Dict,
) -> Dict:
    """Have Claude self-assess the recommendation list. Returns verdict dict."""
    fallback = {
        "quality_score": 0.5,
        "issues": ["critique unavailable"],
        "should_retry": False,
        "suggested_adjustments": None,
    }

    summary = [
        {
            "title": s["title"],
            "artist": s["artist"],
            "genre": s["genre"],
            "mood": s["mood"],
            "score": round(score, 2),
        }
        for s, score, _ in recommendations
    ]

    user_msg = (
        f"User preferences: {json.dumps(user_prefs)}\n\n"
        f"Recommendations to evaluate: {json.dumps(summary)}\n\n"
        "Assess these recommendations now."
    )

    response = _call_claude(_CRITIQUE_SYSTEM, user_msg, max_tokens=512)
    if response is None:
        logger.info("Critique unavailable; using neutral fallback verdict")
        return fallback

    parsed = _extract_json(response)
    if not isinstance(parsed, dict):
        logger.warning("Critique returned non-JSON: %r", response[:200])
        return fallback

    parsed.setdefault("quality_score", 0.5)
    parsed.setdefault("issues", [])
    parsed.setdefault("should_retry", False)
    parsed.setdefault("suggested_adjustments", None)
    logger.info("Critique → quality=%.2f retry=%s issues=%d",
                parsed.get("quality_score", 0.0),
                parsed.get("should_retry"),
                len(parsed.get("issues", [])))
    return parsed


# ──────────────────────────────────────────────────────────────────────
# 3. EXPLAIN — specialized conversational explainer
# ──────────────────────────────────────────────────────────────────────

_EXPLAIN_SYSTEM = """You are an honest music-recommendation explainer.
Given the user's preferences, the system's top recommendations with their
numeric scoring breakdowns, and any external data retrieved from Last.fm,
write a brief, conversational 2-3 paragraph explanation of why these
songs were chosen.

Style:
  - Conversational, second-person ("you'll like...")
  - Reference 1-2 specific scoring factors per pick
  - Honestly call out limitations or compromises
  - Mention Last.fm validation if it was used
  - Under 220 words total
  - Plain text, no markdown headers"""


def generate_explanation(
    recommendations: List[Tuple[Dict, float, str]],
    user_prefs: Dict,
    rag_context: Optional[Dict] = None,
) -> str:
    """Produce a conversational explanation. Falls back to a generic line on failure."""
    summary = [
        {
            "title": s["title"],
            "artist": s["artist"],
            "genre": s["genre"],
            "mood": s["mood"],
            "score": round(score, 2),
            "scoring_reasons": explanation,
        }
        for s, score, explanation in recommendations
    ]

    rag_blob = ""
    if rag_context:
        rag_blob = f"\n\nLast.fm context: {json.dumps(rag_context)[:600]}"

    user_msg = (
        f"User preferences: {json.dumps(user_prefs)}\n\n"
        f"Recommendations: {json.dumps(summary)}"
        f"{rag_blob}\n\n"
        "Write the explanation now."
    )

    response = _call_claude(_EXPLAIN_SYSTEM, user_msg, max_tokens=512)
    if response is None:
        return ("These picks come from your numeric preference profile. See the "
                "scoring breakdown above for the exact factors behind each rank.")
    return response.strip()
