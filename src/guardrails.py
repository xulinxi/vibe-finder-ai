"""
Guardrails for the Vibe Finder AI pipeline.

Implements three categories of safety checks:

    1. INPUT guardrails — sanitize raw user text and validate structured prefs
    2. OUTPUT guardrails — verify recommendations are well-formed and coherent
    3. CONFIDENCE scoring — rate each recommendation high / medium / low

Every check returns either a (bool, list[str]) tuple or a value — never
raises — so the agent can route around problems gracefully.
"""

from typing import Dict, List, Optional, Tuple

from src.config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    MAX_INPUT_LENGTH,
    MAX_POSSIBLE_SCORE,
    VALID_GENRES,
    VALID_MOODS,
    VALID_SCORING_MODES,
)
from src.logger_config import get_logger

logger = get_logger("guardrails")


# ──────────────────────────────────────────────────────────────────────
# INPUT GUARDRAILS
# ──────────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard previous",
    "system prompt",
    "you are now",
    "override your",
    "reveal your prompt",
    "print your instructions",
]


def sanitize_user_input(raw_text: str) -> Tuple[bool, str]:
    """Reject empty, oversized, or injection-flavored input.

    Returns (is_safe, sanitized_text_or_error_message).
    """
    if not raw_text or not raw_text.strip():
        logger.warning("Input rejected: empty")
        return False, "Empty input"

    if len(raw_text) > MAX_INPUT_LENGTH:
        logger.warning("Input rejected: too long (%d chars)", len(raw_text))
        return False, f"Input too long (max {MAX_INPUT_LENGTH} characters)"

    lowered = raw_text.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lowered:
            logger.warning("Input rejected: injection pattern '%s'", pattern)
            return False, "Input rejected: suspicious instruction-override pattern detected"

    return True, raw_text.strip()


def validate_parsed_input(parsed: Dict) -> Tuple[bool, List[str]]:
    """Validate Claude's parsed preference dict.

    Returns (is_valid, list_of_warnings). Warnings do not block the pipeline;
    the agent applies defaults for any field that fails validation.
    """
    warnings: List[str] = []
    required = ["genre", "mood", "energy", "valence", "danceability", "likes_acoustic"]

    for field in required:
        if field not in parsed:
            warnings.append(f"Missing required field: {field}")

    for field in ("energy", "valence", "danceability"):
        if field in parsed:
            val = parsed[field]
            if not isinstance(val, (int, float)):
                warnings.append(f"{field} not numeric: {val!r}")
            elif val < 0.0 or val > 1.0:
                warnings.append(f"{field} out of range [0,1]: {val}")

    if "likes_acoustic" in parsed and not isinstance(parsed["likes_acoustic"], bool):
        warnings.append(f"likes_acoustic not boolean: {parsed['likes_acoustic']!r}")

    if "genre" in parsed and parsed["genre"] not in VALID_GENRES:
        warnings.append(f"Unknown genre: '{parsed['genre']}' (not in catalog)")

    if "mood" in parsed and parsed["mood"] not in VALID_MOODS:
        warnings.append(f"Unknown mood: '{parsed['mood']}'")

    if "scoring_mode" in parsed and parsed["scoring_mode"] not in VALID_SCORING_MODES:
        warnings.append(f"Unknown scoring_mode: '{parsed['scoring_mode']}'")

    if warnings:
        logger.warning("Parsed input validation warnings: %s", warnings)
    return (len(warnings) == 0, warnings)


def detect_hallucination(parsed: Dict, raw_input: str) -> List[str]:
    """Flag artist names that Claude introduced but the user never mentioned.

    Heuristic: every artist in `preferred_artists` should appear (case-insensitive)
    as a substring of the original input. If not, the LLM probably invented it.
    """
    warnings: List[str] = []
    artists = parsed.get("preferred_artists") or []
    raw_lower = raw_input.lower()
    for artist in artists:
        if not isinstance(artist, str):
            continue
        if artist.lower() not in raw_lower:
            warnings.append(f"Possible hallucination: artist '{artist}' not in user input")
    if warnings:
        logger.warning("Hallucination warnings: %s", warnings)
    return warnings


def apply_defaults(parsed: Dict) -> Dict:
    """Fill in missing or invalid fields with safe defaults so scoring can proceed."""
    defaults = {
        "genre": "pop", "mood": "happy", "energy": 0.5, "valence": 0.5,
        "danceability": 0.5, "likes_acoustic": False,
    }
    for key, default in defaults.items():
        if key not in parsed or parsed.get(key) is None:
            parsed[key] = default

    for field in ("energy", "valence", "danceability"):
        try:
            parsed[field] = max(0.0, min(1.0, float(parsed[field])))
        except (TypeError, ValueError):
            parsed[field] = defaults[field]

    if not isinstance(parsed.get("likes_acoustic"), bool):
        parsed["likes_acoustic"] = bool(parsed.get("likes_acoustic"))

    if parsed.get("genre") not in VALID_GENRES:
        logger.info("Genre '%s' not in catalog — keeping as-is for scoring",
                    parsed.get("genre"))

    if parsed.get("scoring_mode") not in VALID_SCORING_MODES:
        parsed["scoring_mode"] = "full-feature"

    # Normalize None list-fields to [] so the rule-based scorer can iterate
    # them safely. Claude's parse schema permits null for these optional
    # fields, but recommender.score_song unconditionally calls set() on them.
    for list_field in ("preferred_mood_tags", "preferred_artists"):
        if parsed.get(list_field) is None:
            parsed[list_field] = []

    return parsed


# ──────────────────────────────────────────────────────────────────────
# OUTPUT GUARDRAILS
# ──────────────────────────────────────────────────────────────────────

def validate_recommendations(
    recommendations: List[Tuple[Dict, float, str]],
    user_prefs: Dict,
    catalog: Optional[List[Dict]] = None,
) -> List[str]:
    """Post-hoc sanity checks on the recommendation list. Returns list of warnings."""
    warnings: List[str] = []

    if not recommendations:
        warnings.append("No recommendations generated")
        return warnings

    avg = sum(s for _, s, _ in recommendations) / len(recommendations)
    if avg < 1.0:
        warnings.append(f"Very low average score ({avg:.2f}); preferences may not match catalog")

    titles = [s["title"] for s, _, _ in recommendations]
    if len(titles) != len(set(titles)):
        warnings.append("Duplicate songs in recommendations")

    for _, score, _ in recommendations:
        if score < 0:
            warnings.append(f"Negative score detected: {score:.2f}")
            break

    if catalog is not None:
        catalog_titles = {s["title"] for s in catalog}
        for song, _, _ in recommendations:
            if song["title"] not in catalog_titles:
                warnings.append(f"Hallucinated title not in catalog: '{song['title']}'")

    genres = {s["genre"] for s, _, _ in recommendations}
    if len(genres) == 1 and len(recommendations) >= 3:
        warnings.append(f"All recommendations share genre '{next(iter(genres))}' — low diversity")

    if recommendations:
        top_genre = recommendations[0][0]["genre"]
        if user_prefs.get("genre") and top_genre != user_prefs["genre"]:
            warnings.append(
                f"Top recommendation genre '{top_genre}' differs from requested '{user_prefs['genre']}'"
            )

    if warnings:
        logger.warning("Output validation warnings: %s", warnings)
    return warnings


# ──────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORING
# ──────────────────────────────────────────────────────────────────────

def _label(ratio: float) -> str:
    if ratio >= CONFIDENCE_HIGH:
        return "high"
    if ratio >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def compute_confidence(
    recommendations: List[Tuple[Dict, float, str]],
    max_score: float = MAX_POSSIBLE_SCORE,
    rag_agreements: Optional[List[bool]] = None,
) -> List[Dict]:
    """Compute high/medium/low confidence per recommendation.

    Confidence measures "how strong is this match for the user", so it must
    correlate with rank — a better-ranked item should never report lower
    confidence than a worse-ranked one.

    Components:
        - magnitude: score / max_possible_score (primary signal)
        - top_gap:   decisiveness over the cutoff item (uniform across the
                     list, so it doesn't invert ranking)
        - agreement: did Last.fm agree? (small bonus when available)
    """
    if not recommendations:
        return []

    n = len(recommendations)
    last_score = recommendations[-1][1]
    top_gap_raw = max(0.0, recommendations[0][1] - last_score)
    top_gap = max(0.0, min(1.0, top_gap_raw / 1.5))

    results = []
    for i, (song, score, explanation) in enumerate(recommendations):
        magnitude = max(0.0, min(1.0, score / max_score)) if max_score > 0 else 0.0

        agrees = None
        if rag_agreements is not None and i < len(rag_agreements):
            agrees = rag_agreements[i]

        if agrees is None:
            confidence = 0.75 * magnitude + 0.25 * top_gap
        else:
            agreement = 1.0 if agrees else 0.0
            confidence = 0.60 * magnitude + 0.20 * top_gap + 0.20 * agreement

        confidence = round(max(0.0, min(1.0, confidence)), 2)
        results.append({
            "song": song,
            "score": round(score, 2),
            "explanation": explanation,
            "confidence": confidence,
            "confidence_label": _label(confidence),
            "rag_agreement": agrees,
        })

    return results
