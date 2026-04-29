"""
Agentic orchestrator for Vibe Finder AI.

Implements a 9-stage pipeline that wraps the original rule-based scoring
engine with retrieval (Last.fm), LLM-powered preference parsing, and an
LLM-powered self-critique loop.

Stages:
    1. SANITIZE — input guardrails (reject empty/oversized/injection)
    2. PARSE    — Claude turns NL into structured prefs
    3. VALIDATE — type + range checks, hallucination detection, defaults
    4. RETRIEVE — Last.fm RAG: similar artists for any user-mentioned artist
    5. SCORE    — call recommender.recommend_songs (the original engine)
    6. BOOST    — add Last.fm match score on top of rule-based score
    7. CRITIQUE — Claude self-assesses; may request a re-score
    8. RETRY    — at most one re-score with a different scoring mode
    9. EXPLAIN  — Claude writes conversational summary; confidence labels added

Every stage logs its input/output, so the pipeline's reasoning is fully
observable. Every external call (Claude, Last.fm) has a fallback path so
the worst-case behavior is "rule-based recommender with warnings".
"""

from typing import Dict, List, Optional, Tuple

from src.config import (
    DEFAULT_K,
    MAX_RETRIES,
    QUALITY_GATE_MIN_SCORE,
    SONGS_CSV,
    has_claude_api,
    has_lastfm_api,
)
from src.guardrails import (
    apply_defaults,
    compute_confidence,
    detect_hallucination,
    sanitize_user_input,
    validate_parsed_input,
    validate_recommendations,
)
from src.lastfm import LastFMClient
from src.llm import (
    critique_recommendations,
    generate_explanation,
    parse_user_input,
)
from src.logger_config import get_logger
from src.recommender import load_songs, recommend_songs

logger = get_logger("agent")


# Conservative fallback parser used when Claude is unavailable.
# Picks a default profile that produces a sensible-looking recommendation
# rather than crashing. Not as good as Claude, but lets the demo run offline.
_KEYWORD_HINTS = {
    "lofi": {"genre": "lofi", "mood": "chill", "energy": 0.4, "valence": 0.55, "danceability": 0.55, "likes_acoustic": True, "scoring_mode": "mood-first"},
    "lo-fi": {"genre": "lofi", "mood": "chill", "energy": 0.4, "valence": 0.55, "danceability": 0.55, "likes_acoustic": True, "scoring_mode": "mood-first"},
    "study": {"genre": "lofi", "mood": "focused", "energy": 0.4, "valence": 0.5, "danceability": 0.4, "likes_acoustic": True, "scoring_mode": "mood-first"},
    "workout": {"genre": "edm", "mood": "energetic", "energy": 0.9, "valence": 0.85, "danceability": 0.9, "likes_acoustic": False, "scoring_mode": "energy-focused"},
    "gym": {"genre": "edm", "mood": "energetic", "energy": 0.9, "valence": 0.85, "danceability": 0.9, "likes_acoustic": False, "scoring_mode": "energy-focused"},
    "party": {"genre": "pop", "mood": "happy", "energy": 0.85, "valence": 0.85, "danceability": 0.9, "likes_acoustic": False, "scoring_mode": "energy-focused"},
    "sad": {"genre": "folk", "mood": "melancholic", "energy": 0.3, "valence": 0.3, "danceability": 0.4, "likes_acoustic": True, "scoring_mode": "mood-first"},
    "metal": {"genre": "metal", "mood": "aggressive", "energy": 0.95, "valence": 0.45, "danceability": 0.5, "likes_acoustic": False, "scoring_mode": "genre-first"},
    "jazz": {"genre": "jazz", "mood": "relaxed", "energy": 0.45, "valence": 0.65, "danceability": 0.55, "likes_acoustic": True, "scoring_mode": "genre-first"},
    "classical": {"genre": "classical", "mood": "somber", "energy": 0.3, "valence": 0.45, "danceability": 0.3, "likes_acoustic": True, "scoring_mode": "genre-first"},
}


def _fallback_parse(text: str) -> Dict:
    """Keyword-based parse for offline/fallback mode (no Claude)."""
    lower = text.lower()
    for kw, profile in _KEYWORD_HINTS.items():
        if kw in lower:
            logger.info("Fallback parse matched keyword '%s'", kw)
            return dict(profile)
    logger.info("Fallback parse using neutral default")
    return {
        "genre": "pop", "mood": "happy", "energy": 0.6, "valence": 0.6,
        "danceability": 0.6, "likes_acoustic": False, "scoring_mode": "balanced",
    }


class RecommendationAgent:
    """Multi-step agentic recommendation pipeline."""

    def __init__(self, songs_path: Optional[str] = None):
        path = str(songs_path) if songs_path else str(SONGS_CSV)
        self.songs = load_songs(path)
        self.lastfm = LastFMClient() if has_lastfm_api() else LastFMClient(api_key=None)
        logger.info(
            "Agent ready · songs=%d · claude=%s · lastfm=%s",
            len(self.songs),
            "on" if has_claude_api() else "off",
            "on" if (has_lastfm_api() and self.lastfm.enabled) else "off",
        )

    # ── Public API ────────────────────────────────────────────────────

    def run(self, user_input: str, k: int = DEFAULT_K, use_rag: bool = True) -> Dict:
        """Execute the full pipeline. Returns a result dict (never raises)."""
        result = {
            "input": user_input,
            "recommendations": [],
            "confidence_scores": [],
            "explanation": "",
            "critique": None,
            "warnings": [],
            "rag_enriched": False,
            "retried": False,
            "parsed_preferences": {},
            "stage_log": [],
            "error": None,
        }

        # Stage 1: SANITIZE
        is_safe, sanitized = sanitize_user_input(user_input)
        result["stage_log"].append(("sanitize", "ok" if is_safe else f"reject: {sanitized}"))
        if not is_safe:
            result["error"] = sanitized
            return result

        # Stage 2: PARSE
        parsed = parse_user_input(sanitized) if has_claude_api() else None
        if parsed is None:
            parsed = _fallback_parse(sanitized)
            result["stage_log"].append(("parse", "fallback (keyword)"))
        else:
            result["stage_log"].append(("parse", "claude"))

        # Stage 3: VALIDATE + hallucination detection
        _, validation_warnings = validate_parsed_input(parsed)
        hallucinations = detect_hallucination(parsed, sanitized)
        parsed = apply_defaults(parsed)
        result["parsed_preferences"] = dict(parsed)
        result["warnings"].extend(validation_warnings)
        result["warnings"].extend(hallucinations)
        result["stage_log"].append(("validate", f"{len(validation_warnings)} warnings"))

        # Stage 4: RETRIEVE (Last.fm RAG)
        rag_context = {}
        if use_rag and self.lastfm.enabled:
            rag_context = self._retrieve_rag(parsed)
            result["rag_enriched"] = bool(rag_context)
            result["stage_log"].append(("retrieve", f"{len(rag_context)} entries"))
        else:
            result["stage_log"].append(("retrieve", "skipped"))

        # Stage 5: SCORE
        # When RAG is on, score a wider candidate pool so the boost can
        # promote songs from outside the rule-based top-k.
        scoring_mode = parsed.get("scoring_mode", "full-feature")
        preferred_artists = parsed.get("preferred_artists") or []
        candidate_k = max(k * 3, 12) if (use_rag and rag_context) else k
        recs = recommend_songs(
            parsed, self.songs, k=candidate_k, mode=scoring_mode, diversity=True,
        )
        result["stage_log"].append(("score", f"{len(recs)} candidates"))

        # Stage 6: BOOST (Last.fm similarity)
        rag_agreements: List[bool] = [False] * len(recs)
        if rag_context and preferred_artists:
            recs, rag_agreements = self._apply_rag_boost(recs, preferred_artists, rag_context)
            result["stage_log"].append(("boost", f"{sum(rag_agreements)} matches"))

        # Truncate to k now that the boost has had a chance to re-rank
        recs = recs[:k]
        rag_agreements = rag_agreements[:k]

        # Stage 7: CRITIQUE
        critique = critique_recommendations(recs, parsed) if has_claude_api() else {
            "quality_score": 0.6, "issues": [], "should_retry": False,
            "suggested_adjustments": None,
        }
        result["critique"] = critique
        result["stage_log"].append(("critique", f"quality={critique.get('quality_score', 0.0):.2f}"))

        # Stage 8: RETRY (one shot)
        if (
            critique.get("should_retry")
            and critique.get("quality_score", 1.0) < QUALITY_GATE_MIN_SCORE
            and isinstance(critique.get("suggested_adjustments"), dict)
            and MAX_RETRIES > 0
        ):
            new_mode = critique["suggested_adjustments"].get("mode", scoring_mode)
            if new_mode != scoring_mode:
                logger.info("Retrying with mode=%s (was %s)", new_mode, scoring_mode)
                recs = recommend_songs(parsed, self.songs, k=k, mode=new_mode, diversity=True)
                result["retried"] = True
                result["stage_log"].append(("retry", f"mode={new_mode}"))
                if rag_context and preferred_artists:
                    recs, rag_agreements = self._apply_rag_boost(recs, preferred_artists, rag_context)

        # Stage 9: EXPLAIN + confidence + output validation
        explanation = generate_explanation(recs, parsed, rag_context) if has_claude_api() else (
            "These picks come from your numeric preference profile. See the "
            "scoring breakdown for each pick's contributing factors."
        )
        confidences = compute_confidence(recs, rag_agreements=rag_agreements if rag_context else None)
        output_warnings = validate_recommendations(recs, parsed, catalog=self.songs)
        result["recommendations"] = recs
        result["confidence_scores"] = confidences
        result["explanation"] = explanation
        result["warnings"].extend(output_warnings)
        result["stage_log"].append(("explain", f"{len(confidences)} scored"))

        logger.info(
            "Pipeline done · k=%d · rag=%s · retried=%s · warnings=%d",
            len(recs), result["rag_enriched"], result["retried"], len(result["warnings"]),
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────

    def _retrieve_rag(self, parsed: Dict) -> Dict:
        """Pull similar-artist data for any artist the user mentioned."""
        ctx: Dict = {}
        artists = parsed.get("preferred_artists") or []
        for artist in artists[:3]:
            if not isinstance(artist, str):
                continue
            similar = self.lastfm.get_similar_artists(artist)
            if similar:
                ctx[f"similar_to::{artist}"] = similar
        return ctx

    def _apply_rag_boost(
        self,
        recs: List[Tuple[Dict, float, str]],
        preferred_artists: List[str],
        rag_context: Dict,
    ) -> Tuple[List[Tuple[Dict, float, str]], List[bool]]:
        """Add a similarity bonus when a recommended song's proxy artist appears
        in the Last.fm similar-artists list for one of the user's preferred artists.

        Returns (boosted_recs, agreements) where agreements[i] is True when
        recs[i] received a Last.fm boost.
        """
        boosted: List[Tuple[Dict, float, str]] = []
        agreements: List[bool] = []

        for song, score, explanation in recs:
            proxy = self.lastfm.proxy_for(song["artist"])
            best_match = 0.0
            matched_via = None
            if proxy:
                proxy_lower = proxy.lower()
                for pref in preferred_artists:
                    key = f"similar_to::{pref}"
                    similar = rag_context.get(key) or []
                    for entry in similar:
                        if entry["name"].lower() == proxy_lower:
                            if entry["match"] > best_match:
                                best_match = entry["match"]
                                matched_via = pref
                            break
            if best_match > 0 and matched_via:
                bonus = best_match * 1.5
                explanation = f"{explanation}; Last.fm: similar to {matched_via} (+{bonus:.2f})"
                boosted.append((song, score + bonus, explanation))
                agreements.append(True)
            else:
                boosted.append((song, score, explanation))
                agreements.append(False)

        ranked = sorted(zip(boosted, agreements), key=lambda x: x[0][1], reverse=True)
        sorted_recs = [item[0] for item in ranked]
        sorted_agreements = [item[1] for item in ranked]
        return sorted_recs, sorted_agreements
