"""
Vibe Finder AI — Command-line entry point.

Three modes:
    python -m src.main                                  # interactive NL mode
    python -m src.main --query "chill lofi for coding"  # one-shot agentic query
    python -m src.main --demo                           # original rule-based demo

The agentic modes go through src.agent.RecommendationAgent (sanitize → parse →
RAG → score → critique → retry → explain). The demo mode reproduces the
original Module 1-3 behavior verbatim — no API keys needed, fully offline.
"""

import argparse
import sys
from typing import Dict

from src.logger_config import setup_logging
from src.recommender import format_results_table, load_songs, recommend_songs


# ──────────────────────────────────────────────────────────────────────
# Original demo mode — preserved from the rule-based project
# ──────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    """Original Module 1-3 demonstration — 4 hard-coded profiles, all modes."""
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    profiles = {
        "Pop Enthusiast": {"genre": "pop", "mood": "happy", "energy": 0.80,
                           "valence": 0.82, "danceability": 0.80, "likes_acoustic": False},
        "Chill Studier": {"genre": "lofi", "mood": "chill", "energy": 0.38,
                          "valence": 0.58, "danceability": 0.55, "likes_acoustic": True},
        "Workout Warrior": {"genre": "edm", "mood": "energetic", "energy": 0.93,
                            "valence": 0.85, "danceability": 0.90, "likes_acoustic": False},
        "Melancholic Folkster": {"genre": "folk", "mood": "melancholic", "energy": 0.30,
                                 "valence": 0.35, "danceability": 0.40, "likes_acoustic": True},
    }

    print("\n" + "=" * 70)
    print("  SCORING MODE: balanced (default)")
    print("=" * 70)
    for name, prefs in profiles.items():
        print(format_results_table(recommend_songs(prefs, songs, k=5), name, prefs))

    print("\n\n" + "#" * 70)
    print("  CHALLENGE 2: Comparing Scoring Modes for 'Pop Enthusiast'")
    print("#" * 70)
    pop_prefs = profiles["Pop Enthusiast"]
    for mode_name in ["genre-first", "mood-first", "energy-focused"]:
        print(f"\n--- Mode: {mode_name} ---")
        recs = recommend_songs(pop_prefs, songs, k=5, mode=mode_name)
        print(format_results_table(recs, f"Pop Enthusiast [{mode_name}]", pop_prefs))

    print("\n\n" + "#" * 70)
    print("  CHALLENGE 3: Diversity Penalty (Chill Studier)")
    print("#" * 70)
    chill = profiles["Chill Studier"]
    print("\n--- WITHOUT diversity penalty ---")
    print(format_results_table(recommend_songs(chill, songs, k=5),
                               "Chill Studier [no diversity]", chill))
    print("\n--- WITH diversity penalty ---")
    print(format_results_table(recommend_songs(chill, songs, k=5, diversity=True),
                               "Chill Studier [diversity ON]", chill))

    print("\n\n" + "#" * 70)
    print("  CHALLENGE 1: Full-Feature Mode (5 new attributes)")
    print("#" * 70)
    full = {
        "genre": "lofi", "mood": "chill", "energy": 0.38, "valence": 0.58,
        "danceability": 0.55, "likes_acoustic": True,
        "preferred_decade": "2020s",
        "preferred_mood_tags": ["nostalgic", "dreamy"],
        "target_instrumental": 0.85,
        "target_lyrics_sentiment": 0.10,
    }
    recs = recommend_songs(full, songs, k=5, mode="full-feature", diversity=True)
    print(format_results_table(recs, "Chill Studier [full-feature + diversity]", full))


# ──────────────────────────────────────────────────────────────────────
# Agent output rendering
# ──────────────────────────────────────────────────────────────────────

def _render_result(result: Dict) -> None:
    """Pretty-print the dict returned by RecommendationAgent.run()."""
    if result.get("error"):
        print(f"\n[Rejected] {result['error']}\n")
        return

    parsed = result.get("parsed_preferences") or {}
    print(format_results_table(
        result["recommendations"],
        profile_name="Vibe Finder AI",
        user_prefs=parsed,
    ))

    print("\nConfidence:")
    for cs in result["confidence_scores"]:
        print(f"  {cs['song']['title']:<30} {cs['confidence_label']:<7} ({cs['confidence']:.2f})")

    if result.get("rag_enriched"):
        print("\n[Enhanced with Last.fm artist similarity data]")
    if result.get("retried"):
        print("[Refined after self-critique]")

    critique = result.get("critique") or {}
    print(f"\nSelf-critique quality: {critique.get('quality_score', 0.0):.2f}")
    if critique.get("issues"):
        print("Critique issues:")
        for issue in critique["issues"]:
            print(f"  - {issue}")

    print(f"\nExplanation:\n{result['explanation']}")

    if result.get("warnings"):
        print("\nWarnings:")
        for w in result["warnings"]:
            print(f"  - {w}")

    print("\nPipeline trace:")
    for stage, detail in result.get("stage_log", []):
        print(f"  · {stage:<10} {detail}")


def run_agent_query(query: str) -> int:
    """Run a single NL query and exit. Returns 0 on success, 1 on error."""
    from src.agent import RecommendationAgent
    agent = RecommendationAgent()
    result = agent.run(query)
    _render_result(result)
    return 0 if not result.get("error") else 1


def run_interactive() -> None:
    """Interactive REPL — accept NL queries until the user types quit."""
    from src.agent import RecommendationAgent
    print("\n" + "=" * 70)
    print("  Vibe Finder AI — Interactive Mode")
    print("  Describe what you're in the mood for. Type 'quit' to exit.")
    print("=" * 70 + "\n")

    agent = RecommendationAgent()
    while True:
        try:
            text = input("\nWhat are you in the mood for? > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            return
        if not text:
            continue
        result = agent.run(text)
        _render_result(result)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Vibe Finder AI — RAG + Agentic music recommender")
    parser.add_argument("--demo", action="store_true",
                        help="Run the original rule-based demo (no API keys needed)")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Single natural-language query, then exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console log verbosity (default: INFO)")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    if args.demo:
        run_demo()
        return 0
    if args.query:
        return run_agent_query(args.query)
    run_interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
