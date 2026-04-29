"""
Test harness for Vibe Finder AI — stretch feature (+2).

Run from project root:

    python -m tests.test_harness                # runs all suites
    python -m tests.test_harness --skip-api     # skips network-dependent suites
    python -m tests.test_harness --json         # machine-readable output

Three suites:

    1. SCENARIOS — runs each entry in data/test_scenarios.json through the
       full agent pipeline and checks expectations
    2. RAG-comparison — same artist-grounded query with and without Last.fm
       enrichment; reports precision@k against the proxy similarity ground truth
    3. SPECIALIZATION-comparison — same NL input parsed by the specialized
       constrained-vocabulary prompt vs. a vanilla baseline prompt; reports
       which produces catalog-valid output

The two comparison suites supply the "measurable difference" evidence
required for the RAG Enhancement (+2) and Fine-Tuning/Specialization (+2)
stretch points.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import RecommendationAgent  # noqa: E402
from src.config import VALID_GENRES, has_claude_api, has_lastfm_api  # noqa: E402
from src.guardrails import sanitize_user_input  # noqa: E402
from src.lastfm import LastFMClient  # noqa: E402
from src.logger_config import setup_logging  # noqa: E402


SCENARIOS_PATH = PROJECT_ROOT / "data" / "test_scenarios.json"


@dataclass
class HarnessResult:
    name: str
    passed: bool
    detail: str
    elapsed_ms: int
    confidence: Optional[float] = None
    extras: Dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Suite 1 — Scenario suite
# ──────────────────────────────────────────────────────────────────────

def _evaluate_scenario(scenario: Dict, agent: RecommendationAgent) -> HarnessResult:
    name = scenario["name"]
    input_text = scenario["input"]
    expected = scenario.get("expected", {})
    start = time.perf_counter()

    if scenario.get("type") == "guardrail":
        is_safe, msg = sanitize_user_invalid_or_skip(input_text)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        passed = (not is_safe) if expected.get("rejected") else is_safe
        detail = f"sanitizer={'rejected' if not is_safe else 'accepted'} ({msg})"
        return HarnessResult(name=name, passed=passed, detail=detail, elapsed_ms=elapsed_ms)

    try:
        result = agent.run(input_text)
    except Exception as e:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return HarnessResult(name=name, passed=False, detail=f"raised {type(e).__name__}: {e}",
                          elapsed_ms=elapsed_ms)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    checks: List[str] = []
    failures: List[str] = []
    confidence: Optional[float] = None

    if result.get("recommendations"):
        top_genre = result["recommendations"][0][0]["genre"]
        if "top_genre" in expected:
            ok = top_genre == expected["top_genre"]
            checks.append(f"top_genre={top_genre}")
            if not ok:
                failures.append(f"expected top_genre={expected['top_genre']}, got {top_genre}")
        if "top_genre_in" in expected:
            ok = top_genre in expected["top_genre_in"]
            checks.append(f"top_genre={top_genre}")
            if not ok:
                failures.append(
                    f"expected top_genre in {expected['top_genre_in']}, got {top_genre}"
                )
        if result.get("confidence_scores"):
            confidence = result["confidence_scores"][0]["confidence"]
            checks.append(f"confidence={confidence:.2f}")
            if "min_confidence" in expected and confidence < expected["min_confidence"]:
                failures.append(f"confidence {confidence:.2f} < min {expected['min_confidence']}")

    if "rag_enriched" in expected:
        actual_rag = result.get("rag_enriched", False)
        checks.append(f"rag={actual_rag}")
        if expected["rag_enriched"] and not actual_rag:
            failures.append("expected RAG enrichment but none occurred (Last.fm disabled?)")

    detail = " | ".join(checks) if checks else "(no checks)"
    if failures:
        detail += "  ← " + "; ".join(failures)
    return HarnessResult(
        name=name,
        passed=not failures,
        detail=detail,
        elapsed_ms=elapsed_ms,
        confidence=confidence,
    )


def sanitize_user_invalid_or_skip(text: str):
    """Wrapper to match the scenario evaluator's interface."""
    return sanitize_user_input(text)


def run_scenarios(agent: RecommendationAgent) -> List[HarnessResult]:
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    return [_evaluate_scenario(s, agent) for s in scenarios]


# ──────────────────────────────────────────────────────────────────────
# Suite 2 — RAG before/after comparison (+2 RAG Enhancement evidence)
# ──────────────────────────────────────────────────────────────────────

def _precision_at_k(recs, lastfm: LastFMClient, target_artist: str, k: int = 5) -> float:
    """Fraction of top-k recommendations whose proxy artist appears in
    Last.fm's similar-artist list for the user's mentioned artist.
    """
    if not lastfm.enabled:
        return 0.0
    similar = lastfm.get_similar_artists(target_artist, limit=100)
    similar_lower = {s["name"].lower() for s in similar}
    if not similar_lower:
        return 0.0
    hits = 0
    for song, _, _ in recs[:k]:
        proxy = lastfm.proxy_for(song["artist"])
        if proxy and proxy.lower() in similar_lower:
            hits += 1
    return hits / max(1, min(k, len(recs)))


def _count_boosts(recs) -> int:
    """How many recs received a Last.fm bonus (visible in their explanation)."""
    return sum(1 for _, _, expl in recs if "Last.fm" in (expl or ""))


def _count_reorders(off_titles, on_titles) -> int:
    """How many of the top-K positions changed between the two rankings."""
    k = min(len(off_titles), len(on_titles))
    return sum(1 for i in range(k) if off_titles[i] != on_titles[i])


def run_rag_comparison(agent: RecommendationAgent) -> Dict:
    """Same artist-grounded query, RAG on vs. RAG off.

    Methodology: parse the query ONCE (so Claude's non-determinism is held
    constant), then run the rule-based scoring once and apply the RAG boost
    on top. This isolates RAG's effect from Claude variability.

    Reports three complementary signals:
        - precision@5 against Last.fm proxy ground truth (the headline metric)
        - boost_count: how many recs received a Last.fm match bonus
        - reorder_count: how many top-5 positions changed between OFF and ON
    """
    from src.guardrails import apply_defaults
    from src.llm import parse_user_input
    from src.recommender import recommend_songs

    query = "Something smooth and romantic like Frank Ocean"
    target_artist = "Frank Ocean"

    parsed = parse_user_input(query)
    if parsed is None:
        return {
            "query": query, "skipped": True, "reason": "Claude parse failed",
            "delta": 0.0, "rag_off": {}, "rag_on": {},
        }
    parsed = apply_defaults(parsed)
    preferred_artists = parsed.get("preferred_artists") or [target_artist]

    # Single rule-based scoring pass on a wider candidate pool so the boost
    # can lift songs from outside the rule-based top-5
    base_full = recommend_songs(
        parsed, agent.songs, k=15,
        mode=parsed.get("scoring_mode", "full-feature"),
        diversity=True,
    )
    base_recs = base_full[:5]

    # Apply the RAG boost on the wider pool, then take top-5
    rag_context = agent._retrieve_rag({"preferred_artists": preferred_artists})
    if rag_context and preferred_artists:
        boosted_full, _ = agent._apply_rag_boost(base_full, preferred_artists, rag_context)
        boosted_recs = boosted_full[:5]
    else:
        boosted_recs = list(base_recs)

    p_off = _precision_at_k(base_recs, agent.lastfm, target_artist)
    p_on = _precision_at_k(boosted_recs, agent.lastfm, target_artist)

    off_titles = [s["title"] for s, _, _ in base_recs[:5]]
    on_titles = [s["title"] for s, _, _ in boosted_recs[:5]]

    return {
        "query": query,
        "parsed_artists": preferred_artists,
        "rag_off": {
            "precision_at_5": round(p_off, 2),
            "top_artists": [s["artist"] for s, _, _ in base_recs[:5]],
        },
        "rag_on": {
            "precision_at_5": round(p_on, 2),
            "rag_enriched": bool(rag_context),
            "boost_count": _count_boosts(boosted_recs),
            "reorder_count": _count_reorders(off_titles, on_titles),
            "top_artists": [s["artist"] for s, _, _ in boosted_recs[:5]],
        },
        "delta": round(p_on - p_off, 2),
    }


# ──────────────────────────────────────────────────────────────────────
# Suite 3 — Specialization vs. baseline (+2 Fine-Tuning evidence)
# ──────────────────────────────────────────────────────────────────────

def run_specialization_comparison() -> Dict:
    """Parse the same input through specialized + baseline prompts; compare."""
    if not has_claude_api():
        return {"skipped": True, "reason": "Claude API key not configured"}

    from src.llm import baseline_parse_user_input, parse_specialization_diff, parse_user_input

    test_input = "I want something kind of dreamy and lo-fi for studying"
    specialized = parse_user_input(test_input)
    baseline = baseline_parse_user_input(test_input)
    diff = parse_specialization_diff(specialized, baseline)

    return {
        "input": test_input,
        "specialized": specialized,
        "baseline": baseline,
        "metrics": diff,
        "interpretation": (
            "Specialized prompt produces catalog-valid genre and a chosen "
            "scoring_mode — both required for the recommender to use the "
            "output directly. Baseline prompt may produce free-text values "
            "that fail validation."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────

def _print_report(scenario_results: List[HarnessResult], rag_result: Dict, spec_result: Dict) -> int:
    bar = "=" * 78
    print(f"\n{bar}")
    print("  VIBE FINDER AI — TEST HARNESS")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Claude={'on' if has_claude_api() else 'off'}  "
          f"Last.fm={'on' if has_lastfm_api() else 'off'}  "
          f"Catalog={len(VALID_GENRES)} genres")
    print(bar)

    print("\n[SUITE 1: Scenario tests]")
    passed = sum(1 for r in scenario_results if r.passed)
    failed = len(scenario_results) - passed
    confidences = [r.confidence for r in scenario_results if r.confidence is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    for r in scenario_results:
        flag = "PASS" if r.passed else "FAIL"
        conf_str = f" conf={r.confidence:.2f}" if r.confidence is not None else ""
        print(f"  [{flag}] {r.name:<38} ({r.elapsed_ms:>4}ms){conf_str}")
        if not r.passed:
            print(f"         · {r.detail}")
        elif r.detail and r.detail != "(no checks)":
            print(f"         · {r.detail}")

    print(f"\n  Scenarios: {passed} passed / {failed} failed   "
          f"avg confidence: {avg_conf:.2f}")

    print("\n[SUITE 2: RAG before/after comparison]")
    if not rag_result.get("rag_on", {}).get("rag_enriched") and not has_lastfm_api():
        print("  SKIPPED — Last.fm API key not configured")
    else:
        print(f"  Query: {rag_result['query']!r}")
        print(f"  RAG OFF  precision@5={rag_result['rag_off']['precision_at_5']}  "
              f"top={rag_result['rag_off']['top_artists']}")
        print(f"  RAG ON   precision@5={rag_result['rag_on']['precision_at_5']}  "
              f"boost_count={rag_result['rag_on']['boost_count']}  "
              f"reorder_count={rag_result['rag_on']['reorder_count']}/5")
        print(f"           top={rag_result['rag_on']['top_artists']}")
        rag_evidence = (
            "ranking changed" if rag_result["rag_on"]["reorder_count"] > 0
            else "no ranking change"
        )
        if rag_result["delta"] > 0:
            rag_evidence = "Δprecision@5 positive"
        print(f"  Δ precision@5 = {rag_result['delta']:+.2f}  ({rag_evidence})")

    print("\n[SUITE 3: Specialization vs. baseline parse]")
    if spec_result.get("skipped"):
        print(f"  SKIPPED — {spec_result.get('reason')}")
    else:
        m = spec_result["metrics"]
        print(f"  Input: {spec_result['input']!r}")
        print(f"  Specialized → genre={spec_result['specialized'].get('genre') if spec_result['specialized'] else None}  "
              f"valid={m['specialized_valid_genre']}  "
              f"has_mode={m['specialized_has_mode']}")
        print(f"  Baseline    → genre={spec_result['baseline'].get('genre') if spec_result['baseline'] else None}  "
              f"valid={m['baseline_valid_genre']}  "
              f"has_mode={m['baseline_has_mode']}")

    print(f"\n{bar}")
    print(f"  SUMMARY: {passed}/{passed+failed} scenarios passed | "
          f"avg conf {avg_conf:.2f} | "
          f"RAG Δ={rag_result.get('delta', 0.0):+.2f}")
    print(bar + "\n")

    return 0 if failed == 0 else 1


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Vibe Finder AI test harness")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip suites that require external APIs (RAG/specialization)")
    parser.add_argument("--json", action="store_true",
                        help="Print results as JSON instead of formatted report")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(level=args.log_level, log_file="test_harness.log")

    agent = RecommendationAgent()
    scenario_results = run_scenarios(agent)

    rag_result: Dict = {"delta": 0.0}
    spec_result: Dict = {"skipped": True, "reason": "skipped (--skip-api)"}
    if not args.skip_api:
        rag_result = run_rag_comparison(agent)
        spec_result = run_specialization_comparison()

    if args.json:
        payload = {
            "scenarios": [r.__dict__ for r in scenario_results],
            "rag_comparison": rag_result,
            "specialization_comparison": spec_result,
        }
        print(json.dumps(payload, indent=2, default=str))
        failed = sum(1 for r in scenario_results if not r.passed)
        return 0 if failed == 0 else 1

    return _print_report(scenario_results, rag_result, spec_result)


if __name__ == "__main__":
    sys.exit(main())
