"""
Centralized configuration for Vibe Finder AI.

Loads API keys from .env (via python-dotenv) and exposes constants used
across the agent pipeline. Keep all magic numbers and strings here so the
rest of the codebase stays free of hardcoded values.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- API keys (loaded from environment / .env) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

# --- Claude API settings ---
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 1024
CLAUDE_TIMEOUT = 30

# --- Last.fm API settings ---
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"
LASTFM_TIMEOUT = 5
LASTFM_SIMILAR_LIMIT = 50
LASTFM_USER_AGENT = "VibeFinderAI/1.0"

# --- Confidence thresholds (score / max_score ratio) ---
MAX_POSSIBLE_SCORE = 8.0
CONFIDENCE_HIGH = 0.55
CONFIDENCE_MEDIUM = 0.30

# --- Pipeline behavior ---
MAX_RETRIES = 1
QUALITY_GATE_MIN_SCORE = 0.40
DEFAULT_K = 5

# --- Catalog constraints (derived from songs.csv) ---
VALID_GENRES = [
    "pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop",
    "hip hop", "folk", "electronic", "blues", "metal", "r&b", "latin",
    "classical", "edm", "country",
]
VALID_MOODS = [
    "happy", "chill", "intense", "focused", "moody", "relaxed", "energetic",
    "melancholic", "euphoric", "nostalgic", "aggressive", "romantic",
    "joyful", "somber", "uplifting", "wistful",
]
VALID_SCORING_MODES = [
    "balanced", "genre-first", "mood-first", "energy-focused", "full-feature",
]

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / ".cache"
SONGS_CSV = DATA_DIR / "songs.csv"
PROXIES_JSON = DATA_DIR / "lastfm_proxies.json"

# --- Input limits ---
MAX_INPUT_LENGTH = 500


def has_claude_api() -> bool:
    """Return True if a Claude API key is configured."""
    return bool(ANTHROPIC_API_KEY)


def has_lastfm_api() -> bool:
    """Return True if a Last.fm API key is configured."""
    return bool(LASTFM_API_KEY)
