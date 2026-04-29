"""Shared pytest fixtures for the Vibe Finder AI test suite."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender import load_songs


@pytest.fixture(scope="session")
def songs():
    return load_songs(str(PROJECT_ROOT / "data" / "songs.csv"))


@pytest.fixture
def pop_profile():
    return {
        "genre": "pop", "mood": "happy", "energy": 0.80,
        "valence": 0.82, "danceability": 0.80, "likes_acoustic": False,
    }


@pytest.fixture
def chill_profile():
    return {
        "genre": "lofi", "mood": "chill", "energy": 0.38,
        "valence": 0.58, "danceability": 0.55, "likes_acoustic": True,
    }


@pytest.fixture
def ghost_genre_profile():
    return {
        "genre": "reggaeton", "mood": "happy", "energy": 0.80,
        "valence": 0.80, "danceability": 0.85, "likes_acoustic": False,
    }
