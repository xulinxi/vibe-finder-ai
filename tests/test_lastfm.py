"""Unit tests for src/lastfm.py — RAG retrieval layer (mocked HTTP)."""

from unittest.mock import MagicMock, patch

import requests

from src.lastfm import LastFMClient, load_proxies


_FAKE_SIMILAR_RESPONSE = {
    "similarartists": {
        "artist": [
            {"name": "Massive Attack", "match": "0.92", "url": "https://example.com/1"},
            {"name": "Portishead", "match": "0.85", "url": "https://example.com/2"},
            {"name": "Burial", "match": "0.71", "url": "https://example.com/3"},
        ]
    }
}

_FAKE_INFO_RESPONSE = {
    "artist": {
        "name": "The Weeknd",
        "stats": {"listeners": "5000000", "playcount": "200000000"},
        "tags": {"tag": [{"name": "rnb"}, {"name": "pop"}]},
    }
}


def _client_with_key():
    return LastFMClient(api_key="fake-key-for-tests")


# ── load_proxies ──────────────────────────────────────────────────────

def test_load_proxies_returns_mapping():
    proxies = load_proxies()
    assert "LoRoom" in proxies
    assert proxies["LoRoom"] == "Nujabes"
    assert "_meta" not in proxies  # underscore keys filtered


# ── get_similar_artists ───────────────────────────────────────────────

def test_get_similar_artists_parses_response():
    client = _client_with_key()
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = _FAKE_SIMILAR_RESPONSE
    with patch("src.lastfm.requests.get", return_value=mock_response):
        results = client.get_similar_artists("The Weeknd")
    assert len(results) == 3
    assert results[0]["name"] == "Massive Attack"
    assert results[0]["match"] == 0.92


def test_get_similar_artists_caches():
    client = _client_with_key()
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = _FAKE_SIMILAR_RESPONSE
    with patch("src.lastfm.requests.get", return_value=mock_response) as mock_get:
        client.get_similar_artists("The Weeknd")
        client.get_similar_artists("The Weeknd")
    assert mock_get.call_count == 1  # second call hits cache


def test_get_similar_artists_handles_timeout():
    client = _client_with_key()
    with patch("src.lastfm.requests.get", side_effect=requests.Timeout):
        results = client.get_similar_artists("The Weeknd")
    assert results == []


def test_get_similar_artists_handles_api_error():
    client = _client_with_key()
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"error": 6, "message": "The artist you supplied could not be found"}
    with patch("src.lastfm.requests.get", return_value=mock_response):
        results = client.get_similar_artists("Nonexistent Artist")
    assert results == []


def test_disabled_client_returns_empty_without_calling():
    # Force-disable by zeroing out the env-var fallback inside this test scope,
    # so the test passes regardless of whether the developer has a real
    # LASTFM_API_KEY in .env on their machine.
    with patch("src.lastfm.LASTFM_API_KEY", None):
        client = LastFMClient(api_key=None)
        assert client.enabled is False
        with patch("src.lastfm.requests.get") as mock_get:
            results = client.get_similar_artists("The Weeknd")
        assert results == []
        mock_get.assert_not_called()


# ── get_artist_info ───────────────────────────────────────────────────

def test_get_artist_info_parses_response():
    client = _client_with_key()
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = _FAKE_INFO_RESPONSE
    with patch("src.lastfm.requests.get", return_value=mock_response):
        info = client.get_artist_info("The Weeknd")
    assert info["name"] == "The Weeknd"
    assert info["listeners"] == 5000000
    assert "rnb" in info["tags"]


# ── proxy lookup helpers ──────────────────────────────────────────────

def test_proxy_for_known_artist():
    client = LastFMClient(api_key=None)
    assert client.proxy_for("LoRoom") == "Nujabes"


def test_proxy_for_unknown_artist_returns_none():
    client = LastFMClient(api_key=None)
    assert client.proxy_for("Madeup Band") is None
