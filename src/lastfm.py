"""
Last.fm API client — the RAG retrieval layer for Vibe Finder AI.

Uses two endpoints (both confirmed working as of 2026):
    artist.getSimilar — returns up to 100 similar artists with a 0-1 match score
    artist.getInfo    — returns artist metadata (tags, listener count, etc.)

The catalog uses fictional artist names, so we map each to a real-world
proxy in data/lastfm_proxies.json. RAG retrieval queries Last.fm for the
proxy artists and uses the returned similarity neighborhoods to:

    1. Validate our content-based ranking against collaborative filtering
    2. Boost songs whose proxy artists appear in Last.fm's similar list
       for the user's mentioned/inferred preferred artists

All methods return None or empty lists on failure — never raise.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.config import (
    LASTFM_API_KEY,
    LASTFM_BASE_URL,
    LASTFM_SIMILAR_LIMIT,
    LASTFM_TIMEOUT,
    LASTFM_USER_AGENT,
    PROXIES_JSON,
)
from src.logger_config import get_logger

logger = get_logger("lastfm")


def load_proxies(path: Path = PROXIES_JSON) -> Dict[str, str]:
    """Load the fictional → real artist mapping. Returns {} if file missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Could not load proxies from %s: %s", path, e)
        return {}


class LastFMClient:
    """Thin wrapper around the Last.fm REST API with caching + graceful failure."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = LASTFM_TIMEOUT):
        self.api_key = api_key or LASTFM_API_KEY
        self.timeout = timeout
        self._cache: Dict[str, object] = {}
        self.proxies = load_proxies()
        self.reverse_proxies = {v.lower(): k for k, v in self.proxies.items()}
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logger.info("Last.fm disabled — no API key configured")

    # ── Public API ────────────────────────────────────────────────────

    def get_similar_artists(self, artist: str, limit: int = LASTFM_SIMILAR_LIMIT) -> List[Dict]:
        """Return list of {name, match, url} dicts. Empty list on any failure."""
        if not self.enabled:
            return []

        cache_key = f"similar::{artist.lower()}::{limit}"
        if cache_key in self._cache:
            logger.debug("Cache HIT  similar(%s)", artist)
            return self._cache[cache_key]  # type: ignore

        params = {
            "method": "artist.getsimilar",
            "artist": artist,
            "api_key": self.api_key,
            "format": "json",
            "limit": limit,
        }
        results = self._fetch(params, parser=self._parse_similar)
        self._cache[cache_key] = results
        logger.info("Last.fm similar(%s) → %d results", artist, len(results))
        return results

    def get_artist_info(self, artist: str) -> Optional[Dict]:
        """Return {name, listeners, playcount, tags} or None on failure."""
        if not self.enabled:
            return None

        cache_key = f"info::{artist.lower()}"
        if cache_key in self._cache:
            logger.debug("Cache HIT  info(%s)", artist)
            return self._cache[cache_key]  # type: ignore

        params = {
            "method": "artist.getinfo",
            "artist": artist,
            "api_key": self.api_key,
            "format": "json",
        }
        info = self._fetch(params, parser=self._parse_info)
        self._cache[cache_key] = info
        if info:
            logger.info("Last.fm info(%s) → %d listeners", artist, info["listeners"])
        return info

    def proxy_for(self, catalog_artist: str) -> Optional[str]:
        """Return the real-world proxy artist for a catalog artist (or None)."""
        return self.proxies.get(catalog_artist)

    def catalog_artist_for_proxy(self, proxy_name: str) -> Optional[str]:
        """Reverse lookup: given a Last.fm artist name, return our catalog artist."""
        return self.reverse_proxies.get(proxy_name.lower())

    def similarity_for_pair(self, catalog_a: str, catalog_b: str) -> float:
        """Cross-paradigm similarity between two catalog artists, via their proxies.

        Returns 0.0 if either artist has no proxy or Last.fm returns no match.
        """
        proxy_a = self.proxy_for(catalog_a)
        proxy_b = self.proxy_for(catalog_b)
        if not proxy_a or not proxy_b or proxy_a == proxy_b:
            return 0.0

        similar = self.get_similar_artists(proxy_a, limit=100)
        target = proxy_b.lower()
        for entry in similar:
            if entry["name"].lower() == target:
                return float(entry.get("match", 0.0))
        return 0.0

    # ── Private helpers ───────────────────────────────────────────────

    def _fetch(self, params: Dict, parser):
        start = time.perf_counter()
        try:
            resp = requests.get(
                LASTFM_BASE_URL,
                params=params,
                timeout=self.timeout,
                headers={"User-Agent": LASTFM_USER_AGENT},
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.warning("Last.fm error %s: %s", data.get("error"), data.get("message"))
                return [] if parser is self._parse_similar else None
            logger.debug("Last.fm OK %s in %dms", params.get("method"), elapsed_ms)
            return parser(data)
        except requests.Timeout:
            logger.warning("Last.fm timeout (%ds) for %s", self.timeout, params.get("artist"))
        except requests.RequestException as e:
            logger.warning("Last.fm request failed: %s", e)
        except (ValueError, KeyError) as e:
            logger.warning("Last.fm parse error: %s", e)
        return [] if parser is self._parse_similar else None

    @staticmethod
    def _parse_similar(data: Dict) -> List[Dict]:
        items = data.get("similarartists", {}).get("artist", [])
        out = []
        for a in items:
            try:
                out.append({
                    "name": a["name"],
                    "match": float(a.get("match", 0.0)),
                    "url": a.get("url", ""),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _parse_info(data: Dict) -> Optional[Dict]:
        artist = data.get("artist")
        if not artist:
            return None
        stats = artist.get("stats", {})
        tags = artist.get("tags", {}).get("tag", [])
        try:
            return {
                "name": artist.get("name", ""),
                "listeners": int(stats.get("listeners", 0)),
                "playcount": int(stats.get("playcount", 0)),
                "tags": [t["name"] for t in tags if "name" in t],
            }
        except (TypeError, ValueError):
            return None
