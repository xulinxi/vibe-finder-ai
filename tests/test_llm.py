"""Unit tests for src/llm.py — Claude integration (mocked client)."""

from unittest.mock import MagicMock, patch

from src.llm import (
    _extract_json,
    critique_recommendations,
    generate_explanation,
    parse_specialization_diff,
    parse_user_input,
)


def _mock_claude_response(text: str):
    response = MagicMock()
    block = MagicMock()
    block.text = text
    response.content = [block]
    return response


# ── _extract_json ─────────────────────────────────────────────────────

def test_extract_json_plain():
    assert _extract_json('{"genre": "pop"}') == {"genre": "pop"}


def test_extract_json_in_code_fence():
    text = '```json\n{"genre": "pop"}\n```'
    assert _extract_json(text) == {"genre": "pop"}


def test_extract_json_in_text():
    text = 'Here is the result: {"genre": "pop", "mood": "happy"} ok?'
    parsed = _extract_json(text)
    assert parsed["genre"] == "pop"
    assert parsed["mood"] == "happy"


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("not json at all") is None


# ── parse_user_input ──────────────────────────────────────────────────

def test_parse_user_input_returns_dict_on_valid_json():
    fake_json = ('{"genre": "lofi", "mood": "chill", "energy": 0.4, "valence": 0.5, '
                 '"danceability": 0.5, "likes_acoustic": true, "scoring_mode": "mood-first"}')
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response(fake_json)
    with patch("src.llm._get_client", return_value=mock_client):
        result = parse_user_input("chill music for studying")
    assert result["genre"] == "lofi"
    assert result["scoring_mode"] == "mood-first"


def test_parse_user_input_returns_none_when_client_unavailable():
    with patch("src.llm._get_client", return_value=None):
        assert parse_user_input("anything") is None


def test_parse_user_input_returns_none_on_garbage_response():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response("not json")
    with patch("src.llm._get_client", return_value=mock_client):
        assert parse_user_input("anything") is None


# ── critique_recommendations ──────────────────────────────────────────

def test_critique_returns_fallback_when_unavailable():
    with patch("src.llm._get_client", return_value=None):
        verdict = critique_recommendations([], {"genre": "pop"})
    assert verdict["should_retry"] is False
    assert verdict["quality_score"] == 0.5


def test_critique_parses_response():
    fake = ('{"quality_score": 0.85, "issues": [], "should_retry": false, '
            '"suggested_adjustments": null}')
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response(fake)
    rec = ({"title": "X", "artist": "Y", "genre": "pop", "mood": "happy"}, 5.0, "")
    with patch("src.llm._get_client", return_value=mock_client):
        verdict = critique_recommendations([rec], {"genre": "pop"})
    assert verdict["quality_score"] == 0.85


# ── generate_explanation ──────────────────────────────────────────────

def test_explanation_returns_fallback_when_unavailable():
    with patch("src.llm._get_client", return_value=None):
        out = generate_explanation([], {"genre": "pop"})
    assert isinstance(out, str)
    assert len(out) > 0


def test_explanation_uses_claude_response():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response("These tracks fit your vibe.")
    rec = ({"title": "X", "artist": "Y", "genre": "pop", "mood": "happy"}, 5.0, "genre match")
    with patch("src.llm._get_client", return_value=mock_client):
        out = generate_explanation([rec], {"genre": "pop"})
    assert "These tracks fit your vibe" in out


# ── parse_specialization_diff ─────────────────────────────────────────

def test_specialization_diff_detects_difference():
    specialized = {"genre": "pop", "mood": "happy", "scoring_mode": "balanced"}
    baseline = {"genre": "fun upbeat music", "mood": "happy"}
    metrics = parse_specialization_diff(specialized, baseline)
    assert metrics["specialized_valid_genre"] is True
    assert metrics["baseline_valid_genre"] is False
    assert metrics["specialized_has_mode"] is True
    assert metrics["baseline_has_mode"] is False
