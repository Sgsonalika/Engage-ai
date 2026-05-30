"""
tests/test_engage_ai.py
------------------------
pytest test suite for all EngageAI modules.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.analyzer import (
    CallToAction,
    ContentCharacteristics,
    EmojiUsage,
    HashtagUsage,
    analyze_characteristics,
    _detect_cta,
    _extract_hashtags,
    _extract_emojis,
    _calculate_sentiment,
    _calculate_readability,
)
from modules.extractor import validate_file, clean_text, ALLOWED_EXTENSIONS
from modules.display import _bar, _sentiment_icon, _readability_icon


# ===========================================================================
# Fixtures
# ===========================================================================

SAMPLE_POST = (
    "Just launched my new side project! 🚀 So excited to share this. "
    "Check it out – link in bio! #startup #socialmedia #design"
)

NEGATIVE_POST = "This is a terrible, awful, and horrible experience. Everything is broken and bad."

LONG_POST = " ".join(["This is a fairly long sentence that contains many words"] * 10)


# ===========================================================================
# extractor tests
# ===========================================================================

class TestValidateFile:
    def test_valid_extensions_accepted(self, tmp_path):
        for ext in [".pdf", ".png", ".jpg", ".jpeg", ".webp"]:
            f = tmp_path / f"test{ext}"
            f.write_bytes(b"x" * 100)
            valid, err = validate_file(str(f))
            assert valid, f"Expected {ext} to be valid, got: {err}"

    def test_invalid_extension_rejected(self, tmp_path):
        f = tmp_path / "test.docx"
        f.write_bytes(b"x")
        valid, err = validate_file(str(f))
        assert not valid
        assert "Unsupported" in err

    def test_nonexistent_file_rejected(self):
        valid, err = validate_file("/nonexistent/path/file.pdf")
        assert not valid
        assert "not found" in err.lower()

    def test_oversized_file_rejected(self, tmp_path):
        f = tmp_path / "big.pdf"
        f.write_bytes(b"x" * (11 * 1024 * 1024))  # 11 MB
        valid, err = validate_file(str(f))
        assert not valid
        assert "too large" in err.lower()

    def test_exactly_max_size_accepted(self, tmp_path):
        f = tmp_path / "ok.pdf"
        f.write_bytes(b"x" * (10 * 1024 * 1024))  # exactly 10 MB
        valid, err = validate_file(str(f))
        assert valid, err


class TestCleanText:
    def test_collapses_blank_lines(self):
        # clean_text allows at most 2 consecutive blank lines (3 \n's total).
        # More than that should be collapsed.
        raw = "Line one\n\n\n\n\n\n\nLine two"
        cleaned = clean_text(raw)
        # Should not have 4+ newlines in a row (i.e. 3+ consecutive blank lines)
        assert "\n\n\n\n" not in cleaned

    def test_normalises_whitespace(self):
        raw = "Hello    world   how   are   you"
        cleaned = clean_text(raw)
        assert "  " not in cleaned

    def test_preserves_content(self):
        raw = "Hello world\nThis is a test."
        cleaned = clean_text(raw)
        assert "Hello world" in cleaned
        assert "This is a test." in cleaned


# ===========================================================================
# analyzer tests
# ===========================================================================

class TestExtractHashtags:
    def test_finds_hashtags(self):
        tags = _extract_hashtags("Love this! #startup #AI #python")
        assert "#startup" in tags
        assert "#AI" in tags
        assert "#python" in tags
        assert len(tags) == 3

    def test_no_hashtags(self):
        tags = _extract_hashtags("No tags here at all.")
        assert tags == []

    def test_deduplicates(self):
        tags = _extract_hashtags("#ai #AI #Ai")
        assert len(tags) == 1


class TestExtractEmojis:
    def test_finds_emojis(self):
        emojis = _extract_emojis("Hello 🚀 world 🎉!")
        assert "🚀" in emojis
        assert "🎉" in emojis

    def test_no_emojis(self):
        emojis = _extract_emojis("Plain text with no emojis here.")
        assert emojis == []

    def test_deduplicates(self):
        emojis = _extract_emojis("🚀 🚀 🚀")
        assert len(emojis) == 1


class TestDetectCTA:
    def test_detects_link_in_bio(self):
        cta = _detect_cta("Check it out! Link in bio!")
        assert cta.detected
        assert cta.cta_text is not None

    def test_detects_shop_now(self):
        cta = _detect_cta("Don't wait – shop now for 50% off!")
        assert cta.detected

    def test_no_cta(self):
        cta = _detect_cta("Today was a beautiful day in the park.")
        assert not cta.detected
        assert cta.cta_text is None


class TestCalculateSentiment:
    def test_positive_words(self):
        words = ["amazing", "great", "love", "fantastic", "happy"]
        assert _calculate_sentiment(words) == "Positive"

    def test_negative_words(self):
        words = ["terrible", "awful", "horrible", "bad", "worst"]
        assert _calculate_sentiment(words) == "Negative"

    def test_neutral_empty(self):
        assert _calculate_sentiment([]) == "Neutral"

    def test_mixed_leans_positive(self):
        words = ["great", "good", "excellent", "bad"]
        assert _calculate_sentiment(words) == "Positive"


class TestCalculateReadability:
    def test_easy(self):
        assert _calculate_readability(50, 8) == "Easy"

    def test_medium(self):
        assert _calculate_readability(150, 15) == "Medium"

    def test_hard(self):
        assert _calculate_readability(500, 25) == "Hard"


class TestAnalyzeCharacteristics:
    def test_returns_correct_type(self):
        result = analyze_characteristics(SAMPLE_POST)
        assert isinstance(result, ContentCharacteristics)

    def test_word_count_positive(self):
        result = analyze_characteristics(SAMPLE_POST)
        assert result.word_count > 0

    def test_hashtags_detected(self):
        result = analyze_characteristics(SAMPLE_POST)
        assert result.hashtag_usage.count == 3

    def test_emoji_detected(self):
        result = analyze_characteristics(SAMPLE_POST)
        assert result.emoji_usage.count >= 1
        assert "🚀" in result.emoji_usage.emojis

    def test_cta_detected(self):
        result = analyze_characteristics(SAMPLE_POST)
        assert result.call_to_action.detected

    def test_sentiment_positive(self):
        result = analyze_characteristics(SAMPLE_POST)
        assert result.sentiment == "Positive"

    def test_sentiment_negative(self):
        result = analyze_characteristics(NEGATIVE_POST)
        assert result.sentiment == "Negative"

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            analyze_characteristics("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            analyze_characteristics("   \n  ")


# ===========================================================================
# display tests
# ===========================================================================

class TestDisplayHelpers:
    def test_bar_full_score(self):
        bar = _bar(100)
        assert "100/100" in bar
        assert "░" not in bar

    def test_bar_zero_score(self):
        bar = _bar(0)
        assert "0/100" in bar
        assert "█" not in bar

    def test_bar_half_score(self):
        bar = _bar(50)
        assert "50/100" in bar

    def test_sentiment_icons(self):
        assert _sentiment_icon("Positive") == "😊"
        assert _sentiment_icon("Negative") == "😟"
        assert _sentiment_icon("Neutral") == "😐"
        assert _sentiment_icon("Unknown") == ""

    def test_readability_icons(self):
        assert _readability_icon("Easy") == "🟢"
        assert _readability_icon("Medium") == "🟡"
        assert _readability_icon("Hard") == "🔴"


# ===========================================================================
# API tests (mocked AI calls)
# ===========================================================================

class TestAPIEndpoints:
    """Tests for the FastAPI endpoints using TestClient."""

    def _get_client(self):
        from fastapi.testclient import TestClient
        from modules.api import app
        return TestClient(app)

    def test_health_endpoint(self):
        client = self._get_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_root_returns_html(self):
        client = self._get_client()
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "EngageAI" in resp.text

    def test_analyze_rejects_wrong_file_type(self, tmp_path):
        client = self._get_client()
        bad_file = tmp_path / "test.exe"
        bad_file.write_bytes(b"not a valid file")
        with open(bad_file, "rb") as f:
            resp = client.post("/api/analyze", files={"file": ("test.exe", f, "application/octet-stream")})
        assert resp.status_code == 400

    @patch("modules.api.extract_text", return_value=(SAMPLE_POST, None))
    @patch("modules.api.generate_engagement_score", new_callable=AsyncMock, return_value={"score": 78})
    @patch("modules.api.provide_improvement_suggestions", new_callable=AsyncMock, return_value={
        "discoverability": ["Add more niche hashtags."],
        "readability": ["Shorten sentences."],
        "interaction": ["Ask a question."],
        "tone": ["Sound more enthusiastic."],
    })
    def test_analyze_success(self, mock_sugg, mock_score, mock_extract, tmp_path):
        client = self._get_client()
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")
        with open(dummy_pdf, "rb") as f:
            resp = client.post("/api/analyze", files={"file": ("test.pdf", f, "application/pdf")})

        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] == 78
        assert "characteristics" in data
        assert "suggestions" in data
        assert "extracted_text" in data
