"""
modules/analyzer.py
--------------------
Analyzes social media content and extracts key characteristics:
  - Word count
  - Average sentence length
  - Sentiment (Positive / Neutral / Negative)
  - Hashtag usage
  - Emoji usage
  - Call-to-Action detection
  - Readability level (Easy / Medium / Hard)

All analysis is rule-based / statistical – no AI API needed for this step.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------

@dataclass
class HashtagUsage:
    count: int
    hashtags: list[str]


@dataclass
class EmojiUsage:
    count: int
    emojis: list[str]


@dataclass
class CallToAction:
    detected: bool
    cta_text: str | None  # The phrase that triggered CTA detection, or None


@dataclass
class ContentCharacteristics:
    word_count: int
    average_sentence_length: float
    sentiment: str          # "Positive" | "Neutral" | "Negative"
    hashtag_usage: HashtagUsage
    emoji_usage: EmojiUsage
    call_to_action: CallToAction
    readability_level: str  # "Easy" | "Medium" | "Hard"


# ---------------------------------------------------------------------------
# Word lists used for sentiment and CTA detection
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "amazing", "awesome", "beautiful", "best", "brilliant", "celebrate",
    "congratulations", "delighted", "excited", "excellent", "fantastic",
    "great", "happy", "incredible", "inspiring", "love", "lovely",
    "magnificent", "outstanding", "perfect", "positive", "proud",
    "thrilled", "wonderful", "win", "winning", "success", "successful",
    "good", "nice", "superb", "glad", "joy", "enjoy", "fun", "blessed",
    "grateful", "thankful", "energized", "motivated", "innovative",
}

NEGATIVE_WORDS = {
    "awful", "bad", "boring", "broken", "cancelled", "crisis", "difficult",
    "disappointing", "disaster", "dreadful", "fail", "failed", "failure",
    "frustrated", "hate", "horrible", "issue", "lost", "missing", "negative",
    "never", "painful", "problem", "sad", "scary", "shocked", "terrible",
    "toxic", "ugly", "upset", "useless", "waste", "worst", "wrong",
    "concerned", "worried", "fear", "danger", "danger",
}

# CTA trigger phrases – order matters (longer phrases first to avoid partial matches)
CTA_PHRASES = [
    "click the link", "link in bio", "link in our bio", "sign up now",
    "get started", "learn more", "subscribe now", "follow us",
    "check it out", "shop now", "buy now", "order now", "register now",
    "join us", "join now", "download now", "get your", "grab yours",
    "don't miss", "act now", "limited time", "call us", "contact us",
    "visit us", "visit our", "tap the link", "swipe up", "dm us",
    "message us", "book now", "reserve your", "claim your", "try now",
    "start today", "start now", "apply now", "read more",
    "watch now", "see more", "find out more", "discover more",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tokenize_words(text: str) -> list[str]:
    """Return a list of lowercase alphabetic words (strips punctuation)."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a simple regex.
    Handles abbreviations poorly on purpose (good enough for social media posts).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Filter out empty strings
    return [s for s in sentences if s.strip()]


def _extract_hashtags(text: str) -> list[str]:
    """Return all unique hashtags found in the text, preserving original casing."""
    found = re.findall(r"#\w+", text)
    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for tag in found:
        lower = tag.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(tag)
    return unique


def _extract_emojis(text: str) -> list[str]:
    """Return all unique emojis found in the text."""
    emoji_list: list[str] = []
    seen: set[str] = set()
    for char in text:
        # Unicode "So" (Symbol, Other) and "Sm" categories cover most emojis.
        # We also check the special emoji block range explicitly.
        cat = unicodedata.category(char)
        code = ord(char)
        is_emoji = (
            cat in ("So", "Sm")
            or (0x1F300 <= code <= 0x1FAFF)   # Misc symbols, emoticons, transport…
            or (0x2600 <= code <= 0x27BF)       # Misc symbols
            or (0x2300 <= code <= 0x23FF)       # Misc technical
        )
        if is_emoji and char not in seen:
            seen.add(char)
            emoji_list.append(char)
    return emoji_list


def _detect_cta(text: str) -> CallToAction:
    """
    Scan the text for known call-to-action phrases (case-insensitive).
    Returns the first match found.
    """
    lower_text = text.lower()
    for phrase in CTA_PHRASES:
        if phrase in lower_text:
            # Find the snippet around the match for context
            idx = lower_text.find(phrase)
            snippet_start = max(0, idx - 5)
            snippet_end = min(len(text), idx + len(phrase) + 30)
            cta_snippet = text[snippet_start:snippet_end].strip()
            return CallToAction(detected=True, cta_text=cta_snippet)
    return CallToAction(detected=False, cta_text=None)


def _calculate_sentiment(words: list[str]) -> str:
    """
    Simple lexicon-based sentiment scoring.
    Returns "Positive", "Neutral", or "Negative".
    """
    if not words:
        return "Neutral"

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

    diff = pos_count - neg_count

    if diff > 0:
        return "Positive"
    elif diff < 0:
        return "Negative"
    else:
        return "Neutral"


def _calculate_readability(word_count: int, avg_sentence_length: float) -> str:
    """
    Classifies readability based on content length and sentence complexity.

    Heuristic:
      - Short posts with short sentences → Easy
      - Medium posts or medium sentence length → Medium
      - Long posts with complex sentences → Hard
    """
    if avg_sentence_length <= 12 and word_count <= 100:
        return "Easy"
    elif avg_sentence_length <= 20 and word_count <= 300:
        return "Medium"
    else:
        return "Hard"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_characteristics(text: str) -> ContentCharacteristics:
    """
    Analyzes the given social media content text and returns a
    ContentCharacteristics object with all computed metrics.

    Args:
        text: The extracted social media post text.

    Returns:
        A ContentCharacteristics dataclass instance.
    """
    if not text or not text.strip():
        raise ValueError("Cannot analyze empty text.")

    # --- Word count ---
    words = _tokenize_words(text)
    word_count = len(words)

    # --- Average sentence length ---
    sentences = _split_sentences(text)
    if sentences:
        sentence_word_counts = [len(_tokenize_words(s)) for s in sentences]
        avg_sentence_length = round(sum(sentence_word_counts) / len(sentences), 1)
    else:
        avg_sentence_length = float(word_count)

    # --- Sentiment ---
    sentiment = _calculate_sentiment(words)

    # --- Hashtags ---
    hashtags = _extract_hashtags(text)
    hashtag_usage = HashtagUsage(count=len(hashtags), hashtags=hashtags)

    # --- Emojis ---
    emojis = _extract_emojis(text)
    emoji_usage = EmojiUsage(count=len(emojis), emojis=emojis)

    # --- Call-to-Action ---
    cta = _detect_cta(text)

    # --- Readability ---
    readability = _calculate_readability(word_count, avg_sentence_length)

    return ContentCharacteristics(
        word_count=word_count,
        average_sentence_length=avg_sentence_length,
        sentiment=sentiment,
        hashtag_usage=hashtag_usage,
        emoji_usage=emoji_usage,
        call_to_action=cta,
        readability_level=readability,
    )
