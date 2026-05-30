"""
modules/display.py
-------------------
Pretty-prints the analysis report to the terminal (used in CLI mode).
"""

from __future__ import annotations

from modules.analyzer import ContentCharacteristics


def _bar(score: int, width: int = 40) -> str:
    """Renders a simple ASCII progress bar for the engagement score."""
    filled = int(score / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score}/100"


def _sentiment_icon(sentiment: str) -> str:
    return {"Positive": "😊", "Neutral": "😐", "Negative": "😟"}.get(sentiment, "")


def _readability_icon(level: str) -> str:
    return {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(level, "")


def print_report(
    extracted_text: str,
    characteristics: ContentCharacteristics,
    score: dict,
    suggestions: dict,
) -> None:
    """
    Prints a formatted analysis report to stdout.

    Args:
        extracted_text: The text extracted from the uploaded file.
        characteristics: Output of analyze_characteristics().
        score: Dict with key "score" (int).
        suggestions: Dict with keys: discoverability, readability, interaction, tone.
    """
    ch = characteristics
    sep = "─" * 60

    print(f"\n{sep}")
    print("  ⚡ ENGAGEAI — ANALYSIS REPORT")
    print(sep)

    # --- Engagement Score ---
    print(f"\n  ENGAGEMENT SCORE")
    print(f"  {_bar(score['score'])}\n")

    # --- Characteristics table ---
    print(f"  CONTENT CHARACTERISTICS")
    print(f"  {'Word count':<30} {ch.word_count}")
    print(f"  {'Avg sentence length':<30} {ch.average_sentence_length} words")
    print(f"  {'Sentiment':<30} {_sentiment_icon(ch.sentiment)} {ch.sentiment}")
    print(f"  {'Readability':<30} {_readability_icon(ch.readability_level)} {ch.readability_level}")
    print(f"  {'Hashtags':<30} {ch.hashtag_usage.count} {' '.join(ch.hashtag_usage.hashtags[:6])}")
    print(f"  {'Emojis':<30} {ch.emoji_usage.count} {'  '.join(ch.emoji_usage.emojis[:8])}")

    cta = ch.call_to_action
    cta_val = f"✓ \"{cta.cta_text}\"" if cta.detected else "✗ None detected"
    print(f"  {'Call to Action':<30} {cta_val}")

    # --- Extracted text preview ---
    print(f"\n  EXTRACTED TEXT (preview)")
    preview = extracted_text[:400].replace("\n", " ")
    if len(extracted_text) > 400:
        preview += "…"
    print(f"  {preview}")

    # --- Suggestions ---
    icons = {
        "discoverability": "🔍 DISCOVERABILITY",
        "readability":     "📖 READABILITY",
        "interaction":     "💬 INTERACTION",
        "tone":            "🎙️  TONE",
    }

    print(f"\n  IMPROVEMENT SUGGESTIONS")
    for key, title in icons.items():
        items = suggestions.get(key, [])
        if not items:
            continue
        print(f"\n  {title}")
        for item in items:
            print(f"    • {item}")

    print(f"\n{sep}\n")
