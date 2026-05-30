from __future__ import annotations

import os
import json
import re
import httpx
from typing import Dict, Any
from modules.analyzer import ContentCharacteristics
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


# ---------------- PROMPT ----------------
def build_prompt(text: str, ch: ContentCharacteristics) -> str:
    return f"""

Return ONLY valid JSON.

Do NOT write:
- explanations
- markdown
- extra text
- prefixes like "Here is"

If you do, it will break the system.

Return ONLY JSON:
{{"score": 0}}

Rules:
- integer 0–100
- no text

Content:
{text}

Stats:
Words: {ch.word_count}
Sentiment: {ch.sentiment}
Hashtags: {ch.hashtag_usage.count}
CTA: {ch.call_to_action.detected}
Readability: {ch.readability_level}
""".strip()

# ---------------- SAFE JSON PARSER ----------------
def extract_json_safe(text: str) -> Dict[str, Any]:
    text = text.strip()

    # 1. direct JSON
    try:
        return json.loads(text)
    except:
        pass

    # 2. remove code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # 3. extract JSON block
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON found: {text[:300]}")

    return json.loads(match.group(0))


# ---------------- MAIN FUNCTION ----------------
async def generate_engagement_score(text: str, ch: ContentCharacteristics) -> Dict[str, int]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    payload = {
        "contents": [{"parts": [{"text": build_prompt(text, ch)}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            f"{GEMINI_API_URL}?key={api_key}",
            json=payload
        )

    if res.status_code != 200:
        raise RuntimeError(f"Gemini error: {res.text[:300]}")

    data = res.json()
    raw = data["candidates"][0]["content"]["parts"][0]["text"]

    parsed = extract_json_safe(raw)

    score = int(parsed.get("score", 0))
    score = max(0, min(100, score))

    return {"score": score}