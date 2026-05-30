'''from __future__ import annotations

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

Do NOT include:
- explanations
- markdown
- introductory text like "Here is"
- any extra characters outside JSON

Return ONLY JSON:

{{
  "discoverability": [],
  "readability": [],
  "interaction": [],
  "tone": []
}}

Rules:
- 2–4 items each
- no explanation

Content:
{text}

Stats:
Words: {ch.word_count}
Sentiment: {ch.sentiment}
Hashtags: {ch.hashtag_usage.count}
Emojis: {ch.emoji_usage.count}
CTA: {ch.call_to_action.detected}
Readability: {ch.readability_level}
""".strip()


# ---------------- SAFE PARSER ----------------
def extract_json_safe(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except:
        pass

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON found: {text[:300]}")

    return json.loads(match.group(0))


# ---------------- MAIN FUNCTION ----------------
async def provide_improvement_suggestions(text: str, ch: ContentCharacteristics) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    payload = {
        "contents": [{"parts": [{"text": build_prompt(text, ch)}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            f"{GEMINI_API_URL}?key={api_key}",
            json=payload
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini error: {resp.text[:300]}")

    data = resp.json()
    raw = data["candidates"][0]["content"]["parts"][0]["text"]

    parsed = extract_json_safe(raw)

    return {
        "discoverability": parsed.get("discoverability", []),
        "readability": parsed.get("readability", []),
        "interaction": parsed.get("interaction", []),
        "tone": parsed.get("tone", [])
    }'''

from __future__ import annotations

import os
import json
import httpx

from typing import Dict, Any
from dotenv import load_dotenv
from modules.analyzer import ContentCharacteristics

load_dotenv()

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)


# ---------------- PROMPT ----------------
def build_prompt(text: str, ch: ContentCharacteristics) -> str:
    return f"""
Return ONLY valid JSON.

Do NOT include:
- explanations
- markdown
- introductory text
- code fences
- extra characters outside JSON

Return ONLY:

{{
  "discoverability": [],
  "readability": [],
  "interaction": [],
  "tone": []
}}

Rules:
- 2 to 4 items per category
- each item must be a short suggestion
- no explanations
- valid JSON only

Content:
{text}

Stats:
Words: {ch.word_count}
Sentiment: {ch.sentiment}
Hashtags: {ch.hashtag_usage.count}
Emojis: {ch.emoji_usage.count}
CTA: {ch.call_to_action.detected}
Readability: {ch.readability_level}
""".strip()


# ---------------- JSON PARSER ----------------
def parse_json_response(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Gemini returned invalid JSON.\n"
            f"Parse Error: {e}\n\n"
            f"Response:\n{raw}"
        )


# ---------------- MAIN FUNCTION ----------------
async def provide_improvement_suggestions(
    text: str,
    ch: ContentCharacteristics
) -> Dict[str, Any]:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": build_prompt(text, ch)
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient(timeout=60) as client:

        response = await client.post(
            f"{GEMINI_API_URL}?key={api_key}",
            json=payload
        )

    # ---------------- ERROR HANDLING ----------------

    if response.status_code == 429:

        try:
            error_data = response.json()

            message = error_data["error"]["message"]

        except Exception:
            message = response.text

        raise RuntimeError(
            f"Gemini quota exceeded.\n{message}"
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API Error ({response.status_code})\n"
            f"{response.text}"
        )

    # ---------------- RESPONSE ----------------

    data = response.json()

    if "candidates" not in data:
        raise RuntimeError(
            f"No candidates returned.\n{json.dumps(data, indent=2)}"
        )

    candidate = data["candidates"][0]

    finish_reason = candidate.get("finishReason")

    if finish_reason not in (None, "STOP"):
        raise RuntimeError(
            f"Generation stopped unexpectedly.\n"
            f"Finish Reason: {finish_reason}\n\n"
            f"Response:\n{json.dumps(data, indent=2)}"
        )

    try:
        raw = candidate["content"]["parts"][0]["text"]

    except Exception:
        raise RuntimeError(
            f"Unexpected Gemini response structure.\n"
            f"{json.dumps(data, indent=2)}"
        )

    parsed = parse_json_response(raw)

    return {
        "discoverability": parsed.get("discoverability", []),
        "readability": parsed.get("readability", []),
        "interaction": parsed.get("interaction", []),
        "tone": parsed.get("tone", [])
    }