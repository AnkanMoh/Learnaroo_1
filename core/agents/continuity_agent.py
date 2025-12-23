from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from tools.genai_client import GenAIClient


def _extract_json_object(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        return text[s : e + 1]
    return text


def _flatten_chars(raw_plan: Dict[str, Any]) -> List[Dict[str, str]]:
    out = []
    for c in (raw_plan.get("characters") or []):
        if isinstance(c, dict):
            out.append(
                {
                    "name": str(c.get("name", "")).strip(),
                    "role": str(c.get("role", "")).strip(),
                    "visual_style": str(c.get("visual_style", "")).strip(),
                }
            )
    return out


def run(client: GenAIClient, raw_plan: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_plan, dict):
        return {"error": "Continuity agent expected a dict lesson plan."}

    # IMPORTANT: never let continuity invent/overwrite schema character_bible
    raw_plan["character_bible"] = None

    chars = _flatten_chars(raw_plan)
    char_names = [c["name"] for c in chars if c["name"]]

    prompt = f"""
You are a Continuity & Style Consistency Agent for kids educational story lessons.

You will EDIT the given lesson plan JSON to improve:
1) Continuity: every scene should naturally connect to the next (smooth transitions, consistent plot thread).
2) Character consistency: the same character must look the same across all scenes.

STRICT OUTPUT:
- Return ONLY valid JSON.
- No markdown. No backticks. No comments.
- Use double quotes for all keys/strings.

SCHEMA RULES:
- Keep the same top-level fields: title, learning_objective, grade_band, category, characters, scenes, character_bible.
- Do NOT remove scenes.
- Do NOT add new characters unless absolutely necessary.
- Keep on_screen_text as [] everywhere (empty lists).
- KEEP character_bible as null (do not build it here).
- ADD a new optional field for each character:
  "style_tokens": a short token string (<= 80 chars) describing that character's fixed look.
  Example: "tiny round face, teal suit, yellow cape, star badge"

STYLE CONSISTENCY REQUIREMENT:
- For every beat visual_prompt, append the relevant character style_tokens
  so the render stays consistent.
- When a beat features multiple characters, include both tokens.

CONTINUITY REQUIREMENT:
- Add a 1-sentence bridge at the END of each scene's last beat narration leading to the next scene.
- Make sure names/actions don't contradict.
- Keep it kid-friendly, Grades 4–6.

Characters present (names): {char_names}

Here is the plan JSON to edit:
{json.dumps(raw_plan, ensure_ascii=False)}
""".strip()

    text = ""
    try:
        text = client.generate_text(prompt)
        json_text = _extract_json_object(text)
        cleaned = json.loads(json_text)

        if not isinstance(cleaned, dict):
            return {"error": "Continuity agent returned non-object JSON."}

        # enforce on_screen_text = [] and character_bible = None
        cleaned["character_bible"] = None
        scenes = cleaned.get("scenes") or []
        for sc in scenes:
            if not isinstance(sc, dict):
                continue
            sc["on_screen_text"] = []
            beats = sc.get("beats")
            if isinstance(beats, list):
                for b in beats:
                    if isinstance(b, dict):
                        b["on_screen_text"] = []

        return cleaned

    except Exception as e:
        head = ""
        try:
            head = (text[:220] + "…") if isinstance(text, str) and len(text) > 220 else str(text)
        except Exception:
            pass
        return {"error": f"Continuity agent failed: {e}. Raw head: {head}"}
