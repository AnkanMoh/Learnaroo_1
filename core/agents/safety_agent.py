from __future__ import annotations

import json
import re
from typing import Any, Dict, Union

from core.safety_policy import KID_SAFE_RULES
from tools.genai_client import GenAIClient


def _extract_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object from an LLM response.
    Handles fences, commentary, etc.
    """
    if not text:
        return ""

    t = text.strip()

    # remove ```json fences if present
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    # locate first {...} object
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


def _force_no_text_overlay(plan: Dict[str, Any]) -> None:
    """
    Enforce on_screen_text = [] for all beats and legacy scenes.
    """
    scenes = plan.get("scenes") or []
    if not isinstance(scenes, list):
        return

    for sc in scenes:
        if not isinstance(sc, dict):
            continue

        # new format: beats
        beats = sc.get("beats")
        if isinstance(beats, list):
            for b in beats:
                if isinstance(b, dict):
                    b["on_screen_text"] = []
        else:
            # legacy format
            if "on_screen_text" in sc:
                sc["on_screen_text"] = []


def _as_dict(raw_plan: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    Accept dict or Pydantic model (LessonPlan), return dict.
    """
    if isinstance(raw_plan, dict):
        return raw_plan
    # Pydantic v2 models have model_dump()
    if hasattr(raw_plan, "model_dump"):
        return raw_plan.model_dump()
    # fallback: try json serialization
    try:
        return json.loads(json.dumps(raw_plan))
    except Exception:
        return {"error": "Safety agent expected dict or LessonPlan-like object."}


def run(client: GenAIClient, raw_plan: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    Content Safety Agent:
    - input: dict or LessonPlan
    - output: dict
    - preserves schema, scenes, beats, characters
    """
    plan = _as_dict(raw_plan)
    if isinstance(plan, dict) and plan.get("error"):
        return plan
    if not isinstance(plan, dict):
        return {"error": "Safety agent expected a dict lesson plan."}

    rules = "\n".join(f"- {r}" for r in KID_SAFE_RULES)

    prompt = f"""
You are a Content Safety Agent for children's educational video lessons (Grades 4–6).

Return ONLY a valid JSON object.
- No markdown.
- No backticks.
- No comments.
- Use double quotes for all keys/strings.
- No trailing commas.

Your task:
1) Make the lesson fully kid-safe and positive.
2) Preserve the SAME structure, scenes, beats, and characters.
   - Do NOT add or remove scenes.
   - Do NOT add or remove beats.
3) Remove or soften anything unsafe/scary/violent.
4) Keep continuity: same characters across scenes unless explicitly introduced.
5) IMPORTANT: keep on_screen_text as [] everywhere (no text overlays in video).

Kid-safe rules:
{rules}

Here is the lesson plan JSON you must edit (keep the same schema):
{json.dumps(plan, ensure_ascii=False)}
""".strip()

    text = ""
    try:
        text = client.generate_text(prompt)
        json_text = _extract_json_object(text)
        cleaned = json.loads(json_text)

        if not isinstance(cleaned, dict):
            return {"error": "Safety agent returned non-object JSON."}

        _force_no_text_overlay(cleaned)
        return cleaned

    except Exception as e:
        head = ""
        if isinstance(text, str) and text:
            head = text[:250] + ("…" if len(text) > 250 else "")
        return {"error": f"Safety agent failed: {e}. Raw head: {head}"}
