# core/agents/scriptwriter.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, List

from tools.genai_client import GenAIClient

try:
    from core.schemas import InputRequest
except Exception:
    InputRequest = Any  # type: ignore


# =========================
# PROMPTS
# =========================

SYSTEM_PROMPT = """
You are a children's animated lesson writer.

STRICT RULES:
- Output ONLY valid JSON.
- Use double quotes everywhere.
- No trailing commas.
- No markdown.
- No comments.
- No explanations.

If you are unsure, simplify — but keep valid JSON.
""".strip()

# NOTE:
# We intentionally DO NOT use .format() on this template because the JSON example contains braces.
# We'll do safe token replacement with .replace().
USER_PROMPT_TEMPLATE = """
Create a short animated lesson as a STORY.

Topic:
"<<PROMPT>>"

Theme (MANDATORY, must drive the entire story):
"<<CATEGORY>>"

STRUCTURE (STRICT):
- 4 to 6 scenes total
- Each scene has exactly 2 beats
- Each beat has:
  - narration (storytelling, not lecture)
  - visual_prompt (must match narration)
  - on_screen_text: []

EDUCATION BALANCE (IMPORTANT):
- Every scene must teach ONE small idea.
- Teach through action + consequence (not lecture).
- Include 1–2 child-friendly definitions across the whole story.

STORY ARC:
- Scene 1: introduce hero + problem (theme-based)
- Middle scenes: explore the idea through actions
- Final scene: resolve the problem + simple takeaway

THEME ENFORCEMENT:
- Everything must happen inside the theme world.
- Do NOT mention real-world scientists.
- Do NOT mention "Newton", "Einstein", etc.

OUTPUT JSON SCHEMA (EXACT):

{
  "title": "string",
  "learning_objective": "string",
  "grade_band": "Grades 4–6",
  "category": "<<CATEGORY>>",
  "characters": [
    {
      "name": "string",
      "description": "string",
      "visual_style": "string"
    }
  ],
  "scenes": [
    {
      "idx": 1,
      "title": "string",
      "beats": [
        {
          "narration": "string",
          "visual_prompt": "string",
          "on_screen_text": []
        },
        {
          "narration": "string",
          "visual_prompt": "string",
          "on_screen_text": []
        }
      ]
    }
  ]
}

Return ONLY the JSON.
""".strip()


# =========================
# JSON SAFETY
# =========================

def _strip_code_fences(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json_object(text: str) -> str:
    """
    Extract the largest {...} JSON object from arbitrary model text.
    """
    if not text:
        return ""
    t = _strip_code_fences(text)

    # Find first '{' and last '}' and slice.
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1].strip()

    # If model forgot braces but started with "title"
    if '"title"' in t and "{" not in t:
        return "{" + t + "}"

    return ""


def _repair_common_json_issues(s: str) -> str:
    """
    Best-effort JSON repair:
    - normalize smart quotes
    - remove trailing commas
    """
    if not s:
        return s

    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    return s


def _safe_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    block = _extract_json_object(raw)
    if not block:
        return None

    block = _repair_common_json_issues(block)

    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def _ensure_scene_structure(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize likely drift:
    - scenes might be dict keyed by "0","1" etc
    - beats might be dict keyed by "0","1"
    """
    scenes = obj.get("scenes")
    if isinstance(scenes, dict):
        # convert dict -> list sorted by key
        items = []
        for k, v in scenes.items():
            items.append((str(k), v))
        items.sort(key=lambda kv: int(re.sub(r"\D+", "", kv[0]) or "0"))
        scenes = [v for _, v in items]
        obj["scenes"] = scenes

    if isinstance(obj.get("scenes"), list):
        new_scenes: List[Dict[str, Any]] = []
        for i, s in enumerate(obj["scenes"], start=1):
            if not isinstance(s, dict):
                continue
            s.setdefault("idx", i)
            s.setdefault("title", f"Scene {i}")

            beats = s.get("beats")
            if isinstance(beats, dict):
                bitems = []
                for k, v in beats.items():
                    bitems.append((str(k), v))
                bitems.sort(key=lambda kv: int(re.sub(r"\D+", "", kv[0]) or "0"))
                beats = [v for _, v in bitems]
                s["beats"] = beats

            # enforce exactly 2 beats
            if isinstance(s.get("beats"), list):
                beats_list = [b for b in s["beats"] if isinstance(b, dict)]
                while len(beats_list) < 2:
                    beats_list.append({
                        "narration": "The hero learns something important.",
                        "visual_prompt": "Hero thinking in the themed world.",
                        "on_screen_text": []
                    })
                beats_list = beats_list[:2]
                for b in beats_list:
                    b.setdefault("narration", "The hero acts and learns.")
                    b.setdefault("visual_prompt", "Themed animated scene.")
                    b.setdefault("on_screen_text", [])
                s["beats"] = beats_list
            else:
                s["beats"] = [
                    {"narration": "The hero faces a challenge.", "visual_prompt": "Themed animated scene.", "on_screen_text": []},
                    {"narration": "The hero discovers the key idea.", "visual_prompt": "Themed animated scene.", "on_screen_text": []},
                ]

            new_scenes.append(s)

        # enforce 4–6 scenes
        if len(new_scenes) < 4:
            for j in range(len(new_scenes) + 1, 5):
                new_scenes.append({
                    "idx": j,
                    "title": f"Scene {j}",
                    "beats": [
                        {"narration": "A new mini challenge appears.", "visual_prompt": "Themed animated scene.", "on_screen_text": []},
                        {"narration": "The hero uses the idea to solve it.", "visual_prompt": "Themed animated scene.", "on_screen_text": []},
                    ]
                })
        obj["scenes"] = new_scenes[:6]

    return obj


def _fallback_lesson(prompt: str, category: str) -> Dict[str, Any]:
    return {
        "title": f"{prompt.title()}",
        "learning_objective": f"Understand the idea of {prompt.lower()} in a {category} story.",
        "grade_band": "Grades 4–6",
        "category": category,
        "characters": [
            {
                "name": "Hero",
                "description": "A brave themed hero who learns by doing.",
                "visual_style": f"{category} style animated character"
            }
        ],
        "scenes": [
            {
                "idx": 1,
                "title": "The Problem",
                "beats": [
                    {"narration": f"A problem appears related to {prompt.lower()}.", "visual_prompt": "Hero sees the problem.", "on_screen_text": []},
                    {"narration": "The hero decides to investigate.", "visual_prompt": "Hero thinking.", "on_screen_text": []},
                ],
            },
            {
                "idx": 2,
                "title": "A Quick Test",
                "beats": [
                    {"narration": "The hero tries a small experiment in-story.", "visual_prompt": "Hero testing.", "on_screen_text": []},
                    {"narration": "The result teaches a simple rule.", "visual_prompt": "Result shown clearly.", "on_screen_text": []},
                ],
            },
            {
                "idx": 3,
                "title": "Using the Rule",
                "beats": [
                    {"narration": "The hero uses the rule to solve part of the problem.", "visual_prompt": "Hero applying the idea.", "on_screen_text": []},
                    {"narration": "It works! The hero gains confidence.", "visual_prompt": "Hero smiles.", "on_screen_text": []},
                ],
            },
            {
                "idx": 4,
                "title": "Final Save",
                "beats": [
                    {"narration": "One last big challenge appears.", "visual_prompt": "Big themed moment.", "on_screen_text": []},
                    {"narration": "The hero solves it and shares the takeaway.", "visual_prompt": "Hero celebrating.", "on_screen_text": []},
                ],
            },
        ],
    }


def run(
    client: GenAIClient,
    req: Optional[InputRequest] = None,
    prompt: Optional[str] = None,
    category: Optional[str] = None,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    # support multiple call styles
    if args and len(args) >= 2:
        prompt = prompt or args[0]
        category = category or args[1]

    if req is not None:
        prompt = prompt or getattr(req, "prompt", None)
        category = category or getattr(req, "category", None)

    prompt = (prompt or "").strip()
    category = (category or "").strip()

    if not prompt or not category:
        return {"error": "Scriptwriter missing prompt or category."}

    final_prompt = (
        USER_PROMPT_TEMPLATE
        .replace("<<PROMPT>>", prompt)
        .replace("<<CATEGORY>>", category)
    )

    # 1) First attempt
    text = client.generate_text(final_prompt, system=SYSTEM_PROMPT)
    parsed = _safe_parse_json(text)

    if isinstance(parsed, dict):
        parsed["category"] = category
        parsed = _ensure_scene_structure(parsed)
        return parsed

    # 2) Retry once, explicitly demand valid JSON
    retry_text = client.generate_text(
        final_prompt + "\n\nIMPORTANT: Your last output was invalid JSON. Return ONLY valid JSON.",
        system=SYSTEM_PROMPT,
    )
    parsed = _safe_parse_json(retry_text)

    if isinstance(parsed, dict):
        parsed["category"] = category
        parsed = _ensure_scene_structure(parsed)
        return parsed

    # 3) Fallback
    return _fallback_lesson(prompt, category)
