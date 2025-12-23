# core/director.py
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import ValidationError

from core.schemas import LessonPlan
from core.agents import scriptwriter, safety_agent, continuity_agent, audio_agent, video_agent
from tools.genai_client import GenAIClient


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ----------------------------
# Character Bible (schema-correct)
# ----------------------------
def _build_character_bible_dict(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a CharacterBible that matches schemas.py exactly:
      {
        "style_lock": str (10..280),
        "character_tokens": [str],
        "negative_tokens": [str]
      }
    """
    chars = plan_dict.get("characters") or []
    main = chars[0] if (isinstance(chars, list) and chars and isinstance(chars[0], dict)) else {}

    main_name = (main.get("name") or "Main Hero").strip()
    main_style = (main.get("visual_style") or "Cute kid-safe 2D cartoon hero with a consistent outfit and colors.").strip()

    # Hard lock: keep it concise but strict
    style_lock = (
        f"MAIN CHARACTER LOCK: {main_name} must look IDENTICAL in every clip. "
        f"{main_style}. "
        "Same face, same hair, same costume, same colors, same proportions, same art style. "
        "Do not redesign. Keep the character model consistent across all scenes."
    )
    style_lock = _truncate(style_lock, 280)
    if len(style_lock) < 10:
        style_lock = "Keep the same character design across all scenes, identical face, outfit, colors."

    # Tokens: short, repeatable anchors
    character_tokens: List[str] = []

    if main_name:
        character_tokens.append(f"{main_name}: same face and outfit every time")
    if main_style:
        character_tokens.append(_truncate(f"{main_name} visual: {main_style}", 140))

    # If your continuity agent adds extra per-character hints, include them
    for c in chars[1:]:
        if isinstance(c, dict):
            nm = (c.get("name") or "").strip()
            vs = (c.get("visual_style") or "").strip()
            if nm and vs:
                character_tokens.append(_truncate(f"{nm} visual: {vs}", 140))

    negative_tokens = [
        "different face",
        "different costume",
        "different hair",
        "new character design",
        "style change",
        "photorealistic",
        "live action",
        "anime",
        "pixar",
        "text on screen",
        "watermark",
        "logo",
    ]

    return {
        "style_lock": style_lock,
        "character_tokens": character_tokens[:8],
        "negative_tokens": negative_tokens[:10],
    }


# ----------------------------
# Schema normalizer (CRITICAL)
# ----------------------------
def _normalize_schema(raw: Any, fallback_category: str, fallback_prompt: str) -> Dict[str, Any]:
    """
    Normalize common schema drift from LLM outputs so LessonPlan can validate.
    Handles:
      - missing category / learning_objective / grade_band
      - scene_number -> idx
      - missing scene.title
      - missing character.visual_style
      - beats missing idx/duration_s/title/on_screen_text
      - list-as-dict outputs
      - clears on_screen_text everywhere (your rule)
    """
    if isinstance(raw, LessonPlan):
        raw = raw.model_dump()

    if not isinstance(raw, dict):
        return {}

    # required top-level
    raw["category"] = raw.get("category") or raw.get("style") or raw.get("theme") or fallback_category
    raw["title"] = raw.get("title") or "Untitled Lesson"
    raw["grade_band"] = raw.get("grade_band") or raw.get("grade") or "Grades 4–6"
    raw["learning_objective"] = (
        raw.get("learning_objective")
        or raw.get("objective")
        or raw.get("learning_goal")
        or raw.get("learningGoal")
        or f"Understand the basics of: {fallback_prompt}"
    )

    # characters list can be dict-like
    chars = raw.get("characters")
    if isinstance(chars, dict):
        try:
            items = sorted(chars.items(), key=lambda kv: int(str(kv[0])))
        except Exception:
            items = list(chars.items())
        chars = [v for _, v in items]
    if not isinstance(chars, list):
        chars = []

    fixed_chars: List[Dict[str, Any]] = []
    for c in chars:
        if not isinstance(c, dict):
            continue
        c["name"] = (c.get("name") or "Character").strip()
        c["role"] = (c.get("role") or "helper").strip()
        c["visual_style"] = (
            c.get("visual_style")
            or c.get("appearance")
            or c.get("look")
            or c.get("description")
            or "Cute kid-safe 2D cartoon character, consistent outfit and colors."
        )
        fixed_chars.append(c)
    raw["characters"] = fixed_chars

    # scenes can be dict-like
    scenes = raw.get("scenes")
    if isinstance(scenes, dict):
        try:
            items = sorted(scenes.items(), key=lambda kv: int(str(kv[0])))
        except Exception:
            items = list(scenes.items())
        scenes = [v for _, v in items]
    if not isinstance(scenes, list):
        scenes = []

    fixed_scenes: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenes, start=1):
        if not isinstance(sc, dict):
            continue

        idx = sc.get("idx")
        if idx is None:
            idx = sc.get("scene_number") or sc.get("sceneIndex") or sc.get("scene_idx") or i
        try:
            sc["idx"] = int(idx)
        except Exception:
            sc["idx"] = i

        sc["title"] = (sc.get("title") or sc.get("setting") or sc.get("summary") or f"Scene {sc['idx']}").strip()

        # ensure on_screen_text empty
        sc["on_screen_text"] = []

        # beats normalize
        beats = sc.get("beats")
        if isinstance(beats, dict):
            try:
                bitems = sorted(beats.items(), key=lambda kv: int(str(kv[0])))
            except Exception:
                bitems = list(beats.items())
            beats = [v for _, v in bitems]

        if isinstance(beats, list) and beats:
            fixed_beats: List[Dict[str, Any]] = []
            for b_i, b in enumerate(beats, start=1):
                if not isinstance(b, dict):
                    continue
                bidx = b.get("idx")
                if bidx is None:
                    bidx = b.get("beat_number") or b.get("beatIndex") or b_i
                try:
                    b["idx"] = int(bidx)
                except Exception:
                    b["idx"] = b_i

                # duration: must be 5/6/7/8; default 6
                dur = b.get("duration_s", 6)
                try:
                    dur = int(dur)
                except Exception:
                    dur = 6
                if dur not in (5, 6, 7, 8):
                    dur = 6
                b["duration_s"] = dur

                # optional title allowed
                if "title" not in b:
                    b["title"] = None

                b["narration"] = (b.get("narration") or "").strip()
                b["visual_prompt"] = (b.get("visual_prompt") or b.get("visual") or b.get("shot") or "").strip()
                b["on_screen_text"] = []  # always empty

                fixed_beats.append(b)

            sc["beats"] = fixed_beats
        else:
            # legacy: keep fields optional but clear overlay text
            sc.setdefault("narration", "")
            sc.setdefault("visual_prompt", "")
            sc["on_screen_text"] = []

        fixed_scenes.append(sc)

    raw["scenes"] = fixed_scenes

    # character_bible must be None or a dict matching CharacterBible schema
    # We'll generate it later, but never keep random shapes from agents.
    raw["character_bible"] = None

    return raw


def _force_required_fields(raw: Dict[str, Any], category: str, prompt: str) -> Dict[str, Any]:
    raw.setdefault("title", "Untitled Lesson")
    raw.setdefault("category", category)
    raw.setdefault("learning_objective", f"Understand the basics of {prompt}")
    raw.setdefault("grade_band", "Grades 4–6")
    raw.setdefault("characters", [])
    raw.setdefault("scenes", [])
    raw.setdefault("character_bible", None)
    return raw


def _coerce_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    # keep within schema max lengths
    raw["title"] = _truncate(str(raw.get("title", "")), 80)
    raw["learning_objective"] = _truncate(str(raw.get("learning_objective", "")), 220)
    raw["grade_band"] = _truncate(str(raw.get("grade_band", "Grades 4–6")), 20)

    chars = raw.get("characters") or []
    if isinstance(chars, list):
        for c in chars:
            if isinstance(c, dict):
                c["name"] = _truncate(str(c.get("name", "Character")), 24)
                c["role"] = _truncate(str(c.get("role", "helper")), 40)
                c["visual_style"] = _truncate(str(c.get("visual_style", "")), 80)

    scenes = raw.get("scenes") or []
    if isinstance(scenes, list):
        for sc in scenes:
            if not isinstance(sc, dict):
                continue
            sc["title"] = _truncate(str(sc.get("title", "")), 80)

            beats = sc.get("beats")
            if isinstance(beats, list):
                for b in beats:
                    if isinstance(b, dict):
                        b["narration"] = _truncate(str(b.get("narration", "")), 700)
                        b["visual_prompt"] = _truncate(str(b.get("visual_prompt", "")), 240)

    return raw


def _call_scriptwriter(client: GenAIClient, prompt: str, category: str) -> Any:
    """
    Supports:
      - scriptwriter.run(client, prompt=..., category=...)
      - scriptwriter.run(client, req)
    """
    # keyword style
    try:
        return scriptwriter.run(client, prompt=prompt, category=category)
    except TypeError:
        pass

    # req style
    try:
        from core.schemas import InputRequest

        req = InputRequest(prompt=prompt, category=category)
        return scriptwriter.run(client, req)
    except Exception as e:
        return {"error": f"Scriptwriter call failed: {e}"}


def _flatten_beats_from_plan(plan: LessonPlan) -> List[Tuple[int, int, str, str]]:
    """
    Returns beats in generation order:
      (scene_idx, beat_idx, narration, visual_prompt)
    """
    out: List[Tuple[int, int, str, str]] = []
    for sc in plan.scenes:
        if sc.beats:
            for b in sc.beats:
                out.append((sc.idx, b.idx, b.narration, b.visual_prompt))
        else:
            out.append((sc.idx, 1, sc.narration or "", sc.visual_prompt or ""))
    return out


def run_pipeline(
    prompt: str,
    category: str,
    gen_video: bool = True,
    report_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    def report(p: float, msg: str):
        if not report_cb:
            return
        try:
            pf = float(p) if p is not None else 0.0
            pf = max(0.0, min(1.0, pf))
            report_cb(pf, str(msg))
        except Exception:
            try:
                report_cb(0.0, str(msg))
            except Exception:
                pass

    prompt = (prompt or "").strip()
    category = (category or "").strip()

    if not prompt:
        return {"error": "Missing prompt"}
    if not category:
        return {"error": "Missing category"}

    client = GenAIClient.from_env()

    run_id = _now_id()
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 1) Scriptwriter
    report(0.05, "🧠 Scriptwriter: generating lesson plan…")
    raw_plan = _call_scriptwriter(client, prompt=prompt, category=category)
    if isinstance(raw_plan, dict) and raw_plan.get("error"):
        return raw_plan

    if isinstance(raw_plan, LessonPlan):
        raw_plan = raw_plan.model_dump()

    if not isinstance(raw_plan, dict):
        return {"error": "Scriptwriter returned unexpected output type."}

    # 2) Safety
    report(0.20, "🛡️ Safety: reviewing content…")
    safe = safety_agent.run(client, raw_plan)
    if isinstance(safe, dict) and safe.get("error"):
        return safe

    # 3) Continuity
    report(0.28, "🧩 Continuity: linking beats + locking style…")
    cont = continuity_agent.run(client, safe if isinstance(safe, dict) else {})
    if isinstance(cont, dict) and cont.get("error"):
        return cont

    safe = cont if isinstance(cont, dict) else safe

    # 4) Normalize + validate
    report(0.32, "📏 Validating lesson schema…")
    try:
        norm = _normalize_schema(safe, fallback_category=category, fallback_prompt=prompt)
        norm = _force_required_fields(norm, category=category, prompt=prompt)
        norm = _coerce_plan(norm)

        # Build Character Bible now (schema-correct)
        norm["character_bible"] = _build_character_bible_dict(norm)

        plan = LessonPlan.model_validate(norm)
    except ValidationError as e:
        try:
            with open(os.path.join(run_dir, "bad_plan.json"), "w", encoding="utf-8") as f:
                json.dump(norm if isinstance(norm, dict) else {}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return {"error": f"LessonPlan validation error: {e}", "run_id": run_id, "run_dir": run_dir}

    # Save final plan
    with open(os.path.join(run_dir, "lesson.json"), "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, ensure_ascii=False, indent=2)

    beats = _flatten_beats_from_plan(plan)
    if not beats:
        return {"error": "No beats produced.", "run_id": run_id, "run_dir": run_dir}

    # 5) Audio
    report(0.35, f"🎧 Audio: generating {len(beats)} beat tracks…")
    try:
        audio_paths = audio_agent.run(plan, run_dir=run_dir, report_cb=report_cb)
    except TypeError:
        audio_paths = audio_agent.run(plan, run_dir=run_dir)

    if isinstance(audio_paths, dict) and audio_paths.get("error"):
        return audio_paths

    if not isinstance(audio_paths, list) or len(audio_paths) < len(beats):
        return {"error": "Audio agent returned unexpected output.", "run_id": run_id, "run_dir": run_dir}

    # 6) Video
    video_out: Optional[Union[str, List[str]]] = None
    if gen_video:
        report(0.55, f"🎬 Video: generating {len(beats)} beat clips with Veo…")
        try:
            video_out = video_agent.run(
                client,
                plan,
                audio_paths,
                run_dir,
                report_cb=report_cb,
            )
        except Exception as e:
            return {"error": str(e), "run_id": run_id, "run_dir": run_dir}

    report(1.0, "✅ Done!")
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "plan": plan.model_dump(),
        "audio_paths": audio_paths,
        "video": video_out,
    }
