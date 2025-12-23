from __future__ import annotations

import os
import subprocess
from typing import Callable, List, Optional, Union

from core.schemas import LessonPlan


def _flatten_narration(plan: LessonPlan) -> List[str]:
    lines: List[str] = []
    for sc in plan.scenes:
        beats = getattr(sc, "beats", None)
        if beats:
            for b in beats:
                lines.append((b.narration or "").strip())
        else:
            lines.append((sc.narration or "").strip())
    return lines


def _say_to_wav(text: str, out_path: str, voice: str = "Samantha", rate: int = 175) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_aiff = out_path.replace(".wav", ".aiff")

    cmd = ["say", "-v", voice, "-r", str(rate), "-o", tmp_aiff, text]
    subprocess.run(cmd, check=True)

    # keep AIFF (MoviePy handles it well)
    # no conversion step here


def run(
    plan: LessonPlan,
    run_dir: str,
    report_cb: Optional[Callable[[float, str], None]] = None,
    voice: str = "Samantha",
    rate: int = 175,
) -> Union[List[str], dict]:
    """
    Returns a list of per-beat audio paths (AIFF).
    Order matches plan scenes then beats (same as video_agent mapping).
    """
    def report(p: float, msg: str):
        if report_cb:
            try:
                report_cb(float(p), msg)
            except Exception:
                try:
                    report_cb(0.0, msg)
                except Exception:
                    pass

    narrations = _flatten_narration(plan)
    if not narrations:
        return {"error": "Audio agent: no narration found."}

    audio_dir = os.path.join(run_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    out_paths: List[str] = []
    total = max(1, len(narrations))

    for i, text in enumerate(narrations, start=1):
        report(0.35 + (i - 1) / total * 0.15, f"🎧 Audio: beat {i}/{total}…")

        if not text:
            text = "Let’s go step by step together!"

        out_path = os.path.join(audio_dir, f"beat_{i:03d}.aiff")
        try:
            _say_to_wav(text, out_path, voice=voice, rate=rate)
        except Exception as e:
            return {"error": f"Audio generation failed at beat {i}: {e}"}

        out_paths.append(out_path)

    report(0.50, "🎧 Audio: done.")
    return out_paths
