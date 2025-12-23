from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips

from core.schemas import LessonPlan
from tools.genai_client import GenAIClient

VEO_ALLOWED_DURATIONS = [5, 6, 7, 8]


def _pick_veo_duration(audio_dur: float) -> int:
    if audio_dur <= 0:
        return 6
    return min(VEO_ALLOWED_DURATIONS, key=lambda d: abs(d - audio_dur))


def _audio_duration_s(path: str) -> float:
    try:
        a = AudioFileClip(path)
        d = float(a.duration or 0)
        a.close()
        return max(0.0, d)
    except Exception:
        return 0.0


def _is_gcs_uri(x: str) -> bool:
    return isinstance(x, str) and x.startswith("gs://")


def _download_gcs(gs_uri: str, local_path: str) -> bool:
    try:
        from google.cloud import storage

        bucket_name, blob_name = gs_uri[5:].split("/", 1)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)

        return os.path.exists(local_path) and os.path.getsize(local_path) > 0
    except Exception as e:
        print("❌ GCS download failed:", repr(e))
        return False


def _ensure_local_mp4(path_or_uri: str, local_path: str) -> Optional[str]:
    if _is_gcs_uri(path_or_uri):
        ok = _download_gcs(path_or_uri, local_path)
        return local_path if ok else None

    if isinstance(path_or_uri, str) and os.path.exists(path_or_uri):
        return path_or_uri

    return None


def _flatten_beats(plan: LessonPlan) -> List[Tuple[int, int, str, str, int]]:
    """
    Returns beats in order:
      (scene_idx, beat_idx, narration, visual_prompt, duration_s)
    """
    out: List[Tuple[int, int, str, str, int]] = []
    for sc in plan.scenes:
        if sc.beats:
            for b in sc.beats:
                out.append((sc.idx, b.idx, b.narration, b.visual_prompt, int(b.duration_s)))
        else:
            out.append((sc.idx, 1, sc.narration or "", sc.visual_prompt or "", 6))
    return out


def _character_bible_to_dict(plan: LessonPlan) -> Dict[str, Any]:
    cb = getattr(plan, "character_bible", None)
    if cb is None:
        return {}

    # pydantic model -> dict
    if hasattr(cb, "model_dump"):
        return cb.model_dump()

    if isinstance(cb, dict):
        return cb

    return {}


def _build_veo_prompt(
    beat_visual: str,
    category: str,
    character_bible: Dict[str, Any],
) -> str:
    """
    HARD-lock character consistency by injecting:
      - style_lock
      - character_tokens
      - negative_tokens
    """
    style_lock = (character_bible.get("style_lock") or "").strip()
    tokens = character_bible.get("character_tokens") or []
    neg = character_bible.get("negative_tokens") or []

    token_line = ""
    if isinstance(tokens, list) and tokens:
        token_line = "CHARACTER TOKENS:\n" + "\n".join([f"- {str(t).strip()}" for t in tokens[:8]])

    neg_line = ""
    if isinstance(neg, list) and neg:
        neg_line = "AVOID:\n" + "\n".join([f"- {str(t).strip()}" for t in neg[:10]])

    return (
        "Kid-friendly 2D animated style, bright colors, simple shapes, smooth motion.\n"
        "No text on screen. No watermark. No logos.\n"
        "No scary visuals. No weapons. No violence.\n"
        "Continuity rule: SAME named character must look IDENTICAL across clips.\n\n"
        f"{style_lock}\n\n"
        f"{token_line}\n\n"
        f"{neg_line}\n\n"
        f"Theme world (MANDATORY): {category}\n"
        f"Shot description: {beat_visual}\n"
        "Camera: gentle, steady, medium shots, clear actions."
    ).strip()


def _sync_video_with_audio(video_path: str, audio_path: str, out_path: str) -> str:
    """
    Hard-sync: loops/trims video to match audio duration, then attaches audio.
    MoviePy v2 API: .with_audio(), .subclipped()
    """
    v = None
    a = None
    try:
        v = VideoFileClip(video_path)
        a = AudioFileClip(audio_path)

        ad = float(a.duration or 0)
        vd = float(v.duration or 0)

        if ad <= 0:
            v.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
            return out_path
        if vd <= 0:
            raise RuntimeError(f"Bad video duration for {video_path}")

        if vd < ad:
            clips = []
            remaining = ad
            while remaining > 0:
                take = min(vd, remaining)
                clips.append(v.subclipped(0, take))
                remaining -= take
            v2 = concatenate_videoclips(clips, method="compose")
        else:
            v2 = v.subclipped(0, ad)

        final = v2.with_audio(a)
        final.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
        return out_path
    finally:
        try:
            if v:
                v.close()
        except Exception:
            pass
        try:
            if a:
                a.close()
        except Exception:
            pass


def run(
    client: GenAIClient,
    plan: LessonPlan,
    audio_paths: List[str],
    run_dir: str,
    report_cb: Optional[Callable[[float, str], None]] = None,
) -> Union[str, List[str]]:
    """
    Beat-wise generation + sync:
      - flatten beats
      - generate Veo clip per beat
      - ensure local (download if gs://)
      - sync clip duration to its beat audio
      - concatenate synced clips into final
    """
    veo_dir = os.path.join(run_dir, "veo")
    os.makedirs(veo_dir, exist_ok=True)

    def report(p: float, msg: str):
        if report_cb:
            try:
                report_cb(max(0.0, min(1.0, float(p))), msg)
            except Exception:
                report_cb(0.0, msg)

    beats = _flatten_beats(plan)
    if not beats:
        return []

    character_bible = _character_bible_to_dict(plan)

    total = max(1, len(beats))
    raw_outputs: List[str] = []
    synced_paths: List[str] = []

    # 1) Generate raw clips per beat/audio
    report(0.55, f"🎬 Veo: generating {len(beats)} beat clips…")

    for i, beat in enumerate(beats):
        scene_idx, beat_idx, narration, visual_prompt, schema_duration = beat

        audio_path = audio_paths[i] if i < len(audio_paths) else ""
        audio_dur = _audio_duration_s(audio_path) if audio_path else 0.0

        # prefer schema duration, else nearest audio
        veo_dur = int(schema_duration) if schema_duration in VEO_ALLOWED_DURATIONS else _pick_veo_duration(audio_dur)

        # If visual_prompt is empty, fall back to narration guidance
        if not visual_prompt.strip():
            visual_prompt = f"Show the actions described: {narration[:140]}"

        veo_prompt = _build_veo_prompt(
            beat_visual=visual_prompt,
            category=str(plan.category),
            character_bible=character_bible,
        )

        report(0.55 + (i / total) * 0.15, f"🎬 Veo: beat {i+1}/{total} ({veo_dur}s)…")

        raw_out = client.generate_video(
            prompt=veo_prompt,
            out_dir=veo_dir,
            duration_seconds=int(veo_dur),
            filename=f"beat_{i+1:03d}_raw.mp4",
            progress_cb=report_cb,
        )
        raw_outputs.append(raw_out)

    # 2) Ensure local + sync each clip to its beat audio
    report(0.72, "🎧 Syncing each beat video to its audio…")

    for i, raw_out in enumerate(raw_outputs):
        local_raw_target = os.path.join(veo_dir, f"beat_{i+1:03d}_raw.mp4")
        local_raw = _ensure_local_mp4(raw_out, local_raw_target)
        if not local_raw:
            report(0.95, "⚠️ Some Veo outputs are only URIs; returning raw outputs list.")
            return raw_outputs

        if i >= len(audio_paths):
            report(0.95, "⚠️ Missing audio for some beats; returning raw outputs list.")
            return raw_outputs

        out_synced = os.path.join(veo_dir, f"beat_{i+1:03d}_synced.mp4")
        report(0.72 + (i / total) * 0.20, f"🔗 Syncing beat {i+1}/{total}…")
        synced = _sync_video_with_audio(local_raw, audio_paths[i], out_synced)
        synced_paths.append(synced)

    # 3) Concatenate synced clips
    report(0.92, "🎞️ Joining all synced beat clips…")
    vclips = []
    try:
        vclips = [VideoFileClip(p) for p in synced_paths]
        final = concatenate_videoclips(vclips, method="compose")
        out_final = os.path.join(run_dir, "final_synced.mp4")
        final.write_videofile(out_final, codec="libx264", audio_codec="aac", logger=None)
    finally:
        for vc in vclips:
            try:
                vc.close()
            except Exception:
                pass

    report(1.0, "✅ Final synced video ready!")
    return out_final
