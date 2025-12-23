# tools/genai_client.py
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

from dotenv import load_dotenv

from google import genai
from google.genai import types

ProgressCb = Optional[Callable[[float, str], None]]


def _report(cb: ProgressCb, p: float, msg: str) -> None:
    if cb:
        try:
            cb(max(0.0, min(1.0, float(p))), str(msg))
        except Exception:
            pass


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else default


def _is_gcs_uri(x: Any) -> bool:
    return isinstance(x, str) and x.startswith("gs://")


def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _extract_video_from_result(res: Any) -> Union[str, bytes]:
    """
    Returns either:
      - gs:// URI (str)
      - raw mp4 bytes (bytes)
      - OR a local/remote mp4 path (str)
    Raises RuntimeError if cannot extract.
    """
    # ---- generated_videos path (common) ----
    try:
        gv = getattr(res, "generated_videos", None)
        if gv and len(gv) > 0:
            video = getattr(gv[0], "video", None)
            if video:
                uri = getattr(video, "uri", None)
                if _is_gcs_uri(uri):
                    return uri

                vb = getattr(video, "video_bytes", None)
                if vb:
                    return vb

                b64f = getattr(video, "bytes_base64_encoded", None)
                if b64f:
                    return base64.b64decode(b64f)
    except Exception:
        pass

    # ---- dict-like fallback ----
    if isinstance(res, dict):
        gv = res.get("generated_videos")
        if isinstance(gv, list) and gv:
            v = gv[0].get("video", {})
            if isinstance(v, dict):
                uri = v.get("uri")
                if _is_gcs_uri(uri):
                    return uri
                if v.get("video_bytes"):
                    return v["video_bytes"]
                if v.get("bytes_base64_encoded"):
                    return base64.b64decode(v["bytes_base64_encoded"])

    # ---- files[] fallback ----
    try:
        files = getattr(res, "files", None)
        if files:
            for f in files:
                if isinstance(f, str) and f.endswith(".mp4"):
                    return f
                if getattr(f, "uri", None):
                    return f.uri
                if getattr(f, "name", None):
                    return f.name
    except Exception:
        pass

    raise RuntimeError("❌ Could not extract video uri/bytes from Veo response.")


def _op_to_operation(op: Any) -> Any:
    """
    Normalizes op handle so operations.get() never receives a bare string.

    Cases seen:
    - op is an Operation-like object (has .name or .done)
    - op is a string operation name
    - op is a dict-like structure containing "name"
    """
    if op is None:
        return op

    # If SDK returned a string operation name
    if isinstance(op, str):
        return types.Operation(name=op)

    # If dict-like
    if isinstance(op, dict) and "name" in op and isinstance(op["name"], str):
        return types.Operation(name=op["name"])

    # If it has "name" attribute already, keep it
    if getattr(op, "name", None):
        return op

    # Some SDK objects store the name under .operation.name or similar
    inner = getattr(op, "operation", None)
    if inner is not None and getattr(inner, "name", None):
        return inner

    return op


def _operation_done(op: Any) -> bool:
    """
    Robust 'done' check across SDK variants.
    """
    try:
        d = getattr(op, "done", None)
        if isinstance(d, bool):
            return d
    except Exception:
        pass

    # Sometimes op has .metadata with state
    md = getattr(op, "metadata", None)
    if isinstance(md, dict):
        state = md.get("state") or md.get("status")
        if isinstance(state, str) and state.lower() in {"succeeded", "success", "done", "completed"}:
            return True

    return False


def _get_operation_result(op: Any) -> Any:
    """
    Extract the completion payload from many possible SDK shapes.
    """
    # Most common:
    for attr in ("result", "response", "_result", "output"):
        try:
            v = getattr(op, attr, None)
            if v is not None:
                return v
        except Exception:
            pass

    # Some variants provide a callable result()
    try:
        if callable(getattr(op, "result", None)):
            return op.result()
    except Exception:
        pass

    # Some place payload under op.response or op.response.generated_videos
    return None


@dataclass
class GenAIClient:
    """
    Supports:
      - Gemini text generation via API key OR Vertex
      - Veo video generation via Vertex ONLY
    """

    text_client: genai.Client
    vertex_client: Optional[genai.Client]
    gemini_model: str
    veo_model: str
    project: Optional[str]
    location: Optional[str]

    @classmethod
    def from_env(cls) -> "GenAIClient":
        load_dotenv()

        gemini_model = _env("GEMINI_MODEL", "gemini-2.0-flash")
        veo_model = _env("VEO_MODEL", "veo-2.0-generate-001")

        api_key = _env("GEMINI_API_KEY")
        project = _env("VERTEX_PROJECT") or _env("GOOGLE_CLOUD_PROJECT") or _env("PROJECT_ID")
        location = _env("VERTEX_LOCATION") or _env("GOOGLE_CLOUD_LOCATION") or _env("LOCATION")

        # text client
        if api_key:
            text_client = genai.Client(api_key=api_key)
        else:
            if not project or not location:
                raise RuntimeError("Missing GEMINI_API_KEY and missing VERTEX_PROJECT/VERTEX_LOCATION for Vertex auth.")
            text_client = genai.Client(vertexai=True, project=project, location=location)

        # vertex client (needed for Veo)
        vertex_client = None
        if project and location:
            vertex_client = genai.Client(vertexai=True, project=project, location=location)

        return cls(
            text_client=text_client,
            vertex_client=vertex_client,
            gemini_model=gemini_model,
            veo_model=veo_model,
            project=project,
            location=location,
        )

    # -----------------------------
    # Text generation (Gemini)
    # -----------------------------
    def generate_text(self, prompt: str, system: Optional[str] = None) -> str:
        contents = prompt if not system else f"{system}\n\n{prompt}"
        resp = self.text_client.models.generate_content(
            model=self.gemini_model,
            contents=contents,
        )
        return (getattr(resp, "text", None) or "").strip()

    def generate_json(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        txt = self.generate_text(prompt, system=system)
        try:
            return json.loads(txt)
        except Exception:
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(txt[start : end + 1])
                except Exception:
                    pass
        return {"error": "Failed to parse JSON from model", "raw": txt}

    # -----------------------------
    # Video generation (Veo)
    # -----------------------------
    def generate_video(
        self,
        prompt: str,
        out_dir: str,
        duration_seconds: int = 6,
        filename: str = "scene.mp4",
        progress_cb: ProgressCb = None,
    ) -> str:
        """
        Generate a video using Veo.
        Returns either:
          - local mp4 path (if bytes returned)
          - OR gs:// URI (if Veo returns a GCS path)
          - OR a provider-returned mp4 path (string)
        """
        if not self.vertex_client:
            raise RuntimeError("Vertex client not initialized. Set VERTEX_PROJECT and VERTEX_LOCATION.")

        _mkdir(out_dir)
        out_path = os.path.join(out_dir, filename)

        def report(p: float, msg: str):
            _report(progress_cb, p, msg)

        report(0.01, "🎬 Veo: submitting render job…")

        # 1) Submit job (may return Operation object OR sometimes a name-like object)
        op = self.vertex_client.models.generate_videos(
            model=self.veo_model,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                duration_seconds=int(duration_seconds),
            ),
        )

        op = _op_to_operation(op)

        # 2) Poll until done
        start = time.time()
        max_poll_seconds = 180
        poll = 0

        while not _operation_done(op):
            poll += 1
            elapsed = time.time() - start
            p = min(0.85, 0.10 + (elapsed / max_poll_seconds) * 0.75)

            report(p, f"⏳ Veo: rendering (poll #{poll}, {int(elapsed)}s)")
            time.sleep(5)

            # operations.get MUST receive an Operation object, not a string
            op = _op_to_operation(op)
            op = self.vertex_client.operations.get(op)

            # soft extend
            if elapsed > max_poll_seconds:
                report(0.86, "⚠️ Veo: taking longer than usual, still waiting…")
                max_poll_seconds += 120

        report(0.90, "📦 Veo: render complete. Extracting output…")

        # 3) Extract result safely
        res = _get_operation_result(op)

        # Some SDKs keep payload deeper; refresh once more if needed
        if res is None:
            try:
                op = self.vertex_client.operations.get(_op_to_operation(op))
                res = _get_operation_result(op)
            except Exception:
                pass

        if res is None:
            raise RuntimeError(f"Veo finished but returned no result. metadata={getattr(op,'metadata',None)}")

        payload = _extract_video_from_result(res)

        # 4) Return uri or write bytes
        if isinstance(payload, str) and _is_gcs_uri(payload):
            report(1.0, "✅ Veo: video ready (GCS URI).")
            return payload

        if isinstance(payload, (bytes, bytearray)):
            with open(out_path, "wb") as f:
                f.write(payload)
            report(1.0, "✅ Veo: video saved locally.")
            return out_path

        # if payload is some other mp4 path string
        if isinstance(payload, str):
            report(1.0, "✅ Veo: video returned as path.")
            return payload

        raise RuntimeError("Veo output payload type unsupported.")
