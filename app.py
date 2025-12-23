# app.py
from __future__ import annotations

import os
import streamlit as st

st.set_page_config(
    page_title="Learnaroo",
    page_icon="🎬",
    layout="centered",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 760px;
}
div[data-testid="stForm"] {
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 14px;
}
footer, header, #MainMenu {
    visibility: hidden;
}
div[role="progressbar"] {
    height: 10px;
    border-radius: 999px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---- Header ----
st.markdown("## Learnaroo")
st.caption("Stories that make learning stick.")

# ---- Input Form ----
with st.form("gen_form", clear_on_submit=False):
    prompt = st.text_input(
        "What should the story teach?",
        placeholder="Example: Explain gravity to a kid using a superhero story",
        max_chars=400,
    )

    category = st.selectbox(
        "Story theme",
        ["superhero", "dinosaur", "space battle", "robot lab", "fairytale"],
        index=0,
    )

    gen_video = st.toggle("Generate video", value=True)

    submitted = st.form_submit_button("Generate", use_container_width=True)

# ---- Run Pipeline ----
if submitted:
    if not prompt.strip():
        st.warning("Please enter a topic to continue.")
        st.stop()

    # Lazy import (but show the REAL error if it fails)
    try:
        from core.director import run_pipeline
    except Exception as e:
        st.error(f"Pipeline import failed: {repr(e)}")
        st.stop()

    progress_bar = st.progress(0)
    status_line = st.empty()
    output_slot = st.empty()

    def report_cb(p: float, msg: str):
        try:
            p = float(p)
        except Exception:
            p = 0.0
        p = max(0.0, min(1.0, p))
        progress_bar.progress(int(p * 100))
        status_line.caption(msg)

    try:
        result = run_pipeline(
            prompt=prompt.strip(),
            category=category,
            gen_video=gen_video,
            report_cb=report_cb,
        )
    except Exception as e:
        progress_bar.empty()
        status_line.empty()
        st.error(f"Generation failed: {repr(e)}")
        st.stop()

    # ---- Clean UI: remove progress once done ----
    progress_bar.empty()
    status_line.empty()

    # ---- Show ONLY the final video ----
    final_video = None

    if isinstance(result, dict):
        video_obj = result.get("video")
        if isinstance(video_obj, str) and video_obj.endswith(".mp4"):
            final_video = video_obj
        else:
            run_dir = result.get("run_dir")
            if isinstance(run_dir, str):
                candidate = os.path.join(run_dir, "final_synced.mp4")
                if os.path.exists(candidate):
                    final_video = candidate

    if final_video and os.path.exists(final_video):
        output_slot.video(final_video)
    else:
        st.error("Final video not found. Veo may have returned a cloud URI instead of a local file.")
