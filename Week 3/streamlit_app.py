"""
Streamlit Frontend — Intelligent Video Analytics Platform
Week 3 - AI Summer Fellowship

Run with:
    streamlit run streamlit_app.py

Uses the same analytics_core.AnalyticsCore engine as app.py (the CLI tool),
so results are identical between the two frontends.
"""

import json
import os
import tempfile
import threading
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from analytics_core import AnalyticsCore, open_capture

st.set_page_config(page_title="Intelligent Video Analytics Platform", layout="wide")


class ThreadedWebcam:
    """
    Reads the webcam continuously in a background thread and always keeps
    only the MOST RECENT frame.

    Why this is needed: the camera pushes frames into its internal buffer
    faster than YOLO inference + Streamlit rendering can consume them. If you
    just call cv2.VideoCapture.read() from the slow consumer, the buffer
    fills up and read() starts returning older and older frames — which
    looks exactly like "freeze for a second, then jump ahead" lag. Reading
    in a separate thread that discards everything except the latest frame
    decouples camera speed from processing speed and eliminates that buildup.
    """

    def __init__(self, index=0):
        self.cap = open_capture(index)
        self.lock = threading.Lock()
        self.frame = None
        self.ok = self.cap.isOpened()
        self.running = self.ok
        if self.ok:
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.ok = True
            else:
                self.ok = False
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1)
        self.cap.release()


# --------------------------------------------------------------------------- #
# Session state defaults
# --------------------------------------------------------------------------- #
defaults = {
    "core": None,
    "line_pts": None,
    "zones": [],
    "webcam_running": False,
    "webcam_slot_claimed": False,
    "history": [],          # list of stats dicts, for the live charts
    "video_first_frame": None,
    "processed_video_path": None,
    "report": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --------------------------------------------------------------------------- #
# Sidebar — configuration
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuration")

source_type = st.sidebar.radio("Video Source", ["Upload Video", "Webcam"])

model_choice = st.sidebar.selectbox(
    "Model", ["yolov8n.pt (pretrained, fast)", "Custom weights (.pt)"]
)
if model_choice.startswith("Custom"):
    custom_model_file = st.sidebar.file_uploader("Upload your .pt weights", type=["pt"])
    model_path = None
    if custom_model_file is not None:
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp_model.write(custom_model_file.read())
        tmp_model.close()
        model_path = tmp_model.name
else:
    model_path = "yolov8n.pt"

tracker_choice = st.sidebar.selectbox("Tracker", ["bytetrack", "botsort"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 0.9, 0.3, 0.05)
display_width = st.sidebar.slider("Video Display Width (px)", 320, 800, 480, 20)
inference_size = st.sidebar.select_slider(
    "Inference Speed vs Accuracy", options=[256, 320, 480, 640], value=480,
    format_func=lambda x: {256: "Fastest", 320: "Fast", 480: "Balanced", 640: "Most Accurate"}[x],
)
st.sidebar.caption(
    f"YOLO runs detection at {inference_size}px internally — lower is much "
    "faster (especially on CPU/webcam) at a small accuracy cost. Doesn't "
    "affect the display size above."
)

device_choice = st.sidebar.selectbox("Compute Device", ["Auto", "CPU", "GPU (CUDA)"])
device_map = {"Auto": "auto", "CPU": "cpu", "GPU (CUDA)": "cuda:0"}
device_arg = device_map[device_choice]

fp16_enabled = st.sidebar.checkbox(
    "Half-precision (fp16) inference", value=False,
    disabled=(device_choice == "CPU"),
    help="Real speedup on a CUDA GPU with tensor cores; no benefit on CPU.",
)
if device_choice == "CPU":
    fp16_enabled = False

draw_zones_enabled = st.sidebar.checkbox("Enable ROI / Counting Line setup", value=True)
record_output = st.sidebar.checkbox("Save annotated output video", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: draw your counting line and ROI zones on the first frame below "
    "before starting. If you skip it, a default horizontal line is used."
)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🎥 Intelligent Video Analytics Platform")
st.caption("Real-time object detection, tracking, counting, and analytics — Week 3 Project")


# --------------------------------------------------------------------------- #
# Helper: resize an annotated frame down to a compact, screen-friendly width
# --------------------------------------------------------------------------- #
def resize_for_display(frame, target_width):
    h, w = frame.shape[:2]
    if w == 0:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, max(1, int(h * scale))))


# --------------------------------------------------------------------------- #
# Helper: canvas → line + polygons
# --------------------------------------------------------------------------- #
def extract_shapes_from_canvas(canvas_result, scale_x, scale_y):
    """Canvas objects: freeform lines (2 pts as the first shape drawn) become
    the counting line if drawn with the 'line' tool; polygons become zones."""
    line = None
    zones = []
    if canvas_result.json_data is None:
        return line, zones

    for obj in canvas_result.json_data.get("objects", []):
        if obj["type"] == "line":
            x1 = (obj["left"] + obj["x1"]) * scale_x
            y1 = (obj["top"] + obj["y1"]) * scale_y
            x2 = (obj["left"] + obj["x2"]) * scale_x
            y2 = (obj["top"] + obj["y2"]) * scale_y
            line = [[int(x1), int(y1)], [int(x2), int(y2)]]
        elif obj["type"] == "path":
            pts = []
            for seg in obj["path"]:
                if len(seg) >= 3:
                    pts.append([int(seg[1] * scale_x + obj.get("left", 0)),
                                int(seg[2] * scale_y + obj.get("top", 0))])
            if len(pts) >= 3:
                zones.append(pts)
    return line, zones


# --------------------------------------------------------------------------- #
# MODE 1: Upload Video
# --------------------------------------------------------------------------- #
if source_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_video.write(uploaded_file.read())
        tmp_video.close()
        video_path = tmp_video.name

        cap = cv2.VideoCapture(video_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        ok, first_frame = cap.read()
        cap.release()

        if ok:
            st.session_state.video_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        # ---- ROI / line setup on the first frame ---- #
        if draw_zones_enabled and st.session_state.video_first_frame is not None:
            st.subheader("1️⃣ Draw Counting Line & ROI Zones")
            st.caption(
                "Select **Line** tool and drag once for the counting line. "
                "Select **Polygon (freeform)** and draw closed shapes for ROI zones. "
                "Skip this step to use a default horizontal line with no zones."
            )
            draw_tool = st.radio("Drawing tool", ["line", "polygon"], horizontal=True)
            canvas_display_width = 700
            scale = canvas_display_width / frame_w
            canvas_display_height = int(frame_h * scale)

            bg_image = Image.fromarray(st.session_state.video_first_frame) \
                if st.session_state.video_first_frame is not None else None

            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.2)",
                stroke_width=3,
                stroke_color="#FF0000" if draw_tool == "line" else "#00FF00",
                background_image=bg_image,
                update_streamlit=True,
                height=canvas_display_height,
                width=canvas_display_width,
                drawing_mode="line" if draw_tool == "line" else "polygon",
                key="canvas_upload",
            )

            if canvas_result is not None:
                line, zones = extract_shapes_from_canvas(canvas_result, 1 / scale, 1 / scale)
                if line:
                    st.session_state.line_pts = line
                if zones:
                    st.session_state.zones = zones

            colA, colB = st.columns(2)
            colA.info(f"Line: {st.session_state.line_pts or 'default (mid-frame horizontal)'}")
            colB.info(f"Zones defined: {len(st.session_state.zones)}")

        st.subheader("2️⃣ Run Analytics")
        start_btn = st.button("▶️ Start Processing", type="primary")

        if start_btn:
            if model_path is None:
                st.error("Please upload custom weights, or switch to the pretrained model.")
                st.stop()

            core = AnalyticsCore(
                model_path=model_path, tracker=tracker_choice, conf=conf_threshold,
                frame_w=frame_w, frame_h=frame_h,
                line=st.session_state.line_pts, zones=st.session_state.zones,
                imgsz=inference_size, device=device_arg, fp16=fp16_enabled,
            )

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            chart_placeholder = st.empty()

            writer = None
            out_path = None
            if record_output:
                os.makedirs("output", exist_ok=True)
                out_path = os.path.join("output", f"annotated_{int(time.time())}.mp4")
                from analytics_core import DASHBOARD_HEIGHT, create_video_writer
                writer, out_path = create_video_writer(out_path, src_fps, (frame_w, frame_h + DASHBOARD_HEIGHT))
                if writer is None:
                    st.warning("Could not open a video writer on this system — recording disabled this run.")
            history = []
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, stats = core.process_frame(frame)  # dashboard baked in, full size
                history.append(stats)

                if writer is not None:
                    writer.write(annotated)  # recorded video keeps full-size dashboard overlay

                frame_idx += 1
                if frame_idx % 2 == 0 or frame_idx == 1:  # update UI every other frame for speed
                    display_frame = resize_for_display(annotated, display_width)
                    frame_placeholder.image(
                        cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB",
                        output_format="JPEG",
                    )
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            if writer is not None:
                writer.release()

            st.session_state.processed_video_path = out_path
            st.session_state.report = core.build_report(uploaded_file.name, model_path)
            st.session_state.history = history

            st.success("✅ Processing complete!")

        # ---- Results section (persists after run) ---- #
        if st.session_state.history:
            st.subheader("3️⃣ Session Analytics")
            df = pd.DataFrame(st.session_state.history)
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(df[["fps"]], height=250)
                st.caption("FPS over time")
            with col2:
                st.line_chart(df[["current_objects"]], height=250)
                st.caption("Objects per frame over time")

            report = st.session_state.report
            st.json(report)

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "⬇️ Download Session Report (JSON)",
                    data=json.dumps(report, indent=2),
                    file_name=f"session_report_{report['run_timestamp']}.json",
                    mime="application/json",
                )
            with dl_col2:
                if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                    video_path = st.session_state.processed_video_path
                    mime = "video/x-msvideo" if video_path.lower().endswith(".avi") else "video/mp4"
                    with open(video_path, "rb") as vf:
                        st.download_button(
                            "⬇️ Download Annotated Video",
                            data=vf.read(),
                            file_name=os.path.basename(video_path),
                            mime=mime,
                        )


# --------------------------------------------------------------------------- #
# MODE 2: Webcam (live, fragment-based — refreshes only the video section,
# not the whole page, so it doesn't flash/flicker)
# --------------------------------------------------------------------------- #
else:
    st.subheader("Live Webcam Analytics")
    st.caption(
        "Runs against the server's local webcam (device 0) using OpenCV. "
        "Works when Streamlit is running on your own machine."
    )

    col_start, col_stop = st.columns(2)
    start_webcam = col_start.button("▶️ Start Webcam", disabled=st.session_state.webcam_running)
    stop_webcam = col_stop.button("⏹️ Stop Webcam", disabled=not st.session_state.webcam_running)

    if start_webcam:
        if model_path is None:
            st.error("Please upload custom weights, or switch to the pretrained model.")
            st.stop()
        st.session_state.webcam_running = True
        st.session_state.core = AnalyticsCore(
            model_path=model_path, tracker=tracker_choice, conf=conf_threshold,
            frame_w=640, frame_h=480, line=None, zones=[],
            imgsz=inference_size, device=device_arg, fp16=fp16_enabled,
        )
        st.session_state.webcam_cap = ThreadedWebcam(0)
        # force a fresh slot-claim on the next fragment call, during this
        # same full rerun (see note inside webcam_loop below)
        st.session_state.webcam_slot_claimed = False

    if stop_webcam:
        st.session_state.webcam_running = False
        cap = st.session_state.get("webcam_cap")
        if cap is not None:
            cap.release()
        st.session_state.webcam_cap = None

    frame_placeholder = st.empty()

    @st.fragment(run_every=0.02)
    def webcam_loop():
        # Claim the placeholder's slot with a write ONLY ONCE (not every
        # tick) — Streamlit needs one write to it during a full page rerun
        # to register a stable position before timer-driven reruns can use
        # it. Writing every tick works too, but since clearing is instant
        # and drawing the next real frame takes 100-300ms (YOLO + tracking),
        # doing it every tick causes a visible blank flash before each frame.
        if not st.session_state.get("webcam_slot_claimed", False):
            frame_placeholder.empty()
            st.session_state.webcam_slot_claimed = True

        if not st.session_state.webcam_running:
            return
        cap = st.session_state.get("webcam_cap")
        ok, frame = (False, None) if cap is None else cap.read()

        if not ok:
            # background thread may just need a moment to grab its first
            # frame — only treat this as fatal after ~2 seconds of failures
            st.session_state.webcam_fail_count = st.session_state.get("webcam_fail_count", 0) + 1
            if st.session_state.webcam_fail_count > 40:
                st.error("Could not read from webcam (device 0). Click Start Webcam again, "
                          "or check that no other app is using the camera.")
                st.session_state.webcam_running = False
                if cap is not None:
                    cap.release()
                st.session_state.webcam_cap = None
            return

        st.session_state.webcam_fail_count = 0
        annotated, stats = st.session_state.core.process_frame(frame)  # dashboard baked in
        display_frame = resize_for_display(annotated, display_width)
        frame_placeholder.image(
            cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB",
            output_format="JPEG",
        )

    webcam_loop()

st.sidebar.markdown("---")
st.sidebar.caption("Week 3 · Intelligent Video Analytics Platform · Streamlit Frontend")
