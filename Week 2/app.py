# ─────────────────────────────────────────────────────────────────────────────
#  Bottle Detection System — AI Summer Fellowship 2026
#  Assignment 5 | Week 2 | Track 1: Computer Vision
#  Model: YOLOv8n | mAP50: 87.1%
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import io
import time

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bottle Detector",
    page_icon="🍾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #080C17; }
.block-container { padding: 1.5rem 2rem 2rem; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0F1623 0%, #111827 100%);
    border: 1px solid #1E2D45;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #06B6D4, #10B981, #06B6D4);
}
.app-title { font-size: 28px; font-weight: 700; color: #F1F5F9; margin: 0; }
.app-title span { color: #06B6D4; }
.app-subtitle { font-size: 13px; color: #64748B; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
.header-badges { display: flex; gap: 8px; }
.hbadge {
    font-size: 11px; font-family: 'JetBrains Mono', monospace;
    padding: 4px 10px; border-radius: 6px; font-weight: 500;
}
.hb-cyan  { background: rgba(6,182,212,.15);  color: #06B6D4; border: 1px solid rgba(6,182,212,.3); }
.hb-green { background: rgba(16,185,129,.15); color: #10B981; border: 1px solid rgba(16,185,129,.3); }
.hb-amber { background: rgba(245,158,11,.15); color: #F59E0B; border: 1px solid rgba(245,158,11,.3); }

/* ── Metric Cards ── */
.metric-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 1.5rem; }
.metric-card {
    background: #111827;
    border: 1px solid #1E2D45;
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.mc-cyan::after  { background: #06B6D4; }
.mc-green::after { background: #10B981; }
.mc-amber::after { background: #F59E0B; }
.mc-purple::after{ background: #8B5CF6; }
.metric-val { font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.mc-cyan  .metric-val { color: #06B6D4; }
.mc-green .metric-val { color: #10B981; }
.mc-amber .metric-val { color: #F59E0B; }
.mc-purple .metric-val { color: #8B5CF6; }
.metric-label { font-size: 11px; color: #64748B; margin-top: 4px; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: .06em; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #1E2D45;
    gap: 4px;
    margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #64748B;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    font-size: 14px;
    padding: 8px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #06B6D4 !important;
    color: #000 !important;
}

/* ── Upload Zone ── */
.uploadedFile { border-radius: 12px !important; }
[data-testid="stFileUploader"] {
    background: #111827;
    border: 2px dashed #1E2D45;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: #06B6D4; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F1623;
    border-right: 1px solid #1E2D45;
}
[data-testid="stSidebar"] .block-container { padding: 1rem; }

.sidebar-section {
    background: #111827;
    border: 1px solid #1E2D45;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
}
.sidebar-label {
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: #06B6D4;
    margin-bottom: 10px;
    font-weight: 600;
}

/* ── Result Panel ── */
.result-panel {
    background: #111827;
    border: 1px solid #1E2D45;
    border-radius: 12px;
    padding: 1rem;
}
.result-label {
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: #64748B;
    margin-bottom: 8px;
}
.result-label.live { color: #10B981; }

/* ── Detection Tags ── */
.det-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.det-tag {
    font-size: 12px; font-family: 'JetBrains Mono', monospace;
    padding: 3px 10px; border-radius: 6px; font-weight: 500;
    background: rgba(6,182,212,.12); color: #06B6D4;
    border: 1px solid rgba(6,182,212,.3);
}

/* ── Info Cards ── */
.info-card {
    background: #111827;
    border: 1px solid #1E2D45;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #94A3B8;
}
.info-card strong { color: #F1F5F9; }

/* ── Buttons ── */
.stButton button {
    background: #06B6D4;
    color: #000;
    border: none;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 20px;
    transition: background .15s;
    width: 100%;
}
.stButton button:hover { background: #0891B2; color: #000; }

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] { padding: 0 4px; }
div[data-baseweb="slider"] div { background: #06B6D4 !important; }

/* ── Select ── */
.stSelectbox select { background: #111827; color: #F1F5F9; border: 1px solid #1E2D45; }

/* ── Status ── */
.status-running { color: #F59E0B; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.status-done { color: #10B981; font-family: 'JetBrains Mono', monospace; font-size: 12px; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #334155;
}
.empty-icon { font-size: 48px; margin-bottom: 12px; }
.empty-title { font-size: 18px; font-weight: 500; color: #4B5563; margin-bottom: 6px; }
.empty-sub { font-size: 13px; color: #374151; }

/* ── Download button ── */
.stDownloadButton button {
    background: #1E293B;
    color: #F1F5F9;
    border: 1px solid #334155;
}
.stDownloadButton button:hover { background: #334155; border-color: #06B6D4; color: #06B6D4; }
</style>
""", unsafe_allow_html=True)

# ── MODEL PATH ────────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\SS\OneDrive\Desktop\University\Internship\Week 2\Bottle-Detection\runs\detect\runs\bottle_v1-3\weights\best.pt"

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    return YOLO(path)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def run_detection(img_bgr, model, conf, iou):
    results = model(img_bgr, conf=conf, iou=iou, verbose=False)
    return results[0]

def draw_boxes(img_bgr, result, box_color, thickness, show_label, show_conf):
    out = img_bgr.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out, 0, []

    detections = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_val = float(box.conf[0])
        cls_id   = int(box.cls[0])
        cls_name = result.names[cls_id]

        color = tuple(int(box_color.lstrip('#')[i:i+2], 16) for i in (0,2,4))
        color_bgr = (color[2], color[1], color[0])

        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness)

        if show_label or show_conf:
            label_parts = []
            if show_label: label_parts.append(cls_name)
            if show_conf:  label_parts.append(f"{conf_val:.2f}")
            label = " ".join(label_parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 8, y1), color_bgr, -1)
            cv2.putText(out, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

        detections.append({"label": cls_name, "confidence": conf_val,
                            "bbox": [x1, y1, x2-x1, y2-y1]})

    return out, len(detections), detections

def img_to_download(img_rgb):
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def metric_cards(n_bottles, avg_conf, fps=None, model_size="6.2 MB"):
    fps_str = f"{fps:.1f}" if fps else "—"
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card mc-cyan">
            <div class="metric-val">{n_bottles}</div>
            <div class="metric-label">Bottles Found</div>
        </div>
        <div class="metric-card mc-green">
            <div class="metric-val">{avg_conf:.0%}</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        <div class="metric-card mc-amber">
            <div class="metric-val">{fps_str}</div>
            <div class="metric-label">FPS</div>
        </div>
        <div class="metric-card mc-purple">
            <div class="metric-val">{model_size}</div>
            <div class="metric-label">Model Size</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🍾 <span>Bottle</span> Detection System</div>
        <div class="app-subtitle">YOLOv8n · mAP50 87.1% · AI Summer Fellowship 2026 · Week 2</div>
    </div>
    <div class="header-badges">
        <span class="hbadge hb-cyan">YOLOv8n</span>
        <span class="hbadge hb-green">mAP50: 87.1%</span>
        <span class="hbadge hb-amber">GPU Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <strong>YOLOv8 Nano</strong><br>
        <span style="font-family:monospace;font-size:11px;color:#64748B">
        Params: 3.0M<br>
        Size: 6.2 MB<br>
        Speed: 9.3ms/img<br>
        Epochs: 50
        </span>
    </div>
    """, unsafe_allow_html=True)

    model_loaded = os.path.exists(MODEL_PATH)
    if model_loaded:
        st.success("✓ Model loaded")
        model = load_model(MODEL_PATH)
    else:
        st.error("Model file not found. Check path in app.py")
        model = None

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Detection Settings</div>', unsafe_allow_html=True)

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.95, value=0.40, step=0.05,
        help="Lower = detect more (may include false positives). Higher = only strong detections."
    )
    iou_thresh = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10, max_value=0.90, value=0.45, step=0.05,
        help="Controls overlap between boxes. Lower = fewer overlapping boxes."
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Box Style</div>', unsafe_allow_html=True)

    box_color   = st.color_picker("Box Color", "#06B6D4")
    box_thick   = st.slider("Box Thickness", 1, 8, 2)
    show_label  = st.checkbox("Show Label", True)
    show_conf   = st.checkbox("Show Confidence", True)

    st.markdown("---")
    st.markdown("""
    <div class="info-card" style="font-size:12px">
        <strong>Model Performance</strong><br>
        <span style="color:#10B981">Precision: 93.5%</span><br>
        <span style="color:#06B6D4">Recall: 79.2%</span><br>
        <span style="color:#F59E0B">mAP50: 87.1%</span><br>
        <span style="color:#8B5CF6">mAP50-95: 62.0%</span>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab_img, tab_video, tab_cam = st.tabs([
    "📷  Image Detection",
    "🎬  Video Detection",
    "📹  Webcam Live"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:
    uploaded = st.file_uploader(
        "Upload an image — JPG, PNG, BMP, WEBP supported",
        type=["jpg","jpeg","png","bmp","webp"],
        label_visibility="collapsed"
    )

    if uploaded is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📷</div>
            <div class="empty-title">Upload an image to detect bottles</div>
            <div class="empty-sub">Supports JPG, PNG, BMP, WEBP — drag and drop or click above</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if model is None:
            st.error("Model not loaded. Cannot run detection.")
        else:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            h, w       = img_bgr.shape[:2]

            with st.spinner("Detecting bottles..."):
                t0     = time.time()
                result = run_detection(img_bgr, model, confidence, iou_thresh)
                t1     = time.time()
                fps    = 1 / (t1 - t0)

            out_bgr, n_det, dets = draw_boxes(
                img_bgr, result, box_color, box_thick, show_label, show_conf
            )
            avg_conf = np.mean([d["confidence"] for d in dets]) if dets else 0.0

            metric_cards(n_det, avg_conf, fps)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="result-label">Original</div>', unsafe_allow_html=True)
                st.image(bgr_to_rgb(img_bgr), use_container_width=True)
            with col2:
                st.markdown('<div class="result-label live">Detected</div>', unsafe_allow_html=True)
                st.image(bgr_to_rgb(out_bgr), use_container_width=True)

            # Detection Details
            if dets:
                st.markdown("---")
                dc1, dc2 = st.columns([2,1])
                with dc1:
                    st.markdown(f"**{n_det} bottle{'s' if n_det>1 else ''} detected**")
                    tags_html = "".join([
                        f'<span class="det-tag">#{i+1} {d["label"]} {d["confidence"]:.0%}</span>'
                        for i, d in enumerate(dets)
                    ])
                    st.markdown(f'<div class="det-tags">{tags_html}</div>', unsafe_allow_html=True)
                with dc2:
                    st.markdown(f"""
                    <div class="info-card" style="margin-top:0">
                        <strong>Image Info</strong><br>
                        <span style="font-family:monospace;font-size:11px;color:#64748B">
                        Size: {w}×{h} px<br>
                        Speed: {(t1-t0)*1000:.1f}ms<br>
                        File: {uploaded.name}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                st.download_button(
                    "💾 Download Result Image",
                    data=img_to_download(bgr_to_rgb(out_bgr)),
                    file_name=f"bottle_detection_{uploaded.name.split('.')[0]}.png",
                    mime="image/png"
                )
            else:
                st.markdown("""
                <div style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);
                border-radius:10px;padding:14px;font-size:13px;color:#F59E0B;margin-top:12px">
                ⚠️ No bottles detected. Try lowering the Confidence Threshold in the sidebar.
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    vid_file = st.file_uploader(
        "Upload a video file",
        type=["mp4","avi","mov","mkv","webm"],
        label_visibility="collapsed",
        key="vid_up"
    )

    if vid_file is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎬</div>
            <div class="empty-title">Upload a video to detect bottles</div>
            <div class="empty-sub">Supports MP4, AVI, MOV, MKV — model processes every frame</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if model is None:
            st.error("Model not loaded.")
        else:
            max_frames = st.slider("Max frames to process", 30, 500, 150, 10,
                help="More frames = more processing time. Start with 150.")

            if st.button("▶ Run Detection on Video", key="run_vid"):
                # Save to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(vid_file.read())
                tfile.close()

                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 25
                w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                process_n    = min(max_frames, total_frames)

                # Output video
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
                writer   = cv2.VideoWriter(out_path, fourcc, fps_vid, (w_vid, h_vid))

                progress_bar  = st.progress(0)
                status_txt    = st.empty()
                preview_slot  = st.empty()
                stats_slot    = st.empty()

                all_counts = []
                frame_idx  = 0

                while frame_idx < process_n:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    result = run_detection(frame, model, confidence, iou_thresh)
                    out_frame, n_det, _ = draw_boxes(
                        frame, result, box_color, box_thick, show_label, show_conf
                    )
                    all_counts.append(n_det)
                    writer.write(out_frame)

                    # Update every 10 frames
                    if frame_idx % 10 == 0:
                        pct = frame_idx / process_n
                        progress_bar.progress(pct)
                        status_txt.markdown(
                            f'<span class="status-running">Processing frame {frame_idx}/{process_n}  '
                            f'— Avg bottles: {np.mean(all_counts):.1f}</span>',
                            unsafe_allow_html=True
                        )
                        preview_slot.image(bgr_to_rgb(out_frame), caption="Live Preview", use_container_width=True)

                    frame_idx += 1

                cap.release()
                writer.release()
                os.unlink(tfile.name)

                progress_bar.progress(1.0)
                status_txt.markdown(
                    '<span class="status-done">✓ Processing complete!</span>',
                    unsafe_allow_html=True
                )

                avg_bottles = np.mean(all_counts) if all_counts else 0
                max_bottles = max(all_counts) if all_counts else 0

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card mc-cyan">
                        <div class="metric-val">{process_n}</div>
                        <div class="metric-label">Frames Processed</div>
                    </div>
                    <div class="metric-card mc-green">
                        <div class="metric-val">{avg_bottles:.1f}</div>
                        <div class="metric-label">Avg Bottles/Frame</div>
                    </div>
                    <div class="metric-card mc-amber">
                        <div class="metric-val">{max_bottles}</div>
                        <div class="metric-label">Max Bottles in Frame</div>
                    </div>
                    <div class="metric-card mc-purple">
                        <div class="metric-val">{sum(all_counts)}</div>
                        <div class="metric-label">Total Detections</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Download processed video
                with open(out_path, "rb") as f:
                    st.download_button(
                        "💾 Download Processed Video",
                        data=f.read(),
                        file_name="bottle_detection_output.mp4",
                        mime="video/mp4"
                    )
                os.unlink(out_path)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
with tab_cam:
    st.markdown("""
    <div class="info-card" style="margin-bottom:12px">
        <strong>How it works:</strong> Take a photo with your webcam → the model detects bottles instantly.
        For continuous detection, click the camera button again.
    </div>
    """, unsafe_allow_html=True)

    cam_img = st.camera_input(
        "Point your camera at bottles and click the capture button",
        label_visibility="collapsed"
    )

    if cam_img is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📹</div>
            <div class="empty-title">Click the camera button above to capture</div>
            <div class="empty-sub">Point your camera at a bottle and capture — detection is instant</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if model is None:
            st.error("Model not loaded.")
        else:
            file_bytes = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Detecting..."):
                t0     = time.time()
                result = run_detection(img_bgr, model, confidence, iou_thresh)
                t1     = time.time()

            out_bgr, n_det, dets = draw_boxes(
                img_bgr, result, box_color, box_thick, show_label, show_conf
            )
            avg_conf = np.mean([d["confidence"] for d in dets]) if dets else 0.0

            metric_cards(n_det, avg_conf, 1/(t1-t0))

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="result-label">Captured Frame</div>', unsafe_allow_html=True)
                st.image(bgr_to_rgb(img_bgr), use_container_width=True)
            with c2:
                st.markdown('<div class="result-label live">Detection Result</div>', unsafe_allow_html=True)
                st.image(bgr_to_rgb(out_bgr), use_container_width=True)

            if dets:
                tags_html = "".join([
                    f'<span class="det-tag">#{i+1} {d["label"]} {d["confidence"]:.0%}</span>'
                    for i, d in enumerate(dets)
                ])
                st.markdown(f'<div class="det-tags">{tags_html}</div>', unsafe_allow_html=True)

                st.download_button(
                    "💾 Save Webcam Result",
                    data=img_to_download(bgr_to_rgb(out_bgr)),
                    file_name="webcam_bottle_detection.png",
                    mime="image/png"
                )
            else:
                st.warning("No bottles detected. Try adjusting the confidence threshold or point at a bottle.")
