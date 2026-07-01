# ─────────────────────────────────────────────────────────────────────────────
#  Vision Toolkit — AI Summer Internship 2026
#  Assignment 3 | Track 1: Computer Vision
#  Built with Python + Streamlit + OpenCV
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── PAGE CONFIGURATION ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Toolkit 2026",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark theme overrides */
.stApp { background-color: #0A0E1A; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label {
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #6B7280 !important;
    font-family: 'JetBrains Mono', monospace;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 20px !important;
    color: #06B6D4 !important;
    font-weight: 700;
}

/* Sidebar */
[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1F2937; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stTextInput label {
    color: #9CA3AF !important;
    font-size: 12px !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .05em;
}

/* Section headers */
.section-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: #06B6D4;
    padding: 8px 0 6px;
    border-bottom: 1px solid #1F2937;
    margin-bottom: 10px;
}

/* Title */
.app-title {
    font-size: 32px;
    font-weight: 800;
    color: #F9FAFB;
    margin-bottom: 2px;
}
.app-title span { color: #06B6D4; }
.app-subtitle {
    font-size: 13px;
    color: #6B7280;
    margin-bottom: 20px;
    font-family: monospace;
}

/* Image panel labels */
.img-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #9CA3AF;
    margin-bottom: 8px;
}
.img-label-live { color: #10B981; }

/* Upload prompt */
.upload-prompt {
    background: #111827;
    border: 2px dashed #1F2937;
    border-radius: 16px;
    padding: 60px;
    text-align: center;
    margin-top: 40px;
}
.upload-prompt h2 { color: #4B5563; font-size: 22px; margin-bottom: 8px; }
.upload-prompt p  { color: #374151; font-size: 14px; }

/* Divider */
hr { border-color: #1F2937 !important; margin: 18px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def hex_to_bgr(hex_color):
    """Convert hex color (#RRGGBB) to OpenCV BGR tuple."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)   # OpenCV uses BGR, not RGB!

def to_display(img):
    """Convert OpenCV image to RGB for Streamlit display."""
    if img is None:
        return None
    if len(img.shape) == 2:           # grayscale
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def format_bytes(size_bytes):
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024*1024):.2f} MB"

def ensure_odd(n):
    """Ensure kernel size is odd (required by OpenCV)."""
    return n if n % 2 == 1 else n + 1

def image_to_download(img):
    """Convert OpenCV image to PNG bytes for download."""
    if len(img.shape) == 2:
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# ─── SESSION STATE INIT ───────────────────────────────────────────────────────
if "original" not in st.session_state:
    st.session_state.original     = None
if "processed" not in st.session_state:
    st.session_state.processed    = None
if "file_name" not in st.session_state:
    st.session_state.file_name    = ""
if "file_size" not in st.session_state:
    st.session_state.file_size    = 0

# ─── APP HEADER ───────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🔬 <span>Vision</span>Toolkit</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">AI Summer Internship 2026 · Track 1: Computer Vision · Week 1</div>',
            unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown('<div class="section-header">📁 Image Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        label_visibility="collapsed"
    )

    # ── Load image when uploaded ──────────────────────────────────────────────
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            st.session_state.original  = img
            st.session_state.processed = img.copy()
            st.session_state.file_name = uploaded_file.name
            st.session_state.file_size = len(file_bytes)
            st.success(f"✓ Loaded: {uploaded_file.name}")

    # ── Only show controls when image is loaded ───────────────────────────────
    if st.session_state.original is not None:
        orig = st.session_state.original
        H, W = orig.shape[:2]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">⚙️ Operation</div>', unsafe_allow_html=True)

        operation = st.selectbox(
            "Choose operation",
            [
                "─── View ───",
                "Original",
                "─── Color ───",
                "Grayscale",
                "Brightness & Contrast",
                "─── Blur ───",
                "Gaussian Blur",
                "Median Blur",
                "─── Threshold ───",
                "Binary Threshold",
                "Adaptive Threshold",
                "─── Edge Detection ───",
                "Canny Edge Detection",
                "─── Transform ───",
                "Rotation",
                "Resize",
                "─── Drawing ───",
                "Draw Rectangle",
                "Draw Circle",
                "Draw Line",
                "Add Text",
            ],
            label_visibility="collapsed"
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Result image starts as a copy of original ─────────────────────────
        result = orig.copy()
        op_label = operation

        # ── ORIGINAL ─────────────────────────────────────────────────────────
        if operation == "Original":
            result = orig.copy()

        # ── GRAYSCALE ─────────────────────────────────────────────────────────
        elif operation == "Grayscale":
            st.markdown('<div class="section-header">🎨 Grayscale</div>', unsafe_allow_html=True)
            st.info("Converts the image to grayscale by removing all color information, keeping only brightness (luminance).")
            result = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

        # ── BRIGHTNESS & CONTRAST ─────────────────────────────────────────────
        elif operation == "Brightness & Contrast":
            st.markdown('<div class="section-header">☀️ Brightness & Contrast</div>', unsafe_allow_html=True)
            brightness = st.slider("Brightness", -150, 150, 0,
                help="Positive = brighter, Negative = darker")
            contrast   = st.slider("Contrast",   -100, 100, 0,
                help="Positive = more contrast, Negative = less contrast")
            alpha = 1.0 + contrast / 100.0
            result = cv2.convertScaleAbs(orig, alpha=alpha, beta=brightness)

        # ── GAUSSIAN BLUR ─────────────────────────────────────────────────────
        elif operation == "Gaussian Blur":
            st.markdown('<div class="section-header">🌫️ Gaussian Blur</div>', unsafe_allow_html=True)
            st.info("Uses a Gaussian kernel to smooth the image. Larger kernel = stronger blur.")
            ksize = st.slider("Kernel Size", 1, 51, 5, 2,
                help="Must be an odd number. Controls how many pixels are averaged together.")
            ksize = ensure_odd(ksize)
            sigma = st.slider("Sigma (strength)", 0.0, 10.0, 0.0, 0.5,
                help="Standard deviation. 0 = auto-calculated from kernel size.")
            result = cv2.GaussianBlur(orig, (ksize, ksize), sigma)

        # ── MEDIAN BLUR ───────────────────────────────────────────────────────
        elif operation == "Median Blur":
            st.markdown('<div class="section-header">🌫️ Median Blur</div>', unsafe_allow_html=True)
            st.info("Replaces each pixel with the median of its neighbors. Great for removing salt-and-pepper noise.")
            ksize = st.slider("Kernel Size", 1, 31, 5, 2,
                help="Size of neighborhood. Larger = stronger blur.")
            ksize = ensure_odd(ksize)
            result = cv2.medianBlur(orig, ksize)

        # ── BINARY THRESHOLD ─────────────────────────────────────────────────
        elif operation == "Binary Threshold":
            st.markdown('<div class="section-header">⬛ Binary Threshold</div>', unsafe_allow_html=True)
            st.info("Pixels above the threshold become white (255), pixels below become black (0).")
            thresh_val = st.slider("Threshold Value", 0, 255, 127,
                help="Pixels brighter than this → white. Darker → black.")
            mode = st.radio("Mode", ["Binary (bright→white)", "Inverse (bright→black)"],
                help="Binary: standard. Inverse: flips black and white.")
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            thresh_type = cv2.THRESH_BINARY if "Binary (" in mode else cv2.THRESH_BINARY_INV
            _, result = cv2.threshold(gray, thresh_val, 255, thresh_type)

        # ── ADAPTIVE THRESHOLD ────────────────────────────────────────────────
        elif operation == "Adaptive Threshold":
            st.markdown('<div class="section-header">⬛ Adaptive Threshold</div>', unsafe_allow_html=True)
            st.info("Calculates different threshold values for different regions of the image. Works better than binary threshold when lighting is uneven.")
            method = st.radio("Method",
                ["Mean C", "Gaussian C"],
                help="Mean: uses average of neighborhood. Gaussian: uses weighted average.")
            block_size = st.slider("Block Size", 3, 51, 11, 2,
                help="Size of neighborhood used to calculate threshold. Must be odd.")
            C = st.slider("C (subtracted from mean)", -20, 20, 2,
                help="A constant subtracted from the mean. Fine-tune the result here.")
            block_size = ensure_odd(block_size)
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            adaptive_method = (cv2.ADAPTIVE_THRESH_MEAN_C
                               if method == "Mean C"
                               else cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            result = cv2.adaptiveThreshold(
                gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, C)

        # ── CANNY EDGE DETECTION ─────────────────────────────────────────────
        elif operation == "Canny Edge Detection":
            st.markdown('<div class="section-header">📐 Canny Edge Detection</div>', unsafe_allow_html=True)
            st.info("Detects edges by finding where pixel intensity changes rapidly. Two thresholds control sensitivity.")
            low  = st.slider("Low Threshold",  0, 255, 50,
                help="Below this value: definitely NOT an edge.")
            high = st.slider("High Threshold", 0, 255, 150,
                help="Above this value: definitely IS an edge. Between low and high: edge only if connected to a strong edge.")
            blur_before = st.checkbox("Apply Gaussian Blur first (recommended)", value=True,
                help="Blurring removes noise before edge detection, reducing false edges.")
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            if blur_before:
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
            result = cv2.Canny(gray, low, high)

        # ── ROTATION ──────────────────────────────────────────────────────────
        elif operation == "Rotation":
            st.markdown('<div class="section-header">🔄 Rotation</div>', unsafe_allow_html=True)
            angle = st.slider("Angle (degrees)", -180, 180, 0,
                help="Positive = counterclockwise, Negative = clockwise.")
            scale = st.slider("Scale", 0.1, 3.0, 1.0, 0.05,
                help="1.0 = original size. 0.5 = half size. 2.0 = double size.")
            center = (W // 2, H // 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            result = cv2.warpAffine(orig, M, (W, H),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))

        # ── RESIZE ────────────────────────────────────────────────────────────
        elif operation == "Resize":
            st.markdown('<div class="section-header">↔️ Resize</div>', unsafe_allow_html=True)
            keep_ratio = st.checkbox("Keep aspect ratio", value=True)
            new_w = st.number_input("New Width (px)",  min_value=1, max_value=8000, value=W)
            if keep_ratio:
                ratio = new_w / W
                new_h = int(H * ratio)
                st.info(f"Height auto-set to: {new_h} px")
            else:
                new_h = st.number_input("New Height (px)", min_value=1, max_value=8000, value=H)
            result = cv2.resize(orig, (int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)

        # ── DRAW RECTANGLE ────────────────────────────────────────────────────
        elif operation == "Draw Rectangle":
            st.markdown('<div class="section-header">⬜ Draw Rectangle</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.number_input("Top-left X", 0, W-1, min(50, W-2))
                y1 = st.number_input("Top-left Y", 0, H-1, min(50, H-2))
            with col2:
                x2 = st.number_input("Bottom-right X", 0, W-1, min(200, W-1))
                y2 = st.number_input("Bottom-right Y", 0, H-1, min(200, H-1))
            color    = st.color_picker("Color", "#00FF00")
            thickness = st.slider("Thickness (px)", 1, 30, 2)
            filled   = st.checkbox("Fill rectangle", False)
            t = -1 if filled else thickness
            result = orig.copy()
            cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)),
                          hex_to_bgr(color), t)

        # ── DRAW CIRCLE ───────────────────────────────────────────────────────
        elif operation == "Draw Circle":
            st.markdown('<div class="section-header">⭕ Draw Circle</div>', unsafe_allow_html=True)
            cx      = st.number_input("Center X", 0, W-1, W//2)
            cy      = st.number_input("Center Y", 0, H-1, H//2)
            radius  = st.number_input("Radius (px)", 1, max(W,H), min(80, min(W,H)//4))
            color   = st.color_picker("Color", "#FF0000")
            thickness = st.slider("Thickness (px)", 1, 30, 2)
            filled  = st.checkbox("Fill circle", False)
            t = -1 if filled else thickness
            result = orig.copy()
            cv2.circle(result, (int(cx), int(cy)), int(radius), hex_to_bgr(color), t)

        # ── DRAW LINE ─────────────────────────────────────────────────────────
        elif operation == "Draw Line":
            st.markdown('<div class="section-header">📏 Draw Line</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.number_input("Start X", 0, W-1, 0)
                y1 = st.number_input("Start Y", 0, H-1, H//2)
            with col2:
                x2 = st.number_input("End X", 0, W-1, W-1)
                y2 = st.number_input("End Y", 0, H-1, H//2)
            color     = st.color_picker("Color", "#0080FF")
            thickness  = st.slider("Thickness (px)", 1, 30, 2)
            result = orig.copy()
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)),
                     hex_to_bgr(color), thickness, cv2.LINE_AA)

        # ── ADD TEXT ──────────────────────────────────────────────────────────
        elif operation == "Add Text":
            st.markdown('<div class="section-header">📝 Add Text</div>', unsafe_allow_html=True)
            text       = st.text_input("Text to add", "Vision Toolkit 2026")
            col1, col2 = st.columns(2)
            with col1:
                tx = st.number_input("X Position", 0, W-1, min(30, W-1))
                ty = st.number_input("Y Position", 0, H-1, min(60, H-1))
            with col2:
                font_scale = st.slider("Font Size", 0.3, 5.0, 1.0, 0.1)
                thickness  = st.slider("Thickness", 1, 10, 2)
            color      = st.color_picker("Color", "#FFFFFF")
            font_names = {
                "Normal":       cv2.FONT_HERSHEY_SIMPLEX,
                "Small":        cv2.FONT_HERSHEY_PLAIN,
                "Complex":      cv2.FONT_HERSHEY_COMPLEX,
                "Duplex":       cv2.FONT_HERSHEY_DUPLEX,
                "Italic":       cv2.FONT_HERSHEY_COMPLEX_SMALL,
            }
            font_choice = st.selectbox("Font Style", list(font_names.keys()))
            result = orig.copy()
            # Add text shadow for readability
            shadow_col = (0, 0, 0)
            cv2.putText(result, text, (int(tx)+2, int(ty)+2),
                        font_names[font_choice], font_scale, shadow_col, thickness+1, cv2.LINE_AA)
            cv2.putText(result, text, (int(tx), int(ty)),
                        font_names[font_choice], font_scale, hex_to_bgr(color), thickness, cv2.LINE_AA)

        # ── UPDATE PROCESSED IMAGE ────────────────────────────────────────────
        if operation not in ["─── View ───", "─── Color ───", "─── Blur ───",
                              "─── Threshold ───", "─── Edge Detection ───",
                              "─── Transform ───", "─── Drawing ───"]:
            st.session_state.processed = result

        # ── RESET BUTTON ──────────────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("↺ Reset to Original", use_container_width=True):
            st.session_state.processed = st.session_state.original.copy()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT AREA
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.original is None:
    # ── Upload prompt when no image loaded ────────────────────────────────────
    st.markdown("""
    <div class="upload-prompt">
        <h2>📷 Upload an Image to Get Started</h2>
        <p>Use the sidebar on the left to upload any image (JPG, PNG, BMP, TIFF, WEBP)</p>
        <br>
        <p style="color:#4B5563; font-size:13px;">
            Supported operations: Grayscale · Blur · Edge Detection ·
            Thresholding · Drawing · Transform · Histogram
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    orig      = st.session_state.original
    processed = st.session_state.processed
    H, W      = orig.shape[:2]
    channels  = 1 if len(orig.shape) == 2 else orig.shape[2]

    # ── IMAGE INFORMATION BAR ─────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Width",     f"{W} px")
    m2.metric("Height",    f"{H} px")
    m3.metric("Channels",  f"{channels} ({'RGB' if channels == 3 else 'Gray' if channels == 1 else 'RGBA'})")
    m4.metric("File Size", format_bytes(st.session_state.file_size))
    m5.metric("Resolution", f"{W}×{H}")
    m6.metric("File", st.session_state.file_name[:14] + ("…" if len(st.session_state.file_name) > 14 else ""))

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── BEFORE / AFTER IMAGE DISPLAY ──────────────────────────────────────────
    col_orig, col_proc = st.columns(2)

    with col_orig:
        st.markdown('<div class="img-label">📷 Original</div>', unsafe_allow_html=True)
        st.image(to_display(orig), use_container_width=True, clamp=True)

    with col_proc:
        st.markdown('<div class="img-label img-label-live">✨ Processed</div>', unsafe_allow_html=True)
        st.image(to_display(processed), use_container_width=True, clamp=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── HISTOGRAM ─────────────────────────────────────────────────────────────
    with st.expander("📊 View Histogram", expanded=False):
        col_h1, col_h2 = st.columns(2)

        def plot_histogram(img_bgr, title):
            """Plot BGR and grayscale histograms."""
            fig, axes = plt.subplots(1, 2, figsize=(10, 3),
                                     facecolor='#111827')
            fig.suptitle(title, color='#9CA3AF', fontsize=12)

            # Grayscale histogram
            if len(img_bgr.shape) == 2:
                gray = img_bgr
            else:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            ax0 = axes[0]
            ax0.set_facecolor('#0A0E1A')
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            ax0.fill_between(range(256), hist.flatten(), color='#06B6D4', alpha=0.7)
            ax0.plot(hist.flatten(), color='#06B6D4', linewidth=0.8)
            ax0.set_title("Grayscale", color='#6B7280', fontsize=10)
            ax0.tick_params(colors='#4B5563', labelsize=7)
            ax0.set_xlim([0, 256])
            for spine in ax0.spines.values():
                spine.set_color('#1F2937')

            # Color histogram
            ax1 = axes[1]
            ax1.set_facecolor('#0A0E1A')
            if len(img_bgr.shape) == 3:
                for i, (col, label) in enumerate([(0,'Blue'),(1,'Green'),(2,'Red')]):
                    h = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
                    colors_map = {0:'#3B82F6', 1:'#10B981', 2:'#EF4444'}
                    ax1.plot(h.flatten(), color=colors_map[i], label=label, linewidth=1, alpha=0.9)
                ax1.legend(fontsize=8, labelcolor='#9CA3AF',
                           facecolor='#1F2937', edgecolor='#374151')
            else:
                ax1.fill_between(range(256), hist.flatten(), color='#8B5CF6', alpha=0.7)
            ax1.set_title("Color Channels", color='#6B7280', fontsize=10)
            ax1.tick_params(colors='#4B5563', labelsize=7)
            ax1.set_xlim([0, 256])
            for spine in ax1.spines.values():
                spine.set_color('#1F2937')

            plt.tight_layout()
            return fig

        with col_h1:
            st.markdown('<div class="img-label">Original Histogram</div>', unsafe_allow_html=True)
            fig1 = plot_histogram(orig, "Original")
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        with col_h2:
            st.markdown('<div class="img-label img-label-live">Processed Histogram</div>',
                        unsafe_allow_html=True)
            p = processed if len(processed.shape) == 3 else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            fig2 = plot_histogram(p, "Processed")
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    # ── DOWNLOAD BUTTON ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        download_bytes = image_to_download(processed)
        st.download_button(
            label="💾  Download Processed Image  (PNG)",
            data=download_bytes,
            file_name="vision_toolkit_output.png",
            mime="image/png",
            use_container_width=True
        )

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#374151;font-size:11px;font-family:monospace;">'
        'Vision Toolkit · AI Summer Internship 2026 · Built with Python + OpenCV + Streamlit'
        '</p>', unsafe_allow_html=True
    )
