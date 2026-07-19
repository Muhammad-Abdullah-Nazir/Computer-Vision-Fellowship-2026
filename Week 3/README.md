# Intelligent Video Analytics Platform — Week 3

Turns YOLO object detection into a full real-time video analytics system:
tracking, unique IDs, line-crossing counting, ROI occupancy, trajectories,
a live dashboard, and video recording.

## 1. Environment Setup (new Week 3 environment)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Verify install:
```bash
python -c "import ultralytics, supervision, cv2, streamlit; print('OK')"
```

## 1b. Using Your NVIDIA GPU

By default, `pip install -r requirements.txt` (via the `ultralytics` dependency)
often installs the **CPU-only** build of PyTorch, even on a machine with an
NVIDIA GPU — pip doesn't know to grab the CUDA build unless you tell it to.
This is the #1 reason "I have a GPU but it's still slow."

**Check what you currently have:**
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```
If this prints `CUDA available: False` and you do have an NVIDIA GPU, follow these steps.

**Step 1 — check your driver's supported CUDA version:**
```bash
nvidia-smi
```
Look at the top-right of the output for something like `CUDA Version: 12.4` —
that's the *maximum* CUDA version your driver supports (you can install an
equal or older CUDA build of PyTorch, not a newer one).

**Step 2 — reinstall PyTorch with the matching CUDA build:**
```bash
pip uninstall torch torchvision torchaudio -y
```
Then go to **https://pytorch.org/get-started/locally/**, select your OS
(Windows), Pip, Python, and the CUDA version from Step 1 — it generates the
exact install command for you. It'll look something like:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
(use whichever `cuXXX` matches your driver — don't guess, use the site's generator)

**Step 3 — verify it worked:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Should print `True` and your GPU's name.

**Step 4 — use it:**
- CLI: `python app.py --source 0 --device cuda:0`
- Streamlit: sidebar → **Compute Device** → **GPU (CUDA)**
- Or just leave it on `auto` / `Auto` — once torch can see CUDA, both
  `app.py` and `streamlit_app.py` will pick the GPU automatically. Either
  way, check your terminal output for `[INFO] Running inference on: ...` —
  it always logs exactly which device it actually used, so there's no
  guessing whether the GPU is really being used.

## 2. Project Structure

```
week3_video_analytics/
├── analytics_core.py    # SHARED engine: detection + tracking + counting + dashboard
│                          (used by both app.py and streamlit_app.py — one source of truth)
├── app.py               # CLI / OpenCV-window frontend
├── streamlit_app.py      # Streamlit web frontend
├── roi_selector.py       # standalone CLI tool to click-define line/zones (for app.py + --config)
├── experiments.py        # Assignment 2: ByteTrack vs BoT-SORT, confidence, resolution experiments
├── requirements.txt
├── configs/               # saved line/zone configs (JSON) per video
├── output/                 # recorded annotated videos
└── reports/                # session reports (JSON), experiment CSV, charts
```

**Why a shared `analytics_core.py`?** Both frontends need identical detection,
tracking, counting, and dashboard logic. Keeping it in one `AnalyticsCore`
class means there's a single pipeline to explain and defend in your technical
interview, and a fix/improvement in one place benefits both UIs.

## 3. Quick Start

**Step 1 — (optional) define your counting line and ROI zones on your actual video:**
```bash
python roi_selector.py --source your_video.mp4 --out configs/my_config.json
```
Click 2 points then press `n` for the line, click 3+ points then `n` for each zone,
press `s` to save, `q` to quit without saving. If you skip this step, `app.py` uses a
default horizontal line through the middle of the frame and no ROI zones.

**Step 2 — run the platform:**
```bash
# Webcam, live dashboard window, ByteTrack, pretrained YOLOv8n
python app.py --source 0

# Video file, BoT-SORT, custom confidence, your own config, save annotated output
python app.py --source your_video.mp4 --tracker botsort --conf 0.35 \
    --config configs/my_config.json --record

# Your own trained weights from Week 1/2
python app.py --source your_video.mp4 --model path/to/best.pt
```
Press `q` in the video window to stop. A JSON session report is written to `reports/`
automatically (total unique objects, IN/OUT counts, zone counts, avg FPS, avg confidence).

**Headless mode** (servers/CI without a display):
```bash
python app.py --source your_video.mp4 --record --headless
```

## 3c. Desktop GUI (recommended for live demos — fastest, no lag)

```bash
python desktop_app.py
```
A native Qt window — no browser, no network round-trip, renders frames the
same direct way `cv2.imshow` does but with real buttons/controls. Use this
one if Streamlit ever feels laggy, especially for webcam.

- **Video Source**: webcam or browse to a file
- **Model & Tracking**: model path (browse for your own `.pt`), tracker,
  confidence, inference size, compute device (Auto/CPU/GPU) — same controls
  as the other two frontends
- **Counting Line / ROI Zones**: click **"1. Load Reference Frame"** to grab
  a frame from your selected source, then **Draw Line** (click 2 points) or
  **Draw Zone** (click 3+ points, then **Finish Zone**) directly on the
  video preview. **Clear All** resets.
- **Start / Stop**: runs the same `AnalyticsCore` engine in a background
  thread so the UI never freezes; live frames stream straight into the
  window as fast as your GPU/CPU can produce them
- On Stop, the session report (JSON) appears in the panel and is also saved
  to `reports/`, same as the other two frontends

This is the same `AnalyticsCore` engine as `app.py` and `streamlit_app.py` —
all three frontends will always agree on detection/tracking/counting results.
Use whichever frontend fits the moment: **Desktop** for the smoothest live
demo, **Streamlit** for a shareable web UI (accept its inherent browser
overhead), **CLI** (`app.py`) for the simplest/fastest option and for
scripting/automation.

## 3b. Streamlit Frontend

```bash
streamlit run streamlit_app.py
```
Opens in your browser at `http://localhost:8501`. Features:
- **Upload Video mode:** upload a file, draw the counting line and ROI zones
  directly on the first frame (canvas: `line` tool for the counting line,
  `polygon` tool for zones), then click **Start Processing**. Live video,
  metrics, and charts update as it processes; download the annotated video
  and JSON report when done.
- **Webcam mode:** runs against the machine Streamlit is running on
  (`device 0`). Click **Start Webcam** / **Stop Webcam**. This only works
  when you run Streamlit locally with a webcam attached — it will not see a
  remote user's camera if you deploy the app to a server.
- Sidebar lets you switch model (pretrained vs your own uploaded `.pt`),
  tracker (ByteTrack/BoT-SORT), confidence threshold, and whether to record
  the output video.

**Note on webcam in the browser:** true browser-camera streaming (capturing
video from a *visitor's* device, not the host machine) needs a different
approach — `streamlit-webrtc` — since it must proxy frames from the browser
over WebRTC. The current webcam mode is the simpler "local demo on your own
machine" version, which fully satisfies the Week 3 live-demo requirement.

## 4. Tracking Experiments (Assignment 2)

```bash
python experiments.py --source your_video.mp4 --model yolov8n.pt --max-frames 300
```
This runs ByteTrack vs BoT-SORT, five confidence thresholds, and three resolutions,
then writes `reports/experiments.csv` plus two comparison charts you can drop straight
into your Assignment 2 write-up.

## 5. How It Works (for Assignment 4 / architecture diagram)

```
Video Source (webcam / file / RTSP)
        │
        ▼
Frame Capture (OpenCV VideoCapture)
        │
        ▼
Object Detection (YOLO forward pass → boxes, classes, confidences)
        │
        ▼
Object Tracking (ByteTrack or BoT-SORT via model.track(),
                  assigns/persists a unique ID per object)
        │
        ▼
Event Processing (LineZone.trigger() → IN/OUT counts,
                   PolygonZone.trigger() → zone occupancy,
                   trajectory buffer update per track ID)
        │
        ▼
Analytics Engine (running FPS, avg confidence, unique object
                   count, per-zone counts, per-frame stats)
        │
        ▼
Dashboard (OpenCV overlay: current/total objects, IN/OUT,
           zone counts, FPS, processing time — rendered live)
        │
        ▼
Reports (annotated .mp4 in output/, JSON session summary
         in reports/, experiment CSV/charts from experiments.py)
```

Key design notes worth mentioning in your architecture write-up:
- Detection and tracking are decoupled conceptually but combined in one call
  (`model.track()`) because ultralytics runs YOLO inference then feeds the
  raw detections into the ByteTrack/BoT-SORT association step each frame.
- Counting (line + zone) is **stateful** — it depends on track IDs persisting
  across frames, which is exactly why tracking (not detection alone) is required
  for accurate counting.
- The dashboard and recording are pure post-processing steps on top of the
  annotated frame — they don't affect detection/tracking accuracy.

## 6. Talking Points for the Technical Interview

- **Detection vs tracking:** detection finds objects in a single frame with no
  memory; tracking links those detections across frames into consistent
  identities over time, which is what makes counting and trajectories possible.
- **Why detection alone can't count accurately:** without persistent IDs, the
  same physical object detected in frame 1 and frame 2 looks like two separate
  objects, so a simple per-frame count will overcount (or double count
  objects that flicker in/out of detection).
- **How ByteTrack assigns IDs:** it associates detections to existing tracks
  using motion (Kalman filter prediction) and IoU overlap, in two passes —
  high-confidence detections first, then a second, more lenient pass on
  low-confidence detections to recover partially-occluded objects, rather
  than discarding them like older trackers did.
- **What causes ID switches:** occlusion, objects crossing paths, fast motion
  with low frame rate, missed detections for several consecutive frames, and
  visually similar objects near each other confusing the association step.
- **Purpose of ROI:** restrict analytics to a specific area of interest (e.g.
  a checkout counter, an entrance, a restricted zone) instead of the whole frame,
  which reduces noise and answers a more specific business question.
- **Mall people-counter design sketch:** camera(s) at entrances → detection +
  tracking → a counting line at each doorway → IN/OUT tallies aggregated into
  a running occupancy number → dashboard/alerting if occupancy nears capacity.
- **Optimizing FPS on low-end hardware:** use a smaller model (e.g. YOLOv8n),
  lower input resolution, reduce inference frequency (track every frame but
  detect every N frames), use a lighter tracker, and consider quantization or
  exporting to a faster runtime (ONNX/TensorRT) if available.
- **Occlusion challenges:** the object's detection may disappear for several
  frames, breaking the association; trackers mitigate this with motion
  prediction (Kalman filter) to "coast" the track through short occlusions,
  but long occlusions or objects that change appearance/direction while
  hidden still commonly cause ID switches.

## 7. Bonus Feature Ideas (pick a couple, not all)

- **Heatmap:** accumulate a 2D histogram of trajectory points over the whole
  session, normalize, and overlay with `cv2.applyColorMap`.
- **Dwell time:** track how many frames/seconds each ID stays inside a zone.
- **Speed estimation:** convert pixel displacement between frames to real
  units using a known reference distance in the scene, divide by frame time.
- **Loitering detection:** flag any track ID whose centroid stays within a
  small radius for longer than a threshold duration.

## 8. Deliverables Checklist Mapping

| Requirement | Where it's covered |
|---|---|
| Real-time detection | `app.py` (`model.track()`) |
| Object tracking + unique IDs | `app.py` (ByteTrack/BoT-SORT via `--tracker`) |
| Object counter | `app.py` dashboard (current/total/in/out) |
| Virtual counting line | `roi_selector.py` + `sv.LineZone` in `app.py` |
| ROI zones | `roi_selector.py` + `sv.PolygonZone` in `app.py` |
| Movement analytics (trajectories + direction) | `_draw_trajectories()` + `_draw_direction_arrows()` in `analytics_core.py` — colored path per ID, plus an arrow and compass label (N/NE/E/etc.) on each object once it has moved enough to trust a heading |
| Analytics dashboard | `_draw_dashboard()` in `analytics_core.py` — 4 lines: counts, IN/OUT + zone counts, FPS/confidence/frame time, and longest dwell per zone |
| Video recording | `--record` flag in `app.py` / checkbox in `streamlit_app.py` |
| Tracking experiments | `experiments.py` → `reports/experiments.csv` + charts |
| Web frontend | `streamlit_app.py` (upload + webcam modes, live dashboard, downloads) |
| **Bonus: Dwell Time** | `_update_zone_dwell()` in `analytics_core.py` — per-track, per-zone timer that starts when an object enters a zone and resets when it leaves; shown live on each object's label (`Z1:3.2s`), on the dashboard ("Longest Dwell"), and in the final session report (`zone_longest_dwell_seconds`) |
