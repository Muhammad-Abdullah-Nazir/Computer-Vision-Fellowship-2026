# Installation Guide

Intelligent Video Analytics Platform — Week 3, AI Summer Fellowship 2026

This guide covers setup on Windows (primary target), with notes for macOS/Linux where it differs. Three frontends are included — CLI, Streamlit (web), and a native Qt desktop app — all sharing one detection/tracking engine (`analytics_core.py`).

---

## 1. Requirements

- Python 3.10–3.12
- A webcam (for live demo) and/or a test video file
- Optional but recommended: an NVIDIA GPU with an up-to-date driver, for real-time performance
- ~2GB free disk space (model weights + dependencies, PyTorch is the largest)

---

## 2. Get the code

```bash
git clone <your-repo-url>
cd <repo-folder>
```
(or extract the project zip into a folder and open a terminal there)

---

## 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```
Your terminal prompt should now start with `(venv)`. Repeat this activation step every time you open a new terminal for this project.

**Windows-only issue:** if activation fails with a "running scripts is disabled" error, run this once in PowerShell (not as admin):
```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Then try activating again.

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```
This installs Ultralytics YOLO, OpenCV, Supervision, Streamlit, PySide6, and their dependencies. It also installs PyTorch — by default this may be a **CPU-only build even on a machine with a GPU** (see Section 5 if you have an NVIDIA GPU).

Verify the install:
```bash
python -c "import ultralytics, supervision, cv2, streamlit; print('OK')"
```

---

## 5. GPU setup (optional, strongly recommended if you have an NVIDIA GPU)

Check what you currently have:
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```
If this prints `CUDA available: False` and you do have an NVIDIA GPU, your PyTorch install is CPU-only — a very common default, not a sign anything is broken.

**Fix it:**
1. Check your driver's max supported CUDA version:
   ```bash
   nvidia-smi
   ```
   Look for `CUDA Version: X.X` in the top-right of the output.

2. Reinstall PyTorch with a matching CUDA build:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   ```
   Go to **https://pytorch.org/get-started/locally/**, select your OS, Pip, Python, and the CUDA version from step 1, and run the exact command it generates. Example:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

3. Verify:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```
   Should print `True` and your GPU's name.

Every frontend logs which device it actually used at startup (`[INFO] Running inference on: ...`), so there's no ambiguity once you're running the app.

---

## 6. Running the platform

### CLI (fastest, simplest — good default)
```bash
python app.py --source 0
```
`--source 0` is your default webcam; use a file path for a video file. Press `q` to quit.

Useful flags:
```bash
python app.py --source 0 --model yolov8n.pt --tracker bytetrack --conf 0.3 --imgsz 640 --device auto --fp16 --record
```
| Flag | Meaning |
|---|---|
| `--model` | `.pt` weights path — pretrained or your own trained model |
| `--tracker` | `bytetrack` or `botsort` |
| `--imgsz` | 256/320/480/640 — inference resolution, lower = faster |
| `--device` | `auto`, `cpu`, or `cuda:0` |
| `--fp16` | half-precision inference (GPU only, no effect on CPU) |
| `--record` | save annotated video to `output/` |
| `--config` | path to a JSON file with a counting line / ROI zones (see below) |

### Desktop App (recommended for live demos — smoothest, no lag)
```bash
python desktop_app.py
```
Click **1. Load Reference Frame**, then **Draw Line** / **Draw Zone** directly on the video preview, then **Start**.

### Streamlit (web frontend)
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`. Supports both video upload and webcam modes, with the same controls (model, tracker, confidence, inference size, device, fp16) in the sidebar.

---

## 7. Defining a counting line / ROI zones from the command line

For the CLI, draw a line/zones once using the interactive selector, then reuse the saved config:
```bash
python roi_selector.py --source your_video.mp4 --out configs/my_config.json
```
Click 2 points + `n` for the line, 3+ points + `n` per zone, `s` to save. Then:
```bash
python app.py --source your_video.mp4 --config configs/my_config.json
```
(The Desktop and Streamlit frontends have their own built-in click-to-draw tools — no separate step needed there.)

---

## 8. Running the experiment / analysis scripts

```bash
# Assignment 2 — tracker/confidence/resolution experiments
python experiments.py --source test_video.mp4 --model yolov8n.pt --device cuda:0 --compare-device

# Assignment 5 — FPS / CPU / memory / inference-time / tracking-accuracy analysis
python performance_analysis.py --source test_video.mp4 --model yolov8n.pt --device cuda:0
```
Both write results to `reports/` (JSON + PNG charts).

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Webcam won't open / grabs no frames | Handled automatically (DirectShow backend + warmup reads) — if it still fails, another app may be holding the camera; close it and retry |
| `torch.cuda.is_available()` is `False` on a GPU machine | See Section 5 — reinstall the CUDA build of PyTorch |
| Recording fails / codec errors | Handled automatically — the writer falls back across MJPG → MP4 → XVID → DIVX and reports which one it used in the console |
| Streamlit webcam feels laggy | Use `desktop_app.py` instead for live demos — it has no browser round-trip |
| Everything is slow | Confirm GPU is actually in use (console log), lower `--imgsz`, or switch to `yolov8n.pt` if using a larger model |

---

## 10. Project structure

```
├── analytics_core.py       # shared engine: detection, tracking, counting, dashboard
├── app.py                  # CLI frontend
├── streamlit_app.py        # web frontend
├── desktop_app.py          # native Qt desktop frontend
├── roi_selector.py         # CLI tool to draw line/zones and save as JSON
├── experiments.py          # Assignment 2 — tracking experiments
├── performance_analysis.py # Assignment 5 — performance analysis
├── requirements.txt
├── configs/                # saved line/zone JSON configs
├── output/                 # recorded annotated videos
└── reports/                # session reports, experiment/performance JSON + charts
```
