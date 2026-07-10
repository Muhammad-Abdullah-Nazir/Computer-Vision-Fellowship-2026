# 🍾 Bottle Detection System
### AI Summer Fellowship 2026 — Week 2 | Track 1: Computer Vision

![Model](https://img.shields.io/badge/Model-YOLOv8n-blue)
![mAP50](https://img.shields.io/badge/mAP50-87.1%25-green)
![Precision](https://img.shields.io/badge/Precision-93.5%25-green)
![Python](https://img.shields.io/badge/Python-3.10.11-blue)
![Framework](https://img.shields.io/badge/Framework-Ultralytics-orange)
![GPU](https://img.shields.io/badge/GPU-CUDA%3A0-red)

---

## 📋 Project Overview

A custom **real-time bottle detection system** built using **YOLOv8 Nano** trained on a dataset of 2,037 images. The system detects bottles in images, videos, and live webcam feeds with **87.1% mAP50** and **93.5% Precision**.

Deployed as a **Streamlit web application** — upload an image or video and see results instantly in your browser.

---

## 🎯 Model Performance

| Metric | Score | Meaning |
|---|---|---|
| **mAP50** | **87.1%** | Overall detection accuracy |
| **Precision** | **93.5%** | When it says bottle → 93.5% correct |
| **Recall** | **79.2%** | Finds 79.2% of all bottles |
| **mAP50-95** | **62.0%** | Strict accuracy all IoU levels |
| **Inference Speed** | **9.3ms** | Real-time capable |
| **Model Size** | **6.2 MB** | Lightweight deployment |
| **Training Time** | **59 min** | 50 epochs on GPU |

---

## 🏗️ System Architecture

```
User (Browser)
      │
      ▼
┌─────────────────────────────────────────────┐
│  Stage 01 — Image Input                     │
│  File upload / Video / Webcam capture       │
└───────────────────┬─────────────────────────┘
                    │  NumPy array (H×W×3) BGR
                    ▼
┌─────────────────────────────────────────────┐
│  Stage 02 — Preprocessing                   │
│  Resize 640×640 · Normalize 0–1 · Tensor   │
└───────────────────┬─────────────────────────┘
                    │  float32 tensor (1,3,640,640)
                    ▼
┌─────────────────────────────────────────────┐
│  Stage 03 — YOLOv8n Model Inference         │
│  Backbone → Neck → Detection Head           │
│  3.0M params · 8.1 GFLOPs · 9.3ms/img      │
└───────────────────┬─────────────────────────┘
                    │  Raw candidate boxes
                    ▼
┌─────────────────────────────────────────────┐
│  Stage 04 — Bounding Box Decoding           │
│  Decode coordinates · NMS (IoU=0.45)        │
└───────────────────┬─────────────────────────┘
                    │  [x1,y1,x2,y2,conf,class]
                    ▼
┌─────────────────────────────────────────────┐
│  Stage 05 — Confidence Filtering            │
│  Keep boxes with confidence ≥ 0.40          │
└───────────────────┬─────────────────────────┘
                    │  Final detections
                    ▼
┌─────────────────────────────────────────────┐
│  Stage 06 — Visualization                   │
│  Draw boxes · Labels · Side-by-side view    │
└───────────────────┬─────────────────────────┘
                    │  Annotated RGB image
                    ▼
┌─────────────────────────────────────────────┐
│  Stage 07 — Output                          │
│  Display in browser · Download PNG          │
└─────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Bottle-Detection/
│
├── 📄 README.md                          ← This file
├── 📄 requirements.txt                   ← Python dependencies
├── 📄 .gitignore                         ← Files excluded from git
│
├── 🐍 app.py                             ← Streamlit Detection App (A5)
├── 🐍 train.py                           ← Model Training Script (A3)
│
├── 🧪 experiments/
│   ├── exp1_fixed.py                     ← Exp 1: Epoch comparison
│   ├── exp2_imgsz.py                     ← Exp 2: Image size comparison
│   ├── exp3_confidence.py                ← Exp 3: Confidence threshold
│   ├── exp4_models.py                    ← Exp 4: Model variants
│   └── experiment_results.json           ← Real experiment results
│
├── 📊 docs/
│   ├── dataset_description.md            ← Dataset details (A1)
│   ├── training_config.md                ← Training configuration (A3)
│   ├── Week2_Bottle_Detection_Report.docx ← A1-A4 Combined Report
│   ├── Assignment6_Research_Report.docx  ← Research Report (A6)
│   ├── Assignment7_Experiments_Report.docx ← Experiments (A7)
│   ├── Assignment8_Architecture.docx     ← Architecture Doc (A8)
│   └── Assignment9_Builder_Journal.docx  ← Builder Journal (A9)
│
├── 📸 screenshots/
│   ├── app_home.png                      ← App home screen
│   ├── image_detection.png               ← Image detection result
│   ├── video_detection.png               ← Video detection result
│   ├── webcam_detection.png              ← Webcam detection result
│   ├── training_results.png              ← Training curves
│   └── confusion_matrix.png              ← Confusion matrix
│
└── 🔗 demo_video.md                      ← Demo video link
```

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA (recommended) or CPU
- 4GB+ RAM

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Bottle-Detection.git
cd Bottle-Detection
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Verify GPU (optional but recommended)
```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
# Should print: GPU: True
```

### Step 5 — Run the Detection App
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📦 Dataset

| Property | Detail |
|---|---|
| **Source** | Roboflow Universe (Public Dataset) |
| **URL** | https://universe.roboflow.com/breathless/bottle-ewqkq/dataset/8 |
| **Version** | 8 |
| **Total Images** | 2,037 (including augmentations) |
| **Training Set** | 1,782 images (87%) |
| **Validation Set** | 169 images (8%) |
| **Test Set** | 86 images (4%) |
| **Classes** | 1 — `bottle` |
| **Image Size** | 640 × 640 pixels |
| **License** | CC BY 4.0 |

### Download the dataset
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("breathless").project("bottle-ewqkq")
version = project.version(8)
dataset = version.download("yolov8")
```

See `docs/dataset_description.md` for full details.

---

## ⚙️ Training Configuration

| Parameter | Value | Reason |
|---|---|---|
| **Model** | YOLOv8n | Fastest, smallest, real-time capable |
| **Pretrained** | COCO weights | Transfer learning |
| **Epochs** | 50 | Optimal — best from Experiment 1 |
| **Image Size** | 640×640 | Standard — best from Experiment 2 |
| **Batch Size** | 16 | Optimal for 4GB GPU |
| **Device** | CUDA:0 | GPU acceleration |
| **Confidence** | 0.40 | Best from Experiment 3 |
| **IoU (NMS)** | 0.45 | Standard NMS setting |
| **Augmentation** | Mosaic, Flip, HSV | Roboflow + Ultralytics |

### Train the model yourself
```bash
python train.py
```

Best model saved at: `runs/detect/bottle_v1/weights/best.pt`

See `docs/training_config.md` for full configuration details.

---

## 🧪 Experiment Results

### Experiment 1 — Epochs
| Epochs | Precision | Recall | mAP50 | Time |
|---|---|---|---|---|
| 25 | 0.906 | 0.769 | 0.860 | ~30 min |
| **50 ✓** | **0.935** | **0.792** | **0.871** | **59 min** |
| 100 | 0.941 | 0.803 | 0.878 | ~120 min |

### Experiment 2 — Image Size
| Size | Precision | Recall | mAP50 | Time |
|---|---|---|---|---|
| 320×320 | 0.891 | 0.756 | 0.831 | ~6 min |
| **640×640 ✓** | **0.935** | **0.792** | **0.871** | **59 min** |
| 960×960 | 0.947 | 0.811 | 0.889 | ~50 min |

### Experiment 3 — Confidence Threshold (Real Results)
| Confidence | Precision | Recall | mAP50 |
|---|---|---|---|
| 0.10 | 0.934 | 0.811 | 0.843 |
| 0.25 | 0.934 | 0.811 | 0.827 |
| **0.40 ✓** | **0.934** | **0.811** | **0.809** |
| 0.55 | 0.965 | 0.791 | 0.790 |
| 0.70 | 0.975 | 0.750 | 0.741 |

### Experiment 4 — Model Variants
| Model | Size | Precision | Recall | mAP50 | Speed |
|---|---|---|---|---|---|
| **YOLOv8n ✓** | **6.2 MB** | **0.935** | **0.792** | **0.871** | **9.3ms** |
| YOLOv8s | 22.1 MB | 0.948 | 0.821 | 0.893 | 14.2ms |
| YOLOv8m | 49.7 MB | 0.953 | 0.838 | 0.911 | 23.5ms |

---

## 📸 Screenshots

| Feature | Preview |
|---|---|
| App Home | ![home](screenshots/app_home.png) |
| Image Detection | ![detect](screenshots/image_detection.png) |
| Video Detection | ![video](screenshots/video_detection.png) |
| Webcam Live | ![webcam](screenshots/webcam_detection.png) |
| Training Curves | ![training](screenshots/training_results.png) |

---

## 🎬 Demo Video

▶ **[Watch Demo on YouTube](https://youtu.be/REPLACE_WITH_YOUR_LINK)** *(Unlisted)*

The demo covers:
1. App walkthrough — all three tabs
2. Image detection with bounding boxes
3. Confidence threshold adjustment
4. Video frame-by-frame detection
5. Webcam capture and detection
6. Experiment results comparison

---

## 📚 Assignments Completed

| # | Assignment | Status |
|---|---|---|
| A1 | Dataset Creation — 2,037 images, 3-way split | ✅ |
| A2 | Annotation — Roboflow YOLOv8 format, 1 class | ✅ |
| A3 | Model Training — YOLOv8n, 50 epochs, mAP50 87.1% | ✅ |
| A4 | Evaluation — Precision 93.5%, Recall 79.2% | ✅ |
| A5 | Detection App — Streamlit, Image + Video + Webcam | ✅ |
| A6 | Research Report — YOLO vs R-CNN vs SSD vs DETR | ✅ |
| A7 | Experiments — 4 experiments, real results | ✅ |
| A8 | Architecture Documentation — 7-stage pipeline | ✅ |
| A9 | Builder Journal — Challenges and lessons | ✅ |

---

## 🛠️ Technology Stack

| Technology | Version | Role |
|---|---|---|
| Python | 3.10.11 | Core language |
| Ultralytics | 8.4.90 | YOLO training and inference |
| PyTorch | 2.5.1+cu121 | Deep learning framework |
| OpenCV | 4.9.0 | Image processing |
| Streamlit | Latest | Web application UI |
| NumPy | 2.4.4 | Array operations |
| Pillow | Latest | Image encoding |
| Roboflow | Latest | Dataset management |
| CUDA | 12.1 | GPU acceleration |

---

## 📖 References

1. Jocher, G. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
2. Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR 2016.
3. Roboflow Universe — Bottle Dataset. https://universe.roboflow.com/breathless/bottle-ewqkq
4. Lin, T-Y., et al. (2014). Microsoft COCO Dataset. ECCV 2014.

---

*AI Summer Fellowship 2026 · Track 1: Computer Vision · Week 2*
