# Dataset Description
## Bottle Detection — Week 2 | AI Summer Fellowship 2026

---

## Source
- **Platform:** Roboflow Universe
- **URL:** https://universe.roboflow.com/breathless/bottle-ewqkq/dataset/8
- **Author:** breathless
- **Version:** 8
- **License:** CC BY 4.0 (Free to use with attribution)

---

## Statistics

| Property | Value |
|---|---|
| Total Images | 2,037 |
| Original Images | 849 |
| Augmented Images | 1,188 |
| Training Set | 1,782 images (87%) |
| Validation Set | 169 images (8%) |
| Test Set | 86 images (4%) |
| Number of Classes | 1 |
| Class Name | bottle |
| Image Size | 640 × 640 pixels |
| Annotation Format | YOLOv8 normalised |

---

## Image Diversity

| Category | Varieties Included |
|---|---|
| Bottle Types | Water, soda, juice, glass, plastic |
| Backgrounds | Indoor table, shelf, floor, outdoor |
| Lighting | Bright daylight, indoor, shadow, low light |
| Angles | Front, side, top-down, angled |
| Occlusion | Partial bottles included |
| Scale | Close-up and distant bottles |

---

## Roboflow Augmentations Applied
- Horizontal flip
- Rotation (±15°)
- Brightness adjustment
- Contrast adjustment
- Blur
- Noise injection
- Hue/Saturation shift
- Mosaic (during YOLOv8 training)

---

## Annotation Format (YOLOv8)
Each image has a corresponding `.txt` label file.
Each line = one bottle bounding box:

```
class_id  x_center  y_center  width  height
0         0.512     0.438     0.234  0.615
```

All values normalised between 0 and 1.
`class_id = 0` always (bottle is the only class).

---

## Folder Structure
```
dataset/bottle-8/
├── train/
│   ├── images/    ← 1,782 images
│   └── labels/    ← 1,782 .txt files
├── valid/
│   ├── images/    ← 169 images
│   └── labels/
├── test/
│   ├── images/    ← 86 images
│   └── labels/
└── data.yaml
```

---

## data.yaml Configuration
```yaml
names:
- bottle
nc: 1
train: /path/to/dataset/bottle-8/train/images
val:   /path/to/dataset/bottle-8/valid/images
test:  /path/to/dataset/bottle-8/test/images
```

---

## Download Instructions
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("breathless").project("bottle-ewqkq")
version = project.version(8)
dataset = version.download("yolov8")
```

*AI Summer Fellowship 2026 · Track 1: Computer Vision · Assignment 1*
