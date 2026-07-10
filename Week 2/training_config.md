# Training Configuration
## Bottle Detection — Week 2 | AI Summer Fellowship 2026

---

## Model
- **Architecture:** YOLOv8 Nano (YOLOv8n)
- **Pretrained Weights:** yolov8n.pt (COCO dataset — 80 classes, 330K images)
- **Training Type:** Transfer Learning (fine-tuning)

---

## Hardware
| Component | Detail |
|---|---|
| GPU | NVIDIA Quadro T1000 (4096 MB VRAM) |
| CUDA Version | 12.1 |
| PyTorch | 2.5.1+cu121 |
| Ultralytics | 8.4.90 |
| OS | Windows 11 |

---

## Training Parameters

| Parameter | Value | Reason |
|---|---|---|
| epochs | 50 | Best from Experiment 1 (diminishing returns after 50) |
| imgsz | 640 | Best from Experiment 2 (standard YOLO resolution) |
| batch | 16 | Optimal for 4GB GPU without OOM error |
| device | 0 | GPU CUDA:0 |
| workers | 2 | Required for Windows multiprocessing stability |
| conf | 0.40 | Best from Experiment 3 |
| iou | 0.45 | Standard NMS IoU threshold |
| lr0 | 0.01 | Initial learning rate (auto) |
| lrf | 0.01 | Final learning rate (auto) |
| momentum | 0.937 | SGD momentum (auto) |
| weight_decay | 0.0005 | L2 regularisation (auto) |
| optimizer | SGD | Auto-selected by Ultralytics |
| AMP | False | Disabled for Quadro T1000 compatibility |

---

## Training Script
```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=r'dataset/bottle-8/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        name='bottle_v1',
        project='runs'
    )

    print("Training complete!")
```

---

## Training Results

| Metric | Value |
|---|---|
| Final mAP50 | 0.871 (87.1%) |
| Final Precision | 0.935 (93.5%) |
| Final Recall | 0.792 (79.2%) |
| Final mAP50-95 | 0.620 (62.0%) |
| Training Time | 0.989 hours (59 minutes) |
| Epochs Completed | 50 / 50 |

---

## Output Files
```
runs/detect/bottle_v1/
├── weights/
│   ├── best.pt     ← Best model (use this for inference)
│   └── last.pt     ← Final epoch model
├── results.csv     ← Per-epoch metrics
├── results.png     ← Training curves
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
└── args.yaml       ← Training config backup
```

---

## Inference
```python
from ultralytics import YOLO

model = YOLO('runs/detect/bottle_v1/weights/best.pt')
results = model('your_image.jpg', conf=0.40, iou=0.45)
results[0].save('output.jpg')
```

*AI Summer Fellowship 2026 · Track 1: Computer Vision · Assignment 3*
