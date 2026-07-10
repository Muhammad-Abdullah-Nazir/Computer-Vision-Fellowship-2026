# ─────────────────────────────────────────────────────────────────
#  EXPERIMENT 4 — Different YOLO Model Variants
#  Tests: YOLOv8n (done) vs YOLOv8s vs YOLOv8m
#  Goal: How model size affects accuracy vs speed trade-off
# ─────────────────────────────────────────────────────────────────

from ultralytics import YOLO
import json, os, time

DATA = r"C:\Users\SS\OneDrive\Desktop\University\Internship\Week 2\Bottle-Detection\dataset\bottle-8\data.yaml"
RESULTS_FILE = "experiment_results.json"

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
else:
    all_results = {}

# Already have YOLOv8n result
all_results["exp4_model_yolov8n"] = {
    "model":      "YOLOv8n",
    "params_M":   3.0,
    "model_size_MB": 6.2,
    "precision":  0.935,
    "recall":     0.792,
    "map50":      0.871,
    "map50_95":   0.620,
    "train_time_hours": 0.989,
    "speed_ms":   9.3,
    "note": "Already trained — bottle_v1-3"
}
print("✓ YOLOv8n result loaded from previous training")

def run_model_experiment(model_name, model_file, exp_name):
    print(f"\n{'='*55}")
    print(f" Training {model_name} → {exp_name}")
    print(f"{'='*55}")

    t_start = time.time()
    model   = YOLO(model_file)
    model.train(
        data=DATA,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        name=exp_name,
        project="runs/experiments",
        verbose=False
    )
    t_end = time.time()
    hours = (t_end - t_start) / 3600

    best    = YOLO(f"runs/experiments/{exp_name}/weights/best.pt")
    val     = best.val(data=DATA, verbose=False)

    # Get model size in MB
    model_path = f"runs/experiments/{exp_name}/weights/best.pt"
    size_mb    = round(os.path.getsize(model_path) / (1024*1024), 1)

    # Get inference speed
    import torch
    info = best.info(verbose=False)

    result = {
        "model":      model_name,
        "precision":  round(float(val.box.mp),    3),
        "recall":     round(float(val.box.mr),    3),
        "map50":      round(float(val.box.map50), 3),
        "map50_95":   round(float(val.box.map),   3),
        "train_time_hours": round(hours, 3),
        "model_size_MB": size_mb,
        "speed_ms":   round(float(val.speed["inference"]), 1),
    }

    print(f"\n  Model      : {model_name}")
    print(f"  Size       : {size_mb} MB")
    print(f"  Precision  : {result['precision']:.3f}")
    print(f"  Recall     : {result['recall']:.3f}")
    print(f"  mAP50      : {result['map50']:.3f}")
    print(f"  mAP50-95   : {result['map50_95']:.3f}")
    print(f"  Speed      : {result['speed_ms']}ms/image")
    print(f"  Train Time : {hours:.2f} hours")
    return result

if __name__ == "__main__":
    print("\n EXPERIMENT 4 — YOLO MODEL VARIANT COMPARISON")
    print(" Already have: YOLOv8n → mAP50 = 0.871")
    print(" Will train:   YOLOv8s (~25 min) and YOLOv8m (~45 min)\n")

    # YOLOv8s — Small model (~25 min)
    print("Step 1/2: Training YOLOv8s — Small (~25 minutes)...")
    all_results["exp4_model_yolov8s"] = run_model_experiment(
        "YOLOv8s", "yolov8s.pt", "exp4_yolov8s"
    )

    # YOLOv8m — Medium model (~45 min)
    print("\nStep 2/2: Training YOLOv8m — Medium (~45 minutes)...")
    all_results["exp4_model_yolov8m"] = run_model_experiment(
        "YOLOv8m", "yolov8m.pt", "exp4_yolov8m"
    )

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*65)
    print(" EXPERIMENT 4 COMPLETE — Results Summary")
    print("="*65)
    print(f" {'Model':<12} {'Size MB':<10} {'Precision':<12} {'Recall':<10} {'mAP50':<10} {'Speed'}")
    print("-"*65)
    for key in ["exp4_model_yolov8n","exp4_model_yolov8s","exp4_model_yolov8m"]:
        if key in all_results:
            r = all_results[key]
            print(f" {r['model']:<12} {r.get('model_size_MB','?'):<10} {r['precision']:<12.3f} {r['recall']:<10.3f} {r['map50']:<10.3f} {r.get('speed_ms','?')}ms")

    print(f"\n✓ All 4 experiments complete!")
    print("  Run: python generate_report.py  to create the Word document")
