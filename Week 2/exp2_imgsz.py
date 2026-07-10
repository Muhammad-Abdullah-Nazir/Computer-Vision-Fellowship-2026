# ─────────────────────────────────────────────────────────────────
#  EXPERIMENT 2 — Different Image Sizes
#  Tests: 320 vs 640 (done) vs 960
#  Goal: How image size affects accuracy and speed
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

# Already have 640 result
all_results["exp2_imgsz_640"] = {
    "imgsz":     640,
    "precision": 0.935,
    "recall":    0.792,
    "map50":     0.871,
    "map50_95":  0.620,
    "train_time_hours": 0.989,
    "note": "Already trained — bottle_v1-3"
}
print("✓ 640px result loaded from previous training")

def run_size_experiment(imgsz, name):
    print(f"\n{'='*55}")
    print(f" Training with imgsz={imgsz} → {name}")
    print(f"{'='*55}")

    batch = 16 if imgsz <= 640 else 8   # reduce batch for larger images

    t_start = time.time()
    model   = YOLO("yolov8n.pt")
    model.train(
        data=DATA,
        epochs=50,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=2,
        name=name,
        project="runs/experiments",
        verbose=False
    )
    t_end = time.time()
    hours = (t_end - t_start) / 3600

    best = YOLO(f"runs/experiments/{name}/weights/best.pt")
    val  = best.val(data=DATA, verbose=False)

    result = {
        "imgsz":     imgsz,
        "batch":     batch,
        "precision": round(float(val.box.mp),    3),
        "recall":    round(float(val.box.mr),    3),
        "map50":     round(float(val.box.map50), 3),
        "map50_95":  round(float(val.box.map),   3),
        "train_time_hours": round(hours, 3),
    }

    print(f"\n  Image Size : {imgsz}×{imgsz}")
    print(f"  Precision  : {result['precision']:.3f}")
    print(f"  Recall     : {result['recall']:.3f}")
    print(f"  mAP50      : {result['map50']:.3f}")
    print(f"  mAP50-95   : {result['map50_95']:.3f}")
    print(f"  Time       : {hours:.2f} hours")
    return result

if __name__ == "__main__":
    print("\n EXPERIMENT 2 — IMAGE SIZE COMPARISON")
    print(" Already have: 640px → mAP50 = 0.871")
    print(" Will train:   320px and 960px\n")

    # 320px — faster, less accurate (~6 min)
    print("Step 1/2: Training with 320×320 images (~6 minutes)...")
    all_results["exp2_imgsz_320"] = run_size_experiment(320, "exp2_imgsz320")

    # 960px — slower, potentially more accurate (~50 min, uses more GPU memory)
    print("\nStep 2/2: Training with 960×960 images (~50 minutes)...")
    all_results["exp2_imgsz_960"] = run_size_experiment(960, "exp2_imgsz960")

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*55)
    print(" EXPERIMENT 2 COMPLETE — Results Summary")
    print("="*55)
    print(f" {'Size':<10} {'Precision':<12} {'Recall':<10} {'mAP50':<10} {'Time'}")
    print("-"*55)
    for key in ["exp2_imgsz_320", "exp2_imgsz_640", "exp2_imgsz_960"]:
        if key in all_results:
            r = all_results[key]
            label = f"{r['imgsz']}px"
            print(f" {label:<10} {r['precision']:<12.3f} {r['recall']:<10.3f} {r['map50']:<10.3f} {r['train_time_hours']:.2f}h")

    print(f"\n✓ Results saved to {RESULTS_FILE}")
    print("  Now run: python exp3_confidence.py")
