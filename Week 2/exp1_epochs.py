# ─────────────────────────────────────────────────────────────────
#  EXPERIMENT 1 — Different Epoch Values
#  Tests: 25 epochs vs 50 epochs (done) vs 100 epochs
#  Goal: Find the best number of training epochs
# ─────────────────────────────────────────────────────────────────

from ultralytics import YOLO
import json, os, time

DATA = r"C:\Users\SS\OneDrive\Desktop\University\Internship\Week 2\Bottle-Detection\dataset\bottle-8\data.yaml"
RESULTS_FILE = "experiment_results.json"

# Load existing results if any
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
else:
    all_results = {}

# ── Already have 50 epochs result — add it manually ──────────────
all_results["exp1_epochs_50"] = {
    "epochs":    50,
    "precision": 0.935,
    "recall":    0.792,
    "map50":     0.871,
    "map50_95":  0.620,
    "train_time_hours": 0.989,
    "note": "Already trained — bottle_v1-3"
}
print("✓ 50 epochs result loaded from previous training")

def run_epoch_experiment(epochs, name):
    print(f"\n{'='*55}")
    print(f" Training with {epochs} epochs → {name}")
    print(f"{'='*55}")

    t_start = time.time()
    model   = YOLO("yolov8n.pt")

    results = model.train(
        data=DATA,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        name=name,
        project="runs/experiments",
        verbose=False
    )

    t_end  = time.time()
    hours  = (t_end - t_start) / 3600

    # Validate
    best   = YOLO(f"runs/experiments/{name}/weights/best.pt")
    val    = best.val(data=DATA, verbose=False)

    result = {
        "epochs":    epochs,
        "precision": round(float(val.box.mp),  3),
        "recall":    round(float(val.box.mr),  3),
        "map50":     round(float(val.box.map50), 3),
        "map50_95":  round(float(val.box.map),  3),
        "train_time_hours": round(hours, 3),
    }

    print(f"\n  Epochs    : {epochs}")
    print(f"  Precision : {result['precision']:.3f}")
    print(f"  Recall    : {result['recall']:.3f}")
    print(f"  mAP50     : {result['map50']:.3f}")
    print(f"  mAP50-95  : {result['map50_95']:.3f}")
    print(f"  Time      : {hours:.2f} hours")

    return result

if __name__ == "__main__":
    print("\n EXPERIMENT 1 — EPOCH COMPARISON")
    print(" Already have: 50 epochs → mAP50 = 0.871")
    print(" Will train:   25 epochs and 100 epochs\n")

    # 25 epochs (~8 minutes)
    print("Step 1/2: Training 25 epochs (~8 minutes)...")
    all_results["exp1_epochs_25"] = run_epoch_experiment(25, "exp1_25epochs")

    # 100 epochs (~35 minutes)
    print("\nStep 2/2: Training 100 epochs (~35 minutes)...")
    all_results["exp1_epochs_100"] = run_epoch_experiment(100, "exp1_100epochs")

    # Save all results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*55)
    print(" EXPERIMENT 1 COMPLETE — Results Summary")
    print("="*55)
    print(f" {'Epochs':<10} {'Precision':<12} {'Recall':<10} {'mAP50':<10} {'Time'}")
    print("-"*55)
    for key in ["exp1_epochs_25", "exp1_epochs_50", "exp1_epochs_100"]:
        if key in all_results:
            r = all_results[key]
            print(f" {r['epochs']:<10} {r['precision']:<12.3f} {r['recall']:<10.3f} {r['map50']:<10.3f} {r['train_time_hours']:.2f}h")

    print(f"\n✓ Results saved to {RESULTS_FILE}")
    print("  Now run: python exp2_imgsz.py")
