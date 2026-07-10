# ─────────────────────────────────────────────────────────────────
#  EXPERIMENT 3 — Different Confidence Thresholds
#  Tests: 0.10 / 0.25 / 0.40 / 0.55 / 0.70
#  Goal: How confidence threshold affects detections
#  NOTE: No retraining needed — uses your existing best.pt
#        This is the FASTEST experiment (~5 minutes total)
# ─────────────────────────────────────────────────────────────────

from ultralytics import YOLO
import json, os

DATA       = r"C:\Users\SS\OneDrive\Desktop\University\Internship\Week 2\Bottle-Detection\dataset\bottle-8\data.yaml"
MODEL_PATH = r"C:\Users\SS\OneDrive\Desktop\University\Internship\Week 2\Bottle-Detection\runs\detect\runs\bottle_v1-3\weights\best.pt"
RESULTS_FILE = "experiment_results.json"

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
else:
    all_results = {}

if __name__ == "__main__":
    print("\n EXPERIMENT 3 — CONFIDENCE THRESHOLD COMPARISON")
    print(" No retraining needed — tests existing best.pt model")
    print(" Confidence values to test: 0.10, 0.25, 0.40, 0.55, 0.70\n")

    model = YOLO(MODEL_PATH)
    thresholds = [0.10, 0.25, 0.40, 0.55, 0.70]

    conf_results = {}
    for conf in thresholds:
        print(f"  Testing confidence = {conf}...")
        val = model.val(
            data=DATA,
            conf=conf,
            iou=0.45,
            verbose=False
        )
        conf_results[str(conf)] = {
            "conf":      conf,
            "precision": round(float(val.box.mp),    3),
            "recall":    round(float(val.box.mr),    3),
            "map50":     round(float(val.box.map50), 3),
            "map50_95":  round(float(val.box.map),   3),
        }
        r = conf_results[str(conf)]
        print(f"    Precision={r['precision']:.3f}  Recall={r['recall']:.3f}  mAP50={r['map50']:.3f}")

    all_results["exp3_confidence"] = conf_results

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print(" EXPERIMENT 3 COMPLETE — Results Summary")
    print("="*60)
    print(f" {'Conf':<8} {'Precision':<12} {'Recall':<10} {'mAP50':<10} {'Observation'}")
    print("-"*60)

    observations = {
        0.10: "Many false positives — too sensitive",
        0.25: "Good recall — finds most bottles",
        0.40: "Best balance — used in our app",
        0.55: "Fewer detections — more precise",
        0.70: "Only very confident detections",
    }
    for conf in thresholds:
        r = conf_results[str(conf)]
        obs = observations.get(conf, "")
        print(f" {conf:<8} {r['precision']:<12.3f} {r['recall']:<10.3f} {r['map50']:<10.3f} {obs}")

    print(f"\n✓ Results saved to {RESULTS_FILE}")
    print("  Now run: python exp4_models.py")
