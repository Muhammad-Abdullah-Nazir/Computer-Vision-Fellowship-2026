"""
Tracking Experiments
Assignment 2 - Week 3

Runs a video through several configurations and logs FPS, detection counts,
unique track counts, and ID-switch proxy metrics so you can compare:

  Experiment 1: ByteTrack
  Experiment 2: BoT-SORT
  Experiment 3: Different confidence thresholds
  Experiment 4: Different video resolutions
  Experiment 5 (bonus): CPU vs GPU, fp16 on vs off

Results are written to reports/experiments.csv and three charts:
  reports/tracker_comparison.png
  reports/confidence_resolution_comparison.png
  reports/device_comparison.png   (only if --device cuda:0 is available)

Usage
-----
python experiments.py --source input.mp4 --model yolov8n.pt --max-frames 300 --device auto
python experiments.py --source input.mp4 --model yolov8s.pt --device cuda:0 --compare-device
"""

import argparse
import csv
import time
import os

import cv2
import numpy as np
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def log_device(device_arg):
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False
    resolved = device_arg if device_arg not in (None, "auto") else ("cuda:0 (auto)" if cuda_ok else "cpu (auto)")
    print(f"[INFO] torch.cuda.is_available() = {cuda_ok}")
    print(f"[INFO] Requested device: {device_arg}  ->  resolved: {resolved}")
    return cuda_ok


def determine_fp16_kwarg(model, frame):
    """Ultralytics renamed half= to quantize='fp16' in newer versions. Figure
    out once which one this installed version accepts, instead of guessing."""
    try:
        model.track(frame, persist=False, quantize="fp16", verbose=False)
        return "quantize"
    except TypeError:
        return "half"


def run_single_experiment(model, source, tracker_yaml, conf, device, fp16, fp16_kwarg,
                           resize_to=None, max_frames=300):
    """Run tracking over up to max_frames frames and collect metrics."""
    try:
        source_val = int(source)
    except ValueError:
        source_val = source
    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source {source}")

    unique_ids = set()
    id_switch_proxy = 0          # counts a new ID appearing on a frame with no detections lost = rough proxy
    prev_ids = set()
    frame_times = []
    det_counts = []
    frame_idx = 0

    track_kwargs = dict(persist=True, tracker=tracker_yaml, conf=conf, device=device, verbose=False)
    if fp16:
        track_kwargs[fp16_kwarg] = "fp16" if fp16_kwarg == "quantize" else True

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if resize_to is not None:
            frame = cv2.resize(frame, resize_to)

        t0 = time.time()
        results = model.track(frame, **track_kwargs)[0]
        dt = time.time() - t0
        frame_times.append(dt)

        ids = set()
        if results.boxes is not None and results.boxes.id is not None:
            ids = set(int(i) for i in results.boxes.id.tolist())
        det_counts.append(len(ids))
        unique_ids.update(ids)

        # crude proxy: brand-new IDs appearing after the first few frames may indicate a switch
        if frame_idx > 5:
            id_switch_proxy += len(ids - prev_ids)
        prev_ids = ids

        frame_idx += 1

    cap.release()

    avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
    return {
        "avg_fps": round(avg_fps, 2),
        "avg_processing_ms": round(np.mean(frame_times) * 1000, 2) if frame_times else 0,
        "frames_processed": frame_idx,
        "unique_track_ids": len(unique_ids),
        "avg_objects_per_frame": round(np.mean(det_counts), 2) if det_counts else 0,
        "new_id_events": id_switch_proxy,  # higher can suggest more ID switching / re-detection
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Video file path (recommended over webcam for repeatable experiments)")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda:0'")
    parser.add_argument("--fp16", action="store_true", help="Run all experiments with half-precision inference")
    parser.add_argument("--compare-device", action="store_true",
                         help="Add bonus Experiment 5: same config on CPU vs GPU, and fp16 on vs off (GPU only)")
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)
    device = None if args.device == "auto" else args.device
    cuda_ok = log_device(args.device)
    model = YOLO(args.model)

    fp16_kwarg = "quantize"
    if args.fp16 or args.compare_device:
        cap = cv2.VideoCapture(args.source)
        ok, probe_frame = cap.read()
        cap.release()
        if ok:
            fp16_kwarg = determine_fp16_kwarg(model, probe_frame)
            print(f"[INFO] fp16 kwarg for this ultralytics version: '{fp16_kwarg}'")

    rows = []

    # Experiment 1 & 2: ByteTrack vs BoT-SORT at default confidence
    for tracker_name, tracker_yaml in [("ByteTrack", "bytetrack.yaml"), ("BoT-SORT", "botsort.yaml")]:
        print(f"[RUN] Tracker experiment: {tracker_name}")
        metrics = run_single_experiment(model, args.source, tracker_yaml, conf=0.3,
                                         device=device, fp16=args.fp16, fp16_kwarg=fp16_kwarg,
                                         max_frames=args.max_frames)
        rows.append({"experiment": "tracker_comparison", "variant": tracker_name, "conf": 0.3, "resolution": "original", **metrics})

    # Experiment 3: confidence thresholds (fixed tracker = ByteTrack)
    for conf in [0.15, 0.25, 0.35, 0.5, 0.65]:
        print(f"[RUN] Confidence experiment: conf={conf}")
        metrics = run_single_experiment(model, args.source, "bytetrack.yaml", conf=conf,
                                         device=device, fp16=args.fp16, fp16_kwarg=fp16_kwarg,
                                         max_frames=args.max_frames)
        rows.append({"experiment": "confidence_threshold", "variant": f"conf_{conf}", "conf": conf, "resolution": "original", **metrics})

    # Experiment 4: resolutions (fixed tracker = ByteTrack, conf = 0.3)
    for label, size in [("240p", (426, 240)), ("480p", (854, 480)), ("720p", (1280, 720))]:
        print(f"[RUN] Resolution experiment: {label}")
        metrics = run_single_experiment(model, args.source, "bytetrack.yaml", conf=0.3,
                                         device=device, fp16=args.fp16, fp16_kwarg=fp16_kwarg,
                                         resize_to=size, max_frames=args.max_frames)
        rows.append({"experiment": "resolution", "variant": label, "conf": 0.3, "resolution": label, **metrics})

    # Experiment 5 (bonus): CPU vs GPU, and fp16 on vs off if GPU available
    device_rows = []
    if args.compare_device:
        variants = [("CPU", "cpu", False)]
        if cuda_ok:
            variants.append(("GPU (fp32)", "cuda:0", False))
            variants.append(("GPU (fp16)", "cuda:0", True))
        for label, dev, use_fp16 in variants:
            print(f"[RUN] Device experiment: {label}")
            metrics = run_single_experiment(model, args.source, "bytetrack.yaml", conf=0.3,
                                             device=dev, fp16=use_fp16, fp16_kwarg=fp16_kwarg,
                                             max_frames=args.max_frames)
            row = {"experiment": "device_comparison", "variant": label, "conf": 0.3, "resolution": "original", **metrics}
            rows.append(row)
            device_rows.append(row)

    # Write CSV
    csv_path = "reports/experiments.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote {csv_path}")

    # Chart 1: tracker comparison (FPS)
    tracker_rows = [r for r in rows if r["experiment"] == "tracker_comparison"]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    names = [r["variant"] for r in tracker_rows]
    fps_vals = [r["avg_fps"] for r in tracker_rows]
    ax1.bar(names, fps_vals, color=["#4C72B0", "#DD8452"])
    ax1.set_ylabel("Average FPS")
    ax1.set_title("ByteTrack vs BoT-SORT: FPS")
    plt.tight_layout()
    plt.savefig("reports/tracker_comparison.png", dpi=150)
    plt.close()

    # Chart 2: confidence + resolution comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    conf_rows = [r for r in rows if r["experiment"] == "confidence_threshold"]
    axes[0].plot([r["conf"] for r in conf_rows], [r["avg_objects_per_frame"] for r in conf_rows], marker="o")
    axes[0].set_xlabel("Confidence threshold")
    axes[0].set_ylabel("Avg objects/frame")
    axes[0].set_title("Confidence Threshold vs Detections")

    res_rows = [r for r in rows if r["experiment"] == "resolution"]
    axes[1].bar([r["variant"] for r in res_rows], [r["avg_fps"] for r in res_rows], color="#55A868")
    axes[1].set_ylabel("Average FPS")
    axes[1].set_title("Resolution vs FPS")
    plt.tight_layout()
    plt.savefig("reports/confidence_resolution_comparison.png", dpi=150)
    plt.close()

    print("[INFO] Charts saved to reports/tracker_comparison.png and reports/confidence_resolution_comparison.png")

    # Chart 3 (bonus): device comparison
    if device_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [r["variant"] for r in device_rows]
        fps_vals = [r["avg_fps"] for r in device_rows]
        colors = ["#C44E52", "#4C72B0", "#55A868"][:len(names)]
        ax.bar(names, fps_vals, color=colors)
        ax.set_ylabel("Average FPS")
        ax.set_title("CPU vs GPU vs GPU+fp16: FPS")
        plt.tight_layout()
        plt.savefig("reports/device_comparison.png", dpi=150)
        plt.close()
        print("[INFO] Bonus chart saved to reports/device_comparison.png")

    print("[INFO] Use these results + charts directly in Assignment 2 documentation.")


if __name__ == "__main__":
    main()
