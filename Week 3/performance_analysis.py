"""
Performance Analysis
Assignment 5 - Week 3

Measures FPS, CPU usage, memory usage, inference time, and tracking-accuracy
proxy metrics while running the platform on a video file. Writes a JSON
report (reports/performance_report.json) and charts (reports/performance_charts.png).

IMPORTANT — Tracking Accuracy honesty note
-------------------------------------------
True MOT accuracy metrics (MOTA, MOTP, IDF1) require ground-truth annotated
video with correct object IDs per frame, which is not available for an
arbitrary test clip. This script instead reports well-established,
ground-truth-free PROXIES:
  - ID switch rate: how often a "new" ID appears where an existing track
    was expected (higher = more fragmentation / lost identities)
  - Average track length: how many frames each ID persists on average
    (higher = more stable tracking, fewer broken/re-started tracks)
These correlate with tracking quality but are NOT the same as benchmark
MOTA/IDF1 scores — that distinction is called out again in the JSON output
and should be kept in the write-up.

CPU/memory sampling note
-------------------------
psutil's cpu_percent() needs a real time interval between calls to be
meaningful — calling it once per frame in a tight, fast loop gives noisy,
unreliable readings. This script instead samples CPU and memory from a
background thread on a fixed wall-clock interval, independent of how fast
frames are being processed.

Warmup frame note
-------------------
The very first inference call includes one-time PyTorch/CUDA
initialization cost (kernel compilation, memory allocation) that can be
10-100x slower than steady-state frames — this is a real, well-known ML
benchmarking artifact, not a bug. Standard practice is to exclude a small
number of warmup frames from timing statistics; this script does that
(default: first 3 frames) and reports both figures so the exclusion is
never hidden.

Usage
-----
python performance_analysis.py --source test_video.mp4 --model yolov8n.pt --device cuda:0
python performance_analysis.py --source test_video.mp4 --model yolov8s.pt --device cuda:0 --fp16 --max-frames 300
"""

import argparse
import json
import os
import threading
import time
from collections import defaultdict

import cv2
import numpy as np
import psutil
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ResourceSampler:
    """Samples this process's CPU% and memory (RSS) on a fixed wall-clock
    interval, in a background thread, independent of the frame loop's speed."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.samples = []  # list of (elapsed_s, process_cpu_pct, system_cpu_pct, mem_mb)
        self._running = False
        self._thread = None
        self._t0 = None

    def start(self):
        self.process.cpu_percent(interval=None)  # prime — first call is meaningless
        psutil.cpu_percent(interval=None)
        self._running = True
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            time.sleep(self.interval)
            proc_cpu = self.process.cpu_percent(interval=None)
            sys_cpu = psutil.cpu_percent(interval=None)
            mem_mb = self.process.memory_info().rss / (1024 * 1024)
            self.samples.append((time.time() - self._t0, proc_cpu, sys_cpu, mem_mb))

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)


def run_performance_analysis(source, model_path, tracker_yaml, conf, imgsz, device, fp16, max_frames, warmup_frames=3):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source {source}")

    sampler = ResourceSampler(interval=0.25)
    sampler.start()

    frame_times = []
    track_frame_counts = defaultdict(int)
    id_switch_events = 0
    prev_ids = set()
    unique_ids = set()

    track_kwargs = dict(persist=True, tracker=tracker_yaml, conf=conf, imgsz=imgsz, device=device, verbose=False)
    if fp16:
        track_kwargs["quantize"] = "fp16"

    frame_idx = 0
    t_start = time.time()
    fp16_fallback_checked = False

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        try:
            results = model.track(frame, **track_kwargs)[0]
        except TypeError:
            if fp16 and not fp16_fallback_checked and "quantize" in track_kwargs:
                track_kwargs.pop("quantize")
                track_kwargs["half"] = True
                fp16_fallback_checked = True
                results = model.track(frame, **track_kwargs)[0]
            else:
                raise
        dt = time.time() - t0
        frame_times.append(dt)

        ids = set()
        if results.boxes is not None and results.boxes.id is not None:
            ids = set(int(i) for i in results.boxes.id.tolist())
        for tid in ids:
            track_frame_counts[tid] += 1
        unique_ids.update(ids)
        if frame_idx > 5:
            id_switch_events += len(ids - prev_ids)
        prev_ids = ids

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t_start
    sampler.stop()

    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    inference_ms = [t * 1000 for t in frame_times]
    steady_ms = inference_ms[warmup_frames:] if len(inference_ms) > warmup_frames else inference_ms
    avg_track_length = float(np.mean(list(track_frame_counts.values()))) if track_frame_counts else 0
    id_switch_rate = (id_switch_events / frame_idx * 100) if frame_idx else 0

    cpu_proc = [s[1] for s in sampler.samples]
    cpu_sys = [s[2] for s in sampler.samples]
    mem_mb = [s[3] for s in sampler.samples]

    report = {
        "config": {
            "model": model_path, "tracker": tracker_yaml, "conf": conf,
            "imgsz": imgsz, "device": device, "fp16": fp16,
        },
        "frames_processed": frame_idx,
        "elapsed_seconds": round(elapsed, 2),
        "fps": {
            "average": round(avg_fps, 2),
            "average_note": "Includes warmup frame(s) — see inference_time_ms.steady_state for the fairer figure.",
        },
        "inference_time_ms": {
            "warmup_frames_excluded": warmup_frames,
            "first_frame_ms": round(inference_ms[0], 2) if inference_ms else None,
            "first_frame_note": "First-call PyTorch/CUDA init cost — expected to be much larger than steady-state; excluded from averages below.",
            "steady_state_average": round(float(np.mean(steady_ms)), 2) if steady_ms else 0,
            "steady_state_p95": round(float(np.percentile(steady_ms, 95)), 2) if steady_ms else 0,
            "steady_state_min": round(min(steady_ms), 2) if steady_ms else 0,
            "steady_state_max": round(max(steady_ms), 2) if steady_ms else 0,
            "steady_state_fps_equivalent": round(1000 / float(np.mean(steady_ms)), 2) if steady_ms else 0,
            "all_frames_average_including_warmup": round(float(np.mean(inference_ms)), 2) if inference_ms else 0,
        },
        "cpu_usage_percent": {
            "process_avg": round(float(np.mean(cpu_proc)), 2) if cpu_proc else None,
            "process_max": round(float(np.max(cpu_proc)), 2) if cpu_proc else None,
            "system_avg": round(float(np.mean(cpu_sys)), 2) if cpu_sys else None,
            "num_logical_cores": psutil.cpu_count(logical=True),
            "note": "process_avg can exceed 100% on a multi-core machine (e.g. 250% = "
                    "2.5 cores fully used) — this is normal psutil behavior, not a bug.",
        },
        "memory_usage_mb": {
            "average": round(float(np.mean(mem_mb)), 2) if mem_mb else None,
            "peak": round(float(np.max(mem_mb)), 2) if mem_mb else None,
            "start": round(mem_mb[0], 2) if mem_mb else None,
            "end": round(mem_mb[-1], 2) if mem_mb else None,
        },
        "tracking_accuracy_proxies": {
            "unique_track_ids": len(unique_ids),
            "avg_track_length_frames": round(avg_track_length, 2),
            "id_switch_events": id_switch_events,
            "id_switch_rate_per_100_frames": round(id_switch_rate, 2),
            "note": "Ground-truth-free proxies, NOT MOTA/IDF1 — see module docstring.",
        },
        "raw_samples": {
            "inference_time_ms": inference_ms,
            "resource_timeline_seconds": [s[0] for s in sampler.samples],
            "cpu_percent": cpu_proc,
            "system_cpu_percent": cpu_sys,
            "memory_mb": mem_mb,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--warmup-frames", type=int, default=3,
                         help="Frames excluded from timing averages (first-call init cost)")
    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    tracker_yaml = "bytetrack.yaml" if args.tracker == "bytetrack" else "botsort.yaml"

    print(f"[INFO] model={args.model} tracker={args.tracker} conf={args.conf} "
          f"imgsz={args.imgsz} device={args.device} fp16={args.fp16}")
    report = run_performance_analysis(args.source, args.model, tracker_yaml, args.conf,
                                       args.imgsz, device, args.fp16, args.max_frames, args.warmup_frames)

    os.makedirs("reports", exist_ok=True)
    summary = {k: v for k, v in report.items() if k != "raw_samples"}
    with open("reports/performance_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("reports/performance_report_full.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + json.dumps(summary, indent=2))
    print("\n[INFO] Summary -> reports/performance_report.json")
    print("[INFO] Full (with raw samples) -> reports/performance_report_full.json")

    raw = report["raw_samples"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(raw["inference_time_ms"], color="#4C81C4", linewidth=0.9)
    axes[0].axhline(report["inference_time_ms"]["steady_state_average"], color="#333333", linestyle="--", linewidth=1,
                     label=f'steady-state avg {report["inference_time_ms"]["steady_state_average"]:.1f} ms')
    wf = report["inference_time_ms"]["warmup_frames_excluded"]
    steady_max = report["inference_time_ms"]["steady_state_max"]
    if wf > 0 and len(raw["inference_time_ms"]) > wf:
        axes[0].axvspan(0, wf, color="orange", alpha=0.15, label=f"warmup ({wf} frames, excluded from stats)")
    # clip y-axis to steady-state range so the chart stays readable even
    # though the warmup frame (often 10-100x larger) is included in the plot
    if steady_max > 0:
        y_top = steady_max * 1.3
        if max(raw["inference_time_ms"]) > y_top:
            axes[0].set_ylim(0, y_top)
            axes[0].text(0.98, 0.96, "warmup frame off-scale\n(see JSON for exact value)",
                          transform=axes[0].transAxes, ha='right', va='top', fontsize=6.5,
                          color="#B8562E", style='italic')
    axes[0].set_title("Inference Time per Frame")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("ms")
    axes[0].legend(fontsize=7)

    if raw["cpu_percent"]:
        axes[1].plot(raw["resource_timeline_seconds"], raw["cpu_percent"], color="#B8562E", label="process")
        axes[1].plot(raw["resource_timeline_seconds"], raw["system_cpu_percent"], color="#B8562E",
                      alpha=0.35, linestyle="--", label="system")
        axes[1].set_title("CPU Usage Over Time")
        axes[1].set_xlabel("Seconds")
        axes[1].set_ylabel("%")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No samples\n(run too short)", ha='center', va='center')
        axes[1].set_title("CPU Usage Over Time")

    if raw["memory_mb"]:
        axes[2].plot(raw["resource_timeline_seconds"], raw["memory_mb"], color="#3D8C5C")
        axes[2].set_title("Process Memory (RSS) Over Time")
        axes[2].set_xlabel("Seconds")
        axes[2].set_ylabel("MB")
    else:
        axes[2].text(0.5, 0.5, "No samples\n(run too short)", ha='center', va='center')
        axes[2].set_title("Process Memory (RSS) Over Time")

    plt.tight_layout()
    plt.savefig("reports/performance_charts.png", dpi=150)
    plt.close()
    print("[INFO] Charts -> reports/performance_charts.png")


if __name__ == "__main__":
    main()
