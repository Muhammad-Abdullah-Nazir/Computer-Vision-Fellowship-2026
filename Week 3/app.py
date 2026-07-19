"""
Intelligent Video Analytics Platform — CLI / OpenCV-window frontend
Week 3 - AI Summer Fellowship

Thin wrapper around analytics_core.AnalyticsCore. See streamlit_app.py for
the web frontend, which uses the exact same engine.

Usage
-----
python app.py --source 0
python app.py --source input.mp4 --tracker botsort --conf 0.35 --record
python app.py --source input.mp4 --model path/to/best.pt --config configs/my_config.json
"""

import argparse
import json
import os

import cv2

from analytics_core import AnalyticsCore, DASHBOARD_HEIGHT, open_capture, create_video_writer

DEFAULT_CONFIG = {"line": None, "zones": []}


def open_source(source):
    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def load_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        print(f"[INFO] Loaded line/zone config from {config_path}")
        return cfg
    print("[INFO] No config file given/found — using default line, no ROI zones.")
    return DEFAULT_CONFIG


def parse_args():
    p = argparse.ArgumentParser(description="Intelligent Video Analytics Platform")
    p.add_argument("--source", default="0", help="Webcam index (e.g. 0), video file path, or RTSP URL")
    p.add_argument("--model", default="yolov8n.pt", help="Path to YOLO weights (.pt)")
    p.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    p.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, choices=[256, 320, 480, 640],
                    help="YOLO inference resolution — lower is much faster (esp. CPU/webcam), small accuracy cost")
    p.add_argument("--device", default="auto",
                    help="Compute device: 'auto' (default), 'cpu', or 'cuda:0' / '0' for GPU")
    p.add_argument("--fp16", action="store_true",
                    help="Half-precision inference — real speedup on CUDA GPUs with tensor cores, no benefit on CPU")
    p.add_argument("--config", default=None, help="Path to JSON config with line/zone coordinates")
    p.add_argument("--record", action="store_true", help="Save annotated output video to output/")
    p.add_argument("--headless", action="store_true", help="Run without opening a display window")
    return p.parse_args()


def main():
    args = parse_args()
    cap = open_source(args.source)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    config = load_config(args.config)
    core = AnalyticsCore(
        model_path=args.model, tracker=args.tracker, conf=args.conf,
        frame_w=frame_w, frame_h=frame_h,
        line=config.get("line"), zones=config.get("zones", []),
        imgsz=args.imgsz, device=args.device, fp16=args.fp16,
    )

    writer = None
    if args.record:
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", f"annotated_{core.build_report(args.source, args.model)['run_timestamp']}.mp4")
        writer, out_path = create_video_writer(out_path, src_fps, (frame_w, frame_h + DASHBOARD_HEIGHT))
        if writer is None:
            print("[WARN] Recording disabled this run — see warnings above for why.")
        else:
            print(f"[INFO] Recording annotated video to {out_path}")

    print("[INFO] Starting analytics loop. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] End of stream.")
                break

            annotated, stats = core.process_frame(frame)

            if writer is not None:
                writer.write(annotated)

            if not args.headless:
                cv2.imshow("Intelligent Video Analytics Platform", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit key pressed.")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        os.makedirs("reports", exist_ok=True)
        report = core.build_report(args.source, args.model)
        out_path = os.path.join("reports", f"session_report_{report['run_timestamp']}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Session report written to {out_path}")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
