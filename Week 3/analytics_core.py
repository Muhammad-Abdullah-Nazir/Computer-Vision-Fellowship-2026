"""
analytics_core.py
Shared detection + tracking + counting + dashboard engine.

Used by:
  - app.py            (CLI / OpenCV-window version)
  - streamlit_app.py  (web frontend)

Keeping this logic in one place means both frontends stay in sync and you
only have to explain / defend ONE analytics pipeline in your interview.
"""

import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

TRAJECTORY_LENGTH = 30
DASHBOARD_HEIGHT = 172  # fixed height for the card-based dashboard layout — recorded video frame size never changes


def open_capture(source, warmup_reads=3):
    """Open a video source robustly.

    On Windows, OpenCV's default webcam backend (MSMF) is known to
    intermittently fail to grab frames right after opening — usually
    showing as 'OnReadSample() is called with error status: -1072873821'.
    DirectShow (CAP_DSHOW) is far more reliable for USB webcams. This tries
    DSHOW first for webcam indices, falls back to the default backend if
    that fails to open (e.g. on non-Windows, or for video files/RTSP where
    DSHOW doesn't apply), and does a few throwaway reads to let the camera
    driver stabilize before real use.
    """
    try:
        source_val = int(source)
        is_webcam = True
    except (TypeError, ValueError):
        source_val = source
        is_webcam = False

    cap = None
    if is_webcam:
        cap = cv2.VideoCapture(source_val, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(source_val)

    if is_webcam and cap.isOpened():
        for _ in range(warmup_reads):
            cap.read()  # discard — lets the driver settle before real frames are used

    return cap


def create_video_writer(out_path, fps, frame_size):
    """Create a cv2.VideoWriter robustly.

    Some OpenCV builds (seen with opencv-python 5.0.0 on Windows) don't
    have a working FFMPEG plugin at all. Explicitly forcing cv2.CAP_FFMPEG
    on a build like that doesn't fail cleanly — it triggers a broken
    internal fallback to an 'image sequence' writer, which then raises an
    assertion error on a normal (non-numbered) filename. So this does NOT
    force any particular backend; instead it tries several codec/container
    combinations, most-likely-to-work-without-FFMPEG first, and lets
    OpenCV auto-pick whatever backend is actually available. Every attempt
    is wrapped in try/except since some builds raise instead of just
    failing to open.

    Returns (writer_or_None, actual_output_path_or_None). Callers must
    check the writer for None (recording unavailable on this system)
    rather than assuming success.
    """
    stem = os.path.splitext(out_path)[0]

    # Sanitize fps: many webcams (especially via DirectShow) report 0 or
    # NaN for cap.get(CAP_PROP_FPS). A plain `fps or 30` fallback does NOT
    # catch NaN, since NaN is truthy in Python — it would silently pass a
    # NaN straight into cv2.VideoWriter, which some codecs (e.g. MJPG)
    # reject with an 'fps >= 1' assertion failure. Guard against all of
    # 0, negative, NaN, and infinite here, once, for every caller.
    if fps is None or not np.isfinite(fps) or fps < 1:
        print(f"[INFO] Reported source FPS ({fps}) is invalid — using 20.0 fps for the recording instead.")
        fps = 20.0

    attempts = [
        (stem + ".avi", "MJPG"),   # Motion JPEG — usually works without FFMPEG on Windows
        (stem + ".mp4", "mp4v"),   # standard MP4 — works if FFMPEG (or a system codec) is present
        (stem + ".avi", "XVID"),   # widely-supported AVI codec
        (stem + ".avi", "DIVX"),   # last resort, older but very broadly supported
    ]

    for path, codec in attempts:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
            if writer is not None and writer.isOpened():
                print(f"[INFO] Recording with codec {codec} -> {path}")
                return writer, path
            if writer is not None:
                writer.release()
        except Exception as e:
            print(f"[WARN] Video writer attempt failed (codec={codec}): {e}")

    print("[WARN] Could not open any video writer on this system — recording disabled this run.")
    return None, None


class AnalyticsCore:
    """
    Wraps one YOLO model + one tracker configuration + one line/zone setup.
    Call process_frame(frame) once per frame; it returns the annotated
    frame (with dashboard baked in) and a dict of current stats.
    """

    def __init__(self, model_path, tracker="bytetrack", conf=0.3,
                 frame_w=None, frame_h=None, line=None, zones=None, imgsz=640,
                 device=None, fp16=False):
        self.model = YOLO(model_path)
        self.tracker_yaml = "bytetrack.yaml" if tracker == "bytetrack" else "botsort.yaml"
        self.conf = conf
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.imgsz = imgsz
        # device: None/"auto" lets ultralytics pick (CUDA if available, else
        # CPU); "cpu" or "cuda:0" forces a specific device. Logged so it's
        # obvious in the console/report which one is actually being used —
        # a silent CPU-only torch install is the #1 cause of "GPU not used".
        self.device = None if device in (None, "auto") else device
        # fp16: half-precision inference. Real speedup on a CUDA GPU with
        # tensor cores (e.g. Turing+); little to no benefit on CPU (PyTorch's
        # CPU half support is limited), so this is a GPU-focused optimization.
        self.fp16 = bool(fp16)
        self._log_device()

        self._setup_line_and_zones(line, zones)

        self.trajectories = defaultdict(lambda: deque(maxlen=TRAJECTORY_LENGTH))
        self.track_classes = {}
        self.all_track_ids = set()
        self.confidences = deque(maxlen=500)
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0
        self.start_time = time.time()

        # Direction: minimum pixel displacement across the trajectory window
        # before we trust it enough to report a heading (avoids jitter on
        # a stationary object being reported as "moving").
        self.direction_min_displacement = 10

        # Dwell time: per zone, {track_id: entry_timestamp}. Reset when the
        # object leaves that zone. zone_max_dwell tracks the longest dwell
        # ever recorded per zone, for the dashboard / session report.
        self.zone_dwell_start = [dict() for _ in self.zones]
        self.zone_max_dwell = [0.0 for _ in self.zones]

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.6)
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(zone=z, thickness=2, text_scale=0.6, opacity=0.15)
            for z in self.zones
        ]

    def _log_device(self):
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except ImportError:
            cuda_ok = False

        if self.device is not None:
            chosen = self.device
        elif cuda_ok:
            chosen = "cuda:0 (auto-selected)"
        else:
            chosen = "cpu (auto-selected — CUDA not available to torch)"

        print(f"[INFO] torch.cuda.is_available() = {cuda_ok}")
        print(f"[INFO] Running inference on: {chosen}")
        if not cuda_ok:
            print("[INFO] If you have an NVIDIA GPU and expected CUDA here, "
                  "your installed torch is very likely the CPU-only build. "
                  "See README section 'Using Your NVIDIA GPU' to fix this.")

        resolved_is_cuda = cuda_ok and (self.device is None or "cuda" in str(self.device))
        if self.fp16 and not resolved_is_cuda:
            print("[INFO] fp16 (half-precision) was requested but no CUDA GPU is in use — "
                  "it provides little/no benefit on CPU. Running anyway, but this is likely a no-op.")
        elif self.fp16:
            print("[INFO] fp16 (half-precision) inference enabled.")

    # ------------------------------------------------------------------ #
    def _setup_line_and_zones(self, line, zones):
        if line is None and self.frame_w and self.frame_h:
            y = self.frame_h // 2
            line = [[0, y], [self.frame_w, y]]
        if line is not None:
            p1, p2 = line
            # ACCURACY: default triggering_anchors checks all 4 box corners,
            # which counts a crossing as soon as ANY corner touches the line
            # (e.g. the edge of a person's bounding box), before the object
            # has actually crossed. Using CENTER matches how the trajectory/
            # direction logic already tracks objects (by centroid), so a
            # crossing is only counted once the object's actual center
            # passes the line — much closer to human intuition of "crossed".
            self.line_zone = sv.LineZone(
                start=sv.Point(x=p1[0], y=p1[1]), end=sv.Point(x=p2[0], y=p2[1]),
                triggering_anchors=(sv.Position.CENTER,),
            )
        else:
            self.line_zone = None

        self.zones = []
        for poly in (zones or []):
            self.zones.append(sv.PolygonZone(polygon=np.array(poly, dtype=np.int32)))

    # ------------------------------------------------------------------ #
    def _update_trajectories(self, detections):
        for xyxy, track_id, class_id in zip(detections.xyxy, detections.tracker_id, detections.class_id):
            if track_id is None:
                continue
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)
            self.trajectories[track_id].append((cx, cy))
            self.all_track_ids.add(track_id)
            self.track_classes[track_id] = self.model.names[int(class_id)]

    def _draw_trajectories(self, frame):
        for track_id, pts in self.trajectories.items():
            if len(pts) < 2:
                continue
            color = self._id_color(track_id)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)
            cv2.circle(frame, pts[-1], 3, color, -1)

    @staticmethod
    def _id_color(track_id):
        rng = np.random.default_rng(int(track_id) * 999)
        return tuple(int(c) for c in rng.integers(60, 255, size=3))

    # ------------------------------------------------------------------ #
    # Movement direction
    # ------------------------------------------------------------------ #
    _COMPASS = [
        ("E", "\u2192"), ("NE", "\u2197"), ("N", "\u2191"), ("NW", "\u2196"),
        ("W", "\u2190"), ("SW", "\u2199"), ("S", "\u2193"), ("SE", "\u2198"),
    ]

    def _smoothed_velocity(self, track_id):
        """Compares the average position of the first few points vs the
        last few points in this track's trajectory window, instead of a
        single endpoint on each side. A single noisy detection (tracker
        jitter) then only shifts a K-point average by ~1/K of its error,
        instead of dominating the result the way a raw two-point endpoint
        comparison would. Returns (dx, dy, total_displacement) or None."""
        pts = self.trajectories.get(track_id)
        if pts is None or len(pts) < 2:
            return None
        pts_arr = np.array(pts, dtype=float)
        k = max(1, min(5, len(pts_arr) // 2))
        start_avg = pts_arr[:k].mean(axis=0)
        end_avg = pts_arr[-k:].mean(axis=0)
        dx, dy = end_avg[0] - start_avg[0], end_avg[1] - start_avg[1]
        total_displacement = (dx ** 2 + dy ** 2) ** 0.5
        return dx, dy, total_displacement

    def _compute_direction(self, track_id):
        """Returns (arrow_symbol, compass_label) or (None, None) if the
        object hasn't moved enough overall to trust a heading (avoids
        jitter on a stationary object being reported as "moving")."""
        result = self._smoothed_velocity(track_id)
        if result is None:
            return None, None
        dx, dy, total_displacement = result
        if total_displacement < self.direction_min_displacement:
            return None, None
        # screen space: y grows downward, so flip dy for a natural compass feel
        angle = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360
        idx = int(round(angle / 45)) % 8
        label, symbol = self._COMPASS[idx]
        return symbol, label

    def _draw_direction_arrows(self, frame):
        for track_id, pts in self.trajectories.items():
            result = self._smoothed_velocity(track_id)
            if result is None:
                continue
            dx, dy, total_displacement = result
            if total_displacement < self.direction_min_displacement:
                continue
            norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            arrow_len = 28
            x1, y1 = pts[-1]
            ex = int(x1 + dx / norm * arrow_len)
            ey = int(y1 + dy / norm * arrow_len)
            color = self._id_color(track_id)
            cv2.arrowedLine(frame, (x1, y1), (ex, ey), color, 2, tipLength=0.4)

    # ------------------------------------------------------------------ #
    # Dwell time (per zone, per track)
    # ------------------------------------------------------------------ #
    def _update_zone_dwell(self, detections, zone_masks, now):
        """zone_masks: list (one per zone) of boolean arrays aligned with
        `detections`, from PolygonZone.trigger(). Returns a dict
        {track_id: {zone_index: dwell_seconds}} for objects currently
        inside at least one zone, for labeling."""
        current_dwell = defaultdict(dict)
        for zi, mask in enumerate(zone_masks):
            inside_ids = set()
            for j, is_inside in enumerate(mask):
                if not is_inside:
                    continue
                track_id = detections.tracker_id[j]
                inside_ids.add(track_id)
                start = self.zone_dwell_start[zi].setdefault(track_id, now)
                dwell = now - start
                current_dwell[track_id][zi] = dwell
                self.zone_max_dwell[zi] = max(self.zone_max_dwell[zi], dwell)
            # anyone previously inside this zone but not anymore: reset their timer
            for track_id in list(self.zone_dwell_start[zi].keys()):
                if track_id not in inside_ids:
                    del self.zone_dwell_start[zi][track_id]
        return current_dwell

    # Palette (BGR, since OpenCV) — dark charcoal background with accent colors
    _BG = (26, 24, 22)
    _CARD_BG = (46, 42, 38)
    _CARD_BORDER = (78, 72, 66)
    _LABEL_GRAY = (150, 145, 140)
    _WHITE = (240, 240, 240)
    _CYAN = (235, 206, 135)     # value accent: general counts
    _GREEN = (108, 201, 124)    # value accent: healthy / good
    _AMBER = (77, 179, 232)     # value accent: caution
    _RED = (86, 86, 231)        # value accent: poor / live dot

    def _perf_color(self, value, good, ok):
        """Green if >= good, amber if >= ok, red otherwise."""
        if value >= good:
            return self._GREEN
        if value >= ok:
            return self._AMBER
        return self._RED

    def _fit_text_scale(self, text, base_scale, thickness, max_width, min_scale=0.28):
        """Shrinks font scale in small steps until text fits max_width, so
        long values (e.g. triple-digit counts on a long session, or a
        narrow frame) never overflow a card's border."""
        scale = base_scale
        while scale > min_scale:
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            if tw <= max_width:
                return scale
            scale -= 0.04
        return min_scale

    def _draw_stat_card(self, dash, x0, y0, card_w, card_h, label, value, sub, value_color):
        x1, y1 = x0 + card_w, y0 + card_h
        cv2.rectangle(dash, (x0, y0), (x1, y1), self._CARD_BG, -1)
        cv2.rectangle(dash, (x0, y0), (x1, y1), self._CARD_BORDER, 1, cv2.LINE_AA)
        avail = card_w - 20

        label_scale = self._fit_text_scale(label, 0.42, 1, avail)
        cv2.putText(dash, label, (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    label_scale, self._LABEL_GRAY, 1, cv2.LINE_AA)

        value_scale = self._fit_text_scale(value, 0.92, 2, avail)
        cv2.putText(dash, value, (x0 + 10, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX,
                    value_scale, value_color, 2, cv2.LINE_AA)

        if sub:
            sub_scale = self._fit_text_scale(sub, 0.38, 1, avail)
            cv2.putText(dash, sub, (x0 + 10, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX,
                        sub_scale, self._LABEL_GRAY, 1, cv2.LINE_AA)

    def _draw_dashboard(self, frame, current_objects, avg_conf, fps, proc_ms):
        h, w = frame.shape[:2]
        in_count = self.line_zone.in_count if self.line_zone else 0
        out_count = self.line_zone.out_count if self.line_zone else 0

        dash = np.full((DASHBOARD_HEIGHT, w, 3), self._BG, dtype=np.uint8)

        # top accent bar + live indicator
        cv2.line(dash, (0, 0), (w, 0), self._CYAN, 3)
        cv2.circle(dash, (16, 18), 5, self._RED, -1, cv2.LINE_AA)
        cv2.putText(dash, "LIVE ANALYTICS", (28, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, self._WHITE, 1, cv2.LINE_AA)

        # row of stat cards
        cards = [
            ("OBJECTS", str(current_objects), f"of {len(self.all_track_ids)} total", self._CYAN),
            ("IN / OUT", f"{in_count} / {out_count}", "line crossings", self._CYAN),
            ("CONFIDENCE", f"{avg_conf * 100:.0f}%",
             "avg detection", self._perf_color(avg_conf, 0.6, 0.4)),
            ("FPS", f"{fps:.1f}", f"{proc_ms:.0f} ms/frame",
             self._perf_color(fps, 15, 5)),
        ]
        margin = 10
        card_top = 36
        card_h = 90
        n = len(cards)
        card_w = max(60, (w - margin * (n + 1)) // n)
        for i, (label, value, sub, color) in enumerate(cards):
            x0 = margin + i * (card_w + margin)
            self._draw_stat_card(dash, x0, card_top, card_w, card_h, label, value, sub, color)

        # secondary info strip: zones + longest dwell
        zone_counts = [int(z.current_count) for z in self.zones]
        zone_txt = "  ".join(f"Zone{i+1}: {c}" for i, c in enumerate(zone_counts)) or "No ROI zones"
        if self.zones:
            dwell_txt = "Longest Dwell:  " + "  ".join(
                f"Zone{i+1} {d:.1f}s" for i, d in enumerate(self.zone_max_dwell)
            )
        else:
            dwell_txt = "Longest Dwell:  --"
        strip_y = card_top + card_h + 22
        cv2.putText(dash, f"{zone_txt}    |    {dwell_txt}", (margin, strip_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, self._LABEL_GRAY, 1, cv2.LINE_AA)

        return np.vstack([dash, frame])

    # ------------------------------------------------------------------ #
    def process_frame(self, frame, draw_dashboard=True):
        """Run detection+tracking+counting on one frame. Returns (annotated_frame, stats)."""
        t0 = time.time()

        track_kwargs = dict(
            persist=True, tracker=self.tracker_yaml, conf=self.conf,
            imgsz=self.imgsz, device=self.device, verbose=False,
        )
        if self.fp16:
            # ultralytics renamed half= to quantize='fp16' in newer versions;
            # try the new name first, fall back to the old one if this
            # installed version doesn't recognize it, then remember which
            # one worked so we don't pay the exception cost every frame.
            fp16_kwarg = getattr(self, "_fp16_kwarg_name", "quantize")
            track_kwargs[fp16_kwarg] = "fp16" if fp16_kwarg == "quantize" else True
        try:
            results = self.model.track(frame, **track_kwargs)[0]
        except TypeError:
            if self.fp16 and track_kwargs.get("quantize") is not None:
                track_kwargs.pop("quantize")
                track_kwargs["half"] = True
                self._fp16_kwarg_name = "half"
                results = self.model.track(frame, **track_kwargs)[0]
            else:
                raise
        detections = sv.Detections.from_ultralytics(results)
        if detections.tracker_id is None:
            # No confirmed track IDs this frame (common for a newly-appeared
            # object on its first frame or two). Shrink ALL fields to empty
            # together so nothing downstream sees mismatched lengths, and
            # explicitly force tracker_id to an empty array (not None) so
            # later code that iterates/zips over it doesn't crash either.
            detections = detections[np.zeros(len(detections), dtype=bool)]
            detections.tracker_id = np.array([], dtype=int)

        if len(detections) > 0:
            self.confidences.extend(detections.confidence.tolist())

        self._update_trajectories(detections)
        if self.line_zone is not None:
            self.line_zone.trigger(detections)

        zone_masks = [zone.trigger(detections) for zone in self.zones]
        now = time.time()
        current_dwell = self._update_zone_dwell(detections, zone_masks, now) if self.zones else {}

        labels = []
        for j, (tid, cid, conf) in enumerate(
            zip(detections.tracker_id, detections.class_id, detections.confidence)
        ):
            label = f"#{tid} {self.model.names[int(cid)]} {conf:.2f}"
            symbol, _ = self._compute_direction(tid)
            if symbol:
                label += f" {symbol}"
            zone_times = current_dwell.get(tid)
            if zone_times:
                # show the zone this object has dwelled in longest, this frame
                zi, secs = max(zone_times.items(), key=lambda kv: kv[1])
                label += f" Z{zi + 1}:{secs:.1f}s"
            labels.append(label)

        annotated = self.box_annotator.annotate(frame.copy(), detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels)
        self._draw_trajectories(annotated)
        self._draw_direction_arrows(annotated)
        if self.line_zone is not None:
            self.line_annotator.annotate(annotated, self.line_zone)
        for zone, zone_ann in zip(self.zones, self.zone_annotators):
            annotated = zone_ann.annotate(annotated)

        proc_ms = (time.time() - t0) * 1000
        fps = 1000 / proc_ms if proc_ms > 0 else 0
        self.fps_history.append(fps)
        avg_fps = float(np.mean(self.fps_history))
        avg_conf = float(np.mean(self.confidences)) if self.confidences else 0.0

        self.frame_count += 1

        stats = {
            "current_objects": len(detections),
            "total_unique_objects": len(self.all_track_ids),
            "objects_in": self.line_zone.in_count if self.line_zone else 0,
            "objects_out": self.line_zone.out_count if self.line_zone else 0,
            "zone_counts": [int(z.current_count) for z in self.zones],
            "zone_max_dwell_seconds": [round(d, 1) for d in self.zone_max_dwell],
            "avg_confidence": avg_conf,
            "fps": avg_fps,
            "processing_ms": proc_ms,
            "frame_count": self.frame_count,
        }

        output_frame = self._draw_dashboard(annotated, len(detections), avg_conf, avg_fps, proc_ms) \
            if draw_dashboard else annotated

        return output_frame, stats

    # ------------------------------------------------------------------ #
    def build_report(self, source_name, model_path):
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        return {
            "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "source": str(source_name),
            "model": model_path,
            "tracker": self.tracker_yaml,
            "confidence_threshold": self.conf,
            "frames_processed": self.frame_count,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(avg_fps, 2),
            "total_unique_objects": len(self.all_track_ids),
            "objects_in": self.line_zone.in_count if self.line_zone else 0,
            "objects_out": self.line_zone.out_count if self.line_zone else 0,
            "zone_final_counts": [int(z.current_count) for z in self.zones],
            "zone_longest_dwell_seconds": [round(d, 1) for d in self.zone_max_dwell],
            "average_confidence": round(float(np.mean(self.confidences)), 3) if self.confidences else None,
        }
