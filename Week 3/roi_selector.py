"""
ROI / Counting-Line Selector
Week 3 - Intelligent Video Analytics Platform

Lets you click points on a frame from your video to define:
  1. A virtual counting LINE (2 points)
  2. Any number of ROI ZONES (polygons, 3+ points each, close with 'n')

Controls
--------
Left click   : add a point
'n'          : finish the current shape and start a new one
'z'          : undo last shape
's'          : save config and quit
'q'          : quit without saving

Usage
-----
python roi_selector.py --source input.mp4 --out configs/my_video_config.json
python roi_selector.py --source 0 --out configs/webcam_config.json
"""

import argparse
import json
import os

import cv2

points_current = []
shapes = []          # list of dicts: {"type": "line"/"zone", "points": [...]}
line_defined = False


def mouse_callback(event, x, y, flags, param):
    global points_current
    if event == cv2.EVENT_LBUTTONDOWN:
        points_current.append([x, y])


def draw_overlay(frame):
    display = frame.copy()

    for shape in shapes:
        pts = shape["points"]
        color = (0, 255, 255) if shape["type"] == "line" else (0, 255, 0)
        if shape["type"] == "line":
            cv2.line(display, tuple(pts[0]), tuple(pts[1]), color, 2)
        else:
            cv2.polylines(display, [cv2_points(pts)], True, color, 2)

    for i, pt in enumerate(points_current):
        cv2.circle(display, tuple(pt), 4, (0, 0, 255), -1)
        if i > 0:
            cv2.line(display, tuple(points_current[i - 1]), tuple(pt), (0, 0, 255), 1)

    help_text = [
        "Click to add points | 'n' finish shape | 'z' undo | 's' save | 'q' quit",
        f"Line defined: {line_defined}   Zones defined: {sum(1 for s in shapes if s['type']=='zone')}",
    ]
    for i, t in enumerate(help_text):
        cv2.putText(display, t, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return display


def cv2_points(pts):
    import numpy as np
    return np.array(pts, dtype=int)


def main():
    global points_current, line_defined

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--out", default="configs/config.json")
    args = parser.parse_args()

    try:
        source_val = int(args.source)
    except ValueError:
        source_val = args.source

    cap = cv2.VideoCapture(source_val)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from source: {args.source}")

    cv2.namedWindow("ROI Selector")
    cv2.setMouseCallback("ROI Selector", mouse_callback)

    while True:
        display = draw_overlay(frame)
        cv2.imshow("ROI Selector", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("n"):
            if len(points_current) == 2 and not line_defined:
                shapes.append({"type": "line", "points": points_current})
                line_defined = True
            elif len(points_current) >= 3:
                shapes.append({"type": "zone", "points": points_current})
            else:
                print("[WARN] Need exactly 2 points for the line, or 3+ for a zone.")
            points_current = []

        elif key == ord("z"):
            if shapes:
                removed = shapes.pop()
                if removed["type"] == "line":
                    line_defined = False
            points_current = []

        elif key == ord("s"):
            config = {
                "line": next((s["points"] for s in shapes if s["type"] == "line"), None),
                "zones": [s["points"] for s in shapes if s["type"] == "zone"],
            }
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(config, f, indent=2)
            print(f"[INFO] Config saved to {args.out}")
            break

        elif key == ord("q"):
            print("[INFO] Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
