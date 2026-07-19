"""
Desktop GUI — Intelligent Video Analytics Platform
Week 3 - AI Summer Fellowship

A native desktop frontend (PySide6 / Qt) that renders frames directly in a
Qt window instead of through a browser. No HTTP/WebSocket round-trip, no
image re-encoding per frame — just the same kind of direct rendering
cv2.imshow uses, but with proper buttons, sliders, and ROI/line drawing.

Uses the exact same analytics_core.AnalyticsCore engine as app.py and
streamlit_app.py, so results are identical across all three frontends.

Run with:
    python desktop_app.py
"""

import json
import os
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QCheckBox, QRadioButton, QButtonGroup, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGroupBox, QMessageBox, QLineEdit, QTextEdit,
)

from analytics_core import AnalyticsCore, open_capture, create_video_writer

DARK_STYLESHEET = """
QMainWindow, QWidget { background-color: #0e1117; color: #e6e6e6; }
QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 10px;
            font-weight: bold; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background-color: #262730; border: 1px solid #444; border-radius: 4px;
              padding: 6px 10px; color: #e6e6e6; }
QPushButton:hover { background-color: #333644; }
QPushButton:disabled { color: #666; }
QPushButton#startBtn { background-color: #1f6f43; }
QPushButton#stopBtn { background-color: #7a2323; }
QComboBox, QDoubleSpinBox, QLineEdit { background-color: #1c1e26; border: 1px solid #444;
              border-radius: 4px; padding: 4px; color: #e6e6e6; }
QLabel#videoLabel { background-color: #000; border: 1px solid #333; }
QLabel#statusLabel { color: #7CFC7C; font-family: Consolas, monospace; }
QTextEdit { background-color: #1c1e26; border: 1px solid #444; color: #cfcfcf;
            font-family: Consolas, monospace; }
"""


class VideoLabel(QLabel):
    """QLabel that reports clicks in ORIGINAL frame coordinates (not widget
    pixel coordinates), accounting for scaling/letterboxing of the pixmap."""
    clicked = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setObjectName("videoLabel")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self.frame_size = None  # (w, h) of the ORIGINAL frame currently shown

    def mousePressEvent(self, event):
        if self.frame_size is None or self.pixmap() is None:
            return
        pm = self.pixmap()
        label_w, label_h = self.width(), self.height()
        pm_w, pm_h = pm.width(), pm.height()
        # pixmap is centered (Qt.AlignCenter) inside the label
        offset_x = (label_w - pm_w) / 2
        offset_y = (label_h - pm_h) / 2
        x = event.position().x() - offset_x
        y = event.position().y() - offset_y
        if not (0 <= x <= pm_w and 0 <= y <= pm_h):
            return  # click outside the actual image area
        scale_x = self.frame_size[0] / pm_w
        scale_y = self.frame_size[1] / pm_h
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)
        self.clicked.emit(orig_x, orig_y)


def to_qpixmap(frame_bgr, target_w, target_h):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class AnalyticsWorker(QThread):
    frame_ready = Signal(np.ndarray)
    finished_run = Signal(dict)
    error = Signal(str)

    def __init__(self, source, model_path, tracker, conf, imgsz, device,
                 line, zones, record, fp16=False):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self.tracker = tracker
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.line = line
        self.zones = zones
        self.record = record
        self.fp16 = fp16
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        cap = open_capture(self.source)
        if not cap.isOpened():
            self.error.emit(f"Could not open video source: {self.source}")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

        try:
            core = AnalyticsCore(
                model_path=self.model_path, tracker=self.tracker, conf=self.conf,
                frame_w=frame_w, frame_h=frame_h, line=self.line, zones=self.zones,
                imgsz=self.imgsz, device=self.device, fp16=self.fp16,
            )
        except Exception as e:
            self.error.emit(f"Failed to load model: {e}")
            cap.release()
            return

        writer = None
        out_path = None
        if self.record:
            from analytics_core import DASHBOARD_HEIGHT
            os.makedirs("output", exist_ok=True)
            out_path = os.path.join("output", f"annotated_{int(time.time())}.mp4")
            writer, out_path = create_video_writer(out_path, src_fps, (frame_w, frame_h + DASHBOARD_HEIGHT))

        while self._running:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, stats = core.process_frame(frame)
            if writer is not None:
                writer.write(annotated)
            self.frame_ready.emit(annotated)

        cap.release()
        if writer is not None:
            writer.release()

        report = core.build_report(self.source, self.model_path)
        report["output_video"] = out_path
        os.makedirs("reports", exist_ok=True)
        report_path = os.path.join("reports", f"session_report_{report['run_timestamp']}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        report["report_path"] = report_path
        self.finished_run.emit(report)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Video Analytics Platform — Desktop")
        self.resize(1200, 750)

        self.worker = None
        self.line_pts = None
        self.zone_points_current = []
        self.zones = []
        self.draw_mode = None  # None / "line" / "zone"
        self.reference_frame = None  # BGR frame used for ROI drawing preview

        self._build_ui()

    # ------------------------------------------------------------------ #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- left control panel ----
        panel = QVBoxLayout()
        panel.setSpacing(10)

        src_box = QGroupBox("Video Source")
        src_layout = QVBoxLayout(src_box)
        self.radio_webcam = QRadioButton("Webcam (device 0)")
        self.radio_file = QRadioButton("Video File")
        self.radio_webcam.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.radio_webcam)
        group.addButton(self.radio_file)
        file_row = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("No file selected")
        self.file_path_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_video)
        file_row.addWidget(self.file_path_edit)
        file_row.addWidget(browse_btn)
        src_layout.addWidget(self.radio_webcam)
        src_layout.addWidget(self.radio_file)
        src_layout.addLayout(file_row)
        panel.addWidget(src_box)

        model_box = QGroupBox("Model && Tracking")
        model_layout = QVBoxLayout(model_box)
        model_row = QHBoxLayout()
        self.model_path_edit = QLineEdit("yolov8n.pt")
        model_browse_btn = QPushButton("Browse...")
        model_browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_path_edit)
        model_row.addWidget(model_browse_btn)
        model_layout.addLayout(model_row)

        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["bytetrack", "botsort"])
        model_layout.addWidget(QLabel("Tracker"))
        model_layout.addWidget(self.tracker_combo)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.30)
        model_layout.addWidget(QLabel("Confidence Threshold"))
        model_layout.addWidget(self.conf_spin)

        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["256 (fastest)", "320 (fast)", "480 (balanced)", "640 (most accurate)"])
        self.imgsz_combo.setCurrentIndex(2)
        model_layout.addWidget(QLabel("Inference Size"))
        model_layout.addWidget(self.imgsz_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CPU", "GPU (CUDA)"])
        model_layout.addWidget(QLabel("Compute Device"))
        model_layout.addWidget(self.device_combo)

        self.fp16_checkbox = QCheckBox("Half-precision (fp16) inference")
        self.fp16_checkbox.setToolTip("Real speedup on a CUDA GPU with tensor cores; no benefit on CPU.")
        model_layout.addWidget(self.fp16_checkbox)

        panel.addWidget(model_box)

        roi_box = QGroupBox("Counting Line / ROI Zones")
        roi_layout = QVBoxLayout(roi_box)
        self.load_ref_btn = QPushButton("1. Load Reference Frame")
        self.load_ref_btn.clicked.connect(self._load_reference_frame)
        roi_layout.addWidget(self.load_ref_btn)

        draw_row = QHBoxLayout()
        self.draw_line_btn = QPushButton("Draw Line")
        self.draw_line_btn.clicked.connect(lambda: self._set_draw_mode("line"))
        self.draw_zone_btn = QPushButton("Draw Zone")
        self.draw_zone_btn.clicked.connect(lambda: self._set_draw_mode("zone"))
        draw_row.addWidget(self.draw_line_btn)
        draw_row.addWidget(self.draw_zone_btn)
        roi_layout.addLayout(draw_row)

        finish_row = QHBoxLayout()
        finish_zone_btn = QPushButton("Finish Zone")
        finish_zone_btn.clicked.connect(self._finish_zone)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_roi)
        finish_row.addWidget(finish_zone_btn)
        finish_row.addWidget(clear_btn)
        roi_layout.addLayout(finish_row)

        self.roi_status_label = QLabel("Line: none | Zones: 0")
        roi_layout.addWidget(self.roi_status_label)
        panel.addWidget(roi_box)

        opts_box = QGroupBox("Output")
        opts_layout = QVBoxLayout(opts_box)
        self.record_checkbox = QCheckBox("Save annotated output video")
        self.record_checkbox.setChecked(True)
        opts_layout.addWidget(self.record_checkbox)
        panel.addWidget(opts_box)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("\u25b6  Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("\u25a0  Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        panel.addLayout(btn_row)

        self.status_label = QLabel("Idle.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        panel.addWidget(self.status_label)

        self.report_box = QTextEdit()
        self.report_box.setReadOnly(True)
        self.report_box.setMaximumHeight(160)
        panel.addWidget(self.report_box)

        panel.addStretch()

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(340)
        root.addWidget(panel_widget)

        # ---- right video panel ----
        self.video_label = VideoLabel()
        self.video_label.clicked.connect(self._on_video_click)
        root.addWidget(self.video_label, stretch=1)

    # ------------------------------------------------------------------ #
    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.file_path_edit.setText(path)
            self.radio_file.setChecked(True)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Weights", "", "Weights (*.pt)")
        if path:
            self.model_path_edit.setText(path)

    def _current_source(self):
        if self.radio_webcam.isChecked():
            return "0"
        return self.file_path_edit.text() or "0"

    # ---- ROI drawing ---- #
    def _load_reference_frame(self):
        source = self._current_source()
        cap = open_capture(source)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            QMessageBox.warning(self, "Error", f"Could not read a frame from source: {source}")
            return
        self.reference_frame = frame
        self._clear_roi(redraw_only=True)
        self._refresh_preview()

    def _set_draw_mode(self, mode):
        if self.reference_frame is None:
            QMessageBox.information(self, "Load a frame first",
                                      "Click '1. Load Reference Frame' first so you have "
                                      "something to draw the line/zone on.")
            return
        self.draw_mode = mode
        self.zone_points_current = []
        self.status_label.setText(f"Draw mode: {mode}. Click on the video to add points.")

    def _finish_zone(self):
        if self.draw_mode == "zone" and len(self.zone_points_current) >= 3:
            self.zones.append(self.zone_points_current)
        self.zone_points_current = []
        self.draw_mode = None
        self._update_roi_status()
        self._refresh_preview()

    def _clear_roi(self, redraw_only=False):
        self.line_pts = None
        self.zones = []
        self.zone_points_current = []
        self.draw_mode = None
        self._update_roi_status()
        if not redraw_only:
            self._refresh_preview()

    def _update_roi_status(self):
        self.roi_status_label.setText(f"Line: {'set' if self.line_pts else 'none'} | Zones: {len(self.zones)}")

    def _on_video_click(self, x, y):
        if self.draw_mode == "line":
            if self.line_pts is None:
                self.line_pts = [[x, y]]
            elif len(self.line_pts) == 1:
                self.line_pts.append([x, y])
                self.draw_mode = None
                self._update_roi_status()
        elif self.draw_mode == "zone":
            self.zone_points_current.append([x, y])
        else:
            return
        self._refresh_preview()

    def _refresh_preview(self):
        if self.reference_frame is None:
            return
        preview = self.reference_frame.copy()
        if self.line_pts and len(self.line_pts) == 2:
            cv2.line(preview, tuple(self.line_pts[0]), tuple(self.line_pts[1]), (0, 255, 255), 2)
        for zone in self.zones:
            pts = np.array(zone, dtype=np.int32)
            cv2.polylines(preview, [pts], True, (0, 255, 0), 2)
        if self.zone_points_current:
            for i, pt in enumerate(self.zone_points_current):
                cv2.circle(preview, tuple(pt), 4, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(preview, tuple(self.zone_points_current[i-1]), tuple(pt), (0, 0, 255), 1)
        self._show_frame(preview)

    # ------------------------------------------------------------------ #
    def _show_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self.video_label.frame_size = (w, h)
        pixmap = to_qpixmap(frame_bgr, self.video_label.width(), self.video_label.height())
        self.video_label.setPixmap(pixmap)

    # ------------------------------------------------------------------ #
    def _start(self):
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Error", "Please specify a model path.")
            return

        imgsz = int(self.imgsz_combo.currentText().split()[0])
        device_map = {"Auto": "auto", "CPU": "cpu", "GPU (CUDA)": "cuda:0"}
        device = device_map[self.device_combo.currentText()]

        self.worker = AnalyticsWorker(
            source=self._current_source(),
            model_path=model_path,
            tracker=self.tracker_combo.currentText(),
            conf=self.conf_spin.value(),
            imgsz=imgsz,
            device=device,
            line=self.line_pts,
            zones=self.zones,
            record=self.record_checkbox.isChecked(),
            fp16=self.fp16_checkbox.isChecked(),
        )
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.finished_run.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.load_ref_btn.setEnabled(False)  # avoid a second camera handle while worker holds it
        self.status_label.setText("Running...")
        self.report_box.clear()

    def _stop(self):
        if self.worker is not None:
            self.worker.stop()
        self.stop_btn.setEnabled(False)

    def _on_finished(self, report):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.load_ref_btn.setEnabled(True)
        self.status_label.setText("Stopped. Session report ready below.")
        self.report_box.setPlainText(json.dumps(report, indent=2))

    def _on_error(self, message):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.load_ref_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", message)
        self.status_label.setText(f"Error: {message}")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
