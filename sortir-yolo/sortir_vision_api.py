"""API Langsung OpenCV Vision untuk Tim Robotik (100% Standalone tanpa Middleware).

Dibuat khusus agar rekan tim robotik di laptop Windows dapat langsung mengakses data
koordinat objek (Koin / Tutup Botol) yang sudah terkalibrasi secara presisi milimeter (mm),
termasuk perhitungan posisi kamera & robot yang saling berseberangan.

PILIHAN INTEGRASI UNTUK TIM ROBOTIK:

PILIHAN A (Import Langsung ke Skrip Robot):
    from sortir_vision_api import SortirVisionAPI

    vision = SortirVisionAPI(source=0)  # Membuka kamera & memuat model YOLO
    target = vision.get_target()        # Ambil objek terbaik di conveyor
    if target:
        print(f"Ambil: {target.label} di X={target.x_mm}, Y={target.y_mm}, Z={target.z_mm}")
        print(f"Taruh di wadah: X={target.drop_x_mm}, Y={target.drop_y_mm}")

PILIHAN B (Baca File JSON IPC Real-time):
    Jika OpenCV dijalankan terpisah via 'python 05_detect.py' untuk memantau video:
    from sortir_vision_api import read_target_json

    data = read_target_json()
    if data and data.get("status") == "TARGET_READY":
        target = data["target"]
        # target["robot_mm"]["x"], target["robot_mm"]["y"], target["robot_mm"]["z"]
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2

from src.camera import Camera, parse_source
from src.config import DEFAULT_CALIB_PATH, DEFAULT_TARGET_JSON, DEFAULT_WEIGHTS, OUTPUT_DIR
from src.coordinate_transform import (
    RobotCoordinateTransformer,
    RobotTarget,
    choose_best_target,
    export_target_json,
)
from src.detector import Detector

TARGET_JSON_PATH = DEFAULT_TARGET_JSON


def read_target_json(json_path: Path | str = TARGET_JSON_PATH) -> dict[str, Any] | None:
    """Membaca target terbaru yang diekspor OpenCV secara aman."""
    p = Path(json_path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class SortirVisionAPI:
    """Modul OpenCV Vision mandiri untuk integrasi langsung dengan kontroler robot."""

    def __init__(
        self,
        source: int | str = 0,
        weights: Path | str = DEFAULT_WEIGHTS,
        calib_path: Path | str = DEFAULT_CALIB_PATH,
        conf: float = 0.45,
        imgsz: int = 1280,
        width: int = 1280,
        height: int = 720,
        edge_margin: int = 15,
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.roi = roi

        # Inisialisasi YOLO Detector
        self.detector = Detector(
            weights=Path(weights),
            conf=conf,
            imgsz=imgsz,
            device="auto",
            smart_filter=True,
            track=True,
            edge_margin=edge_margin,
        )

        # Inisialisasi Transformasi Koordinat Robot (Berseberangan / Homography)
        self.transformer = RobotCoordinateTransformer(config_path=calib_path)

        # Inisialisasi Kamera
        parsed = parse_source(str(source))
        self.camera = Camera(parsed, width=width, height=height).open(auto_fallback=True)

    def read_frame(self) -> Any:
        """Membaca 1 frame dari kamera."""
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return None
        return frame

    def get_target(self) -> RobotTarget | None:
        """Memproses frame kamera terbaru dan mengembalikan RobotTarget terbaik untuk diambil."""
        frame = self.read_frame()
        if frame is None:
            return None

        h, w = frame.shape[:2]
        self.detector.roi = self.roi
        detections = self.detector.detect(frame, apply_smart_filter=True)

        if not detections:
            export_target_json(None, [], TARGET_JSON_PATH)
            return None

        # Konversi setiap deteksi piksel ke koordinat fisik robot (mm)
        robot_targets: list[RobotTarget] = []
        for d in detections:
            rt = self.transformer.to_robot_target(
                class_name=d.class_name,
                label=d.label,
                confidence=d.confidence,
                pixel_center=d.center,
                frame_width=w,
                frame_height=h,
                track_id=d.track_id,
                catatan=d.catatan,
            )
            robot_targets.append(rt)

        best = choose_best_target(robot_targets, roi=self.roi)
        # Ekspor ke JSON secara atomic agar sinkron dengan proses lain
        export_target_json(best, robot_targets, TARGET_JSON_PATH)
        return best

    def get_all_targets(self) -> list[RobotTarget]:
        """Mengembalikan seluruh objek yang terdeteksi di kamera dengan koordinat robot (mm)."""
        frame = self.read_frame()
        if frame is None:
            return []

        h, w = frame.shape[:2]
        self.detector.roi = self.roi
        detections = self.detector.detect(frame, apply_smart_filter=True)

        robot_targets: list[RobotTarget] = []
        for d in detections:
            rt = self.transformer.to_robot_target(
                class_name=d.class_name,
                label=d.label,
                confidence=d.confidence,
                pixel_center=d.center,
                frame_width=w,
                frame_height=h,
                track_id=d.track_id,
                catatan=d.catatan,
            )
            robot_targets.append(rt)

        best = choose_best_target(robot_targets, roi=self.roi)
        export_target_json(best, robot_targets, TARGET_JSON_PATH)
        return robot_targets

    def release(self) -> None:
        """Tutup koneksi kamera."""
        if self.camera:
            self.camera.release()
