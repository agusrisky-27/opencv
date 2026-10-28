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
        conf: float = 0.25,
        imgsz: int = 1280,
        width: int = 1280,
        height: int = 720,
        edge_margin: int = 0,
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

    def set_status(self, status: str, message: str = "") -> bool:
        """Memperbarui status sistem ke file JSON secara atomic (misal CONVEYOR_MOVING, WAITING_STATIONARY)."""
        return export_target_json(None, [], TARGET_JSON_PATH, status=status, message=message)

    def inspect_stationary_object(
        self,
        target_classes: list[str] | None = None,
        save_snapshot: bool = True,
        settle_delay: float = 0.6,
        flush_frames: int = 4,
        verify_motion: bool = True,
        max_motion_px: float = 3.5,
    ) -> RobotTarget | None:
        """Dipanggil saat konveyor BERHENTI (dipicu sensor Infrared / Proximity / Auto-Vision).

        Mengatasi inersia mekanis konveyor & getaran benda (Stop-and-Go Settling Delay):
        1. Menahan pengiriman data ke robot (status: 'WAITING_STATIONARY') selama benda deselerasi.
        2. Menunggu 'settle_delay' (default 0.6 detik, jeda optimal agar getaran & selip belt reda).
        3. Membuang buffer antrian frame kamera lama ('flush_frames') agar tidak membaca foto usang saat masih gerak.
        4. Verifikasi Zero-Motion ('verify_motion'): Membandingkan 2 frame berurutan. Jika pergeseran centroid
           <= max_motion_px (3.5 px), benda terbukti 100% diam total.
        5. Mengambil 1 snapshot foto tajam resolusi tinggi tanpa motion blur.
        6. Jika kelas target cocok: Kunci koordinat mm robot & ekspor 'TARGET_READY' ke JSON.
        7. Jika bukan kelas target: Ekspor 'NO_TARGET' dan kembalikan None agar konveyor langsung resume.
        """
        # Lapisan 1: Beritahu robot bahwa benda baru berhenti & sedang menunggu stabil
        export_target_json(
            None,
            [],
            TARGET_JSON_PATH,
            status="WAITING_STATIONARY",
            message=f"Menunggu inersia konveyor reda ({settle_delay:.2f}s)...",
        )

        # Lapisan 2: Jeda inersia fisik (0.5 - 0.8 detik)
        if settle_delay > 0:
            time.sleep(settle_delay)

        # Lapisan 3: Kuras buffer internal kamera (karena USB cam menyimpan 3-5 frame lama di hardware)
        for _ in range(max(1, flush_frames)):
            self.read_frame()

        frame = self.read_frame()
        if frame is None:
            export_target_json(None, [], TARGET_JSON_PATH, status="NO_TARGET", message="Kamera gagal membaca frame")
            return None

        # Lapisan 4: Verifikasi keheningan gerak (Zero-Motion Verification)
        if verify_motion:
            det1 = self.detector.detect(frame, apply_smart_filter=True)
            if det1:
                c1 = det1[0].center
                time.sleep(0.06)
                # Flush 1 frame dan baca frame verifikasi
                self.read_frame()
                frame_verify = self.read_frame()
                if frame_verify is not None:
                    det2 = self.detector.detect(frame_verify, apply_smart_filter=True)
                    if det2:
                        c2 = det2[0].center
                        shift = ((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2) ** 0.5
                        if shift > max_motion_px:
                            # Masih ada getaran/selip, beri toleransi tambahan 0.2 detik
                            time.sleep(0.2)
                            for _ in range(2):
                                self.read_frame()
                            frame = self.read_frame() or frame_verify
                        else:
                            frame = frame_verify

        h, w = frame.shape[:2]
        self.detector.roi = self.roi
        detections = self.detector.detect(frame, apply_smart_filter=True)

        if save_snapshot:
            snap_path = OUTPUT_DIR / "snapshot_ir_terakhir.jpg"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snap_path), frame)

        if not detections:
            export_target_json(None, [], TARGET_JSON_PATH, status="NO_TARGET", message="Tidak ada objek di zona ambil")
            return None

        robot_targets: list[RobotTarget] = []
        for d in detections:
            # Filter hanya kelas yang ingin diambil oleh robot
            if target_classes is not None and d.class_name not in target_classes:
                continue

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
        if best is not None:
            export_target_json(
                best,
                robot_targets,
                TARGET_JSON_PATH,
                status="TARGET_READY",
                message=f"Benda {best.label} diam terverifikasi. Koordinat terkunci.",
            )
        else:
            export_target_json(
                None,
                robot_targets,
                TARGET_JSON_PATH,
                status="NO_TARGET",
                message="Objek bukan target sortir yang ingin diambil.",
            )
        return best

    def release(self) -> None:
        """Tutup koneksi kamera."""
        if self.camera:
            self.camera.release()
