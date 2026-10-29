"""Transformasi Sistem Koordinat OpenCV Kamera ke Sistem Koordinat Robot RNV3.

Menangani kasus fisik nyata:
1. Posisi Kamera dan Robot BERSEBERANGAN (Facing Each Other):
   - Orientasi sumbu terbalik/berotasi 180° terhadap sudut pandang kamera.
   - Inversi sumbu X (kamera kanan = robot kiri).
   - Inversi/relokasi sumbu Y terhadap basis robot.
2. Kalibrasi Homography 4-Titik (Perspective Transform):
   - Mengatasi distorsi kemiringan sudut kamera (tilt angle / trapesium).
   - Memberikan akurasi milimeter (sub-milimeter) pada bidang kerja conveyor.
3. Ekspor Data Koordinat Real-time (Atomic JSON IPC):
   - Agar tim robotik dapat membaca koordinat langsung di laptop Windows tanpa middleware.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DROP_COORDINATES


@dataclass
class RobotTarget:
    """Struktur data target objek lengkap dengan koordinat robot fisik (mm)."""
    class_name: str
    label: str
    confidence: float
    pixel_center: tuple[int, int]
    x_mm: float
    y_mm: float
    z_mm: float
    drop_x_mm: float
    drop_y_mm: float
    drop_z_mm: float
    track_id: int | None = None
    catatan: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "pixel_center": {"u": self.pixel_center[0], "v": self.pixel_center[1]},
            "robot_mm": {"x": round(self.x_mm, 2), "y": round(self.y_mm, 2), "z": round(self.z_mm, 2)},
            "drop_mm": {"x": round(self.drop_x_mm, 2), "y": round(self.drop_y_mm, 2), "z": round(self.drop_z_mm, 2)},
            "track_id": self.track_id,
            "catatan": self.catatan,
        }


class RobotCoordinateTransformer:
    """Mengonversi titik piksel (u, v) ke koordinat fisik robot (X, Y, Z) dalam milimeter (mm)."""

    def __init__(
        self,
        config_path: Path | str | None = None,
        berseberangan: bool = True,
        flip_x: bool = True,
        flip_y: bool = False,
        scale_x_mm_per_px: float = 0.35,
        scale_y_mm_per_px: float = 0.35,
        origin_robot_x_mm: float = 0.0,
        origin_robot_y_mm: float = 200.0,
        z_heights: dict[str, float] | None = None,
    ) -> None:
        self.config_path = Path(config_path) if config_path else None
        self.berseberangan = berseberangan
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.scale_x = scale_x_mm_per_px
        self.scale_y = scale_y_mm_per_px
        self.origin_x = origin_robot_x_mm
        self.origin_y = origin_robot_y_mm
        self.z_heights = z_heights or {"koin": 2.0, "tutup_botol": 12.0, "charger": 28.0}
        self.drop_coords = dict(DROP_COORDINATES)
        self.homography_matrix: np.ndarray | None = None
        self.calibration_points: list[dict[str, Any]] = []

        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)

    def load_config(self, path: Path | str) -> bool:
        """Memuat parameter kalibrasi dari file JSON."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.berseberangan = data.get("berseberangan", self.berseberangan)
            self.flip_x = data.get("flip_x", self.flip_x)
            self.flip_y = data.get("flip_y", self.flip_y)
            self.scale_x = float(data.get("scale_x_mm_per_px", self.scale_x))
            self.scale_y = float(data.get("scale_y_mm_per_px", self.scale_y))
            self.origin_x = float(data.get("origin_robot_x_mm", self.origin_x))
            self.origin_y = float(data.get("origin_robot_y_mm", self.origin_y))
            if "z_heights" in data and isinstance(data["z_heights"], dict):
                self.z_heights.update({k: float(v) for k, v in data["z_heights"].items()})
            if "drop_coordinates" in data and isinstance(data["drop_coordinates"], dict):
                self.drop_coords.update(data["drop_coordinates"])
            if "calibration_points" in data:
                self.calibration_points = data["calibration_points"]
            if "homography_matrix" in data and data["homography_matrix"]:
                self.homography_matrix = np.array(data["homography_matrix"], dtype=np.float64)
            return True
        except Exception as exc:
            print(f"[Peringatan] Gagal memuat kalibrasi dari {p}: {exc}")
            return False

    def save_config(self, path: Path | str | None = None) -> bool:
        """Menyimpan parameter kalibrasi ke file JSON."""
        target = Path(path) if path else self.config_path
        if not target:
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "berseberangan": self.berseberangan,
            "flip_x": self.flip_x,
            "flip_y": self.flip_y,
            "scale_x_mm_per_px": self.scale_x,
            "scale_y_mm_per_px": self.scale_y,
            "origin_robot_x_mm": self.origin_x,
            "origin_robot_y_mm": self.origin_y,
            "z_heights": self.z_heights,
            "drop_coordinates": self.drop_coords,
            "calibration_points": self.calibration_points,
            "homography_matrix": self.homography_matrix.tolist() if self.homography_matrix is not None else None,
        }
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as exc:
            print(f"[Error] Gagal menyimpan kalibrasi ke {target}: {exc}")
            return False

    def set_homography_points(
        self,
        pixel_points: list[tuple[float, float]],
        robot_points: list[tuple[float, float]],
    ) -> bool:
        """Menghitung matriks Homography dari minimal 4 pasang titik kalibrasi (piksel vs robot mm)."""
        if len(pixel_points) < 4 or len(robot_points) < 4:
            raise ValueError("Kalibrasi Homography membutuhkan minimal 4 titik pasangan!")

        src_pts = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(robot_points, dtype=np.float32).reshape(-1, 1, 2)

        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return False

        self.homography_matrix = H
        self.calibration_points = [
            {"pixel": [float(p[0]), float(p[1])], "robot": [float(r[0]), float(r[1])]}
            for p, r in zip(pixel_points, robot_points)
        ]
        return True

    def pixel_to_robot(
        self,
        pixel_center: tuple[int, int],
        class_name: str,
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> tuple[float, float, float]:
        """Konversi titik piksel (u, v) ke koordinat fisik robot (X, Y, Z) dalam mm."""
        u, v = float(pixel_center[0]), float(pixel_center[1])

        # 1. Jika Homography Matrix tersedia, gunakan pemetaan perspektif planar (akurasi tertinggi)
        if self.homography_matrix is not None:
            pt = np.array([[[u, v]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
            xr = float(transformed[0][0][0])
            yr = float(transformed[0][0][1])
        else:
            # 2. Pemetaan parametrik terbalik (Berseberangan)
            center_u = frame_width / 2.0
            center_v = frame_height / 2.0

            delta_u = u - center_u
            delta_v = v - center_v

            # Jika kamera berseberangan dengan robot:
            # - Kanan kamera adalah Kiri robot (flip_x)
            # - Bawah kamera adalah Mendekati robot atau Menjauhi robot (flip_y)
            sign_x = -1.0 if self.flip_x else 1.0
            sign_y = -1.0 if self.flip_y else 1.0

            xr = self.origin_x + (sign_x * delta_u * self.scale_x)
            yr = self.origin_y + (sign_y * delta_v * self.scale_y)

        # Ketinggian Z (ketinggian pick spesifik per kelas benda)
        zr = self.z_heights.get(class_name, 5.0)
        return round(xr, 2), round(yr, 2), round(zr, 2)

    def to_robot_target(
        self,
        class_name: str,
        label: str,
        confidence: float,
        pixel_center: tuple[int, int],
        frame_width: int = 1280,
        frame_height: int = 720,
        track_id: int | None = None,
        catatan: str = "",
    ) -> RobotTarget:
        """Membuat objek RobotTarget lengkap dengan pick point & drop point."""
        xr, yr, zr = self.pixel_to_robot(pixel_center, class_name, frame_width, frame_height)
        drop = self.drop_coords.get(class_name, {"x": 0.0, "y": 0.0, "z": 50.0})

        return RobotTarget(
            class_name=class_name,
            label=label,
            confidence=confidence,
            pixel_center=pixel_center,
            x_mm=xr,
            y_mm=yr,
            z_mm=zr,
            drop_x_mm=float(drop.get("x", 0.0)),
            drop_y_mm=float(drop.get("y", 0.0)),
            drop_z_mm=float(drop.get("z", 50.0)),
            track_id=track_id,
            catatan=catatan,
        )


def choose_best_target(
    targets: list[RobotTarget],
    roi: tuple[int, int, int, int] | None = None,
) -> RobotTarget | None:
    """Memilih target terbaik untuk diambil lengan robot.

    Prioritas:
    1. Berada di dalam ROI / area conveyor kerja aktif robot (jika ROI aktif).
    2. Tingkat kepercayaan (confidence) tertinggi.
    """
    if not targets:
        return None
    valid = targets
    if roi is not None:
        x1, y1, x2, y2 = roi
        valid = [t for t in targets if (x1 <= t.pixel_center[0] <= x2 and y1 <= t.pixel_center[1] <= y2)]
        if not valid:
            return None
    return max(valid, key=lambda t: t.confidence)


def export_target_json(
    best_target: RobotTarget | None,
    all_targets: list[RobotTarget],
    output_path: Path | str,
    status: str | None = None,
    message: str | None = None,
) -> bool:
    """Menulis status target ke file JSON secara atomic (aman dibaca tim robotik di Windows).

    Format status:
    - TARGET_READY: Benda diam total, koordinat valid, siap diambil robot.
    - WAITING_STATIONARY: Benda baru berhenti / konveyor deselerasi (menunggu getaran inersia reda).
    - CONVEYOR_MOVING: Konveyor sedang berjalan, dilarang mengambil objek bergerak.
    - NO_TARGET: Area sortir kosong atau objek bukan termasuk kelas target.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if status is not None:
        status_str = status
    else:
        status_str = "TARGET_READY" if best_target is not None else "NO_TARGET"

    payload = {
        "status": status_str,
        "message": message or ("Koordinat siap diambil" if status_str == "TARGET_READY" else "Menunggu status siap"),
        "timestamp": time.time(),
        "waktu_baca": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": best_target.to_dict() if best_target else None,
        "total_objek": len(all_targets),
        "daftar_objek": [t.to_dict() for t in all_targets],
    }

    # Gunakan temporary file lalu rename (atomic write) untuk menghindari race-condition di Windows
    tmp_file = p.with_suffix(".tmp")
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        # Di Windows, replace atomic
        if p.exists():
            os.replace(tmp_file, p)
        else:
            tmp_file.rename(p)
        return True
    except Exception:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception:
                pass
        return False
