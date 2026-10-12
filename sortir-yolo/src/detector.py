"""Pembungkus model YOLO: frame masuk -> daftar Detection keluar."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils import (
    clamp_bbox,
    compute_center,
    display_label,
    normalize_point,
    pick_device,
    point_in_roi,
)


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str                      # nama kelas di model, mis. 'tutup_botol'
    label: str                           # nama tampilan, mis. 'Tutup Botol'
    confidence: float
    bbox: tuple[int, int, int, int]      # (x1, y1, x2, y2) piksel
    center: tuple[int, int]              # (cx, cy) piksel
    center_norm: tuple[float, float]     # (cx, cy) skala 0..1
    catatan: str = ""                    # catatan pasca-proses (mis. 'koreksi warna')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_geometry(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    min_ar: float = 0.65,
    max_ar: float = 1.50,
    min_size_px: int = 20,
    max_area_frac: float = 0.22,
) -> bool:
    """Validasi fisik objek: koin dan tutup botol berbentuk bundar/elips dan ukurannya wajar.

    Menolak deteksi palsu:
    - Jendela / perabot / pintu (rasio aspek terlalu lonjong atau kotak terlalu besar)
    - Wajah manusia / kepala (ukuran kotak terlalu besar melebihi proporsi benda)
    - Noise piksel mikroskopis (< min_size_px)
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < min_size_px or h < min_size_px:
        return False
    # Ukuran koin/tutup botol di kamera tidak mungkin melebihi 45% lebar/tinggi frame
    if w > frame_width * 0.45 or h > frame_height * 0.45:
        return False
    # Luas kotak tidak boleh melebihi fraksi maksimal luas frame
    if (w * h) > (frame_width * frame_height * max_area_frac):
        return False
    # Aspek rasio w/h: bundar/elips mendekati 1.0 (toleransi sudut miring 0.65 - 1.50)
    ar = w / float(max(h, 1))
    if ar < min_ar or ar > max_ar:
        return False
    return True


def ciri_warna(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, float] | None:
    """Ciri warna bagian tengah benda (elips 60% kotak, hindari tepi/latar)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 10:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    px1, py1 = max(0, int(cx - w * 0.3)), max(0, int(cy - h * 0.3))
    px2, py2 = min(frame.shape[1], int(cx + w * 0.3)), min(frame.shape[0], int(cy + h * 0.3))
    patch = frame[py1:py2, px1:px2]
    if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
        return None
    ph, pw = patch.shape[:2]
    yy, xx = np.ogrid[:ph, :pw]
    inside = ((xx - pw / 2) / (pw / 2)) ** 2 + ((yy - ph / 2) / (ph / 2)) ** 2 <= 1.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0][inside].astype(np.float32) * 2.0  # derajat 0..360
    sat = hsv[..., 1][inside].astype(np.float32) / 255.0
    val = hsv[..., 2][inside].astype(np.float32) / 255.0
    ok = val > 0.15
    if ok.sum() < 20:
        return None
    pekat = (sat > 0.45) & ok
    frac = float(pekat.sum() / ok.sum())
    if pekat.any():
        rad = np.deg2rad(hue[pekat])
        hue_mean = float(np.rad2deg(math.atan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)
    else:
        hue_mean = 0.0
    return {"frac_pekat": frac, "hue": hue_mean, "sat": float(np.median(sat[ok]))}


def tutup_berwarna(frame: np.ndarray, bbox: tuple[int, int, int, int], min_frac: float = 0.6) -> bool:
    """True jika benda jelas berwarna pekat (biru/merah/hijau/ungu) -> pasti bukan koin logam."""
    c = ciri_warna(frame, bbox)
    if c is None or c["frac_pekat"] < min_frac:
        return False
    # Koin logam rupiah: perak/abu-abu (saturasi rendah) atau kuning emas (hue 10-50).
    return not (10.0 <= c["hue"] <= 50.0)


class Detector:
    """Muat model sekali, lalu panggil detect(frame) berulang kali."""

    def __init__(
        self,
        weights: str | Path | None = None,
        conf: float = 0.65,
        imgsz: int = 640,
        iou: float = 0.45,
        device: str = "auto",
        model: Any = None,
        smart_filter: bool = False,
        min_ar: float = 0.65,
        max_ar: float = 1.50,
        max_area_frac: float = 0.22,
        min_size_px: int = 20,
        cek_warna: bool = True,
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        self.device = pick_device(device)
        self.smart_filter = smart_filter
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.max_area_frac = max_area_frac
        self.min_size_px = min_size_px
        self.cek_warna = cek_warna
        self.roi = roi
        if model is None:
            if weights is None or not Path(weights).is_file():
                raise FileNotFoundError(
                    f"File model tidak ditemukan: {weights}\n"
                    "Jalankan training dulu (04_train.py / notebook Colab), lalu taruh best.pt "
                    "di folder models/ atau pakai --weights <path>."
                )
            from ultralytics import YOLO  # import di sini agar test tanpa ultralytics tetap jalan

            model = YOLO(str(weights))
        self.model = model

    def detect(self, frame: np.ndarray, apply_smart_filter: bool | None = None) -> list[Detection]:
        """Deteksi semua objek di satu frame (BGR). Urut dari confidence tertinggi."""
        height, width = frame.shape[:2]
        use_filter = self.smart_filter if apply_smart_filter is None else apply_smart_filter
        results = self.model.predict(
            frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, agnostic_nms=True, max_det=30, verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names
        name_to_id = {str(v): int(k) for k, v in names.items()}
        detections: list[Detection] = []
        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            bbox = clamp_bbox((x1, y1, x2, y2), width, height)
            center = compute_center(bbox)

            # 1. Filter ROI (jika aktif)
            if self.roi is not None and not point_in_roi(center, self.roi):
                continue

            # 2. Filter Geometri Cerdas (aspek rasio, ukuran, batas area)
            if use_filter and not validate_geometry(
                bbox, width, height,
                min_ar=self.min_ar,
                max_ar=self.max_ar,
                min_size_px=self.min_size_px,
                max_area_frac=self.max_area_frac,
            ):
                continue

            class_id = int(box.cls[0])
            class_name = str(names[class_id])
            confidence = float(box.conf[0])
            label = display_label(class_name)
            catatan = ""

            # 3. Koreksi warna pekat (koin salah tebak tutup botol)
            if (self.cek_warna and class_name == "koin" and "tutup_botol" in name_to_id
                    and tutup_berwarna(frame, bbox)):
                class_id = name_to_id["tutup_botol"]
                class_name = "tutup_botol"
                label = display_label("tutup_botol")
                catatan = "koreksi warna"

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    center=center,
                    center_norm=normalize_point(center, width, height),
                    catatan=catatan,
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
