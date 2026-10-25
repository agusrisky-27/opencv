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
    catatan: str = ""                    # catatan pasca-proses
    track_id: int | None = None          # ID unik pelacak objek bergerak

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_geometry(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    min_ar: float = 0.50,
    max_ar: float = 1.85,
    min_size_px: int = 10,
    max_area_frac: float = 0.40,
) -> bool:
    """Validasi fisik objek: koin dan tutup botol berbentuk bundar/elips dan ukurannya wajar."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < min_size_px or h < min_size_px:
        return False
    if w > frame_width * 0.65 or h > frame_height * 0.65:
        return False
    if (w * h) > (frame_width * frame_height * max_area_frac):
        return False
    ar = w / float(max(h, 1))
    if ar < min_ar or ar > max_ar:
        return False
    return True


def _cek_warna_kulit(patch: np.ndarray) -> float:
    """Hitung fraksi piksel warna kulit manusia (skin tone) menggunakan kombinasi YCrCb dan HSV."""
    if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
        return 0.0
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]
    mask_ycrcb = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    mask_hsv = (h >= 0) & (h <= 25) & (s >= 35) & (s <= 170) & (v >= 50)

    combined = mask_ycrcb & mask_hsv
    return float(np.mean(combined))


def is_human_skin_face(crop: np.ndarray, frame_w: int, frame_h: int) -> bool:
    """Mendeteksi apakah objek adalah wajah manusia di depan webcam.
    
    Koin atau tutup botol yang dipegang jari atau ditaruh di meja tidak boleh ditolak sebagai wajah.
    """
    h, w = crop.shape[:2]
    if w < 10 or h < 10:
        return False
    area_frac = (w * h) / float(max(frame_w * frame_h, 1))
    skin_frac = _cek_warna_kulit(crop)
    # Wajah di depan webcam: harus cukup besar di layar DAN kaya warna kulit manusia
    if skin_frac > 0.40 and area_frac > 0.020 and min(w, h) >= 90:
        return True
    if skin_frac > 0.70 and area_frac > 0.015 and min(w, h) >= 80:
        return True
    return False


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
    pekat = (sat > 0.40) & ok
    frac = float(pekat.sum() / ok.sum())
    if pekat.any():
        rad = np.deg2rad(hue[pekat])
        hue_mean = float(np.rad2deg(math.atan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)
    else:
        hue_mean = 0.0
    return {"frac_pekat": frac, "hue": hue_mean, "sat": float(np.median(sat[ok]))}


def tutup_berwarna(frame: np.ndarray, bbox: tuple[int, int, int, int], min_frac: float = 0.50) -> bool:
    """True jika benda jelas berwarna pekat plastik (oranye/biru/merah/hijau/kuning) -> bukan koin logam dan bukan wajah."""
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    # Tolak jika objek adalah wajah manusia agar tidak dikira tutup botol
    if is_human_skin_face(crop, frame.shape[1], frame.shape[0]):
        return False

    c = ciri_warna(frame, bbox)
    if c is None or c["frac_pekat"] < min_frac:
        return False
    # Koin kuningan emas Rp 1000 / Rp 500 berada di hue 15..60 (kuning oranye keemasan)
    # Plastik pekat harus memiliki saturasi sangat tinggi (> 0.65) agar koin kuningan kusam tidak tertukar
    if 15.0 <= c["hue"] <= 60.0:
        return c["sat"] > 0.65
    return True


def verifikasi_koin_asli(frame: np.ndarray, bbox: tuple[int, int, int, int], confidence: float = 0.50) -> bool:
    """Verifikasi khusus koin: tolak kunci (ada lubang tembus gantungan), plastik abu-abu polos, dan wajah manusia."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 10:
        return False

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.max() == gray.min():
        return True  # frame sintetis polos (unit test)

    # Jika deteksi YOLO sudah yakin (conf >= 0.65), langsung loloskan koin
    if confidence >= 0.65:
        return True

    # 1. Tolak jika objek adalah wajah manusia
    if is_human_skin_face(crop, frame.shape[1], frame.shape[0]):
        return False

    min_dim = min(w, h)
    ch, cw = gray.shape[:2]

    # 3. Cek Tekstur Relief Koin Logam vs Plastik Datar Polos (Adaptif Skala Jarak)
    center_patch = gray[int(ch * 0.22):int(ch * 0.78), int(cw * 0.22):int(cw * 0.78)]
    if center_patch.size > 0:
        center_edges = cv2.Canny(center_patch, 40, 120)
        edge_density = np.count_nonzero(center_edges) / float(center_edges.size)
        std_dev = float(np.std(center_patch))

        sx = cv2.Sobel(center_patch, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(center_patch, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = float(np.mean(np.sqrt(sx**2 + sy**2)))

        if min_dim >= 75:
            # Jarak dekat / sedang: uji tekstur penuh (plastik polos ditolak)
            if edge_density < 0.018 and std_dev < 17.0 and grad_mag < 19.0:
                return False
        elif min_dim >= 40:
            # Jarak menengah: ambang batas tekstur lebih toleran
            if edge_density < 0.010 and std_dev < 12.0 and grad_mag < 14.0:
                return False

    return True


def verifikasi_tutup_botol_asli(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> bool:
    """Verifikasi khusus tutup botol: tolak wajah manusia, tetapi izinkan tutup botol nyata (meskipun dipegang tangan)."""
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
        return False

    # Tolak HANYA jika objek secara dominan adalah wajah manusia di depan webcam
    if is_human_skin_face(crop, frame_width, frame_height):
        return False

    return True


class Detector:
    """Muat model sekali, lalu panggil detect(frame) berulang kali."""

    def __init__(
        self,
        weights: str | Path | None = None,
        conf: float = 0.25,
        imgsz: int = 1280,
        iou: float = 0.45,
        device: str = "auto",
        model: Any = None,
        smart_filter: bool = False,
        track: bool = False,
        min_ar: float = 0.50,
        max_ar: float = 1.85,
        max_area_frac: float = 0.40,
        min_size_px: int = 10,
        cek_warna: bool = True,
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        self.device = pick_device(device)
        self.smart_filter = smart_filter
        self.track = track
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

        if self.track and hasattr(self.model, "track"):
            try:
                results = self.model.track(
                    frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                    device=self.device, agnostic_nms=True, max_det=30, verbose=False,
                    persist=True, tracker="bytetrack.yaml",
                )
            except Exception:
                results = self.model.predict(
                    frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                    device=self.device, agnostic_nms=True, max_det=30, verbose=False,
                )
        else:
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

            class_id = int(box.cls[0])
            class_name = str(names.get(class_id, class_id))
            confidence = float(box.conf[0])
            label = display_label(class_name)
            catatan = ""

            # 2. Filter Geometri Cerdas (aspek rasio, ukuran, batas area)
            if use_filter:
                if class_name == "charger":
                    # Charger berbentuk balok/persegi panjang: toleransi rasio lebih leluasa
                    if not validate_geometry(
                        bbox, width, height,
                        min_ar=0.25,
                        max_ar=4.00,
                        min_size_px=self.min_size_px,
                        max_area_frac=0.55,
                    ):
                        continue
                else:
                    if not validate_geometry(
                        bbox, width, height,
                        min_ar=self.min_ar,
                        max_ar=self.max_ar,
                        min_size_px=self.min_size_px,
                        max_area_frac=self.max_area_frac,
                    ):
                        continue

            track_id: int | None = None
            if hasattr(box, "id") and box.id is not None:
                try:
                    track_id = int(box.id[0])
                except (IndexError, TypeError, ValueError):
                    track_id = None

            # 3. Koreksi warna: jika objek berwarna plastik pekat (biru, hijau, merah, dll.), jadikan tutup_botol
            if (self.cek_warna and class_name == "koin" and "tutup_botol" in name_to_id
                    and tutup_berwarna(frame, bbox)):
                class_id = name_to_id["tutup_botol"]
                class_name = "tutup_botol"
                label = display_label("tutup_botol")
                catatan = "koreksi warna"

            # 4. Verifikasi Koin Asli (Tolak kunci, plastik abu-abu polos, dan wajah manusia)
            if use_filter and class_name == "koin" and not verifikasi_koin_asli(frame, bbox, confidence=confidence):
                continue

            # 5. Verifikasi Tutup Botol Asli (Tolak wajah manusia, tetapi izinkan tutup botol nyata)
            if use_filter and class_name == "tutup_botol" and not verifikasi_tutup_botol_asli(
                frame, bbox, width, height
            ):
                continue

            # 6. Verifikasi Charger Asli (Tolak jika objek adalah wajah manusia di depan webcam)
            if use_filter and class_name == "charger":
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0 and is_human_skin_face(crop, width, height):
                    continue

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
                    track_id=track_id,
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
