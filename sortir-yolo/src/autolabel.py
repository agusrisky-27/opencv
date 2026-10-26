"""Label otomatis (kotak) untuk gambar yang hanya berisi SATU benda.

Dipakai untuk:
  - dataset 'tutup botol' dari Roboflow yang tipenya klasifikasi (tidak punya kotak)
  - foto sendiri yang diambil di latar polos (mis. di atas conveyor)

Dua metode:
  - 'cv'    : segmentasi sederhana berbasis warna latar (cepat, tanpa download model,
              hanya cocok untuk latar polos).
  - 'world' : YOLO-World (deteksi zero-shot berbasis teks, mis. "bottle cap").
              Lebih fleksibel untuk latar ramai, butuh unduh model (sebaiknya di Colab).
Hasilnya SELALU perlu dicek mata lewat gambar preview.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# kotak ternormalisasi: (cx, cy, w, h) skala 0..1
NormBox = tuple[float, float, float, float]


def box_from_background(
    img_bgr: np.ndarray,
    max_side: int = 320,
    max_bg_spread: float = 22.0,
    min_area: float = 0.01,
    max_area: float = 0.90,
) -> NormBox | None:
    """Cari kotak benda dengan asumsi latar relatif polos (warna tepi gambar ~ seragam).

    Return None kalau latar tidak polos atau tidak ada benda yang jelas.
    """
    h, w = img_bgr.shape[:2]
    if h < 16 or w < 16:
        return None
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(img_bgr, (max(16, int(w * scale)), max(16, int(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    lab_raw = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    # warna latar = median piksel di tepi gambar
    t = max(2, int(0.05 * min(sh, sw)))
    ring = np.zeros((sh, sw), dtype=bool)
    ring[:t, :] = ring[-t:, :] = ring[:, :t] = ring[:, -t:] = True
    bg_pixels = lab_raw[ring]
    bg = np.median(bg_pixels, axis=0)
    # keseragaman latar diukur dari gambar ASLI (sebelum blur), supaya latar bertekstur ketahuan
    spread = float(np.mean(np.std(bg_pixels, axis=0)))
    if spread > max_bg_spread:
        return None  # latar ramai -> metode ini tidak cocok

    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

    dist = np.sqrt(((lab - bg) ** 2).sum(axis=2))
    dist8 = np.clip(dist, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(otsu), 15.0)
    mask = (dist > thr).astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, sw // 40) | 1, max(3, sw // 40) | 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    frac = cv2.contourArea(best) / float(sh * sw)
    if frac < min_area or frac > max_area:
        return None
    x, y, bw, bh = cv2.boundingRect(best)
    return ((x + bw / 2) / sw, (y + bh / 2) / sh, bw / sw, bh / sh)


def segment_charger_box(img_bgr: np.ndarray, yolo_model: Any = None) -> NormBox | None:
    """Segmentasi universal kepala charger (Mendukung conveyor hijau kantor, alas hitam, maupun meja terang).

    Bekerja adaptif:
    - Jika terdeteksi conveyor hijau: mencari objek charger putih/terang di atas belt hijau,
      menghindari laptop, rel konveyor, dan pantulan meja di tepi frame.
    - Jika latar gelap (alas hitam): mencari objek terang (charger putih/warna).
    - Jika latar terang (meja/kertas putih): mencari objek gelap (charger hitam/abu-abu).
    - Fallback ke model YOLO jika tersedia dan segmentasi CV terhalang.
    """
    h, w = img_bgr.shape[:2]
    if h < 16 or w < 16:
        return None

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1. Deteksi konveyor hijau
    green = cv2.inRange(hsv, (30, 25, 20), (95, 255, 255))
    green_pct = np.count_nonzero(green) / (w * h)

    if green_pct > 0.08:
        # MODE CONVEYOR HIJAU (Kantor)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        skin = (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 173) & (ycrcb[:, :, 2] >= 77) & (ycrcb[:, :, 2] <= 127)
        is_non_green = cv2.bitwise_not(green)

        # Charger putih/terang: saturasi rendah, brightness cukup, bukan hijau, bukan kulit
        white = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 125) & (is_non_green > 0) & (~skin)
        pad = 12
        white_inner = np.zeros((h, w), dtype=np.uint8)
        white_inner[pad : h - pad, pad : w - pad] = white[pad : h - pad, pad : w - pad]

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(white_inner, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 10000 or a > 100000:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if x <= pad + 1 or y <= pad + 1 or (x + bw) >= (w - pad - 1) or (y + bh) >= (h - pad - 1):
                continue
            aspect = bw / max(1, bh)
            if not (0.35 < aspect < 2.8 and 180 < bw < 580 and 140 < bh < 480):
                continue
            box_skin = skin[y : y + bh, x : x + bw]
            if np.count_nonzero(box_skin) / max(1, box_skin.size) >= 0.22:
                continue
            valid.append((a, x, y, bw, bh))

        if valid:
            valid.sort(key=lambda item: item[0], reverse=True)
            _, x, y, bw, bh = valid[0]
            return ((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)

        # Fallback ke model YOLO jika tersedia
        if yolo_model is not None:
            try:
                res = yolo_model.predict(img_bgr, conf=0.35, verbose=False)[0]
                charger_boxes = [b for b in res.boxes if yolo_model.names.get(int(b.cls[0])) == "charger"]
                if charger_boxes:
                    b = max(charger_boxes, key=lambda item: float(item.conf[0]))
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    if 0.15 < bw < 0.55 and 0.15 < bh < 0.55:
                        return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw, bh)
            except Exception:
                pass
        return None

    # 2. MODE UMUM / ALAS GELAP / MEJA TERANG
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Analisis warna tepi latar belakang (margin 5%)
    t = max(2, int(0.05 * min(h, w)))
    border_pixels = np.concatenate([
        gray[:t, :].flatten(),
        gray[-t:, :].flatten(),
        gray[:, :t].flatten(),
        gray[:, -t:].flatten(),
    ])
    bg_median = float(np.median(border_pixels))

    # Tentukan strategi ambang batas berdasarkan warna latar
    if bg_median < 110:
        thresh_types = [(cv2.THRESH_BINARY, [130, 115, 145, 100, 85])]
    else:
        thresh_types = [(cv2.THRESH_BINARY_INV, [int(bg_median - 30), int(bg_median - 50), 90, 110])]

    # Tambahkan opsi invers sebagai cadangan
    thresh_types.append((cv2.THRESH_BINARY_INV if bg_median < 110 else cv2.THRESH_BINARY, [120, 100, 140]))

    for t_type, thrs in thresh_types:
        for thr in thrs:
            _, mask = cv2.threshold(blurred, thr, 255, t_type)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if (w * h * 0.01) < area < (w * h * 0.80):
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw < w * 0.92 and bh < h * 0.92:
                        valid.append((area, x, y, bw, bh))
            if valid:
                valid.sort(key=lambda item: item[0], reverse=True)
                _, x, y, bw, bh = valid[0]
                return ((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)

    # Fallback Edge-based jika kontras warna latar vs benda rendah
    edges = cv2.Canny(blurred, 30, 100)
    k_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_edge)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (w * h * 0.01) < area < (w * h * 0.80):
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < w * 0.92 and bh < h * 0.92:
                valid.append((area, x, y, bw, bh))
    if valid:
        valid.sort(key=lambda item: item[0], reverse=True)
        _, x, y, bw, bh = valid[0]
        return ((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)

    return None


class WorldLabeler:
    """Label otomatis memakai YOLO-World (zero-shot, berbasis teks)."""

    def __init__(
        self,
        prompts: Sequence[str] = ("bottle cap",),
        weights: str = "yolov8s-worldv2.pt",
        conf: float = 0.03,
        device: str = "auto",
    ) -> None:
        from ultralytics import YOLOWorld  # butuh ultralytics (+ paket CLIP, dipasang otomatis)

        from .utils import pick_device

        self.model = YOLOWorld(weights)
        self.model.set_classes(list(prompts))
        self.conf = conf
        self.device = pick_device(device)

    def box(self, img_bgr: np.ndarray) -> NormBox | None:
        """Kotak dengan confidence tertinggi (asumsi: satu benda per gambar)."""
        res = self.model.predict(img_bgr, conf=self.conf, device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        i = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
        h, w = img_bgr.shape[:2]
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0.01 or bh <= 0.01:
            return None
        return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw, bh)


class ModelLabeler:
    """Label otomatis memakai model yang sudah ada (mis. models/best.pt)."""

    def __init__(
        self,
        weights: str | Path = "models/best.pt",
        conf: float = 0.15,
        device: str = "auto",
    ) -> None:
        from ultralytics import YOLO
        from .utils import pick_device

        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = pick_device(device)

    def box(self, img_bgr: np.ndarray, target_class: str | None = None) -> NormBox | None:
        """Kotak dengan target_class atau confidence tertinggi."""
        res = self.model.predict(img_bgr, conf=self.conf, device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        best_box = None
        best_conf = -1.0
        for b in res.boxes:
            c_id = int(b.cls[0])
            c_name = self.model.names.get(c_id, "")
            conf = float(b.conf[0])
            if target_class and c_name != target_class:
                continue
            if conf > best_conf:
                best_conf = conf
                best_box = b
        if best_box is None:
            return None
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        h, w = img_bgr.shape[:2]
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0.01 or bh <= 0.01:
            return None
        return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw, bh)


def write_yolo_label(path: Path, class_id: int, box: NormBox) -> None:
    cx, cy, w, h = box
    path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


def draw_preview(img_bgr: np.ndarray, box: NormBox | None, size: int = 256) -> np.ndarray:
    """Gambar kecil dengan kotak merah untuk dicek manual."""
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    out = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[: out.shape[0], : out.shape[1]] = out
    if box is not None:
        cx, cy, bw, bh = box
        x1, y1 = int((cx - bw / 2) * out.shape[1]), int((cy - bh / 2) * out.shape[0])
        x2, y2 = int((cx + bw / 2) * out.shape[1]), int((cy + bh / 2) * out.shape[0])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        cv2.putText(canvas, "TANPA KOTAK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas


def save_preview_grid(tiles: list[np.ndarray], out_path: Path, cols: int = 6) -> None:
    if not tiles:
        return
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
