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


def segment_charger_box(img_bgr: np.ndarray) -> NormBox | None:
    """Segmentasi khusus kepala charger (benda terang pada alas gelap atau sebaliknya)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Coba beberapa ambang batas kecerahan
    for thr in [130, 115, 145, 100]:
        _, mask = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY)
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
