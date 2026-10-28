"""Fungsi bantu: geometri (titik tengah), menggambar di frame, simpan file."""
from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np

from .config import BOX_COLOR, DISPLAY_LABELS

BBox = tuple[int, int, int, int]


class _DetectionLike(Protocol):
    label: str
    confidence: float
    bbox: BBox
    center: tuple[int, int]


# ----------------------------------------------------------------- geometri
def compute_center(bbox: BBox) -> tuple[int, int]:
    """Titik tengah kotak: cx=(x1+x2)/2, cy=(y1+y2)/2 (dibulatkan ke piksel)."""
    x1, y1, x2, y2 = bbox
    return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))


def normalize_point(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    """Ubah koordinat piksel ke skala 0.0 - 1.0 terhadap ukuran frame."""
    if width <= 0 or height <= 0:
        raise ValueError("width dan height harus > 0")
    return round(point[0] / width, 4), round(point[1] / height, 4)


def clamp_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> BBox:
    """Pastikan kotak tidak keluar dari frame."""
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width - 1, round(x2))))
    y2 = int(max(0, min(height - 1, round(y2))))
    return x1, y1, x2, y2


def display_label(class_name: str) -> str:
    """Nama yang tampil di layar (mis. 'tutup_botol' -> 'Tutup Botol')."""
    return DISPLAY_LABELS.get(class_name, class_name.replace("_", " ").title())


def pick_device(requested: str | None = "auto") -> str:
    """Pilih device otomatis: GPU NVIDIA ('0'), Apple Silicon ('mps'), atau 'cpu'."""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # torch belum terpasang / error lain
        pass
    return "cpu"


# ----------------------------------------------------------------- ROI & Penstabil
def point_in_roi(point: tuple[int, int], roi: BBox | None) -> bool:
    """Cek apakah titik (x, y) berada di dalam kotak ROI (x1, y1, x2, y2)."""
    if roi is None:
        return True
    px, py = point
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= px <= rx2 and ry1 <= py <= ry2


def draw_roi(frame: np.ndarray, roi: BBox | None, active: bool = True) -> None:
    """Gambar batas area kerja (ROI) di frame."""
    if roi is None or not active:
        return
    rx1, ry1, rx2, ry2 = roi
    color = (255, 200, 0)  # Cyan / Kuning-biru terang
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
    _text_with_outline(frame, "ZONA AKTIF (ROI)", (rx1 + 8, ry1 + 22), 0.6)


def iou_xyxy(a: BBox, b: BBox) -> float:
    """Hitung Intersection over Union (IoU) antara dua kotak (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / float(max(1, union))


class Penstabil:
    """Label tiap benda diputuskan lewat SUARA TERBANYAK beberapa frame terakhir (anti flicker).
    
    Benda dilacak antar-frame lewat kombinasi IoU kotak, jarak titik tengah adaptif, dan
    ekstrapolasi kecepatan gerak (tahan goyangan kamera & gerakan konveyor cepat).
    """

    def __init__(
        self,
        jendela: int = 5,
        iou_min: float = 0.15,
        max_dist_px: float = 300.0,
        umur_maks: int = 4,
        smooth_factor: float = 0.70,
    ) -> None:
        self.jendela = jendela
        self.iou_min = iou_min
        self.max_dist_px = max_dist_px
        self.umur_maks = umur_maks
        self.smooth_factor = smooth_factor
        self._tracks: list[dict] = []
        self._next_id = 1

    def update(self, dets: list) -> list:
        from collections import defaultdict, deque
        from dataclasses import replace

        for t in self._tracks:
            t["umur"] += 1

        pairs = []
        for i, d in enumerate(dets):
            d_center = getattr(d, "center", compute_center(d.bbox))
            for j, t in enumerate(self._tracks):
                if getattr(d, "track_id", None) is not None and t.get("track_id") == d.track_id:
                    pairs.append((100.0, i, j))
                    continue
                t_box = t["bbox"]
                iou = iou_xyxy(d.bbox, t_box)
                t_center = compute_center(t_box)
                dist = ((d_center[0] - t_center[0]) ** 2 + (d_center[1] - t_center[1]) ** 2) ** 0.5
                if iou >= self.iou_min:
                    pairs.append((iou + 2.0, i, j))
                elif dist <= self.max_dist_px:
                    score = max(0.05, 1.0 - (dist / self.max_dist_px))
                    if t.get("last_class_id") == d.class_id:
                        score += 0.5
                    pairs.append((score, i, j))

        pairs.sort(reverse=True, key=lambda x: x[0])
        match: dict[int, int] = {}
        used_t: set[int] = set()
        for v, i, j in pairs:
            if i in match or j in used_t:
                continue
            match[i] = j
            used_t.add(j)

        out: list = []
        for i, d in enumerate(dets):
            if i in match:
                t = self._tracks[match[i]]
                prev_box = t["bbox"]
                # Penghalusan posisi kotak (exponential moving average) agar tidak bergetar saat kamera goyang
                cur_box = d.bbox
                cur_c = compute_center(cur_box)
                prev_c = compute_center(prev_box)
                dist_move = math.hypot(cur_c[0] - prev_c[0], cur_c[1] - prev_c[1])
                # Jika benda sedang bergerak cepat di conveyor / digerakkan tangan, snap langsung tanpa lag
                s = 1.0 if dist_move > 15.0 else self.smooth_factor
                smoothed_bbox = (
                    int(round(s * cur_box[0] + (1.0 - s) * prev_box[0])),
                    int(round(s * cur_box[1] + (1.0 - s) * prev_box[1])),
                    int(round(s * cur_box[2] + (1.0 - s) * prev_box[2])),
                    int(round(s * cur_box[3] + (1.0 - s) * prev_box[3])),
                )
                smoothed_center = compute_center(smoothed_bbox)
                vx = smoothed_center[0] - prev_c[0]
                vy = smoothed_center[1] - prev_c[1]
                t["vx"] = 0.6 * vx + 0.4 * t.get("vx", 0.0)
                t["vy"] = 0.6 * vy + 0.4 * t.get("vy", 0.0)
            else:
                tid = getattr(d, "track_id", None) or self._next_id
                self._next_id += 1
                t = {
                    "votes": deque(maxlen=self.jendela),
                    "track_id": tid,
                    "hits": 0,
                    "vx": 0.0,
                    "vy": 0.0,
                }
                self._tracks.append(t)
                smoothed_bbox = d.bbox
                smoothed_center = d.center

            t["bbox"], t["umur"] = smoothed_bbox, 0
            t["hits"] = t.get("hits", 0) + 1
            t["votes"].append((d.class_id, d.class_name, d.confidence))
            t["last_class_id"] = d.class_id

            tid = getattr(d, "track_id", None) or t.get("track_id")

            # Perbarui koordinat ternormalisasi jika center berubah
            c_norm = d.center_norm
            if d.center[0] > 0 and d.center[1] > 0 and (smoothed_center != d.center):
                rx = d.center_norm[0] / float(d.center[0])
                ry = d.center_norm[1] / float(d.center[1])
                c_norm = (round(smoothed_center[0] * rx, 4), round(smoothed_center[1] * ry, 4))

            # Putuskan suara terbanyak
            votes = t["votes"]
            skor: dict[tuple[int, str], float] = defaultdict(float)
            for cid, cname, conf in votes:
                skor[(cid, cname)] += conf
            (best_cid, best_cname), _ = max(skor.items(), key=lambda kv: kv[1])

            if best_cid == d.class_id:
                d_out = replace(d, bbox=smoothed_bbox, center=smoothed_center, center_norm=c_norm, track_id=tid)
            else:
                conf_avg = float(np.mean([c for ci, _, c in votes if ci == best_cid]))
                d_out = replace(
                    d,
                    bbox=smoothed_bbox,
                    center=smoothed_center,
                    center_norm=c_norm,
                    class_id=best_cid,
                    class_name=best_cname,
                    label=display_label(best_cname),
                    confidence=conf_avg,
                    catatan="suara terbanyak",
                    track_id=tid,
                )
            t["last_d"] = d_out
            out.append(d_out)

        # Anti-flicker & Coasting: Jika benda sempat hilang 1-3 frame karena motion blur / goyangan kamera
        matched_track_indices = set(match.values())
        for idx, t in enumerate(self._tracks):
            if idx not in matched_track_indices and 1 <= t["umur"] <= 3 and t.get("hits", 0) >= 2 and "last_d" in t:
                last_d = t["last_d"]
                vx = int(round(t.get("vx", 0.0)))
                vy = int(round(t.get("vy", 0.0)))
                extrapolated_box = (
                    last_d.bbox[0] + vx,
                    last_d.bbox[1] + vy,
                    last_d.bbox[2] + vx,
                    last_d.bbox[3] + vy,
                )
                extrapolated_center = compute_center(extrapolated_box)
                c_norm = last_d.center_norm
                if last_d.center[0] > 0 and last_d.center[1] > 0:
                    rx = last_d.center_norm[0] / float(last_d.center[0])
                    ry = last_d.center_norm[1] / float(last_d.center[1])
                    c_norm = (round(extrapolated_center[0] * rx, 4), round(extrapolated_center[1] * ry, 4))
                coasted_d = replace(
                    last_d,
                    bbox=extrapolated_box,
                    center=extrapolated_center,
                    center_norm=c_norm,
                    catatan="stabil",
                )
                out.append(coasted_d)
                t["bbox"] = extrapolated_box

        self._tracks = [t for t in self._tracks if t["umur"] <= self.umur_maks]
        return out


# ------------------------------------------------------------ menggambar
def _text_with_outline(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_detection(
    frame: np.ndarray,
    det: _DetectionLike,
    color: tuple[int, int, int] = BOX_COLOR,
    show_center: bool = True,
    show_persen: bool = True,
) -> None:
    """Gambar kotak merah + judul (nama benda & %) + titik tengah di frame (in-place)."""
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    tid = getattr(det, "track_id", None)
    prefix = f"#{tid} " if tid is not None else ""
    title = f"{prefix}{det.label}" + (f" {int(round(det.confidence * 100))}%" if show_persen else "")
    catatan = getattr(det, "catatan", "")
    if catatan:
        title += f" [{catatan}]"

    (tw, th), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    top = y1 - th - base - 6
    if top < 0:  # kotak mepet atas -> taruh judul di dalam kotak
        top = y1
    cv2.rectangle(frame, (x1, top), (x1 + tw + 10, top + th + base + 6), color, -1)
    cv2.putText(frame, title, (x1 + 5, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    if show_center:
        cx, cy = det.center
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 5, color, -1)
        _text_with_outline(frame, f"({cx}, {cy})", (cx + 10, cy - 8))


def draw_overlay(
    frame: np.ndarray,
    fps: float,
    count: int,
    conf: float | None = None,
    filter_active: bool | None = None,
    roi_active: bool | None = None,
    track_active: bool | None = None,
) -> None:
    """FPS dan status sistem di pojok kiri atas, bantuan tombol di pojok bawah."""
    status_parts = [f"FPS: {fps:.1f}", f"Objek: {count}"]
    if conf is not None:
        status_parts.append(f"Conf: {conf:.2f}")
    if filter_active is not None:
        status_parts.append(f"Filter: {'ON' if filter_active else 'OFF'}")
    if track_active is not None:
        status_parts.append(f"Track: {'ON' if track_active else 'OFF'}")
    if roi_active is not None:
        status_parts.append(f"ROI: {'ON' if roi_active else 'OFF'}")

    _text_with_outline(frame, " | ".join(status_parts), (10, 25), 0.65)
    _text_with_outline(
        frame,
        "q: keluar | s: foto | t: track | f: filter | r: ROI | w: warna | +/-: conf",
        (10, frame.shape[0] - 10),
        0.5,
    )


# ------------------------------------------------------------------ file
def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def save_screenshot(frame: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"deteksi_{timestamp()}.jpg"
    cv2.imwrite(str(path), frame)
    return path


CSV_HEADER = ["waktu", "kelas", "confidence", "cx", "cy", "x1", "y1", "x2", "y2"]


def append_csv(detections: Iterable[_DetectionLike], path: Path) -> int:
    """Tambahkan koordinat deteksi ke file CSV. Return jumlah baris yang ditulis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    written = 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(CSV_HEADER)
        now = datetime.now().isoformat(timespec="seconds")
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            writer.writerow([now, d.label, f"{d.confidence:.3f}", d.center[0], d.center[1], x1, y1, x2, y2])
            written += 1
    return written
