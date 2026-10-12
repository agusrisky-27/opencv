#!/usr/bin/env python3
"""Deteksi Koin / Tutup Botol (YOLOv8) - SEMUA dalam satu file.

    python sortir.py unduh                  # 1. unduh dataset Roboflow
    python sortir.py foto                   # 2. ambil foto sendiri (sangat disarankan)
    python sortir.py label tutup-botol      # 3. beri kotak otomatis dataset tutup botol
    python sortir.py label sendiri          #    beri kotak otomatis foto sendiri
    python sortir.py gabung                 # 4. gabung + buat data sintetis (koin & tutup botol satu gambar)
    python sortir.py latih                  # 5. training (disarankan di Google Colab)
    python sortir.py deteksi                # 6. deteksi webcam: kotak merah + judul + titik tengah

Tombol saat deteksi: q / ESC keluar | s screenshot | c simpan koordinat ke outputs/koordinat.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import shutil
import sys
import time
import zipfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml

# =============================================================================
# 1. KONFIGURASI  (ubah di sini)
# =============================================================================
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "datasets_raw"                 # dataset mentah hasil unduhan
MERGED_DIR = ROOT / "dataset_gabungan"          # dataset final untuk training
MODELS_DIR = ROOT / "models"                    # taruh best.pt di sini
OUTPUT_DIR = ROOT / "outputs"                   # screenshot, csv, gambar preview
RUNS_DIR = ROOT / "runs"                        # hasil training
OWN_RAW_DIR = RAW_DIR / "foto_sendiri_mentah"   # foto sendiri per kelas (+ folder 'latar')
OWN_DIR = RAW_DIR / "data_sendiri"              # foto sendiri + label YOLO
LATAR_DIR = OWN_RAW_DIR / "latar"               # foto latar KOSONG (tangan/meja/conveyor tanpa benda)
TUTUP_BOTOL_YOLO = RAW_DIR / "tutup_botol_yolo"
DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"

CLASS_NAMES = ["koin", "tutup_botol"]           # urutan = ID kelas. JANGAN diubah setelah training.
DISPLAY_LABELS = {"koin": "Koin", "tutup_botol": "Tutup Botol"}
PROMPTS = {"koin": "coin", "tutup_botol": "bottle cap"}   # teks untuk YOLO-World
BOX_COLOR = (0, 0, 255)                         # merah (BGR)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": ("train",), "valid": ("valid", "val", "validation"), "test": ("test",)}

DATASETS = [
    {"key": "koin", "workspace": "arfan-faris-pramatya", "project": "koin-rupiah-mvpr",
     "version": None, "kind": "detection", "target_class": "koin"},
    {"key": "tutup_botol", "workspace": "bottlecapsorting", "project": "tutup-botol",
     "version": None, "kind": "classification", "target_class": "tutup_botol"},
]

BBox = tuple[int, int, int, int]
NormBox = tuple[float, float, float, float]     # (cx, cy, w, h) skala 0..1


# =============================================================================
# 2. FUNGSI BANTU
# =============================================================================
def compute_center(bbox: BBox) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))


def normalize_point(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("width dan height harus > 0")
    return round(point[0] / width, 4), round(point[1] / height, 4)


def clamp_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    c = lambda v, hi: int(max(0, min(hi - 1, round(v))))  # noqa: E731
    return c(x1, width), c(y1, height), c(x2, width), c(y2, height)


def display_label(class_name: str) -> str:
    return DISPLAY_LABELS.get(class_name, class_name.replace("_", " ").title())


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_device(requested: str | None = "auto") -> str:
    """GPU NVIDIA ('0'), Apple Silicon ('mps') atau 'cpu'."""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    if not folder.is_dir():
        return []
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.suffix.lower() in IMAGE_EXTS)


def _text(frame: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, det: "Detection", persen: bool = False,
                   color: tuple[int, int, int] = BOX_COLOR) -> None:
    """Kotak merah + judul + titik tengah. Persentase hanya tampil kalau persen=True."""
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    title = det.label + (f" {int(round(det.confidence * 100))}%" if persen else "")
    (tw, th), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    top = y1 - th - base - 6
    if top < 0:
        top = y1
    cv2.rectangle(frame, (x1, top), (x1 + tw + 10, top + th + base + 6), color, -1)
    cv2.putText(frame, title, (x1 + 5, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cx, cy = det.center
    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
    cv2.circle(frame, (cx, cy), 5, color, -1)
    _text(frame, f"({cx}, {cy})", (cx + 10, cy - 8))


def draw_overlay(frame: np.ndarray, fps: float, count: int) -> None:
    _text(frame, f"FPS: {fps:.1f}", (10, 25), 0.7)
    _text(frame, f"Objek: {count}", (10, 50), 0.7)
    _text(frame, "q: keluar | s: screenshot | c: simpan koordinat", (10, frame.shape[0] - 10), 0.5)


def save_screenshot(frame: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"deteksi_{timestamp()}.jpg"
    cv2.imwrite(str(path), frame)
    return path


CSV_HEADER = ["waktu", "kelas", "confidence", "cx", "cy", "x1", "y1", "x2", "y2"]


def append_csv(detections: Iterable["Detection"], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    written = 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        now = datetime.now().isoformat(timespec="seconds")
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            w.writerow([now, d.label, f"{d.confidence:.3f}", d.center[0], d.center[1], x1, y1, x2, y2])
            written += 1
    return written


class CameraError(RuntimeError):
    """Kamera tidak bisa dibuka."""


def parse_source(value: str | int) -> int | str:
    if isinstance(value, int):
        return value
    return int(value) if value.strip().isdigit() else value


class Camera:
    def __init__(self, source: int | str = 0, width: int | None = None, height: int | None = None) -> None:
        self.source, self.width, self.height = source, width, height
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> "Camera":
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.release()
            raise CameraError(
                f"Tidak bisa membuka sumber video: {self.source!r}\n"
                "- Coba nomor kamera lain: --source 1 atau --source 2\n"
                "- Tutup aplikasi lain yang memakai kamera (Zoom, Meet, dll)\n"
                "- macOS: beri izin kamera untuk Terminal/VS Code di "
                "System Settings > Privacy & Security > Camera")
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            raise CameraError("Kamera belum dibuka.")
        ok, frame = self.cap.read()
        return bool(ok), frame if ok else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# =============================================================================
# 3. DETEKTOR  (model + pemeriksa warna + penstabil label)
# =============================================================================
@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    label: str
    confidence: float
    bbox: BBox
    center: tuple[int, int]
    center_norm: tuple[float, float]
    catatan: str = ""        # mis. 'koreksi warna' / 'suara terbanyak'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ciri_warna(frame: np.ndarray, bbox: BBox) -> dict[str, float] | None:
    """Ciri warna bagian TENGAH benda (elips 60% kotak, supaya tangan/latar di tepi tidak ikut)."""
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
    hue = hsv[..., 0][inside].astype(np.float32) * 2.0          # derajat 0..360
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


def tutup_berwarna(frame: np.ndarray, bbox: BBox, min_frac: float = 0.6) -> bool:
    """True kalau benda JELAS berwarna pekat (biru/merah/hijau/ungu/pink...) -> pasti bukan koin.

    Koin rupiah itu logam: abu-abu/perak (saturasi rendah) atau emas/kuning (hue ~10-50).
    Warna kuning-oranye sengaja TIDAK dikoreksi (mirip koin emas) -> diserahkan ke model.
    """
    c = ciri_warna(frame, bbox)
    if c is None or c["frac_pekat"] < min_frac:
        return False
    return not (10.0 <= c["hue"] <= 50.0)


def validate_geometry(
    bbox: BBox,
    frame_width: int,
    frame_height: int,
    min_ar: float = 0.65,
    max_ar: float = 1.50,
    min_size_px: int = 20,
    max_area_frac: float = 0.22,
) -> bool:
    """Validasi fisik objek: koin dan tutup botol berbentuk bundar/elips dan ukurannya wajar.

    Menolak:
    - Jendela / pintu (terlalu lonjong atau terlalu besar)
    - Wajah manusia (terlalu besar melebihi proporsi benda)
    - Noise piksel mikroskopis (< min_size_px)
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < min_size_px or h < min_size_px:
        return False
    if w > frame_width * 0.45 or h > frame_height * 0.45:
        return False
    if (w * h) > (frame_width * frame_height * max_area_frac):
        return False
    ar = w / float(max(h, 1))
    if ar < min_ar or ar > max_ar:
        return False
    return True


class Detector:
    """Muat model sekali, lalu detect(frame) berulang kali."""

    def __init__(self, weights: str | Path | None = None, conf: float = 0.5, imgsz: int = 640,
                 iou: float = 0.45, device: str = "auto", model: Any = None,
                 cek_warna: bool = True, smart_filter: bool = False,
                 min_ar: float = 0.65, max_ar: float = 1.50, max_area_frac: float = 0.22,
                 min_size_px: int = 20, roi: BBox | None = None) -> None:
        self.conf, self.imgsz, self.iou, self.cek_warna = conf, imgsz, iou, cek_warna
        self.smart_filter = smart_filter
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.max_area_frac = max_area_frac
        self.min_size_px = min_size_px
        self.roi = roi
        self.device = pick_device(device)
        if model is None:
            if weights is None or not Path(weights).is_file():
                raise FileNotFoundError(
                    f"File model tidak ditemukan: {weights}\n"
                    "Jalankan training dulu (python sortir.py latih / notebook Colab), lalu taruh best.pt "
                    "di folder models/ atau pakai --weights <path>.")
            from ultralytics import YOLO

            model = YOLO(str(weights))
        self.model = model

    def detect(self, frame: np.ndarray, apply_smart_filter: bool | None = None) -> list[Detection]:
        height, width = frame.shape[:2]
        use_filter = self.smart_filter if apply_smart_filter is None else apply_smart_filter
        # agnostic_nms=True: satu benda = satu kotak = satu label (tidak bisa jadi 'Koin' DAN 'Tutup Botol')
        results = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                     device=self.device, agnostic_nms=True, max_det=30, verbose=False)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        names = result.names
        name_to_id = {str(v): int(k) for k, v in names.items()}
        dets: list[Detection] = []
        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            bbox = clamp_bbox((x1, y1, x2, y2), width, height)
            center = compute_center(bbox)

            # 1. Filter ROI
            if self.roi is not None:
                rx1, ry1, rx2, ry2 = self.roi
                if not (rx1 <= center[0] <= rx2 and ry1 <= center[1] <= ry2):
                    continue

            # 2. Filter Geometri Cerdas
            if use_filter and not validate_geometry(
                bbox, width, height,
                min_ar=self.min_ar, max_ar=self.max_ar,
                min_size_px=self.min_size_px, max_area_frac=self.max_area_frac,
            ):
                continue

            cid = int(box.cls[0])
            cname = str(names[cid])
            det = Detection(cid, cname, display_label(cname), float(box.conf[0]), bbox, center,
                            normalize_point(center, width, height))
            if (self.cek_warna and cname == "koin" and "tutup_botol" in name_to_id
                    and tutup_berwarna(frame, bbox)):
                det = replace(det, class_id=name_to_id["tutup_botol"], class_name="tutup_botol",
                              label=display_label("tutup_botol"), catatan="koreksi warna")
            dets.append(det)
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets


class Penstabil:
    """Label tiap benda diputuskan lewat SUARA TERBANYAK beberapa frame terakhir (anti berkedip).

    Benda dilacak antar-frame lewat tumpang-tindih kotak (IoU). Koordinat TIDAK diubah.
    """

    def __init__(self, jendela: int = 7, iou_min: float = 0.25, umur_maks: int = 6) -> None:
        self.jendela, self.iou_min, self.umur_maks = jendela, iou_min, umur_maks
        self._tracks: list[dict[str, Any]] = []

    def update(self, dets: list[Detection]) -> list[Detection]:
        for t in self._tracks:
            t["umur"] += 1
        pairs = sorted(((iou_xyxy(d.bbox, t["bbox"]), i, j)
                        for i, d in enumerate(dets) for j, t in enumerate(self._tracks)), reverse=True)
        match: dict[int, int] = {}
        used_t: set[int] = set()
        for v, i, j in pairs:
            if v < self.iou_min:
                break
            if i in match or j in used_t:
                continue
            match[i] = j
            used_t.add(j)
        out: list[Detection] = []
        for i, d in enumerate(dets):
            if i in match:
                t = self._tracks[match[i]]
            else:
                t = {"votes": deque(maxlen=self.jendela)}
                self._tracks.append(t)
            t["bbox"], t["umur"] = d.bbox, 0
            t["votes"].append((d.class_id, d.class_name, d.confidence))
            out.append(self._putuskan(d, t["votes"]))
        self._tracks = [t for t in self._tracks if t["umur"] <= self.umur_maks]
        return out

    @staticmethod
    def _putuskan(d: Detection, votes: deque) -> Detection:
        skor: dict[tuple[int, str], float] = defaultdict(float)
        for cid, cname, conf in votes:
            skor[(cid, cname)] += conf
        (cid, cname), _ = max(skor.items(), key=lambda kv: kv[1])
        if cid == d.class_id:
            return d
        conf = float(np.mean([c for i, _, c in votes if i == cid]))
        return replace(d, class_id=cid, class_name=cname, label=display_label(cname),
                       confidence=conf, catatan="suara terbanyak")


# =============================================================================
# 4. DATASET: baca, konversi, gabung, cek kebocoran
# =============================================================================
def find_split_dirs(dataset_dir: Path) -> dict[str, tuple[Path, Path]]:
    found: dict[str, tuple[Path, Path]] = {}
    for split, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            images = dataset_dir / alias / "images"
            if images.is_dir():
                found[split] = (images, dataset_dir / alias / "labels")
                break
    return found


def read_class_names(dataset_dir: Path) -> list[str]:
    p = dataset_dir / "data.yaml"
    if not p.is_file():
        return []
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    return [str(n) for n in names]


def group_key(filename: str) -> str:
    """Kunci 'foto asli': turunan augmentasi Roboflow punya awalan sama sebelum '.rf.'."""
    name = Path(filename).name
    return name.split(".rf.")[0] if ".rf." in name else Path(name).stem


def convert_label_line(line: str, target_id: int) -> str | None:
    """Ubah baris label YOLO (kotak atau poligon) jadi 'id cx cy w h' ternormalisasi."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        nums = [float(p) for p in parts[1:]]
    except ValueError:
        return None
    if len(nums) == 4:
        cx, cy, w, h = nums
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    elif len(nums) >= 6 and len(nums) % 2 == 0:
        xs, ys = nums[0::2], nums[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    else:
        return None
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
    w, h = x2 - x1, y2 - y1
    if w <= 1e-4 or h <= 1e-4:
        return None
    return f"{target_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}"


def dedup_lines(lines: list[str], thr: float = 0.8) -> list[str]:
    """Buang kotak ganda untuk benda yang sama (mis. koin yang dilabeli '500' DAN 'KOIN-RUPIAH-MVPR')."""
    def xyxy(line: str) -> tuple[float, float, float, float]:
        _, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    keep: list[str] = []
    for line in sorted(lines, key=lambda s: -float(s.split()[3]) * float(s.split()[4])):
        if all(not (l.split()[0] == line.split()[0] and iou_xyxy(xyxy(l), xyxy(line)) >= thr) for l in keep):
            keep.append(line)
    return keep


@dataclass
class Source:
    key: str
    path: Path
    target_class: str | None     # semua kelas asli -> 1 kelas final; None = cocokkan lewat nama


@dataclass
class _Entry:
    key: str
    split: str
    image: Path
    lines: list[str]


@dataclass
class MergeReport:
    instances: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    images: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    leak_dropped: int = 0
    skipped_lines: int = 0
    dup_dropped: int = 0
    warnings: list[str] = field(default_factory=list)


def _holdout_valid(entries: list[_Entry], key: str, fraction: float, seed: int) -> None:
    train = [e for e in entries if e.key == key and e.split == "train"]
    groups = sorted({group_key(e.image.name) for e in train})
    if len(groups) < 5:
        return
    rng = random.Random(seed)
    rng.shuffle(groups)
    held = set(groups[: max(1, int(len(groups) * fraction))])
    for e in train:
        if group_key(e.image.name) in held:
            e.split = "valid"


def merge_datasets(sources: list[Source], out_dir: Path, class_names: list[str], seed: int = 0,
                   drop_leak: bool = True, valid_fraction: float = 0.15) -> MergeReport:
    report = MergeReport()
    entries: list[_Entry] = []
    for src in sources:
        splits = find_split_dirs(src.path)
        if not splits:
            report.warnings.append(f"[{src.key}] tidak ada folder train/valid/test di {src.path} -> dilewati")
            continue
        src_names = read_class_names(src.path)
        for split, (img_dir, lbl_dir) in splits.items():
            for img in list_images(img_dir):
                lines: list[str] = []
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.is_file():
                    for raw in lbl.read_text(encoding="utf-8").splitlines():
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            sid = int(float(raw.split()[0]))
                        except (ValueError, IndexError):
                            report.skipped_lines += 1
                            continue
                        target = src.target_class if src.target_class is not None else (
                            src_names[sid] if 0 <= sid < len(src_names) else None)
                        if target not in class_names:
                            report.skipped_lines += 1
                            continue
                        conv = convert_label_line(raw, class_names.index(target))
                        if conv is None:
                            report.skipped_lines += 1
                            continue
                        lines.append(conv)
                    before = len(lines)
                    lines = dedup_lines(lines)
                    report.dup_dropped += before - len(lines)
                entries.append(_Entry(src.key, split, img, lines))
        if "valid" not in splits and "train" in splits:
            _holdout_valid(entries, src.key, valid_fraction, seed)
            report.warnings.append(f"[{src.key}] tidak punya split valid -> "
                                   f"{int(valid_fraction * 100)}% foto asli diambil dari train")

    held = {(e.key, group_key(e.image.name)) for e in entries if e.split in ("valid", "test")}
    leaked = [e for e in entries if e.split == "train" and (e.key, group_key(e.image.name)) in held]
    if leaked:
        if drop_leak:
            ids = {id(e) for e in leaked}
            entries = [e for e in entries if id(e) not in ids]
            report.leak_dropped = len(leaked)
            report.warnings.append(f"Kebocoran data: {len(leaked)} gambar train dibuang karena foto aslinya "
                                   "ada di valid/test")
        else:
            report.warnings.append(f"PERINGATAN kebocoran data: {len(leaked)} gambar train sefoto dengan valid/test")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "valid", "test"):
        if any(e.split == split for e in entries):
            (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    for e in entries:
        new_name = f"{e.key}__{e.image.name}"
        shutil.copy2(e.image, out_dir / e.split / "images" / new_name)
        (out_dir / e.split / "labels" / f"{Path(new_name).stem}.txt").write_text(
            "\n".join(e.lines) + ("\n" if e.lines else ""), encoding="utf-8")
        report.images[e.split][e.key] += 1
        for line in e.lines:
            report.instances[e.split][class_names[int(line.split()[0])]] += 1
    write_data_yaml(out_dir, class_names, has_test=any(e.split == "test" for e in entries))
    return report


def write_data_yaml(out_dir: Path, class_names: list[str], has_test: bool = False) -> Path:
    cfg: dict = {"path": str(out_dir.resolve()), "train": "train/images", "val": "valid/images",
                 "names": {i: n for i, n in enumerate(class_names)}}
    if has_test:
        cfg["test"] = "test/images"
    path = out_dir / "data.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def count_instances(labels_dir: Path, class_names: list[str]) -> dict[str, int]:
    out = {n: 0 for n in class_names}
    for f in labels_dir.glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out[class_names[int(line.split()[0])]] += 1
    return out


def balance_warnings(instances: dict, class_names: list[str]) -> list[str]:
    msgs: list[str] = []
    train, valid = instances.get("train", {}), instances.get("valid", {})
    for name in class_names:
        nt, nv = train.get(name, 0), valid.get(name, 0)
        if nt == 0:
            msgs.append(f"Kelas '{name}' TIDAK punya data train -> tidak akan terdeteksi.")
        elif nt < 100:
            msgs.append(f"Kelas '{name}' hanya {nt} instance di train (disarankan >= 100-300). "
                        "Tambah foto sendiri: python sortir.py foto")
        if nt > 0 and nv == 0:
            msgs.append(f"Kelas '{name}' tidak punya data valid -> metrik kelas ini tidak bisa dihitung.")
    counts = [train.get(n, 0) for n in class_names if train.get(n, 0) > 0]
    if len(counts) >= 2 and max(counts) / min(counts) > 5:
        msgs.append(f"Data tidak seimbang ({max(counts) / min(counts):.1f}x). Kelas yang sedikit kurang akurat.")
    return msgs


def format_report(report: MergeReport, class_names: list[str], notes: Sequence[str] = ()) -> str:
    lines = ["", "Jumlah INSTANCE (kotak) per kelas:", f"{'kelas':<14}{'train':>8}{'valid':>8}{'test':>8}"]
    for n in class_names:
        row = [report.instances.get(s, {}).get(n, 0) for s in ("train", "valid", "test")]
        lines.append(f"{n:<14}{row[0]:>8}{row[1]:>8}{row[2]:>8}")
    lines += ["", "Jumlah GAMBAR per split / sumber:"]
    for s in ("train", "valid", "test"):
        if report.images.get(s):
            lines.append(f"  {s:<6} total={sum(report.images[s].values())}  "
                         f"({', '.join(f'{k}={v}' for k, v in sorted(report.images[s].items()))})")
    if report.skipped_lines:
        lines.append(f"\nBaris label rusak/tidak dikenal yang dilewati: {report.skipped_lines}")
    if report.dup_dropped:
        lines.append(f"Kotak ganda pada benda yang sama dibuang: {report.dup_dropped}")
    allnotes = list(report.warnings) + list(notes)
    if allnotes:
        lines.append("\nCATATAN:")
        lines += [f"  - {w}" for w in allnotes]
    return "\n".join(lines)


# =============================================================================
# 5. LABEL OTOMATIS (satu benda per gambar)
# =============================================================================
def box_from_background(img_bgr: np.ndarray, max_side: int = 320, max_bg_spread: float = 22.0,
                        min_area: float = 0.01, max_area: float = 0.90) -> NormBox | None:
    """Kotak benda di latar polos (warna tepi gambar ~ seragam). None kalau latar ramai/tak ada benda."""
    h, w = img_bgr.shape[:2]
    if h < 16 or w < 16:
        return None
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(img_bgr, (max(16, int(w * scale)), max(16, int(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    lab_raw = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    t = max(2, int(0.05 * min(sh, sw)))
    ring = np.zeros((sh, sw), dtype=bool)
    ring[:t, :] = ring[-t:, :] = ring[:, :t] = ring[:, -t:] = True
    bg_px = lab_raw[ring]
    bg = np.median(bg_px, axis=0)
    if float(np.mean(np.std(bg_px, axis=0))) > max_bg_spread:
        return None
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    dist8 = np.clip(np.sqrt(((lab - bg) ** 2).sum(axis=2)), 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (dist8 > max(float(otsu), 15.0)).astype(np.uint8) * 255
    k_close = max(3, sw // 40) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    frac = cv2.contourArea(best) / float(sh * sw)
    if frac < min_area or frac > max_area:
        return None
    x, y, bw, bh = cv2.boundingRect(best)
    return ((x + bw / 2) / sw, (y + bh / 2) / sh, bw / sw, bh / sh)


def kotak_wajar(box: NormBox | None, width: int, height: int) -> bool:
    """Koin & tutup botol dilihat dari atas itu bulat: kotak sangat lonjong / sangat besar = label salah."""
    if box is None:
        return False
    _, _, bw, bh = box
    if bw * bh > 0.90 or bw * bh < 0.002:
        return False
    return 0.45 <= (bw * width) / max(bh * height, 1e-6) <= 2.2


class WorldLabeler:
    """YOLO-World (zero-shot, berbasis teks, mis. 'bottle cap'). Sebaiknya di Colab."""

    def __init__(self, prompts: Sequence[str], weights: str = "yolov8s-worldv2.pt",
                 conf: float = 0.10, device: str = "auto") -> None:
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(weights)
        self.model.set_classes(list(prompts))
        self.conf, self.device = conf, pick_device(device)

    def box(self, img: np.ndarray) -> NormBox | None:
        return _best_box(self.model.predict(img, conf=self.conf, device=self.device, verbose=False)[0], img)


class ModelLabeler:
    """Pakai model YOLO kamu sendiri HANYA untuk mencari lokasi benda (kelasnya diabaikan).

    Cocok untuk foto sendiri di latar ramai (tangan, ruangan): model lama biasanya sudah
    bisa menemukan benda bulatnya, hanya salah menamai. Nama kelas diambil dari nama folder foto.
    """

    def __init__(self, weights: Path, conf: float = 0.25, device: str = "auto") -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf, self.device = conf, pick_device(device)

    def box(self, img: np.ndarray) -> NormBox | None:
        return _best_box(self.model.predict(img, conf=self.conf, device=self.device,
                                            agnostic_nms=True, verbose=False)[0], img)


def _best_box(res: Any, img: np.ndarray) -> NormBox | None:
    if res.boxes is None or len(res.boxes) == 0:
        return None
    i = int(res.boxes.conf.argmax())
    x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
    h, w = img.shape[:2]
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    if bw <= 0.01 or bh <= 0.01:
        return None
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw, bh)


def write_yolo_label(path: Path, class_id: int, box: NormBox) -> None:
    cx, cy, w, h = box
    path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


def draw_preview(img: np.ndarray, box: NormBox | None, size: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    out = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[: out.shape[0], : out.shape[1]] = out
    if box is not None:
        cx, cy, bw, bh = box
        cv2.rectangle(canvas, (int((cx - bw / 2) * out.shape[1]), int((cy - bh / 2) * out.shape[0])),
                      (int((cx + bw / 2) * out.shape[1]), int((cy + bh / 2) * out.shape[0])), (0, 0, 255), 2)
    else:
        cv2.putText(canvas, "TANPA KOTAK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas


def save_preview_grid(tiles: list[np.ndarray], out_path: Path, cols: int = 6) -> None:
    if not tiles:
        return
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i: i + cols]
        row += [np.zeros_like(tiles[0])] * (cols - len(row))
        rows.append(np.hstack(row))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))


# =============================================================================
# 6. DATA SINTETIS: koin & tutup botol ditempel ke SEMUA jenis latar
#    Obat untuk akar masalah 'latar = kelas' (lihat README).
# =============================================================================
def _alpha_elips(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (max(1, int(w * 0.48)), max(1, int(h * 0.48))), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), max(1.0, 0.03 * min(h, w)))


def kumpulkan_potongan(out_dir: Path, class_names: list[str], rng: random.Random,
                       per_class: int = 300) -> dict[int, list[np.ndarray]]:
    """Potong benda (hanya dari split TRAIN) berdasarkan kotak labelnya."""
    img_dir, lbl_dir = out_dir / "train" / "images", out_dir / "train" / "labels"
    cuts: dict[int, list[np.ndarray]] = {i: [] for i in range(len(class_names))}
    files = sorted(lbl_dir.glob("*.txt"))
    rng.shuffle(files)
    imgs = {p.stem: p for p in list_images(img_dir)}
    for lbl in files:
        if all(len(v) >= per_class for v in cuts.values()):
            break
        if lbl.stem.startswith(("sint__", "latar__")) or lbl.stem not in imgs:
            continue
        rows = [l.split() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows or all(len(cuts[int(r[0])]) >= per_class for r in rows):
            continue
        img = cv2.imread(str(imgs[lbl.stem]))
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = []
        for r in rows:
            cid, cx, cy, bw, bh = int(r[0]), *(float(v) for v in r[1:5])
            boxes.append((cid, (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
        for k, (cid, x1, y1, x2, y2) in enumerate(boxes):
            if len(cuts[cid]) >= per_class:
                continue
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) < 28 or not 0.5 <= bw / bh <= 2.0:
                continue
            if any(iou_xyxy((x1, y1, x2, y2), b[1:]) > 0.05 for j, b in enumerate(boxes) if j != k):
                continue
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue
            s = 256 / max(crop.shape[:2])
            if s < 1:
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cuts[cid].append(crop)
    return cuts


def kumpulkan_latar(out_dir: Path, latar_dir: Path | None, rng: random.Random,
                    n_dataset: int = 120, n_latar: int = 120) -> tuple[list[np.ndarray], int]:
    """Latar = (1) foto latar kosong milik sendiri, (2) gambar dataset yang bendanya dihapus (inpaint)."""
    bgs: list[np.ndarray] = []

    def kecil(im: np.ndarray, side: int = 800) -> np.ndarray:
        s = side / max(im.shape[:2])
        return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im

    n_own = 0
    if latar_dir is not None:
        files = list_images(latar_dir)
        rng.shuffle(files)
        for p in files[:n_latar]:
            im = cv2.imread(str(p))
            if im is not None:
                bgs.append(kecil(im))
                n_own += 1
    img_dir, lbl_dir = out_dir / "train" / "images", out_dir / "train" / "labels"
    files = [p for p in list_images(img_dir) if not p.stem.startswith(("sint__", "latar__"))]
    rng.shuffle(files)
    for p in files[:n_dataset]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = kecil(im)
        H, W = im.shape[:2]
        mask = np.zeros((H, W), np.uint8)
        lbl = lbl_dir / f"{p.stem}.txt"
        if lbl.is_file():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    _, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
                    cv2.ellipse(mask, (int(cx * W), int(cy * H)), (int(bw * W * 0.62), int(bh * H * 0.62)),
                                0, 0, 360, 255, -1)
        bgs.append(cv2.inpaint(im, mask, 5, cv2.INPAINT_TELEA) if mask.any() else im)
    return bgs, n_own


def _latar_polos(rng: random.Random, size: int) -> np.ndarray:
    """Permukaan polos/gradasi + noise (meja, conveyor, dinding)."""
    c1 = np.array([rng.randint(30, 235) for _ in range(3)], np.float32)
    c2 = np.clip(c1 + np.array([rng.randint(-35, 35) for _ in range(3)], np.float32), 0, 255)
    t = np.linspace(0, 1, size, dtype=np.float32)
    t = np.tile(t, (size, 1)) if rng.random() < 0.5 else np.tile(t[:, None], (1, size))
    img = c1[None, None] * (1 - t[..., None]) + c2[None, None] * t[..., None]
    img += np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(1, 6), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _rotasi_potongan(crop: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Putar potongan + alpha elips, lalu pangkas ke kotak tepat objek. Return (bgr, alpha)."""
    h, w = crop.shape[:2]
    alpha = _alpha_elips(h, w)
    s = int(math.ceil(math.hypot(h, w))) + 4
    pad_img = np.zeros((s, s, 3), np.uint8)
    pad_a = np.zeros((s, s), np.float32)
    oy, ox = (s - h) // 2, (s - w) // 2
    pad_img[oy:oy + h, ox:ox + w], pad_a[oy:oy + h, ox:ox + w] = crop, alpha
    M = cv2.getRotationMatrix2D((s / 2, s / 2), angle, 1.0)
    img = cv2.warpAffine(pad_img, M, (s, s), flags=cv2.INTER_LINEAR)
    a = cv2.warpAffine(pad_a, M, (s, s), flags=cv2.INTER_LINEAR)
    ys, xs = np.where(a > 0.5)
    if len(xs) == 0:
        return crop, alpha
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return img[y1:y2, x1:x2], a[y1:y2, x1:x2]


def buat_adegan(cuts: dict[int, list[np.ndarray]], bgs: list[np.ndarray], rng: random.Random,
                size: int = 640, kosong: bool = False) -> tuple[np.ndarray, list[str]]:
    """Satu gambar sintetis: latar acak + 1-4 benda dari kelas acak. Return (gambar, baris label YOLO)."""
    if bgs and rng.random() < 0.85:
        bg = bgs[rng.randrange(len(bgs))]
        s = size / min(bg.shape[:2]) * rng.uniform(1.0, 1.6)
        bg = cv2.resize(bg, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        y0, x0 = rng.randint(0, bg.shape[0] - size), rng.randint(0, bg.shape[1] - size)
        canvas = bg[y0:y0 + size, x0:x0 + size].astype(np.float32)
        if rng.random() < 0.5:
            canvas = canvas[:, ::-1]
        if rng.random() < 0.3:
            canvas = canvas[::-1]
        canvas = np.ascontiguousarray(canvas)
    else:
        canvas = _latar_polos(rng, size).astype(np.float32)
    avail = [c for c, v in cuts.items() if v]
    placed: list[tuple[int, int, int, int]] = []
    labels: list[str] = []
    for _ in range(0 if kosong or not avail else rng.randint(1, 4)):
        cid = rng.choice(avail)
        obj, a = _rotasi_potongan(rng.choice(cuts[cid]), rng.uniform(0, 360))
        d = rng.uniform(0.07, 0.40) * size
        sc = d / max(obj.shape[:2])
        obj = cv2.resize(obj, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        a = cv2.resize(a, (obj.shape[1], obj.shape[0]), interpolation=cv2.INTER_LINEAR)
        oh, ow = obj.shape[:2]
        if oh < 12 or ow < 12 or oh >= size or ow >= size:
            continue
        obj = obj.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)   # kecerahan/kontras
        if rng.random() < 0.5:
            sig = rng.uniform(0.3, 1.5)
            obj = cv2.GaussianBlur(obj, (0, 0), sig)
        for _try in range(25):
            x, y = rng.randint(0, size - ow), rng.randint(0, size - oh)
            if all(iou_xyxy((x, y, x + ow, y + oh), p) <= 0.02 for p in placed):
                break
        else:
            continue
        roi = canvas[y:y + oh, x:x + ow]
        canvas[y:y + oh, x:x + ow] = roi * (1 - a[..., None]) + obj * a[..., None]
        ys, xs = np.where(a > 0.5)
        bx1, bx2, by1, by2 = x + xs.min(), x + xs.max() + 1, y + ys.min(), y + ys.max() + 1
        placed.append((x, y, x + ow, y + oh))
        labels.append(f"{cid} {(bx1 + bx2) / 2 / size:.6f} {(by1 + by2) / 2 / size:.6f} "
                      f"{(bx2 - bx1) / size:.6f} {(by2 - by1) / size:.6f}")
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(1, 5), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(60, 95)])
    return (cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img), labels


def buat_sintetis(out_dir: Path, class_names: list[str], n: int, latar_dir: Path | None = LATAR_DIR,
                  seed: int = 0, preview: Path | None = None) -> dict[str, Any]:
    """Tambahkan n gambar sintetis ke out_dir/train. Valid/test TIDAK disentuh (metrik tetap jujur)."""
    rng = random.Random(seed)
    cuts = kumpulkan_potongan(out_dir, class_names, rng)
    ada = {class_names[i]: len(v) for i, v in cuts.items()}
    if not any(ada.values()) or n <= 0:
        return {"dibuat": 0, "potongan": ada, "latar_sendiri": 0}
    bgs, n_own = kumpulkan_latar(out_dir, latar_dir, rng)
    (out_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    tiles: list[np.ndarray] = []
    for i in range(n):
        img, labels = buat_adegan(cuts, bgs, rng, kosong=(rng.random() < 0.05))
        name = f"sint__{i:05d}"
        cv2.imwrite(str(out_dir / "train" / "images" / f"{name}.jpg"), img)
        (out_dir / "train" / "labels" / f"{name}.txt").write_text("\n".join(labels) + ("\n" if labels else ""),
                                                                    encoding="utf-8")
        if preview is not None and i < 12:
            tile = img.copy()
            for line in labels:
                cid, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = (float(v) * img.shape[0] for v in (cx, cy, bw, bh))
                cv2.rectangle(tile, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)),
                              (0, 0, 255), 2)
                cv2.putText(tile, class_names[int(cid)], (int(cx - bw / 2), max(15, int(cy - bh / 2) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(cv2.resize(tile, (320, 320)))
    if preview is not None:
        save_preview_grid(tiles, preview, cols=4)
    return {"dibuat": n, "potongan": ada, "latar_sendiri": n_own, "latar_total": len(bgs)}


def tambah_latar_kosong(out_dir: Path, latar_dir: Path, maks: int = 150, seed: int = 0) -> int:
    """Foto latar kosong (tangan/meja tanpa benda) jadi contoh NEGATIF: 'ini bukan benda'."""
    files = list_images(latar_dir)
    random.Random(seed).shuffle(files)
    (out_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    for p in files[:maks]:
        shutil.copy2(p, out_dir / "train" / "images" / f"latar__{p.name}")
        (out_dir / "train" / "labels" / f"latar__{p.stem}.txt").write_text("", encoding="utf-8")
    return min(len(files), maks)


# =============================================================================
# 7. PERINTAH
# =============================================================================
# ---- unduh -----------------------------------------------------------------
API_KEY_HELP = """
API key Roboflow belum ditemukan. Cara mendapatkannya:
  1. Login di https://app.roboflow.com (gratis)
  2. Settings workspace > "API Keys" (atau "Roboflow API") > salin "Private API Key"
Lalu pilih salah satu:
  - file .env di folder proyek berisi:  ROBOFLOW_API_KEY=kunci_kamu
  - export ROBOFLOW_API_KEY=kunci_kamu   (Mac/Linux)
  - di Colab: menu Secrets (ikon kunci), nama ROBOFLOW_API_KEY
"""


def get_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        try:
            from google.colab import userdata  # type: ignore

            key = (userdata.get("ROBOFLOW_API_KEY") or "").strip()
        except Exception:
            pass
    if not key and sys.stdin.isatty():
        import getpass

        key = getpass.getpass("Tempel API key Roboflow (tidak terlihat): ").strip()
    if not key:
        raise SystemExit(API_KEY_HELP)
    return key


def _latest_version(project: Any) -> int | None:
    try:
        versions = project.versions()
    except Exception:
        return None
    nums = []
    for v in versions or []:
        for attr in ("version", "id", "name"):
            raw = getattr(v, attr, None)
            m = re.search(r"(\d+)$", str(raw).rstrip("/").split("/")[-1]) if raw is not None else None
            if m:
                nums.append(int(m.group(1)))
                break
    return max(nums) if nums else None


def cmd_unduh(args: argparse.Namespace) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    keys = [d["key"] for d in DATASETS]
    for item in args.zip:
        key, _, path = item.partition("=")
        if key not in keys or not Path(path).is_file():
            print(f"[zip] lewati '{item}': key harus salah satu dari {keys} dan file harus ada")
            continue
        dest = RAW_DIR / key
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
        kids = [c for c in dest.iterdir() if c.is_dir()]
        if not (dest / "train").exists() and len(kids) == 1 and (kids[0] / "train").exists():
            for it in kids[0].iterdir():
                shutil.move(str(it), str(dest / it.name))
            kids[0].rmdir()
        print(f"[zip] {key}: diekstrak ke {dest}")

    has_data = lambda p: p.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in p.rglob("*"))  # noqa: E731
    specs = [d for d in DATASETS if not args.only or d["key"] == args.only]
    pending = [d for d in specs if args.force or not has_data(RAW_DIR / d["key"])]
    rf = None
    if pending:
        from roboflow import Roboflow

        rf = Roboflow(api_key=get_api_key())
    failed = []
    for spec in specs:
        dest = RAW_DIR / spec["key"]
        print(f"\n[{spec['key']}] {spec['workspace']}/{spec['project']} ({spec['kind']})")
        try:
            if rf is not None and spec in pending:
                shutil.rmtree(dest, ignore_errors=True)
                project = rf.workspace(spec["workspace"]).project(spec["project"])
                ver = spec["version"] or _latest_version(project)
                if ver is None:
                    raise RuntimeError("project belum punya versi. Fork Dataset > Generate > Download (YOLOv8), "
                                       f"lalu: python sortir.py unduh --zip {spec['key']}=<file.zip>")
                fmt = "yolov8" if spec["kind"] == "detection" else "folder"
                print(f"  versi {ver}, format {fmt} -> {dest}")
                project.version(ver).download(fmt, location=str(dest))
            else:
                print(f"  sudah ada di {dest}")
            if spec["kind"] == "detection":
                print(f"  kelas asli: {read_class_names(dest)} -> semua digabung jadi '{spec['target_class']}'")
        except Exception as exc:
            failed.append(spec["key"])
            print(f"  GAGAL: {exc}")
    print("\n" + ("Selesai tanpa error." if not failed else f"Selesai, gagal: {failed}"))
    return 1 if failed else 0


# ---- foto ------------------------------------------------------------------
def cmd_foto(args: argparse.Namespace) -> int:
    """Ambil foto sendiri. 1/2 = kelas | 0 = LATAR KOSONG | SPASI foto | a otomatis | q selesai."""
    try:
        cam = Camera(parse_source(args.source), args.width, args.height).open()
    except CameraError as exc:
        print(exc)
        return 1
    folders = [OWN_RAW_DIR / n for n in CLASS_NAMES] + [LATAR_DIR]
    names = [display_label(n) for n in CLASS_NAMES] + ["LATAR KOSONG (tanpa benda)"]
    keys = {ord(str(i + 1)): i for i in range(len(CLASS_NAMES))}
    keys[ord("0")] = len(CLASS_NAMES)
    cur, auto, last = 0, False, 0.0
    count = lambda f: len(list(f.glob("*.jpg"))) if f.is_dir() else 0  # noqa: E731
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                break

            def simpan() -> None:
                folders[cur].mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folders[cur] / f"{timestamp()}.jpg"), frame)

            if auto and time.time() - last >= args.interval:
                simpan()
                last = time.time()
            view = frame.copy()
            st = f"{names[cur]} | foto: {count(folders[cur])} | auto: {'ON' if auto else 'off'}"
            cv2.putText(view, st, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, st, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, "1/2 kelas | 0 latar kosong | SPASI foto | a auto | q selesai",
                        (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Ambil foto sendiri", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k in keys:
                cur = keys[k]
            elif k == ord(" "):
                simpan()
            elif k == ord("a"):
                auto = not auto
    finally:
        cam.release()
        cv2.destroyAllWindows()
    print("\nJumlah foto:")
    for n, f in zip(names, folders):
        print(f"  {n:<28}{count(f)}")
    print("\nLanjut: python sortir.py label sendiri")
    return 0


# ---- label -----------------------------------------------------------------
def _collect_by_split(src_dir: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for split, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            d = src_dir / alias
            if d.is_dir():
                result[split] = list_images(d, recursive=True)
                break
    return result or {"train": list_images(src_dir, recursive=True)}


def _pilih_labeler(metode: str, prompts: Sequence[str], args: argparse.Namespace) -> tuple[Any, str]:
    if metode == "model":
        return ModelLabeler(args.weights, device=args.device), "model"
    if metode in ("auto", "world"):
        try:
            return WorldLabeler(prompts, device=args.device), "world"
        except Exception as exc:
            if metode == "world":
                raise SystemExit(f"YOLO-World gagal dimuat: {exc}")
            print(f"YOLO-World tidak tersedia ({exc.__class__.__name__}); pakai metode 'cv' (latar polos saja).")
    return None, "cv"


def cmd_label(args: argparse.Namespace) -> int:
    tiles: list[np.ndarray] = []
    if args.target == "tutup-botol":
        src, out = RAW_DIR / "tutup_botol", TUTUP_BOTOL_YOLO
        if not src.is_dir():
            print(f"Folder {src} tidak ada. Jalankan dulu: python sortir.py unduh")
            return 1
        labeler, metode = _pilih_labeler(args.metode, ["bottle cap"], args)
        print(f"Metode label otomatis: {metode}")
        images = _collect_by_split(src)
        shutil.rmtree(out, ignore_errors=True)
        stats = {"ok": 0, "ditolak": 0}
        total = sum(len(v) for v in images.values())
        step, seen = max(1, total // 36), 0
        class_id = CLASS_NAMES.index("tutup_botol")
        for split, files in images.items():
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)
            for p in (files[: args.limit] if args.limit else files):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                box = labeler.box(img) if labeler else None
                if box is None:
                    box = box_from_background(img)
                seen += 1
                if seen % step == 0 and len(tiles) < 36:
                    tiles.append(draw_preview(img, box))
                if not kotak_wajar(box, img.shape[1], img.shape[0]):
                    stats["ditolak"] += 1          # tanpa kotak / kotak tidak wajar -> jangan dipakai
                    continue
                prefix = p.parent.name if p.parent.name not in SPLIT_ALIASES[split] else "x"
                name = f"{prefix}__{p.name}"
                shutil.copy2(p, out / split / "images" / name)
                write_yolo_label(out / split / "labels" / f"{Path(name).stem}.txt", class_id, box)
                stats["ok"] += 1
        (out / "data.yaml").write_text(yaml.safe_dump({"names": ["tutup_botol"]}), encoding="utf-8")
        preview = OUTPUT_DIR / "preview_autolabel_tutup_botol.jpg"
        save_preview_grid(tiles, preview)
        print(f"\nBerhasil diberi kotak: {stats['ok']} | ditolak: {stats['ditolak']}\nHasil: {out}")
        print(f"PREVIEW (cek manual!): {preview}")
        return 0 if stats["ok"] else 1

    # ---- foto sendiri
    if not OWN_RAW_DIR.is_dir():
        print(f"{OWN_RAW_DIR} belum ada. Jalankan dulu: python sortir.py foto")
        return 1
    shutil.rmtree(OWN_DIR, ignore_errors=True)
    worlds: dict[str, Any] = {}
    model_labeler = None
    stats = {}
    for cid, name in enumerate(CLASS_NAMES):
        files = list_images(OWN_RAW_DIR / name)
        if not files:
            continue
        if args.metode == "model":
            model_labeler = model_labeler or ModelLabeler(args.weights, device=args.device)
            labeler = model_labeler
        elif args.metode == "world":
            labeler = worlds.setdefault(name, WorldLabeler([PROMPTS[name]], device=args.device))
        else:
            labeler = None
        stats[name] = {"ok": 0, "lewat": 0}
        for i, p in enumerate(files):
            img = cv2.imread(str(p))
            if img is None:
                continue
            box = labeler.box(img) if labeler else box_from_background(img)
            if box is None and labeler:
                box = box_from_background(img)
            if i % 15 == 0 and len(tiles) < 36:
                tiles.append(draw_preview(img, box))
            if not kotak_wajar(box, img.shape[1], img.shape[0]):
                stats[name]["lewat"] += 1
                continue
            # bagi per BLOK foto berurutan (bukan acak): foto berdekatan hampir sama -> valid jadi 'contekan'
            split = "valid" if (i // args.block) % args.valid_every == args.valid_every - 1 else "train"
            (OWN_DIR / split / "images").mkdir(parents=True, exist_ok=True)
            (OWN_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
            out_name = f"{name}__{p.name}"
            shutil.copy2(p, OWN_DIR / split / "images" / out_name)
            write_yolo_label(OWN_DIR / split / "labels" / f"{Path(out_name).stem}.txt", cid, box)
            stats[name]["ok"] += 1
    if not stats:
        print("Tidak ada foto di", OWN_RAW_DIR)
        return 1
    (OWN_DIR / "data.yaml").write_text(yaml.safe_dump({"names": CLASS_NAMES}), encoding="utf-8")
    preview = OUTPUT_DIR / "preview_autolabel_foto_sendiri.jpg"
    save_preview_grid(tiles, preview)
    for n, s in stats.items():
        print(f"{n:<12} berhasil: {s['ok']:>4} | dilewati: {s['lewat']:>4}")
    print(f"\nHasil: {OWN_DIR}\nPREVIEW (cek manual!): {preview}\nBerikutnya: python sortir.py gabung")
    return 0


# ---- gabung ----------------------------------------------------------------
def build_sources() -> tuple[list[Source], list[str]]:
    sources: list[Source] = []
    missing: list[str] = []
    for d in DATASETS:
        if d["kind"] == "classification":
            path, key, hint = TUTUP_BOTOL_YOLO, "tutup_botol_yolo", "python sortir.py label tutup-botol"
        else:
            path, key, hint = RAW_DIR / d["key"], d["key"], f"python sortir.py unduh --only {d['key']}"
        if path.is_dir():
            sources.append(Source(key, path, d["target_class"]))
        else:
            missing.append(f"{d['key']}: belum ada -> jalankan `{hint}`")
    if OWN_DIR.is_dir():
        sources.append(Source("data_sendiri", OWN_DIR, None))
    return sources, missing


def cmd_gabung(args: argparse.Namespace) -> int:
    sources, missing = build_sources()
    for m in missing:
        print(f"[lewati] {m}")
    if not sources:
        print("Tidak ada dataset yang bisa digabung.")
        return 1
    report = merge_datasets(sources, args.out, CLASS_NAMES, seed=args.seed, drop_leak=not args.keep_leak)
    notes: list[str] = []
    n_latar = tambah_latar_kosong(args.out, LATAR_DIR, seed=args.seed) if LATAR_DIR.is_dir() else 0
    if n_latar:
        notes.append(f"{n_latar} foto latar kosong ditambahkan sebagai contoh negatif.")
    else:
        notes.append("Belum ada foto LATAR KOSONG (python sortir.py foto, tombol 0). Sangat membantu: "
                     "tangan/meja/conveyor tanpa benda.")
    if args.sintetis > 0:
        info = buat_sintetis(args.out, CLASS_NAMES, args.sintetis, LATAR_DIR if n_latar else None,
                             seed=args.seed, preview=OUTPUT_DIR / "preview_sintetis.jpg")
        notes.append(f"{info['dibuat']} gambar SINTETIS dibuat (potongan: {info['potongan']}, "
                     f"latar sendiri: {info.get('latar_sendiri', 0)}). Cek outputs/preview_sintetis.jpg")
    # hitung ulang instance dari label di disk (sudah termasuk sintetis)
    for split in ("train", "valid", "test"):
        ld = args.out / split / "labels"
        if ld.is_dir():
            report.instances[split] = count_instances(ld, CLASS_NAMES)
            report.images[split] = {"total": len(list(ld.glob("*.txt")))}
    notes += balance_warnings(report.instances, CLASS_NAMES)
    print(format_report(report, CLASS_NAMES, notes))
    print(f"\nDataset siap: {args.out / 'data.yaml'}")
    return 0


# ---- latih -----------------------------------------------------------------
def cmd_latih(args: argparse.Namespace) -> int:
    if not args.data.is_file():
        print(f"{args.data} tidak ditemukan. Jalankan dulu: python sortir.py gabung")
        return 1
    from ultralytics import YOLO

    device = pick_device(args.device)
    print(f"Device: {device} | data: {args.data}")
    if device == "cpu":
        print("PERINGATAN: training di CPU sangat lambat. Pakai Google Colab (Runtime > T4 GPU).")
    project = args.project.resolve()
    if args.resume:
        model = YOLO(str(project / args.name / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
            project=str(project), name=args.name, exist_ok=True, patience=args.patience,
            workers=args.workers, save_period=10, plots=True, cos_lr=True,
            # benda bulat dilihat dari atas -> boleh dibalik; warna adalah ciri penting -> jangan digeser banyak
            flipud=0.5, fliplr=0.5, hsv_h=0.01, hsv_s=0.5, hsv_v=0.4, scale=0.6, mixup=0.1, close_mosaic=10,
        )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        print("best.pt tidak ditemukan, training mungkin gagal.")
        return 1
    print(f"\nEvaluasi model terbaik: {best}")
    m = YOLO(str(best)).val(data=str(args.data), device=device, plots=True)
    print(f"\nmAP50 = {m.box.map50:.3f} | mAP50-95 = {m.box.map:.3f}")
    print(f"{'kelas':<14}{'precision':>10}{'recall':>9}{'mAP50':>8}{'mAP50-95':>10}")
    for i, ci in enumerate(m.box.ap_class_index):
        p, r, a50, a = m.box.class_result(i)
        print(f"{CLASS_NAMES[int(ci)]:<14}{p:>10.3f}{r:>9.3f}{a50:>8.3f}{a:>10.3f}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, DEFAULT_WEIGHTS)
    print(f"\nModel disalin ke: {DEFAULT_WEIGHTS}")
    print("Metrik tinggi di data valid BUKAN jaminan akurat di kamera nyata -> tes langsung: python sortir.py deteksi")
    return 0


# ---- deteksi ---------------------------------------------------------------
WINDOW = "Deteksi Koin / Tutup Botol"


def print_detections(dets: list[Detection]) -> None:
    stamp = time.strftime("%H:%M:%S")
    for d in dets:
        extra = f" | {d.catatan}" if d.catatan else ""
        print(f"[{stamp}] {d.label} | conf={d.confidence:.2f} | center={d.center} | "
              f"center_norm={d.center_norm} | bbox={d.bbox}{extra}")


def annotate(frame: np.ndarray, dets: list[Detection], fps: float, persen: bool = False) -> None:
    for d in dets:
        draw_detection(frame, d, persen=persen)
    draw_overlay(frame, fps, len(dets))


def run_image(detector: Detector, path: Path, show: bool, persen: bool) -> int:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Tidak bisa membaca gambar: {path}")
        return 1
    dets = detector.detect(frame)
    print_detections(dets) if dets else print("Tidak ada objek terdeteksi.")
    annotate(frame, dets, 0.0, persen)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"hasil_{path.stem}.jpg"
    cv2.imwrite(str(out), frame)
    print(f"Hasil disimpan: {out}")
    if show:
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def parse_roi_arg(val: str | None, width: int, height: int) -> BBox | None:
    if not val:
        return None
    val = val.strip().lower()
    if val == "auto":
        return int(width * 0.15), int(height * 0.20), int(width * 0.85), int(height * 0.90)
    try:
        parts = [int(p.strip()) for p in val.split(",")]
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], parts[3]
    except Exception:
        pass
    print(f"Format ROI tidak valid: {val}. Gunakan 'x1,y1,x2,y2' atau 'auto'.")
    return None


def run_stream(detector: Detector, source: int | str, args: argparse.Namespace) -> int:
    try:
        cam = Camera(source, args.width, args.height).open()
    except CameraError as exc:
        print(exc)
        return 1

    smart_filter_on = not args.no_smart_filter
    roi_coords = parse_roi_arg(args.roi, args.width, args.height)
    roi_active = roi_coords is not None
    current_conf = detector.conf
    stab = None if args.tanpa_stabil else Penstabil(jendela=args.jendela)

    print("\nKontrol Keyboard:")
    print("  [q] / [ESC] : Keluar")
    print("  [s]         : Simpan tangkapan layar (screenshot)")
    print("  [c]         : Simpan koordinat objek ke CSV")
    print("  [f]         : Toggle Smart Filter (ON/OFF)")
    print("  [r]         : Toggle ROI Zona Kerja (ON/OFF)")
    print("  [+] / [-]   : Naikkan / turunkan confidence threshold\n")

    prev, fps, last_print = time.time(), 0.0, 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("Frame habis / kamera berhenti.")
                break
            if args.flip:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            if roi_coords is None:
                roi_coords = (int(w * 0.15), int(h * 0.20), int(w * 0.85), int(h * 0.90))

            detector.roi = roi_coords if roi_active else None
            detector.conf = current_conf
            dets = detector.detect(frame, apply_smart_filter=smart_filter_on)
            if stab is not None:
                dets = stab.update(dets)
            now = time.time()
            inst = 1.0 / max(now - prev, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            prev = now
            if dets and now - last_print >= args.print_every:
                print_detections(dets)
                last_print = now
            if roi_active and roi_coords is not None:
                rx1, ry1, rx2, ry2 = roi_coords
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
            annotate(frame, dets, fps, args.persen)
            cv2.imshow(WINDOW, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                print(f"Screenshot: {save_screenshot(frame, OUTPUT_DIR)}")
            if k == ord("c"):
                n = append_csv(dets, OUTPUT_DIR / "koordinat.csv")
                print(f"{n} koordinat disimpan ke {OUTPUT_DIR / 'koordinat.csv'}")
            if k in (ord("f"), ord("F")):
                smart_filter_on = not smart_filter_on
                print(f"[Smart Filter] -> {'ON' if smart_filter_on else 'OFF'}")
            if k in (ord("r"), ord("R")):
                roi_active = not roi_active
                print(f"[ROI Conveyor] -> {'ON' if roi_active else 'OFF'}")
            if k in (ord("+"), ord("=")):
                current_conf = min(0.95, round(current_conf + 0.05, 2))
                print(f"[Confidence] -> {current_conf:.2f}")
            if k in (ord("-"), ord("_")):
                current_conf = max(0.10, round(current_conf - 0.05, 2))
                print(f"[Confidence] -> {current_conf:.2f}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return 0


def cmd_deteksi(args: argparse.Namespace) -> int:
    try:
        det = Detector(
            args.weights,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            cek_warna=not args.tanpa_cek_warna,
            smart_filter=not args.no_smart_filter,
            min_ar=args.min_ar,
            max_ar=args.max_ar,
            max_area_frac=args.max_area,
            min_size_px=args.min_size,
        )
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print(f"Model: {args.weights} | device: {det.device} | conf: {args.conf} | "
          f"cek warna: {'ON' if det.cek_warna else 'off'} | penstabil: {'off' if args.tanpa_stabil else 'ON'} | "
          f"smart filter: {'ON' if det.smart_filter else 'OFF'}")
    source = parse_source(args.source)
    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS:
        return run_image(det, Path(source), not args.no_show, args.persen)
    return run_stream(det, source, args)


# =============================================================================
# 8. CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sortir.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("unduh", help="unduh dataset Roboflow")
    p.add_argument("--only", choices=[d["key"] for d in DATASETS])
    p.add_argument("--force", action="store_true")
    p.add_argument("--zip", action="append", default=[], metavar="KEY=PATH", help="pakai zip hasil unduh manual")
    p.set_defaults(fn=cmd_unduh)

    p = sub.add_parser("foto", help="ambil foto sendiri lewat webcam")
    p.add_argument("--source", default="0")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--interval", type=float, default=0.4)
    p.set_defaults(fn=cmd_foto)

    p = sub.add_parser("label", help="kotak otomatis: 'tutup-botol' atau 'sendiri'")
    p.add_argument("target", choices=["tutup-botol", "sendiri"])
    p.add_argument("--metode", choices=["auto", "world", "cv", "model"], default="auto",
                   help="world=YOLO-World | cv=latar polos | model=pakai best.pt lama sebagai pencari lokasi")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--block", type=int, default=10)
    p.add_argument("--valid-every", type=int, default=5)
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("gabung", help="gabung dataset + data sintetis")
    p.add_argument("--out", type=Path, default=MERGED_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sintetis", type=int, default=800, help="jumlah gambar sintetis (0 = matikan)")
    p.add_argument("--keep-leak", action="store_true")
    p.set_defaults(fn=cmd_gabung)

    p = sub.add_parser("latih", help="training YOLOv8")
    p.add_argument("--data", type=Path, default=MERGED_DIR / "data.yaml")
    p.add_argument("--model", default="yolov8s.pt", help="yolov8n.pt (tercepat) / yolov8s.pt (lebih akurat)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--project", type=Path, default=RUNS_DIR)
    p.add_argument("--name", default="sortir")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(fn=cmd_latih)

    p = sub.add_parser("deteksi", help="deteksi real-time (webcam / gambar / video)")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--source", default="0", help="nomor kamera atau path gambar/video")
    p.add_argument("--conf", type=float, default=0.65, help="threshold confidence (default: 0.65)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--print-every", type=float, default=0.5)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--persen", action="store_true", help="tampilkan persentase confidence di judul")
    p.add_argument("--no-smart-filter", action="store_true", help="matikan filter geometri cerdas")
    p.add_argument("--tanpa-cek-warna", action="store_true", help="matikan koreksi warna")
    p.add_argument("--tanpa-stabil", action="store_true", help="matikan suara-terbanyak antar-frame")
    p.add_argument("--jendela", type=int, default=7, help="jumlah frame untuk suara terbanyak")
    p.add_argument("--roi", default=None, help="area aktif 'x1,y1,x2,y2' atau 'auto'")
    p.add_argument("--min-ar", type=float, default=0.65, help="aspek rasio minimum")
    p.add_argument("--max-ar", type=float, default=1.50, help="aspek rasio maksimum")
    p.add_argument("--max-area", type=float, default=0.22, help="fraksi luas maksimal terhadap frame")
    p.add_argument("--min-size", type=int, default=20, help="ukuran minimum piksel")
    p.set_defaults(fn=cmd_deteksi)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
