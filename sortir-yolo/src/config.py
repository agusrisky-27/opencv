"""Konfigurasi pusat proyek.

Semua nama kelas, label tampilan, dan sumber dataset diatur di sini supaya
skrip lain tidak perlu hardcode.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Folder kerja
RAW_DIR = ROOT / "datasets_raw"          # hasil unduhan Roboflow (mentah)
MERGED_DIR = ROOT / "dataset_gabungan"   # dataset final untuk training
MODELS_DIR = ROOT / "models"             # tempat best.pt untuk dipakai deteksi
OUTPUT_DIR = ROOT / "outputs"            # screenshot, csv, preview
RUNS_DIR = ROOT / "runs"                 # hasil training Ultralytics
OWN_RAW_DIR = RAW_DIR / "foto_sendiri_mentah"  # foto sendiri per kelas (hasil extra_capture_data.py)
OWN_DIR = RAW_DIR / "data_sendiri"       # foto sendiri + label YOLO (hasil extra_autolabel_own.py)

DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"

# 3 kelas final (urutan = ID kelas 0, 1, 2). JANGAN diubah setelah training.
CLASS_NAMES: list[str] = ["koin", "tutup_botol", "charger"]

# Teks yang tampil di layar untuk tiap kelas
DISPLAY_LABELS: dict[str, str] = {
    "koin": "Koin",
    "tutup_botol": "Tutup Botol",
    "charger": "Charger",
}

# Warna kotak default (BGR). Merah sesuai standar proyek.
BOX_COLOR = (0, 0, 255)

# Palet warna opsional per kelas (BGR) jika ingin membedakan warna visual tiap benda
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "koin": (0, 215, 255),         # Emas / Kuning (BGR)
    "tutup_botol": (255, 140, 0),  # Biru / Cyan (BGR)
    "charger": (0, 255, 0),        # Hijau Terang (BGR)
}

# ==============================================================================
# // [ISI KOORDINAT WADAH MENARUH BARANG DI SINI] (dalam satuan milimeter - mm)
# Silakan ganti angka X, Y, Z di bawah ini sesuai posisi fisik wadah sortir Anda:
# ==============================================================================
DROP_COORDINATES: dict[str, dict[str, float]] = {
    "koin": {
        "x": -150.0,  # // ISI KOORDINAT X WADAH KOIN DI SINI (misal: -150.0 mm)
        "y": 100.0,   # // ISI KOORDINAT Y WADAH KOIN DI SINI (misal: 100.0 mm)
        "z": 45.0,    # // ISI KOORDINAT Z WADAH KOIN DI SINI (ketinggian aman drop)
    },
    "tutup_botol": {
        "x": 150.0,   # // ISI KOORDINAT X WADAH TUTUP BOTOL DI SINI (misal: 150.0 mm)
        "y": 100.0,   # // ISI KOORDINAT Y WADAH TUTUP BOTOL DI SINI (misal: 100.0 mm)
        "z": 55.0,    # // ISI KOORDINAT Z WADAH TUTUP BOTOL DI SINI (ketinggian aman drop)
    },
    "charger": {
        "x": 0.0,     # // ISI KOORDINAT X WADAH CHARGER DI SINI (misal: 0.0 mm)
        "y": 180.0,   # // ISI KOORDINAT Y WADAH CHARGER DI SINI (misal: 180.0 mm)
        "z": 70.0,    # // ISI KOORDINAT Z WADAH CHARGER DI SINI (ketinggian aman drop)
    },
}

# Sumber dataset Roboflow Universe.
#   kind = "detection"       -> sudah punya kotak, diunduh format yolov8
#   kind = "classification"  -> TIDAK punya kotak; diberi kotak otomatis (02_autolabel_tutup_botol.py)
#   target_class             -> semua kelas asli di dataset itu digabung jadi 1 kelas final
#   version                  -> None = pakai versi terbaru; isi angka kalau mau memaksa
DATASETS: list[dict] = [
    {
        "key": "koin",
        "workspace": "arfan-faris-pramatya",
        "project": "koin-rupiah-mvpr",
        "version": None,
        "kind": "detection",
        "target_class": "koin",
    },
    {
        "key": "tutup_botol",
        "workspace": "bottlecapsorting",
        "project": "tutup-botol",
        "version": None,
        "kind": "classification",
        "target_class": "tutup_botol",
    },
    {
        "key": "bottle_cap_v2",
        "workspace": "work2-rjplo",
        "project": "bottle-cap-oiuwp",
        "version": 2,
        "kind": "detection",
        "target_class": "tutup_botol",
    },
    {
        "key": "charger",
        "workspace": "khamalraaj",
        "project": "chager-r5xqk",
        "version": 2,
        "kind": "detection",
        "target_class": "charger",
    },
]

# Hasil label otomatis tutup botol disimpan di sini (format YOLO), lalu ikut digabung.
TUTUP_BOTOL_YOLO_KEY = "tutup_botol_yolo"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
