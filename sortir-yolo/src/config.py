"""Konfigurasi pusat proyek.

Semua nama kelas, label tampilan, dan sumber dataset diatur di sini supaya
skrip lain tidak perlu hardcode.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Folder kerja
RAW_DIR = ROOT / "datasets_raw"          # hasil unduhan Roboflow (mentah)
MERGED_DIR = ROOT / "dataset_gabungan"   # dataset final 3 kelas untuk training
MODELS_DIR = ROOT / "models"             # tempat best.pt untuk dipakai deteksi
OUTPUT_DIR = ROOT / "outputs"            # screenshot, csv, preview
RUNS_DIR = ROOT / "runs"                 # hasil training Ultralytics
OWN_RAW_DIR = RAW_DIR / "foto_sendiri_mentah"  # foto sendiri per kelas (hasil extra_capture_data.py)
OWN_DIR = RAW_DIR / "data_sendiri"       # foto sendiri + label YOLO (hasil extra_autolabel_own.py)

DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"

# 2 kelas final (urutan = ID kelas 0, 1). JANGAN diubah setelah training.
CLASS_NAMES: list[str] = ["koin", "tutup_botol"]

# Teks yang tampil di layar untuk tiap kelas
DISPLAY_LABELS: dict[str, str] = {
    "koin": "Koin",
    "tutup_botol": "Tutup Botol",
}

# Warna kotak (BGR). Merah sesuai permintaan.
BOX_COLOR = (0, 0, 255)

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
]

# Hasil label otomatis tutup botol disimpan di sini (format YOLO), lalu ikut digabung.
TUTUP_BOTOL_YOLO_KEY = "tutup_botol_yolo"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
