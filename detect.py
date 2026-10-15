#!/usr/bin/env python3
"""Runner shortcut untuk menjalankan deteksi real-time langsung dari root workspace.

Contoh:
    python detect.py                     # webcam default (0)
    python detect.py --source 1          # kamera USB eksternal
    python detect.py --warna-kelas       # warna kotak berbeda (koin=emas, tutup=cyan)
    python detect.py --source foto.jpg   # uji satu gambar

Atau Anda tetap bisa menjalankan langsung dari folder sortir-yolo:
    cd sortir-yolo && python 05_detect.py
"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SORTIR_DIR = ROOT / "sortir-yolo"

if str(SORTIR_DIR) not in sys.path:
    sys.path.insert(0, str(SORTIR_DIR))

# Jika argumen --source berupa path file yang ada dari folder pemanggil, ubah ke path absolut
for i, arg in enumerate(sys.argv):
    if arg == "--source" and i + 1 < len(sys.argv):
        cand = Path(sys.argv[i + 1])
        if cand.exists():
            sys.argv[i + 1] = str(cand.resolve())
    elif arg.startswith("--source="):
        val = arg.split("=", 1)[1]
        cand = Path(val)
        if cand.exists():
            sys.argv[i] = f"--source={cand.resolve()}"

# Ubah cwd ke sortir-yolo agar output & relative path internal tetap konsisten
os.chdir(SORTIR_DIR)

spec = importlib.util.spec_from_file_location("detect_mod", SORTIR_DIR / "05_detect.py")
if spec and spec.loader:
    detect_mod = importlib.util.module_from_spec(spec)
    sys.modules["detect_mod"] = detect_mod
    spec.loader.exec_module(detect_mod)

    if __name__ == "__main__":
        sys.exit(detect_mod.main())
