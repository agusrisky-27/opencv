#!/usr/bin/env python3
"""Langkah 3 - Gabungkan semua dataset menjadi satu dataset final.

    python 03_merge_datasets.py

Yang dilakukan:
  - semua kelas asli di tiap dataset dipetakan ke kelas final: koin / tutup_botol
  - gambar hasil augmentasi dari foto asli yang sama TIDAK dibiarkan ada di train sekaligus valid/test
  - menulis dataset_gabungan/data.yaml dan mencetak laporan jumlah data per kelas
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import CLASS_NAMES, DATASETS, MERGED_DIR, OWN_DIR, RAW_DIR, TUTUP_BOTOL_YOLO_KEY
from src.dataset_tools import Source, format_report, merge_datasets


def build_sources() -> tuple[list[Source], list[str]]:
    sources: list[Source] = []
    missing: list[str] = []
    for d in DATASETS:
        if d["kind"] == "classification":
            path, key = RAW_DIR / TUTUP_BOTOL_YOLO_KEY, TUTUP_BOTOL_YOLO_KEY
            hint = "python 02_autolabel_tutup_botol.py"
        else:
            path, key = RAW_DIR / d["key"], d["key"]
            hint = f"python 01_download_datasets.py --only {d['key']}"
        if path.is_dir():
            sources.append(Source(key, path, d["target_class"]))
        else:
            missing.append(f"{d['key']}: belum ada -> jalankan `{hint}`")
    if OWN_DIR.is_dir():  # foto sendiri (opsional), kelas dicocokkan lewat nama di data.yaml
        sources.append(Source("data_sendiri", OWN_DIR, None))
    return sources, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=MERGED_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-leak", action="store_true", help="jangan buang gambar train yang sefoto dengan valid/test")
    args = ap.parse_args()

    sources, missing = build_sources()
    for m in missing:
        print(f"[lewati] {m}")
    if not sources:
        print("Tidak ada dataset yang bisa digabung.")
        return 1

    report = merge_datasets(sources, args.out, CLASS_NAMES, seed=args.seed, drop_leak=not args.keep_leak)
    print(format_report(report, CLASS_NAMES))
    total = sum(sum(v.values()) for v in report.images.values())
    if total == 0:
        print("Hasil gabungan kosong.")
        return 1
    print(f"\nDataset siap: {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
