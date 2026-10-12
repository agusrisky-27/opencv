#!/usr/bin/env python3
"""EXTRA - Beri kotak otomatis pada foto sendiri (hasil extra_capture_data.py).

Asumsi: tiap foto berisi SATU benda di latar polos (mis. permukaan conveyor).
Foto yang kotaknya tidak bisa ditemukan dilewati. Selalu cek gambar preview-nya!

    python extra_autolabel_own.py
    python extra_autolabel_own.py --method world     # latar tidak polos (butuh ultralytics + unduh model)

Pembagian train/valid dilakukan per BLOK foto berurutan (bukan acak), karena foto yang
diambil berdekatan hampir sama -> kalau diacak, valid jadi 'contekan' dan metrik menipu.
Setelah ini jalankan lagi: python 03_merge_datasets.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import yaml

from src.autolabel import WorldLabeler, box_from_background, draw_preview, save_preview_grid, write_yolo_label
from src.config import CLASS_NAMES, IMAGE_EXTS, OUTPUT_DIR, OWN_DIR, OWN_RAW_DIR

PROMPTS = {"koin": "coin", "korek_api": "lighter", "tutup_botol": "bottle cap"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["cv", "world"], default="cv")
    ap.add_argument("--block", type=int, default=10, help="ukuran blok foto berurutan")
    ap.add_argument("--valid-every", type=int, default=5, help="tiap blok ke-N jadi valid (5 = 20%)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not OWN_RAW_DIR.is_dir():
        print(f"{OWN_RAW_DIR} belum ada. Jalankan dulu: python extra_capture_data.py")
        return 1

    world: dict[str, WorldLabeler] = {}
    if OWN_DIR.exists():
        shutil.rmtree(OWN_DIR)
    stats: dict[str, dict[str, int]] = {}
    tiles = []

    for class_id, name in enumerate(CLASS_NAMES):
        folder = OWN_RAW_DIR / name
        if not folder.is_dir():
            continue
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        stats[name] = {"ok": 0, "lewat": 0}
        labeler = None
        if args.method == "world":
            labeler = world.setdefault(name, WorldLabeler([PROMPTS[name]], device=args.device))
        for i, path in enumerate(files):
            img = cv2.imread(str(path))
            if img is None:
                continue
            box = labeler.box(img) if labeler else box_from_background(img)
            if i % 15 == 0 and len(tiles) < 36:
                tiles.append(draw_preview(img, box))
            if box is None:
                stats[name]["lewat"] += 1
                continue
            split = "valid" if (i // args.block) % args.valid_every == args.valid_every - 1 else "train"
            (OWN_DIR / split / "images").mkdir(parents=True, exist_ok=True)
            (OWN_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
            out_name = f"{name}__{path.name}"
            shutil.copy2(path, OWN_DIR / split / "images" / out_name)
            write_yolo_label(OWN_DIR / split / "labels" / f"{Path(out_name).stem}.txt", class_id, box)
            stats[name]["ok"] += 1

    if not stats:
        print("Tidak ada foto ditemukan di", OWN_RAW_DIR)
        return 1
    with open(OWN_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"names": CLASS_NAMES}, f)

    preview = OUTPUT_DIR / "preview_autolabel_foto_sendiri.jpg"
    save_preview_grid(tiles, preview)
    for name, s in stats.items():
        print(f"{name:<12} berhasil: {s['ok']:>4} | dilewati (kotak tak ketemu): {s['lewat']:>4}")
    print(f"\nHasil: {OWN_DIR}\nPREVIEW (cek manual!): {preview}")
    print("Berikutnya: python 03_merge_datasets.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
