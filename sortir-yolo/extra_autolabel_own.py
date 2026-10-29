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
from typing import Any

import cv2
import yaml

from src.autolabel import (
    ModelLabeler,
    WorldLabeler,
    box_from_background,
    draw_preview,
    save_preview_grid,
    segment_charger_box,
    write_yolo_label,
)
from src.config import (
    CLASS_NAMES,
    DEFAULT_WEIGHTS,
    IMAGE_EXTS,
    LATAR_CAPTURE_DIR,
    OUTPUT_DIR,
    OWN_DIR,
    OWN_RAW_DIR,
    get_capture_dir,
)

PROMPTS = {
    "koin": ["coin", "round metal coin", "rupiah coin", "money"],
    "tutup_botol": ["bottle cap", "plastic cap", "bottle lid"],
    "charger": ["phone charger", "wall adapter", "power adapter", "usb charger", "charger"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_method = "model" if DEFAULT_WEIGHTS.exists() else "world"
    ap.add_argument(
        "--method",
        choices=["model", "world", "cv"],
        default=default_method,
        help=f"metode autolabel: 'model' (best.pt), 'world' (YOLO-World), 'cv' (latar polos). Default: {default_method}",
    )
    ap.add_argument("--conf", type=float, default=0.15, help="confidence threshold (default: 0.15)")
    ap.add_argument("--block", type=int, default=10, help="ukuran blok foto berurutan")
    ap.add_argument("--valid-every", type=int, default=5, help="tiap blok ke-N jadi valid (5 = 20%%)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    has_dirs = any(get_capture_dir(n).is_dir() for n in CLASS_NAMES) or OWN_RAW_DIR.is_dir()
    if not has_dirs:
        print("Folder foto belum ada. Jalankan dulu: python capture.py")
        return 1

    if OWN_DIR.exists():
        shutil.rmtree(OWN_DIR)
    OWN_DIR.mkdir(parents=True, exist_ok=True)

    labelers: dict[str, Any] = {}
    stats: dict[str, dict[str, int]] = {}
    tiles = []

    for class_id, name in enumerate(CLASS_NAMES):
        folder = get_capture_dir(name)
        if not folder.is_dir() or not any(folder.iterdir()):
            folder = OWN_RAW_DIR / name
        if not folder.is_dir():
            continue
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            continue
        stats[name] = {"ok": 0, "lewat": 0}

        if args.method == "model":
            labeler = labelers.setdefault("model", ModelLabeler(DEFAULT_WEIGHTS, conf=args.conf, device=args.device))
        elif args.method == "world":
            labeler = labelers.setdefault(name, WorldLabeler(PROMPTS.get(name, [name]), conf=0.03, device=args.device))
        else:
            labeler = None

        for i, path in enumerate(files):
            img = cv2.imread(str(path))
            if img is None:
                continue

            if name == "charger":
                model_inst = labeler.model if (args.method == "model" and labeler is not None) else None
                box = segment_charger_box(img, yolo_model=model_inst)
            elif args.method == "model":
                box = labeler.box(img, target_class=name)
                if box is None:
                    box = box_from_background(img)
            elif args.method == "world":
                box = labeler.box(img)
            else:
                box = box_from_background(img)

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

    # Proses sampel negatif (Latar Kosong / Meja / Tangan) dengan empty label (.txt kosong)
    latar_files = []
    if LATAR_CAPTURE_DIR.is_dir():
        latar_files = sorted(p for p in LATAR_CAPTURE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if latar_files:
        stats["latar"] = {"ok": 0, "lewat": 0}
        for i, path in enumerate(latar_files):
            split = "valid" if (i // args.block) % args.valid_every == args.valid_every - 1 else "train"
            (OWN_DIR / split / "images").mkdir(parents=True, exist_ok=True)
            (OWN_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
            out_name = f"latar__{path.name}"
            shutil.copy2(path, OWN_DIR / split / "images" / out_name)
            # Standar YOLO: buat file teks kosong (0 byte) untuk background negative sample
            (OWN_DIR / split / "labels" / f"{Path(out_name).stem}.txt").write_text("", encoding="utf-8")
            stats["latar"]["ok"] += 1

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
