#!/usr/bin/env python3
"""Langkah 2 - Beri kotak otomatis pada dataset 'tutup botol'.

Dataset tutup-botol di Roboflow bertipe KLASIFIKASI (hanya label per gambar, tanpa kotak),
sedangkan YOLO deteksi butuh kotak. Skrip ini membuat kotak otomatis (1 kelas: tutup_botol).

    python 02_autolabel_tutup_botol.py                 # auto: YOLO-World jika bisa, kalau tidak metode 'cv'
    python 02_autolabel_tutup_botol.py --method world  # paksa YOLO-World (disarankan di Colab)
    python 02_autolabel_tutup_botol.py --method cv     # tanpa unduh model; cocok untuk latar polos

HASIL OTOMATIS TIDAK PERNAH SEMPURNA: buka gambar preview di outputs/ dan cek dulu.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import yaml

from src.autolabel import WorldLabeler, box_from_background, draw_preview, save_preview_grid, write_yolo_label
from src.config import IMAGE_EXTS, OUTPUT_DIR, RAW_DIR, TUTUP_BOTOL_YOLO_KEY
from src.dataset_tools import SPLIT_ALIASES


def collect_images(src_dir: Path) -> dict[str, list[Path]]:
    """Kumpulkan gambar per split. Kalau tidak ada folder split, semua dianggap 'train'."""
    result: dict[str, list[Path]] = {}
    for split, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            d = src_dir / alias
            if d.is_dir():
                result[split] = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
                break
    if not result:
        result["train"] = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=RAW_DIR / "tutup_botol")
    ap.add_argument("--output", type=Path, default=RAW_DIR / TUTUP_BOTOL_YOLO_KEY)
    ap.add_argument("--method", choices=["auto", "world", "cv"], default="auto")
    ap.add_argument("--prompts", nargs="+", default=["bottle cap"], help="teks untuk YOLO-World")
    ap.add_argument("--conf", type=float, default=0.10, help="threshold YOLO-World")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah gambar per split (untuk uji coba)")
    ap.add_argument("--preview", type=int, default=36, help="jumlah gambar di preview")
    args = ap.parse_args()

    if not args.input.is_dir():
        print(f"Folder {args.input} tidak ada. Jalankan dulu: python 01_download_datasets.py")
        return 1

    labeler = None
    method = args.method
    if method in ("auto", "world"):
        try:
            labeler = WorldLabeler(args.prompts, conf=args.conf, device=args.device)
            method = "world"
        except Exception as exc:
            if args.method == "world":
                print(f"YOLO-World gagal dimuat: {exc}")
                return 1
            print(f"YOLO-World tidak tersedia ({exc.__class__.__name__}); pakai metode 'cv'.")
            method = "cv"
    print(f"Metode label otomatis: {method}")

    images = collect_images(args.input)
    if args.output.exists():
        shutil.rmtree(args.output)

    stats = {"ok": 0, "tanpa_kotak": 0, "gagal_baca": 0}
    tiles = []
    all_count = sum(len(v) for v in images.values())
    step = max(1, all_count // max(1, args.preview))
    seen = 0

    for split, files in images.items():
        if args.limit:
            files = files[: args.limit]
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in files:
            img = cv2.imread(str(img_path))
            if img is None:
                stats["gagal_baca"] += 1
                continue
            box = labeler.box(img) if labeler is not None else None
            if box is None:  # world gagal / metode cv
                box = box_from_background(img)
            seen += 1
            if seen % step == 0 and len(tiles) < args.preview:
                tiles.append(draw_preview(img, box))
            if box is None or box[2] * box[3] > 0.98:
                stats["tanpa_kotak"] += 1
                continue
            # awali nama dengan nama folder kelas asli agar unik (mis. 'class2__birutua_fokus2.jpg')
            prefix = img_path.parent.name if img_path.parent.name not in SPLIT_ALIASES[split] else "x"
            new_name = f"{prefix}__{img_path.name}"
            shutil.copy2(img_path, args.output / split / "images" / new_name)
            write_yolo_label(args.output / split / "labels" / f"{Path(new_name).stem}.txt", 0, box)
            stats["ok"] += 1

    with open(args.output / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"names": ["tutup_botol"]}, f)

    preview_path = OUTPUT_DIR / "preview_autolabel_tutup_botol.jpg"
    save_preview_grid(tiles, preview_path)
    print(f"\nBerhasil diberi kotak : {stats['ok']}")
    print(f"Dilewati (tanpa kotak): {stats['tanpa_kotak']}")
    print(f"Gagal dibaca          : {stats['gagal_baca']}")
    print(f"Hasil label           : {args.output}")
    print(f"PREVIEW (cek manual!) : {preview_path}")
    if stats["ok"] == 0:
        print("Tidak ada gambar yang berhasil diberi kotak. Coba --method world di Colab.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
