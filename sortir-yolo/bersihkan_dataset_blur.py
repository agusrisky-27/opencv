#!/usr/bin/env python3
"""Pembersih Otomatis Foto Blur / Buram pada Dataset Manual (YOLOv8 Sortir).

Memanfaatkan algoritma Two-Stage Laplacian Sharpness (Global + Local Patch Variance)
untuk mendeteksi foto yang buram akibat motion blur / out-of-focus.

Cara Penggunaan:
    # 1. Cek dulu tanpa memindahkan file (Hanya scan & laporkan):
    python bersihkan_dataset_blur.py --dry-run

    # 2. Pindahkan foto blur ke folder karantina 'datasets_raw/sampah_blur/' (Aman):
    python bersihkan_dataset_blur.py

    # 3. Langsung hapus permanen foto blur:
    python bersihkan_dataset_blur.py --delete

    # 4. Hanya periksa folder tertentu:
    python bersihkan_dataset_blur.py --folder charger_capture_manual
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from src.config import MANUAL_CAPTURE_DIRS, RAW_DIR


def deteksi_blur(
    img_bgr: np.ndarray,
    ambang_global: float = 4.5,
    ambang_patch: float = 25.0,
    patch_size: int = 128,
    stride: int = 64,
) -> tuple[bool, float, float]:
    """Deteksi apakah gambar buram/blur secara presisi.

    Returns:
        (is_blur, skor_global, skor_patch_maksimal)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Jika varians global sangat kecil (< ambang_global), jelas buram parah
    if g_var < ambang_global:
        return True, g_var, 0.0

    # Jika varians global sedang (4.5 s/d 10.0), periksa apakah ada objek tajam di salah satu patch
    max_p = 0.0
    if g_var < 10.0:
        h, w = gray.shape
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                v = float(cv2.Laplacian(gray[y : y + patch_size, x : x + patch_size], cv2.CV_64F).var())
                if v > max_p:
                    max_p = v
        # Jika patch tertajam pun tidak mencapai ambang batas, gambar buram
        if max_p < ambang_patch:
            return True, g_var, max_p
    else:
        max_p = g_var

    return False, g_var, max_p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="hanya periksa dan laporkan tanpa memindahkan atau menghapus file.",
    )
    ap.add_argument(
        "--delete",
        action="store_true",
        help="hapus permanen file blur (default: dipindahkan ke datasets_raw/sampah_blur/).",
    )
    ap.add_argument(
        "--threshold-global",
        type=float,
        default=4.5,
        help="ambang batas varians global (default: 4.5).",
    )
    ap.add_argument(
        "--threshold-patch",
        type=float,
        default=25.0,
        help="ambang batas ketajaman patch lokal (default: 25.0).",
    )
    ap.add_argument(
        "--folder",
        type=str,
        default=None,
        help="nama folder target di datasets_raw (misal: charger_capture_manual).",
    )
    args = ap.parse_args()

    quarantine_base = RAW_DIR / "sampah_blur"

    # Tentukan folder-folder target
    targets: dict[str, Path] = {}
    if args.folder:
        p = RAW_DIR / args.folder
        if not p.is_dir():
            print(f"[Error] Folder tidak ditemukan: {p}")
            return 1
        targets[args.folder] = p
    else:
        for name, folder in MANUAL_CAPTURE_DIRS.items():
            if folder.is_dir():
                targets[folder.name] = folder

    print("=" * 72)
    print("🔍 SISTEM FILTER & PEMBERSIH FOTO BURAM (PRECISION BLUR DETECTOR)")
    print(f"Ambang Batas Global : {args.threshold_global:.1f} | Patch: {args.threshold_patch:.1f}")
    print(f"Mode Operasi        : {'[DRY-RUN] Hanya Cek' if args.dry_run else ('[HAPUS] Hapus Permanen' if args.delete else '[KARANTINA] Pindah ke sampah_blur/')}")
    print("=" * 72)

    total_scanned = 0
    total_blurred = 0

    for label, folder in targets.items():
        files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))
        if not files:
            continue

        print(f"\n📁 Memeriksa Folder: {folder.name} ({len(files)} foto)...")
        blur_in_folder = []

        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue

            total_scanned += 1
            is_blur, g_score, p_score = deteksi_blur(
                img,
                ambang_global=args.threshold_global,
                ambang_patch=args.threshold_patch,
            )

            if is_blur:
                blur_in_folder.append((p, g_score, p_score))

        if not blur_in_folder:
            print(f"   ✔ Luar biasa! Semua {len(files)} foto tajam dan siap latih.")
            continue

        print(f"   ⚠️  Ditemukan {len(blur_in_folder)} foto buram/motion-blur:")
        total_blurred += len(blur_in_folder)

        quarantine_dir = quarantine_base / folder.name
        if not args.dry_run and not args.delete:
            quarantine_dir.mkdir(parents=True, exist_ok=True)

        for p, g_s, p_s in blur_in_folder:
            if args.dry_run:
                print(f"      [BLUR] {p.name:<28} (Global: {g_s:>4.1f}, Patch: {p_s:>4.1f}) -> Terdeteksi")
            elif args.delete:
                p.unlink()
                print(f"      [DIHAPUS] {p.name:<28} (Global: {g_s:>4.1f}) -> Dihapus permanen")
            else:
                shutil.move(str(p), str(quarantine_dir / p.name))
                print(f"      [KARANTINA] {p.name:<28} -> sampah_blur/{folder.name}/")

    print("\n" + "=" * 72)
    print("📊 REKAP HASIL FILTER KETAJAMAN DATASET:")
    print(f"  Total Foto Diperiksa : {total_scanned} foto")
    print(f"  Foto Tajam & Bagus   : {total_scanned - total_blurred} foto ({(total_scanned - total_blurred)/max(total_scanned, 1)*100:.1f}%)")
    print(f"  Foto Buram / Blur    : {total_blurred} foto ({total_blurred/max(total_scanned, 1)*100:.1f}%)")
    if not args.dry_run and not args.delete and total_blurred > 0:
        print(f"\n  📁 Lokasi Karantina : {quarantine_base}")
        print("     (Foto blur dipisahkan dengan aman agar Anda bisa review sewaktu-waktu di Finder)")
    print("=" * 72 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
