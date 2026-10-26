#!/usr/bin/env python3
"""Program Pengambilan Foto Manual untuk Dataset Sortir AI (YOLOv8).

Menyimpan foto langsung ke folder manual terpisah di datasets_raw/:
  - datasets_raw/koin_capture_manual/          (Tombol 1)
  - datasets_raw/tutup_botol_capture_manual/   (Tombol 2)
  - datasets_raw/charger_capture_manual/       (Tombol 3)
  - datasets_raw/latar_negatif_manual/         (Tombol 0)

Cara Penggunaan:
    python capture.py --source 0               # Webcam bawaan laptop Mac
    python capture.py --source 1               # Webcam USB / iPhone

Kontrol Keyboard:
    1 = Target Koin
    2 = Target Tutup Botol
    3 = Target Charger
    0 = Target Latar Negatif (meja/conveyor kosong, tangan, background)
    SPASI = Ambil 1 foto manual
    a = Auto-capture tiap interval detik (Toggle ON/OFF)
    q / ESC = Selesai dan simpan
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from src.camera import Camera, CameraError, parse_source
from src.config import CLASS_NAMES, LATAR_CAPTURE_DIR, get_capture_dir
from src.utils import display_label, timestamp


def count_photos(folder: Path) -> int:
    """Menghitung jumlah foto (jpg, png, jpeg) di dalam folder."""
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--source", default="0", help="nomor kamera (default: 0 untuk webcam USB Rexus)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--interval", type=float, default=0.4, help="jeda mode auto-capture (detik)")
    args = ap.parse_args()

    try:
        cam = Camera(parse_source(args.source), args.width, args.height).open()
    except CameraError as exc:
        print(f"[Error Kamera] {exc}")
        return 1

    current = 0  # 0: koin, 1: tutup_botol, 2: charger, -1: latar negatif
    auto = False
    last_auto = 0.0
    last_saved_msg = ""
    last_saved_time = 0.0

    # Petakan tombol angka ke indeks kelas
    key_to_class = {ord(str(i + 1)): i for i in range(len(CLASS_NAMES))}
    key_to_class[ord("0")] = -1  # -1 = Latar kosong / sampel negatif

    print("\n" + "=" * 65)
    print("📸 PENGAMBIL FOTO DATASET MANUAL (YOLOv8 SORTIR)")
    print("=" * 65)
    print("Folder target aktif:")
    for n in CLASS_NAMES:
        print(f"  [{CLASS_NAMES.index(n)+1}] {display_label(n):<12} -> {get_capture_dir(n)}")
    print(f"  [0] Latar Negatif -> {LATAR_CAPTURE_DIR}")
    print("=" * 65)
    print("Tombol kontrol: [1/2/3/0] Ganti Target | [SPASI] Foto | [A] Auto | [Q] Keluar\n")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                break

            if current == -1:
                display_name = "LATAR NEGATIF (conveyor kosong, tangan, ruangan)"
                folder = LATAR_CAPTURE_DIR
            else:
                name = CLASS_NAMES[current]
                display_name = display_label(name)
                folder = get_capture_dir(name)

            def save_photo() -> None:
                nonlocal last_saved_msg, last_saved_time
                folder.mkdir(parents=True, exist_ok=True)
                fname = f"{timestamp()}.jpg"
                out_path = folder / fname
                cv2.imwrite(str(out_path), frame)
                last_saved_msg = f"Tersimpan: {folder.name}/{fname}"
                last_saved_time = time.time()
                print(f"[FOTO DISIMPAN] -> {folder.name}/{fname} (Total: {count_photos(folder)})")

            now = time.time()
            if auto and (now - last_auto >= args.interval):
                save_photo()
                last_auto = now

            view = frame.copy()
            total_current = count_photos(folder)

            # Banner status atas
            status_text = f"Target: {display_name} | Folder: {folder.name}/ | Foto: {total_current}"
            auto_text = f"Auto: {'ON (' + str(args.interval) + 's)' if auto else 'OFF'}"

            # Kotak background semi-transparan untuk header
            cv2.rectangle(view, (0, 0), (view.shape[1], 45), (20, 20, 20), -1)
            cv2.putText(view, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, auto_text, (view.shape[1] - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0) if auto else (180, 180, 180), 2, cv2.LINE_AA)

            # Indikator notifikasi foto baru tersimpan (tampil 0.6 detik)
            if now - last_saved_time < 0.6:
                cv2.rectangle(view, (0, 47), (view.shape[1], 80), (0, 160, 0), -1)
                cv2.putText(view, f"✔ {last_saved_msg}", (12, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # Petunjuk keyboard bawah
            cv2.rectangle(view, (0, view.shape[0] - 35), (view.shape[1], view.shape[0]), (20, 20, 20), -1)
            petunjuk = "[1] Koin | [2] Tutup Botol | [3] Charger | [0] Latar | [SPASI] Ambil Foto | [A] Auto | [Q] Selesai"
            cv2.putText(view, petunjuk, (12, view.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Capture Dataset Manual - YOLO Sortir", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in key_to_class:
                current = key_to_class[key]
                t_name = "Latar Negatif" if current == -1 else display_label(CLASS_NAMES[current])
                print(f"[Target Ganti] -> {t_name}")
            elif key == ord(" "):
                save_photo()
            elif key == ord("a"):
                auto = not auto
                print(f"[Auto-Capture] {'AKTIF (tiap ' + str(args.interval) + ' detik)' if auto else 'NON-AKTIF'}")

    finally:
        cam.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 65)
    print("📊 REKAP FOTO DI FOLDER MANUAL:")
    print("=" * 65)
    for n in CLASS_NAMES:
        f_dir = get_capture_dir(n)
        print(f"  {f_dir.name:<30} : {count_photos(f_dir):>4} foto")
    print(f"  {LATAR_CAPTURE_DIR.name:<30} : {count_photos(LATAR_CAPTURE_DIR):>4} foto")
    print("=" * 65)
    print("Langkah berikutnya untuk pelabelan otomatis:")
    print("  python extra_autolabel_own.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
