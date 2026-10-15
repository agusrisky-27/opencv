#!/usr/bin/env python3
"""EXTRA - Ambil foto sendiri lewat webcam untuk memperbaiki akurasi.

Dataset Roboflow itu foto orang lain (latar & kamera berbeda). Model paling akurat kalau
dilatih juga dengan foto di kondisi NYATA: kamera yang sama, latar yang sama (mis. di atas
conveyor), pencahayaan yang sama.

    python extra_capture_data.py --source 0

Tombol:  1 = koin | 2 = tutup_botol | 0 = latar kosong / negatif
         SPASI = ambil 1 foto | a = ambil otomatis tiap 0.4 detik (on/off) | q = selesai
Tips: 1 benda per foto, letakkan di latar polos, ubah posisi/putaran/jarak/cahaya di antara
foto. Target minimal 100-200 foto per kelas. Lalu jalankan extra_autolabel_own.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from src.camera import Camera, CameraError, parse_source
from src.config import CLASS_NAMES, OWN_RAW_DIR
from src.utils import display_label, timestamp


def count_photos(folder: Path) -> int:
    return len(list(folder.glob("*.jpg"))) if folder.is_dir() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--interval", type=float, default=0.4, help="jeda mode otomatis (detik)")
    args = ap.parse_args()

    try:
        cam = Camera(parse_source(args.source), args.width, args.height).open()
    except CameraError as exc:
        print(exc)
        return 1

    current, auto, last_auto = 0, False, 0.0
    latar_dir = OWN_RAW_DIR / "latar"
    key_to_class = {ord(str(i + 1)): i for i in range(len(CLASS_NAMES))}
    key_to_class[ord("0")] = -1  # -1 = Latar kosong / contoh negatif

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                break
            if current == -1:
                display_name = "LATAR KOSONG / NEGATIF (wajah, jendela, meja)"
                folder = latar_dir
            else:
                name = CLASS_NAMES[current]
                display_name = display_label(name)
                folder = OWN_RAW_DIR / name

            def save() -> None:
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / f"{timestamp()}.jpg"), frame)

            now = time.time()
            if auto and now - last_auto >= args.interval:
                save()
                last_auto = now

            view = frame.copy()
            status = f"Target: {display_name} | foto: {count_photos(folder)} | auto: {'ON' if auto else 'off'}"
            cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, "1/2 kelas | 0 latar kosong | SPASI foto | a auto | q selesai",
                        (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Ambil foto sendiri", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in key_to_class:
                current = key_to_class[key]
            elif key == ord(" "):
                save()
            elif key == ord("a"):
                auto = not auto
    finally:
        cam.release()
        cv2.destroyAllWindows()

    print("\nJumlah foto per folder:")
    for n in CLASS_NAMES:
        print(f"  {n:<12} {count_photos(OWN_RAW_DIR / n)}")
    print(f"  {'latar':<12} {count_photos(latar_dir)}")
    print("\nLanjut: python extra_autolabel_own.py (untuk kelas 1/2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
