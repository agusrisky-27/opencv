#!/usr/bin/env python3
"""Contoh Program untuk Rekan Tim Robotik (Windows).

Menunjukkan 2 cara mudah robot membaca koordinat objek dari OpenCV:

Metode 1: Membaca file JSON real-time ('outputs/target_robot.json')
          (Cocok jika OpenCV dijalankan di terminal 1 via 'python 05_detect.py',
           dan robot dijalankan di terminal 2).

Metode 2: Menggunakan modul SortirVisionAPI langsung
          (Cocok jika robot ingin memanggil deteksi kamera langsung dari kode Python).

Jalankan contoh:
    python contoh_baca_koordinat_robot.py --mode json
    python contoh_baca_koordinat_robot.py --mode api --source 0
"""
from __future__ import annotations

import argparse
import time

from sortir_vision_api import SortirVisionAPI, read_target_json


def jalankan_mode_json() -> None:
    print("==========================================================")
    print(" 🤖 MEMBACA KOORDINAT DARI FILE REAL-TIME ('outputs/target_robot.json')")
    print("==========================================================")
    print("Pastikan 'python 05_detect.py' sedang berjalan di terminal lain.")
    print("Menunggu deteksi objek... (Tekan Ctrl+C untuk keluar)\n")

    last_ts = 0.0
    try:
        while True:
            data = read_target_json()
            if data and data.get("status") == "TARGET_READY":
                ts = data.get("timestamp", 0.0)
                if ts != last_ts:
                    last_ts = ts
                    target = data["target"]
                    kelas = target["label"]
                    r = target["robot_mm"]
                    drop = target["drop_mm"]

                    print(f"🎯 OBJEK TERDETEKSI: {kelas}")
                    print(f"   Posisi Pick Conveyor : X={r['x']} mm, Y={r['y']} mm, Z={r['z']} mm")
                    print(f"   Posisi Wadah Drop    : X={drop['x']} mm, Y={drop['y']} mm, Z={drop['z']} mm")
                    print("   --> Kirim instruksi G-Code ke robot: G0 X{:.1f} Y{:.1f} Z{:.1f}".format(
                        r['x'], r['y'], r['z']
                    ))
                    print("-" * 55)

            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nBerhenti.")


def jalankan_mode_api(source: int | str) -> None:
    print("==========================================================")
    print(" 🤖 MEMANGGIL OPENCV VISION API SECARA LANGSUNG")
    print("==========================================================")
    print("Menginisialisasi kamera dan model AI YOLO...")

    vision = SortirVisionAPI(source=source)
    print("Vision siap! Membaca target objek... (Tekan Ctrl+C untuk keluar)\n")

    try:
        while True:
            target = vision.get_target()
            if target:
                print(f"🎯 Target Ditemukan: {target.label} (Conf: {target.confidence*100:.1f}%)")
                print(f"   Koordinat Robot : X={target.x_mm} mm, Y={target.y_mm} mm, Z={target.z_mm} mm")
                print(f"   Koordinat Drop  : X={target.drop_x_mm} mm, Y={target.drop_y_mm} mm, Z={target.drop_z_mm} mm")
                print("   --> Simulasi instruksi robot dijalankan...\n")
                time.sleep(1.0)
            else:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nBerhenti.")
    finally:
        vision.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Contoh Integrasi Koordinat Robotik")
    parser.add_argument("--mode", choices=["json", "api"], default="json",
                        help="Mode baca koordinat: 'json' (baca file) atau 'api' (panggil vision langsung)")
    parser.add_argument("--source", default="0", help="Index kamera (khusus mode api)")
    args = parser.parse_args()

    if args.mode == "json":
        jalankan_mode_json()
    else:
        jalankan_mode_api(args.source)


if __name__ == "__main__":
    main()
