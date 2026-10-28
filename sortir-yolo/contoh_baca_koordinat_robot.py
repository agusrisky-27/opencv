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
    print("Pastikan 'sortir_otomatis.py' atau 'detect.py' sedang berjalan.")
    print("Menunggu deteksi objek... (Tekan Ctrl+C untuk keluar)\n")

    last_ts = 0.0
    last_status = ""
    try:
        while True:
            data = read_target_json()
            if data:
                status = data.get("status", "NO_TARGET")
                msg = data.get("message", "")

                # Cegah spam cetak status yang sama
                if status != last_status:
                    if status == "CONVEYOR_MOVING":
                        print(f"⏩ [STATUS: CONVEYOR_MOVING] {msg} -> Robot STANDBY (Data ditahan).")
                    elif status == "WAITING_STATIONARY":
                        print(f"⏳ [STATUS: WAITING_STATIONARY] {msg} -> Robot MENUNGGU getaran inersia reda.")
                    elif status == "NO_TARGET":
                        print(f"⛔ [STATUS: NO_TARGET] {msg} -> Robot STANDBY di posisi HOME.")
                    last_status = status

                if status == "TARGET_READY":
                    ts = data.get("timestamp", 0.0)
                    if ts != last_ts:
                        last_ts = ts
                        target = data["target"]
                        kelas = target["label"]
                        r = target["robot_mm"]
                        drop = target["drop_mm"]

                        print("\n" + "=" * 60)
                        print(f"🎯 [DATA DITERIMA OLEH ROBOT] Target Terkunci: {kelas}")
                        print(f"   📍 Titik Ambil di Konveyor : X={r['x']} mm, Y={r['y']} mm, Z={r['z']} mm")
                        print(f"   📦 Titik Wadah Sortir      : X={drop['x']} mm, Y={drop['y']} mm, Z={drop['z']} mm")
                        print("-" * 60)
                        print("   🦾 [SIMULASI EKSEKUSI GERAK LENGAN ROBOT]:")
                        print(f"      1. G0 X{r['x']} Y{r['y']} Z{r['z'] + 25:.1f} F100  -> Meluncur aman di atas objek")
                        print(f"      2. G0 Z{r['z']:.1f} F40                            -> Ujung vakum turun menyentuh objek")
                        print(f"      3. M1                                              -> Pompa Vakum ON (Menghisap)")
                        print(f"      4. G0 Z{r['z'] + 30:.1f} F80                       -> Mengangkat objek dari sabuk")
                        print(f"      5. G0 X{drop['x']} Y{drop['y']} F100               -> Bergerak ke kotak wadah {kelas}")
                        print(f"      6. M2 / M6                                         -> Pompa Vakum OFF (Lepas ke wadah)")
                        print(f"      7. G0 X0 Y150 Z30 F100                             -> Kembali standby ke posisi HOME")
                        print("   --> ✅ SIMULASI SUKSES: Objek berhasil dipindahkan!")
                        print("=" * 60 + "\n")

            time.sleep(0.1)
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


def jalankan_mode_infrared(source: int | str, target_classes: list[str], settle_delay: float = 0.6) -> None:
    print("==========================================================")
    print(" 🤖 MODE INSPEKSI STATIS (STOP-AND-GO INFRARED TRIGGER)")
    print("==========================================================")
    print("Sistem menunggu sinyal Infrared saat benda berhenti di konveyor.")
    print(f"Settling Delay : {settle_delay:.2f} detik (Peredam inersia konveyor)")
    print(f"Kelas target sortir yang ingin diambil robot: {target_classes}")
    print("Menginisialisasi kamera dan model AI YOLO...")

    vision = SortirVisionAPI(source=source)
    print("Vision siap!")
    print("👉 Tekan [ENTER] di terminal untuk simulasi 'Sensor Infrared Terpicu & Konveyor Berhenti'")
    print("   (Tekan Ctrl+C untuk keluar)\n")

    try:
        urutan = 1
        while True:
            input(f"[IR TRIGGER #{urutan}] Tekan ENTER untuk simulasi benda tiba & konveyor berhenti...")
            print(f"⏳ Konveyor berhenti! Menahan data & meredam inersia mekanis ({settle_delay:.2f}s)...")

            # SEKALI LIHAT (Single Snapshot) dengan Settling Delay & Motion Verification:
            target = vision.inspect_stationary_object(
                target_classes=target_classes,
                save_snapshot=True,
                settle_delay=settle_delay,
                verify_motion=True,
            )

            if target:
                print(f"✅ OBJEK TARGET VALID & DIAM TOTAL: {target.label} (Confidence: {target.confidence*100:.1f}%)")
                print(f"   Titik Ambil Pasti (Statis) : X={target.x_mm} mm, Y={target.y_mm} mm, Z={target.z_mm} mm")
                print(f"   Titik Wadah Sortir         : X={target.drop_x_mm} mm, Y={target.drop_y_mm} mm, Z={target.drop_z_mm} mm")
                print("   --> 🦾 Lengan robot mengeksekusi Pick & Place (titik terkunci statis, tidak berubah-ubah).")
                print("   --> 🟢 Gerak robot selesai. Konveyor dinyalakan kembali.\n")
            else:
                print("⛔ BUKAN TARGET SORTIR / AREA KOSONG.")
                print("   --> ⏩ Lengan robot diam. Konveyor langsung dinyalakan kembali mencari benda berikutnya.\n")

            urutan += 1
    except KeyboardInterrupt:
        print("\nBerhenti.")
    finally:
        vision.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Contoh Integrasi Koordinat Robotik")
    parser.add_argument("--mode", choices=["json", "api", "infrared"], default="infrared",
                        help="Mode baca koordinat: 'infrared' (single-shot statis), 'json' (baca file) atau 'api' (stream)")
    parser.add_argument("--source", default="0", help="Index kamera (khusus mode api/infrared)")
    parser.add_argument("--target-classes", nargs="+", default=["koin", "charger"],
                        help="Daftar kelas yang ingin diambil robot (default: koin charger)")
    parser.add_argument("--settle-delay", type=float, default=0.6,
                        help="Jeda settling time (detik) peredam inersia konveyor (default: 0.6s)")
    args = parser.parse_args()

    if args.mode == "json":
        jalankan_mode_json()
    elif args.mode == "api":
        jalankan_mode_api(args.source)
    else:
        jalankan_mode_infrared(args.source, args.target_classes, settle_delay=args.settle_delay)


if __name__ == "__main__":
    main()
