#!/usr/bin/env python3
"""Contoh Program Klien Pengendali Robot (100% Lokal Tanpa Internet).

Jalankan skrip ini di laptop yang mengendalikan robot (baik Windows maupun Mac):
    python klien_robot_local.py

Alur Kerja:
1. Skrip ini secara berkala memanggil server middleware lokal (http://localhost:8000).
2. Ketika kamera mendeteksi koin atau tutup botol, skrip ini menerima:
   - Titik Ambil (x, y, z)
   - Titik Wadah Taruh (drop_x, drop_y, drop_z)
3. Fungsi eksekusi robot dipanggil untuk menggerakkan servo/lengan vakum.
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import requests
except ImportError:
    print("[ERROR] Library 'requests' belum terpasang.")
    print("Silakan jalankan: pip install requests")
    sys.exit(1)


def gerakkan_robot_ke_titik(x: float, y: float, z: float, aksi_vakum: str | None = None) -> None:
    """// TEMPAT ANDA MENGHUBUNGKAN KODE MOTOR / SERVO ROBOT.

    Ganti isi fungsi ini dengan perintah ke robot Anda (misalnya lewat PySerial,
    Arduino G-Code, atau library lengan robot).
    """
    print(f"  [ROBOT MOVE] Bergerak ke posisi: X={x:.1f} mm, Y={y:.1f} mm, Z={z:.1f} mm")
    if aksi_vakum == "SEDOT":
        print("  [ROBOT VAKUM] >>> SUCTION ON (Menyedot Objek)")
    elif aksi_vakum == "LEPAS":
        print("  [ROBOT VAKUM] >>> SUCTION OFF (Melepas Objek ke Wadah)")
    # Beri sedikit jeda simulasi fisik pergerakan robot
    time.sleep(0.5)


def posisi_home() -> None:
    """Posisi istirahat / standby robot."""
    print("  [ROBOT HOME] Kembali ke posisi standby (Home)\n")
    time.sleep(0.2)


def laporkan_status(server_url: str, state: str, aksi: str) -> None:
    """Melaporkan kondisi robot ke dashboard server middleware."""
    try:
        requests.post(
            f"{server_url}/api/robot/status",
            json={"state": state, "current_action": aksi},
            timeout=1.0,
        )
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Klien Robotik Pengambil Target Lokal")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="URL middleware server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Jeda waktu (detik) periksa antrean objek (default: 0.5s)",
    )
    args = parser.parse_args()

    server_url = args.server.rstrip("/")
    print("==========================================================")
    print(" 🤖 KLIEN ROBOT SORTIR OTOMATIS (100% LOKAL)")
    print("==========================================================")
    print(f" Menghubungkan ke Server: {server_url}")
    print(" Menunggu target objek dari penglihatan AI (YOLO)...")
    print(" Tekan Ctrl+C untuk berhenti.\n")

    laporkan_status(server_url, "IDLE", "Robot Standby menunggu objek")

    try:
        while True:
            try:
                # 1. Panggil endpoint target ke middleware
                response = requests.get(f"{server_url}/api/robot/target", timeout=2.0)
                if response.status_code == 200:
                    data = response.json()

                    # Cek jika ada perintah gerak otomatis dari Vision AI
                    if data.get("available") is True and data.get("aksi") == "AMBIL":
                        kelas = data.get("kelas", "objek")
                        pick_x = data.get("x", 0.0)
                        pick_y = data.get("y", 0.0)
                        pick_z = data.get("z", 2.0)
                        drop_x = data.get("drop_x", 0.0)
                        drop_y = data.get("drop_y", 0.0)
                        drop_z = data.get("drop_z", 40.0)

                        print("----------------------------------------------------------")
                        print(f"🎯 TARGET DITERIMA: {kelas.upper()}")
                        print(f"   Titik Ambil Conveyor : X={pick_x:.1f}, Y={pick_y:.1f}, Z={pick_z:.1f}")
                        print(f"   Titik Wadah Tujuan   : X={drop_x:.1f}, Y={drop_y:.1f}, Z={drop_z:.1f}")

                        laporkan_status(server_url, "BUSY", f"Mengambil {kelas}")

                        # --- LANGKAH 1: Ambil barang di conveyor ---
                        gerakkan_robot_ke_titik(pick_x, pick_y, pick_z, aksi_vakum="SEDOT")

                        # --- LANGKAH 2: Angkat dan bawa ke wadah sortir ---
                        laporkan_status(server_url, "BUSY", f"Menaruh {kelas} ke wadah")
                        gerakkan_robot_ke_titik(drop_x, drop_y, drop_z, aksi_vakum="LEPAS")

                        # --- LANGKAH 3: Kembali ke posisi Home ---
                        posisi_home()
                        laporkan_status(server_url, "IDLE", f"Selesai sortir {kelas}")

            except requests.exceptions.RequestException:
                # Jika server belum dinyalakan, tunggu tanpa crash
                pass

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[STOP] Program robot dihentikan.")
        laporkan_status(server_url, "IDLE", "Robot Offline")


if __name__ == "__main__":
    main()
