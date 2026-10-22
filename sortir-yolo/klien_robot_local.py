#!/usr/bin/env python3
"""Program Klien Pengendali Robot RNV3 (100% Lokal & Terintegrasi).

Mendukung Robot Arm RNV3 (Cartesian Robot berbasis Arduino G-Code via PySerial).

Alur Kerja:
1. Skrip ini secara berkala meminta target objek ke Local Middleware Server (http://localhost:8000).
2. Ketika kamera OpenCV mendeteksi Koin atau Tutup Botol, middleware memberikan koordinat target.
3. Skrip ini menerjemahkan koordinat menjadi instruksi G-Code (G0, M1/M2) ke Robot RNV3.
4. Robot mengambil objek di conveyor dan memindahkannya ke wadah sortir yang tepat.

Penggunaan di Laptop Rekan (Windows):
    python klien_robot_local.py --port COM3           # Hubungkan ke Robot RNV3 di COM3
    python klien_robot_local.py --mock                # Mode simulasi (tanpa colok robot fisik)
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

try:
    import requests
except ImportError:
    print("[ERROR] Library 'requests' belum terpasang.")
    print("Silakan jalankan: pip install requests")
    sys.exit(1)

try:
    import serial
except ImportError:
    serial = None  # type: ignore


class RNV3Robot:
    """Driver antarmuka ke hardware Robot Arm RNV3 via PySerial (G-Code)."""

    def __init__(self, port: str | None = None, baud: int = 115200, mock: bool = False) -> None:
        self.port = port
        self.baud = baud
        self.mock = mock or (port is None)
        self.ser: Any = None

        if not self.mock:
            if serial is None:
                print("⚠️ Library 'pyserial' belum terpasang. Berjalan dalam mode simulasi.")
                self.mock = True
            else:
                try:
                    print(f"🔌 Menghubungkan ke Robot RNV3 pada port {self.port} (baud {self.baud})...")
                    self.ser = serial.Serial(self.port, self.baud, timeout=1.0)
                    time.sleep(1.5)  # Beri waktu Arduino me-reset
                    print("✅ Terhubung ke Robot RNV3!")
                except Exception as exc:
                    print(f"❌ Gagal membuka serial {self.port}: {exc}")
                    print("💡 Beralih ke MODE SIMULASI agar sistem tetap bisa diuji.")
                    self.mock = True

    def send_and_wait(self, command: str, timeout_sec: float = 10.0) -> None:
        """Kirim perintah G-Code ke robot RNV3 dan tunggu balasan 'ok' dari Arduino."""
        cmd_clean = command.strip()
        if self.mock or self.ser is None:
            print(f"   [ROBOT G-CODE SIMULASI] -> {cmd_clean}")
            time.sleep(0.3)
            return

        try:
            payload = (cmd_clean + "\r").encode("utf-8")
            self.ser.write(payload)
            start_t = time.time()
            while time.time() - start_t < timeout_sec:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    if "ok" in line.lower():
                        break
        except Exception as exc:
            print(f"⚠️ Error komunikasi serial: {exc}")

    def calibrate_home(self) -> None:
        """Kirim perintah G28 (Homing) ke Robot RNV3."""
        print("🏠 Memulai Homing Robot (G28)...")
        self.send_and_wait("G28", timeout_sec=15.0)
        print("✅ Robot RNV3 siap di posisi Home.")

    def gerakkan_ke_titik(self, x: float, y: float, z: float, speed: float = 70.0, aksi_vakum: str | None = None) -> None:
        """Kirim perintah koordinat Kartesian ke Robot RNV3."""
        print(f"  [ROBOT MOVE] Bergerak ke posisi: X={x:.1f}, Y={y:.1f}, Z={z:.1f} (Speed: {speed})")
        self.send_and_wait(f"G0 X{x:.2f} Y{y:.2f} Z{z:.2f} F{speed:.1f}")

        if aksi_vakum == "SEDOT":
            print("  [ROBOT VAKUM] >>> SUCTION ON (M1: Menyedot Objek)")
            self.send_and_wait("M1")
            time.sleep(0.3)
        elif aksi_vakum == "LEPAS":
            print("  [ROBOT VAKUM] >>> SUCTION OFF (M2: Melepas Objek ke Wadah)")
            self.send_and_wait("M2")
            # Katup pelepas tekanan cepat (M6 / M7 sesuai RNV3)
            self.send_and_wait("M6")
            time.sleep(0.2)

    def posisi_standby(self, y: float = 150.0, z: float = 30.0) -> None:
        """Kembali ke posisi aman di atas conveyor."""
        print("  [ROBOT STANDBY] Kembali ke posisi standby aman...\n")
        self.send_and_wait(f"G0 X0 Y{y:.1f} Z{z:.1f} F80")

    def shutdown(self) -> None:
        """Matikan stepper motor saat program berhenti."""
        if not self.mock and self.ser:
            try:
                print("🛑 Mematikan stepper motor robot (M18)...")
                self.send_and_wait("M18")
                self.ser.close()
            except Exception:
                pass


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
    parser = argparse.ArgumentParser(description="Klien Robot RNV3 Sortir Otomatis")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="URL middleware server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Port Serial Arduino robot di Windows (contoh: COM3, COM4, COM11)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Baudrate serial robot RNV3 (default: 115200)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Jalankan dalam mode simulasi tanpa robot fisik tercolok",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.4,
        help="Jeda waktu (detik) periksa antrean objek (default: 0.4s)",
    )
    args = parser.parse_args()

    server_url = args.server.rstrip("/")
    print("==========================================================")
    print(" 🤖 KLIEN ROBOT RNV3 SORTIR OTOMATIS (100% LOKAL)")
    print("==========================================================")
    print(f" Menghubungkan ke Middleware: {server_url}")

    # Inisialisasi Hardware Robot RNV3
    robot = RNV3Robot(port=args.port, baud=args.baud, mock=args.mock)
    robot.calibrate_home()

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
                        pick_y = data.get("y", 200.0)
                        pick_z = data.get("z", -20.0)
                        drop_x = data.get("drop_x", 100.0)
                        drop_y = data.get("drop_y", 200.0)
                        drop_z = data.get("drop_z", 0.0)

                        print("----------------------------------------------------------")
                        print(f"🎯 TARGET DITERIMA DARI AI: {kelas.upper()}")
                        print(f"   Titik Ambil Conveyor : X={pick_x:.1f}, Y={pick_y:.1f}, Z={pick_z:.1f}")
                        print(f"   Titik Wadah Tujuan   : X={drop_x:.1f}, Y={drop_y:.1f}, Z={drop_z:.1f}")

                        laporkan_status(server_url, "BUSY", f"Mengambil {kelas}")

                        # --- LANGKAH 1: Dekati posisi atas objek ---
                        robot.gerakkan_ke_titik(pick_x, pick_y, z=10.0, speed=80.0)

                        # --- LANGKAH 2: Turun dan hisap benda dengan vakum (M1) ---
                        robot.gerakkan_ke_titik(pick_x, pick_y, pick_z, speed=30.0, aksi_vakum="SEDOT")

                        # --- LANGKAH 3: Angkat objek ke atas ---
                        robot.gerakkan_ke_titik(pick_x, pick_y, z=20.0, speed=60.0)

                        # --- LANGKAH 4: Pindahkan ke atas wadah sortir ---
                        laporkan_status(server_url, "BUSY", f"Menaruh {kelas} ke wadah")
                        robot.gerakkan_ke_titik(drop_x, drop_y, z=20.0, speed=70.0)

                        # --- LANGKAH 5: Lepaskan vakum (M2) ke dalam wadah ---
                        robot.gerakkan_ke_titik(drop_x, drop_y, drop_z, speed=40.0, aksi_vakum="LEPAS")

                        # --- LANGKAH 6: Kembali ke posisi Standby ---
                        robot.posisi_standby()
                        laporkan_status(server_url, "IDLE", f"Selesai sortir {kelas}")

            except requests.exceptions.RequestException:
                # Jika server belum dinyalakan, tunggu tanpa crash
                pass

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[STOP] Program robot dihentikan pengguna.")
    finally:
        robot.shutdown()
        laporkan_status(server_url, "IDLE", "Robot Offline")


if __name__ == "__main__":
    main()

