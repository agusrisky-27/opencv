#!/usr/bin/env python3
"""Sistem Sortir Otomatis Penuh (100% Hands-Free / Tanpa Sentuh Laptop).

Dirancang untuk skenario kasus nyata di lapangan / demo:
1. Menunggu benda tiba di konveyor (bisa via Sensor Infrared Serial atau Deteksi Otomatis Kamera).
2. Konveyor berhenti diam -> Kamera mengambil 1 snapshot statis (koordinat pasti & terkunci).
3. Mengecek apakah benda termasuk kelas target sortir yang ingin diambil:
   - JIKA YA: Mengunci koordinat mm robot (X, Y, Z), mengekspor target, dan robot memindahkan ke wadah.
   - JIKA BUKAN: Mengabaikan benda dan konveyor langsung jalan kembali membawa benda berikutnya.

Jalankan:
    # 1. Mode Otomatis Kamera (Paling Mudah: Kamera otomatis tahu saat benda berhenti di zona ambil):
    python sortir_otomatis.py --mode auto-vision --source 0

    # 2. Mode Sensor Infrared Serial (Jika sensor IR & relay konveyor terhubung ke Arduino USB):
    python sortir_otomatis.py --mode serial --port /dev/tty.usbmodem1234 --baud 9600

    # 3. Mode Server REST API (Jika script robot di Windows memanggil via jaringan lokal Wi-Fi/LAN):
    python sortir_otomatis.py --mode server --port 8000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import DEFAULT_CALIB_PATH, DEFAULT_TARGET_JSON, DEFAULT_WEIGHTS, OUTPUT_DIR
from src.coordinate_transform import (
    RobotCoordinateTransformer,
    RobotTarget,
    choose_best_target,
    export_target_json,
)
from src.detector import Detector
from sortir_vision_api import SortirVisionAPI


class AutoSortirController:
    """Pengontrol sortir mandiri yang mengelola siklus Stop-and-Go secara otomatis."""

    def __init__(
        self,
        source: int | str = 0,
        weights: Path | str = DEFAULT_WEIGHTS,
        calib_path: Path | str = DEFAULT_CALIB_PATH,
        conf: float = 0.45,
        target_classes: list[str] | None = None,
        edge_margin: int = 15,
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.target_classes = target_classes or ["koin", "charger"]
        self.roi = roi

        print("🤖 [AutoSortir] Memuat modul Vision AI...")
        self.vision = SortirVisionAPI(
            source=source,
            weights=weights,
            calib_path=calib_path,
            conf=conf,
            edge_margin=edge_margin,
            roi=roi,
        )
        print(f"✅ [AutoSortir] Vision Siap! Kelas target yang akan diambil: {self.target_classes}")

    def inspect_once(self) -> RobotTarget | None:
        """Satu kali snapshot statis saat benda diam (koordinat terkunci)."""
        return self.vision.inspect_stationary_object(
            target_classes=self.target_classes,
            save_snapshot=True,
            flush_frames=3,
        )

    def run_auto_vision_loop(self, min_stationary_seconds: float = 0.3) -> None:
        """Mode Otomatis Murni: Kamera mendeteksi otomatis saat ada benda yang berhenti di konveyor."""
        print("\n" + "=" * 65)
        print(" 🚀 MODE OTOMATIS PENUH (AUTO-VISION TRIGGER: HANDS-FREE)")
        print("=" * 65)
        print("Sistem berjalan 100% otomatis tanpa menyentuh keyboard!")
        print("Kamera mengawasi konveyor -> Saat benda berhenti, AI langsung mengunci koordinat.\n")

        last_center: tuple[int, int] | None = None
        stationary_start: float = 0.0
        already_processed = False

        try:
            while True:
                frame = self.vision.read_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                h, w = frame.shape[:2]
                detections = self.vision.detector.detect(frame, apply_smart_filter=True)

                if not detections:
                    last_center = None
                    stationary_start = 0.0
                    already_processed = False
                    time.sleep(0.05)
                    continue

                top_det = detections[0]
                cx, cy = top_det.center

                # Cek apakah posisi benda diam / berhenti di konveyor (delta pergerakan < 8 piksel)
                if last_center is not None:
                    dist = np.hypot(cx - last_center[0], cy - last_center[1])
                    if dist < 8.0:
                        # Benda sedang diam di konveyor
                        if stationary_start == 0.0:
                            stationary_start = time.time()
                        elif time.time() - stationary_start >= min_stationary_seconds:
                            # BENDA SUDAH BERHENTI DIAM TOTAL:
                            if not already_processed:
                                print(f"\n🚦 [BENDA DIAM TERDETEKSI] Objek: {top_det.label}")
                                print("   📸 Mengambil 1 snapshot foto statis & mengunci koordinat fisik...")

                                # Ambil snapshot statis presisi tinggi
                                target = self.inspect_once()

                                if target is not None:
                                    print(f"   🎯 TARGET VALID: {target.label} (Conf: {target.confidence * 100:.1f}%)")
                                    print(f"      Titik Ambil Robot : X={target.x_mm} mm, Y={target.y_mm} mm, Z={target.z_mm} mm")
                                    print(f"      Titik Wadah Drop  : X={target.drop_x_mm} mm, Y={target.drop_y_mm} mm, Z={target.drop_z_mm} mm")
                                    print("      --> ✅ Koordinat telah diekspor ke outputs/target_robot.json")
                                    print("      --> 🦾 Robot mengeksekusi pengambilan ke wadah...")
                                else:
                                    print(f"   ⏩ Objek '{top_det.label}' bukan target sortir yang ingin diambil.")
                                    print("      --> Konveyor lanjut membawa objek berikutnya (diabaikan).")

                                already_processed = True
                    else:
                        # Benda masih bergerak / meluncur di konveyor
                        stationary_start = 0.0
                        already_processed = False

                last_center = (cx, cy)
                time.sleep(0.04)

        except KeyboardInterrupt:
            print("\n[AutoSortir] Berhenti.")
        finally:
            self.vision.release()

    def run_serial_ir_loop(self, port: str, baud: int = 9600) -> None:
        """Mode Serial IR: Terhubung ke Arduino / ESP32 yang membaca sensor Infrared fisik."""
        import serial  # import lokal

        print("\n" + "=" * 65)
        print(f" 🔌 MODE SENSOR INFRARED SERIAL (Port: {port}, Baud: {baud})")
        print("=" * 65)
        print("Menghubungkan ke Arduino konveyor...")

        try:
            ser = serial.Serial(port, baud, timeout=1.0)
            time.sleep(2.0)  # Tunggu reset Arduino
            print("✅ Terhubung ke Arduino! Menunggu trigger sensor IR...\n")
        except Exception as exc:
            print(f"❌ Gagal membuka port serial {port}: {exc}")
            return

        try:
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    # Jika Arduino mengirim sinyal trigger (misal: "IR_TRIGGER" atau "STOP")
                    if "IR" in line or "TRIGGER" in line or "OBJECT" in line or line == "1":
                        print("\n🚦 [SENSOR IR TERPICU] Benda tiba di depan sensor!")
                        print("   Mengambil 1 snapshot statis...")

                        target = self.inspect_once()

                        if target is not None:
                            print(f"   🎯 TARGET VALID: {target.label}")
                            print(f"      Koordinat: X={target.x_mm}, Y={target.y_mm}, Z={target.z_mm} mm")
                            # Kirim instruksi balik ke Arduino / Robot
                            ser.write(f"PICK:{target.class_name}:{target.x_mm}:{target.y_mm}:{target.z_mm}\n".encode())
                        else:
                            print("   ⏩ Bukan target sortir. Mengirim sinyal konveyor jalan...")
                            ser.write(b"RESUME\n")

                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\n[AutoSortir] Berhenti.")
        finally:
            ser.close()
            self.vision.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="Sistem Sortir Otomatis Penuh (Hands-Free)")
    ap.add_argument("--mode", choices=["auto-vision", "serial", "server"], default="auto-vision",
                    help="Mode operasi: 'auto-vision' (deteksi benda diam otomatis), 'serial' (sensor IR Arduino), atau 'server' (REST API)")
    ap.add_argument("--source", default="0", help="Nomor indeks webcam USB (0, 1, ..)")
    ap.add_argument("--target-classes", nargs="+", default=["koin", "charger"],
                    help="Daftar kelas yang ingin diambil robot (default: koin charger)")
    ap.add_argument("--port", default="/dev/ttyUSB0", help="Port serial Arduino (khusus mode serial)")
    ap.add_argument("--baud", type=int, default=9600, help="Baudrate serial Arduino")
    args = ap.parse_args()

    controller = AutoSortirController(
        source=args.source,
        target_classes=args.target_classes,
    )

    if args.mode == "auto-vision":
        controller.run_auto_vision_loop()
    elif args.mode == "serial":
        controller.run_serial_ir_loop(port=args.port, baud=args.baud)
    elif args.mode == "server":
        print("Menjalankan API Server Uvicorn...")
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI(title="OpenCV Sortir Autonomous API")

        @app.get("/target")
        def get_current_target() -> dict[str, Any]:
            t = controller.inspect_once()
            return {"status": "TARGET_READY" if t else "NO_TARGET", "target": t.to_dict() if t else None}

        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
