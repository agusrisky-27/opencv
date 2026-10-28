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
        conf: float = 0.25,
        target_classes: list[str] | None = None,
        edge_margin: int = 0,
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.target_classes = target_classes or ["koin", "tutup_botol", "charger"]
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

    def inspect_once(self, settle_delay: float = 0.6, verify_motion: bool = True) -> RobotTarget | None:
        """Satu kali snapshot statis saat benda diam (koordinat terkunci)."""
        return self.vision.inspect_stationary_object(
            target_classes=self.target_classes,
            save_snapshot=True,
            settle_delay=settle_delay,
            flush_frames=4,
            verify_motion=verify_motion,
        )

    def run_auto_vision_loop(
        self,
        min_stationary_seconds: float = 0.6,
        motion_threshold_px: float = 3.5,
    ) -> None:
        """Mode Otomatis Murni: Kamera mendeteksi otomatis saat ada benda yang berhenti di konveyor.

        - Saat benda bergerak: Data tidak dikirim (status 'CONVEYOR_MOVING').
        - Saat benda berhenti: Menunggu jeda stabilitas (min_stationary_seconds = 0.6 detik).
        - Setelah diam total terverifikasi: Mengunci koordinat fisik (status 'TARGET_READY').
        """
        print("\n" + "=" * 65)
        print(" 🚀 MODE OTOMATIS PENUH (AUTO-VISION TRIGGER: HANDS-FREE)")
        print("=" * 65)
        print("Sistem berjalan 100% otomatis tanpa menyentuh keyboard!")
        print("Kamera mengawasi konveyor -> Saat benda berhenti stabil, AI mengunci koordinat.\n")

        self.vision.set_status("CONVEYOR_MOVING", "Konveyor berjalan, memantau objek...")
        last_center: tuple[int, int] | None = None
        stationary_start: float = 0.0
        already_processed = False
        last_motion_state = "MOVING"

        try:
            while True:
                frame = self.vision.read_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                h, w = frame.shape[:2]
                detections = self.vision.detector.detect(frame, apply_smart_filter=True)

                if not detections:
                    if last_center is not None:
                        self.vision.set_status("CONVEYOR_MOVING", "Konveyor kosong / mencari objek...")
                    last_center = None
                    stationary_start = 0.0
                    already_processed = False
                    time.sleep(0.05)
                    continue

                top_det = detections[0]
                cx, cy = top_det.center

                # Cek apakah posisi benda diam / berhenti di konveyor
                if last_center is not None:
                    dist = np.hypot(cx - last_center[0], cy - last_center[1])
                    if dist <= motion_threshold_px:
                        # Benda mulai atau sedang diam di konveyor
                        if stationary_start == 0.0:
                            stationary_start = time.time()
                            if last_motion_state != "WAITING":
                                self.vision.set_status(
                                    "WAITING_STATIONARY",
                                    f"Objek '{top_det.label}' mulai berhenti, menunggu stabil...",
                                )
                                last_motion_state = "WAITING"

                        elapsed = time.time() - stationary_start
                        if elapsed >= min_stationary_seconds:
                            # BENDA SUDAH BERHENTI DIAM TOTAL (STABIL):
                            if not already_processed:
                                print(f"\n🚦 [BENDA DIAM TERVERIFIKASI] Objek: {top_det.label} (Diam selama {elapsed:.2f}s)")
                                print("   📸 Mengambil 1 snapshot foto statis & mengunci koordinat fisik...")

                                # Ambil snapshot statis presisi tinggi
                                target = self.inspect_once(settle_delay=0.1, verify_motion=True)

                                if target is not None:
                                    print(f"   🎯 TARGET VALID: {target.label} (Conf: {target.confidence * 100:.1f}%)")
                                    print(f"      Titik Ambil Robot : X={target.x_mm} mm, Y={target.y_mm} mm, Z={target.z_mm} mm")
                                    print(f"      Titik Wadah Drop  : X={target.drop_x_mm} mm, Y={target.drop_y_mm} mm, Z={target.drop_z_mm} mm")
                                    print("      --> ✅ Koordinat telah diekspor ke outputs/target_robot.json (TARGET_READY)")
                                    print("      --> 🦾 Robot mengeksekusi pengambilan ke wadah...")
                                else:
                                    print(f"   ⏩ Objek '{top_det.label}' bukan target sortir yang ingin diambil.")
                                    print("      --> Konveyor lanjut membawa objek berikutnya (diabaikan).")

                                already_processed = True
                                last_motion_state = "PROCESSED"
                    else:
                        # Benda masih meluncur / bergerak di konveyor -> TAHAN DATA
                        if last_motion_state != "MOVING":
                            self.vision.set_status(
                                "CONVEYOR_MOVING",
                                f"Objek '{top_det.label}' sedang bergerak di konveyor (data ditahan).",
                            )
                            last_motion_state = "MOVING"
                        stationary_start = 0.0
                        already_processed = False

                last_center = (cx, cy)
                time.sleep(0.04)

        except KeyboardInterrupt:
            print("\n[AutoSortir] Berhenti.")
        finally:
            self.vision.release()

    def run_serial_ir_loop(
        self,
        port: str,
        baud: int = 9600,
        settle_delay: float = 0.6,
    ) -> None:
        """Mode Serial IR: Terhubung ke Arduino / ESP32 yang membaca sensor Infrared fisik.

        Alur Stop-and-Go Settling:
        1. Konveyor berjalan -> Status 'CONVEYOR_MOVING', data ditahan.
        2. Sensor IR terpicu -> Konveyor dimatikan seketika oleh Arduino.
        3. Menunggu 'settle_delay' (default 0.6s) agar inersia motor & selip benda reda total.
        4. Verifikasi keheningan gerak kamera (Zero-Motion) & ambil snapshot tajam.
        5. Ekspor koordinat 'TARGET_READY' ke robot.
        6. Setelah selesai, Arduino melanjutkan konveyor ('RESUME').
        """
        import serial  # import lokal

        print("\n" + "=" * 65)
        print(f" 🔌 MODE SENSOR INFRARED SERIAL (Port: {port}, Baud: {baud})")
        print("=" * 65)
        print(f"Parameter Settling Delay : {settle_delay:.2f} detik (Peredam inersia konveyor)")
        print("Menghubungkan ke Arduino konveyor...")

        try:
            ser = serial.Serial(port, baud, timeout=1.0)
            time.sleep(2.0)  # Tunggu reset Arduino
            print("✅ Terhubung ke Arduino! Menunggu trigger sensor IR...\n")
        except Exception as exc:
            print(f"❌ Gagal membuka port serial {port}: {exc}")
            return

        self.vision.set_status("CONVEYOR_MOVING", "Konveyor berjalan, menunggu sinyal sensor IR...")

        try:
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    # Sinyal bahwa robot telah selesai melakukan pick & place
                    if "DONE" in line or "PICKED" in line:
                        print("🦾 [ROBOT SELESAI] Siklus ambil selesai. Menyalakan kembali konveyor...")
                        ser.write(b"RESUME\n")
                        self.vision.set_status("CONVEYOR_MOVING", "Robot selesai, konveyor kembali berjalan.")
                        continue

                    # Jika Arduino mengirim sinyal trigger (misal: "IR_TRIGGER", "OBJECT", atau "1")
                    if "IR" in line or "TRIGGER" in line or "OBJECT" in line or line == "1":
                        print(f"\n🚦 [SENSOR IR TERPICU] Benda tiba di depan sensor! (Pesan: '{line}')")
                        print(f"   ⏳ Menahan pengiriman data ke robot...")
                        print(f"   ⏳ Menunggu jeda inersia konveyor ({settle_delay:.2f}s) & verifikasi gerak diam...")

                        # Status WAITING_STATIONARY: robot tidak boleh bergerak selama jeda inersia
                        self.vision.set_status(
                            "WAITING_STATIONARY",
                            f"Sensor IR aktif. Menunggu inersia reda ({settle_delay:.2f}s)...",
                        )

                        # Ambil snapshot statis presisi tinggi dengan jeda inersia
                        target = self.inspect_once(settle_delay=settle_delay, verify_motion=True)

                        if target is not None:
                            print(f"   🎯 TARGET VALID: {target.label} (Conf: {target.confidence * 100:.1f}%)")
                            print(f"      Titik Ambil Robot : X={target.x_mm} mm, Y={target.y_mm} mm, Z={target.z_mm} mm")
                            print(f"      Titik Wadah Drop  : X={target.drop_x_mm} mm, Y={target.drop_y_mm} mm, Z={target.drop_z_mm} mm")
                            print("      --> ✅ Koordinat TERKUNCI & diekspor ke outputs/target_robot.json (TARGET_READY)")
                            print("      --> 📡 Mengirim instruksi PICK ke Arduino/Robot Controller...")

                            cmd = f"PICK:{target.class_name}:{target.x_mm}:{target.y_mm}:{target.z_mm}\n"
                            ser.write(cmd.encode())
                        else:
                            print("   ⏩ Objek bukan target sortir atau area kosong.")
                            print("      --> Mengirim sinyal RESUME agar konveyor jalan kembali...")
                            ser.write(b"RESUME\n")
                            self.vision.set_status("CONVEYOR_MOVING", "Bukan target sortir, konveyor lanjut jalan.")

                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\n[AutoSortir] Berhenti.")
        finally:
            ser.close()
            self.vision.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="Sistem Sortir Otomatis Penuh (Hands-Free)")
    ap.add_argument("--mode", choices=["auto-vision", "serial"], default="auto-vision",
                    help="Mode operasi: 'auto-vision' (deteksi benda diam otomatis), atau 'serial' (sensor IR Arduino)")
    ap.add_argument("--source", default="0", help="Nomor indeks webcam USB (0, 1, ..)")
    ap.add_argument("--target-classes", nargs="+", default=["koin", "charger"],
                    help="Daftar kelas yang ingin diambil robot (default: koin charger)")
    ap.add_argument("--port", default="/dev/ttyUSB0", help="Port serial Arduino (khusus mode serial)")
    ap.add_argument("--baud", type=int, default=9600, help="Baudrate serial Arduino")
    ap.add_argument("--settle-delay", type=float, default=0.6,
                    help="Jeda inersia (detik) saat konveyor berhenti sebelum snapshot diambil (default: 0.6s)")
    ap.add_argument("--stationary-time", type=float, default=0.6,
                    help="Durasi (detik) benda harus diam di kamera sebelum terkunci (khusus auto-vision, default: 0.6s)")
    ap.add_argument("--motion-thresh", type=float, default=3.5,
                    help="Batas toleransi gerak piksel (default: 3.5 px)")
    args = ap.parse_args()

    controller = AutoSortirController(
        source=args.source,
        target_classes=args.target_classes,
    )

    if args.mode == "auto-vision":
        controller.run_auto_vision_loop(
            min_stationary_seconds=args.stationary_time,
            motion_threshold_px=args.motion_thresh,
        )
    elif args.mode == "serial":
        controller.run_serial_ir_loop(
            port=args.port,
            baud=args.baud,
            settle_delay=args.settle_delay,
        )


if __name__ == "__main__":
    main()
