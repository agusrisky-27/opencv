#!/usr/bin/env python3
"""Deteksi real-time DUAL KAMERA (Webcam + ESP32-CAM).

Menjalankan dua kamera sekaligus dalam satu tampilan layar berdampingan (split-screen):
- Kamera 1: Webcam USB / Laptop (source 0 atau 1)
- Kamera 2: ESP32-CAM via Wi-Fi HTTP Stream (misal: http://192.168.1.50:81/stream)

Menggunakan multi-threading agar stream Wi-Fi ESP32-CAM zero-latency dan tidak
menghambat kecepatan FPS Webcam USB.

Contoh Penggunaan:
    # 1 Webcam USB + 1 ESP32-CAM
    python 05_detect_dual.py --source1 0 --source2 http://192.168.1.50:81/stream

    # 2 Webcam USB (misal webcam laptop 0 dan webcam USB eksternal 1)
    python 05_detect_dual.py --source1 0 --source2 1

    # Ubah threshold confidence & warna per kelas
    python 05_detect_dual.py --source1 0 --source2 http://192.168.1.50:81/stream --conf 0.4 --warna-kelas

Tombol Keyboard:
    [q] / [ESC] : Keluar
    [s]         : Simpan screenshot kedua kamera
    [c]         : Simpan koordinat objek ke CSV
    [m]         : Toggle MQTT Sender (ON/OFF)
    [w]         : Toggle Warna per Kelas (Kuning=Koin, Biru=Tutup)
    [+] / [-]   : Naikkan / turunkan confidence threshold
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.camera import CameraError, ThreadedCamera, parse_source
from src.config import BOX_COLOR, CLASS_COLORS, DEFAULT_WEIGHTS, OUTPUT_DIR
from src.detector import Detection, Detector
from src.middleware_client import MiddlewareSender
from src.mqtt_client import MqttSender
from src.utils import append_csv, draw_detection, save_screenshot

WINDOW_DUAL = "Deteksi Dual Kamera - Sortir YOLO (Webcam + ESP32-CAM)"


def annotate_camera_view(
    frame: np.ndarray,
    detections: list[Detection],
    cam_name: str,
    fps: float,
    conf: float,
    warna_kelas: bool = False,
) -> np.ndarray:
    """Beri bounding box dan header nama kamera pada frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Gambar bounding box objek
    for det in detections:
        color = CLASS_COLORS.get(det.class_name, BOX_COLOR) if warna_kelas else BOX_COLOR
        draw_detection(canvas, det, color=color, show_persen=True)

    # Header bar di atas frame
    bar_height = 36
    cv2.rectangle(canvas, (0, 0), (w, bar_height), (20, 20, 20), -1)
    header_text = f"[{cam_name}] FPS: {fps:4.1f} | Objek: {len(detections)} | Conf: {conf:.2f}"
    cv2.putText(
        canvas,
        header_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def make_waiting_frame(name: str, width: int = 640, height: int = 480) -> np.ndarray:
    """Frame placeholder saat kamera sedang loading atau reconnecting."""
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        f"Menghubungkan ke {name}...",
        (30, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    return placeholder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source1", default="0", help="Kamera 1 (default: 0 untuk Webcam USB/Internal)")
    ap.add_argument(
        "--source2",
        default="http://192.168.1.50:81/stream",
        help="Kamera 2 (default: http://192.168.1.50:81/stream untuk ESP32-CAM)",
    )
    ap.add_argument("--label1", default="Kamera 1 (Webcam USB)", help="Label nama tampilan kamera 1")
    ap.add_argument("--label2", default="Kamera 2 (ESP32-CAM)", help="Label nama tampilan kamera 2")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path ke weights model YOLO")
    ap.add_argument("--conf", type=float, default=0.30, help="Confidence threshold (default: 0.30)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--width", type=int, default=640, help="Lebar frame display tiap kamera")
    ap.add_argument("--height", type=int, default=480, help="Tinggi frame display tiap kamera")
    ap.add_argument("--flip1", action="store_true", help="Cerminkan kamera 1 secara horizontal")
    ap.add_argument("--flip2", action="store_true", help="Cerminkan kamera 2 secara horizontal")
    ap.add_argument("--warna-kelas", action="store_true", help="Bedakan warna kotak (Koin=Emas, Tutup=Cyan)")
    ap.add_argument("--no-smart-filter", action="store_true", help="Matikan filter rasio & ukuran")
    ap.add_argument("--no-mqtt", action="store_true", help="Matikan pengiriman MQTT")
    ap.add_argument("--mqtt-broker", default="p8f1fa01.ala.asia-southeast1.emqxsl.com")
    ap.add_argument("--mqtt-port", type=int, default=8883)
    ap.add_argument("--mqtt-user", default="admin_pabrik")
    ap.add_argument("--mqtt-pass", default="rahasia123")
    ap.add_argument("--mqtt-topic", default="sortir/koin/agusrisky")
    ap.add_argument("--mqtt-cooldown", type=float, default=2.0)
    ap.add_argument("--middleware", default=None, help="URL server middleware lokal (misal: http://localhost:8000)")
    args = ap.parse_args()

    # Load YOLO Detector
    try:
        detector = Detector(
            args.weights,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            smart_filter=not args.no_smart_filter,
            track=False,
        )
    except FileNotFoundError as exc:
        print(f"Error memuat model: {exc}")
        return 1

    print("==================================================")
    print(" MENJALANKAN DUAL CAMERA (WEBCAM + ESP32-CAM)")
    print("==================================================")
    print(f" [Kamera 1] : {args.source1} ({args.label1})")
    print(f" [Kamera 2] : {args.source2} ({args.label2})")
    print(f" [Model]    : {args.weights}")
    print(f" [Conf]     : {args.conf}")
    print(f" [MQTT]     : {'ON' if not args.no_mqtt else 'OFF'}")
    print("==================================================")
    print("Kontrol Keyboard:")
    print("  [q] / [ESC] : Keluar")
    print("  [s]         : Simpan tangkapan layar (screenshot dual)")
    print("  [c]         : Simpan koordinat kedua kamera ke CSV")
    print("  [m]         : Toggle MQTT (ON/OFF)")
    print("  [w]         : Toggle Warna per Kelas (Kuning=Koin, Biru=Tutup)")
    print("  [+] / [-]   : Naikkan / turunkan confidence threshold\n")

    # Inisialisasi ThreadedCamera untuk kedua sumber
    src1 = parse_source(args.source1)
    src2 = parse_source(args.source2)

    try:
        print(f"Membuka {args.label1}...")
        cam1 = ThreadedCamera(src1, args.width, args.height).open(auto_fallback=True)
        print(f"[OK] {args.label1} terhubung.")
    except CameraError as e:
        print(f"[GAGAL] {args.label1}: {e}")
        return 1

    try:
        print(f"Membuka {args.label2}...")
        cam2 = ThreadedCamera(src2, args.width, args.height).open(auto_fallback=False)
        print(f"[OK] {args.label2} terhubung.")
    except CameraError as e:
        print(f"[GAGAL] {args.label2}: {e}")
        print("\nTips koneksi ESP32-CAM:")
        print("1. Pastikan ESP32-CAM dan laptop terhubung ke WiFi/hotspot yang sama.")
        print("2. Cek Serial Monitor Arduino untuk melihat IP ESP32-CAM yang aktif.")
        print("3. Cek apakah URL stream bisa dibuka di browser: http://<IP_ESP32>:81/stream\n")
        cam1.release()
        return 1

    mqtt_sender = MqttSender(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        username=args.mqtt_user,
        password=args.mqtt_pass,
        topic=args.mqtt_topic,
        enabled=not args.no_mqtt,
        cooldown=args.mqtt_cooldown,
    )
    middleware_sender = MiddlewareSender(base_url=args.middleware, enabled=bool(args.middleware))

    warna_kelas_on = bool(args.warna_kelas)
    current_conf = args.conf
    smart_filter_on = not args.no_smart_filter

    prev1, fps1 = time.time(), 0.0
    prev2, fps2 = time.time(), 0.0

    target_h = args.height
    target_w = args.width

    try:
        while True:
            ok1, raw1 = cam1.read()
            ok2, raw2 = cam2.read()

            all_detections: list[Detection] = []

            # Proses Kamera 1
            if ok1 and raw1 is not None:
                if args.flip1:
                    raw1 = cv2.flip(raw1, 1)
                now1 = time.time()
                fps1 = 0.9 * fps1 + 0.1 * (1.0 / max(now1 - prev1, 1e-6)) if fps1 else 1.0 / max(now1 - prev1, 1e-6)
                prev1 = now1

                detector.conf = current_conf
                dets1 = detector.detect(raw1, apply_smart_filter=smart_filter_on)
                all_detections.extend(dets1)

                if dets1 and mqtt_sender.enabled:
                    for d in dets1:
                        mqtt_sender.send_detection(f"{d.class_name}_cam1", d.confidence, d.track_id)

                if dets1 and middleware_sender.enabled:
                    for d in dets1:
                        z_val = 2.0 if "koin" in d.class_name else 12.0
                        middleware_sender.send_detection(
                            class_name=f"{d.class_name}_cam1",
                            confidence=d.confidence,
                            x_mm=float(d.center[0]),
                            y_mm=float(d.center[1]),
                            z_mm=z_val,
                            track_id=d.track_id,
                            aksi="AMBIL",
                        )

                view1 = annotate_camera_view(raw1, dets1, args.label1, fps1, current_conf, warna_kelas_on)
            else:
                view1 = make_waiting_frame(args.label1, target_w, target_h)

            # Proses Kamera 2
            if ok2 and raw2 is not None:
                if args.flip2:
                    raw2 = cv2.flip(raw2, 1)
                now2 = time.time()
                fps2 = 0.9 * fps2 + 0.1 * (1.0 / max(now2 - prev2, 1e-6)) if fps2 else 1.0 / max(now2 - prev2, 1e-6)
                prev2 = now2

                detector.conf = current_conf
                dets2 = detector.detect(raw2, apply_smart_filter=smart_filter_on)
                all_detections.extend(dets2)

                if dets2 and mqtt_sender.enabled:
                    for d in dets2:
                        mqtt_sender.send_detection(f"{d.class_name}_cam2", d.confidence, d.track_id)

                if dets2 and middleware_sender.enabled:
                    for d in dets2:
                        z_val = 2.0 if "koin" in d.class_name else 12.0
                        middleware_sender.send_detection(
                            class_name=f"{d.class_name}_cam2",
                            confidence=d.confidence,
                            x_mm=float(d.center[0]),
                            y_mm=float(d.center[1]),
                            z_mm=z_val,
                            track_id=d.track_id,
                            aksi="AMBIL",
                        )

                view2 = annotate_camera_view(raw2, dets2, args.label2, fps2, current_conf, warna_kelas_on)
            else:
                view2 = make_waiting_frame(args.label2, target_w, target_h)

            # Normalisasi ukuran agar seragam sebelum digabungkan (side-by-side)
            v1_resized = cv2.resize(view1, (target_w, target_h))
            v2_resized = cv2.resize(view2, (target_w, target_h))

            # Garis pemisah vertikal di tengah
            divider = np.full((target_h, 4, 3), 100, dtype=np.uint8)

            # Gabungkan berdampingan
            combined = cv2.hconcat([v1_resized, divider, v2_resized])

            # Footer status global
            footer = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
            status_text = (
                f"MQTT: {'ON' if mqtt_sender.enabled else 'OFF'} | "
                f"Warna: {'Kelas' if warna_kelas_on else 'Standar'} | "
                f"Filter: {'ON' if smart_filter_on else 'OFF'} | "
                f"Tekan [s] Screenshot | [c] CSV | [q] Keluar"
            )
            cv2.putText(
                footer,
                status_text,
                (15, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            ) 
            final_display = cv2.vconcat([combined, footer])

            cv2.imshow(WINDOW_DUAL, final_display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                p = save_screenshot(final_display, OUTPUT_DIR)
                print(f"Screenshot dual disimpan: {p}")
            if key == ord("c"):
                n = append_csv(all_detections, OUTPUT_DIR / "koordinat_dual.csv")
                print(f"{n} koordinat disimpan ke {OUTPUT_DIR / 'koordinat_dual.csv'}")
            if key in (ord("m"), ord("M")):
                mqtt_sender.enabled = not mqtt_sender.enabled
                print(f"[MQTT] -> {'ON' if mqtt_sender.enabled else 'OFF'}")
            if key in (ord("w"), ord("W")):
                warna_kelas_on = not warna_kelas_on
                print(f"[Warna Kelas] -> {'ON' if warna_kelas_on else 'OFF'}")
            if key in (ord("+"), ord("=")):
                current_conf = min(0.95, round(current_conf + 0.05, 2))
                print(f"[Confidence] -> {current_conf:.2f}")
            if key in (ord("-"), ord("_")):
                current_conf = max(0.10, round(current_conf - 0.05, 2))
                print(f"[Confidence] -> {current_conf:.2f}")

    finally:
        cam1.release()
        cam2.release()
        mqtt_sender.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
