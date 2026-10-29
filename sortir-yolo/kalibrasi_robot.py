#!/usr/bin/env python3
"""Alat Kalibrasi Koordinat Kamera ke Robot RNV3 (Kamera & Robot Berseberangan).

Memungkinkan tim robotik menyelaraskan titik piksel kamera OpenCV dengan koordinat
fisik lengan robot (X, Y, Z mm) secara interaktif di laptop Windows.

Cara Penggunaan:
1. Jalankan skrip ini:
     python kalibrasi_robot.py --source 0
2. Klik 4 titik acuan di meja/conveyor (misal 4 sudut atau 4 titik yang bisa dijangkau robot).
3. Masukkan koordinat fisik robot (X, Y dalam mm) di jendela terminal saat diminta.
4. Matriks Homography presisi tinggi akan langsung dihitung & disimpan ke 'robot_calib.json'.
5. Gerakkan kursor mouse di video untuk melihat koordinat robot fisik secara langsung!

Tombol Cepat (Mode Manual):
  [b]       : Toggle mode Berseberangan (ON/OFF)
  [x]       : Toggle Flip sumbu X (kiri-kanan)
  [y]       : Toggle Flip sumbu Y (maju-mundur)
  [r]       : Reset titik kalibrasi
  [s]       : Simpan konfigurasi kalibrasi
  [q] / ESC : Keluar
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera, CameraError, ThreadedCamera, parse_source
from src.config import DEFAULT_CALIB_PATH
from src.coordinate_transform import RobotCoordinateTransformer

WINDOW = "Kalibrasi OpenCV -> Robot RNV3"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="Index webcam (0, 1) atau video stream")
    ap.add_argument("--config", default=str(DEFAULT_CALIB_PATH), help=f"Path file konfigurasi kalibrasi (default: {DEFAULT_CALIB_PATH.name})")
    ap.add_argument("--width", type=int, default=1280, help="Lebar kamera")
    ap.add_argument("--height", type=int, default=720, help="Tinggi kamera")
    args = ap.parse_args()

    config_path = Path(args.config)
    transformer = RobotCoordinateTransformer(config_path=config_path)

    source = parse_source(args.source)
    try:
        cam = Camera(source, args.width, args.height).open(auto_fallback=True)
    except CameraError as exc:
        print(f"[ERROR] Gagal membuka kamera: {exc}")
        return 1

    clicked_pixel_pts: list[tuple[int, int]] = []
    robot_pts: list[tuple[float, float]] = []
    current_mouse: tuple[int, int] = (args.width // 2, args.height // 2)
    pending_pixel_click: tuple[int, int] | None = None

    def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
        nonlocal current_mouse, pending_pixel_click
        current_mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pixel_pts) < 4 and pending_pixel_click is None:
            pending_pixel_click = (x, y)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    print("\n========================================================")
    print(" 🛠️ ALAT KALIBRASI KAMERA OPENCV -> ROBOT RNV3")
    print("========================================================")
    print(" * Klik 4 titik di area kerja conveyor / meja robot.")
    print(" * Masukkan koordinat robot mm pada terminal.")
    print(" * Arahkan mouse untuk melihat koordinat robot secara real-time.")
    print(" Tekan 'q' untuk selesai & simpan.")
    print("========================================================\n")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                break

            h, w = frame.shape[:2]

            # Jika ada klik baru yang menunggu input koordinat robot dari terminal
            if pending_pixel_click is not None:
                px, py = pending_pixel_click
                idx = len(clicked_pixel_pts) + 1
                cv2.circle(frame, (px, py), 9, (0, 0, 255), -1)
                cv2.putText(frame, f">> Titik #{idx} terpilih! Masukkan X & Y (mm) di terminal...", (20, h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.imshow(WINDOW, frame)
                cv2.waitKey(1)

                print(f"\n[Titik #{idx}] Terpilih di piksel ({px}, {py}).")
                print(f"Masukkan koordinat fisik Robot (mm) untuk Titik #{idx}:")
                try:
                    rx_str = input(f"   X_robot (mm) Titik #{idx}: ").strip()
                    ry_str = input(f"   Y_robot (mm) Titik #{idx}: ").strip()
                    rx = float(rx_str)
                    ry = float(ry_str)
                    clicked_pixel_pts.append((px, py))
                    robot_pts.append((rx, ry))
                    print(f"✅ Titik #{idx} dicatat: Piksel=({px}, {py}) -> Robot=({rx:.1f}, {ry:.1f}) mm")

                    if len(clicked_pixel_pts) == 4:
                        print("\n🎯 Menghitung Matriks Homography 4-Titik...")
                        ok_h = transformer.set_homography_points(clicked_pixel_pts, robot_pts)
                        if ok_h:
                            transformer.save_config(config_path)
                            print(f"🎉 SUKSES! Matriks Kalibrasi tersimpan di '{config_path}'")
                        else:
                            print("❌ Gagal menghitung Homography. Periksa kembali input titik.")
                except Exception as e:
                    print(f"⚠️ Input dibatalkan / format salah: {e}")
                finally:
                    pending_pixel_click = None

            # Gambar titik-titik kalibrasi yang sudah diklik
            for i, (px, py) in enumerate(clicked_pixel_pts):
                cv2.circle(frame, (px, py), 7, (0, 255, 0), -1)
                lbl = f"P{i+1}"
                if i < len(robot_pts):
                    lbl += f" (X:{robot_pts[i][0]:.0f}, Y:{robot_pts[i][1]:.0f})"
                cv2.putText(frame, lbl, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if len(clicked_pixel_pts) >= 2:
                for i in range(len(clicked_pixel_pts) - 1):
                    cv2.line(frame, clicked_pixel_pts[i], clicked_pixel_pts[i+1], (0, 255, 255), 1)
                if len(clicked_pixel_pts) == 4:
                    cv2.line(frame, clicked_pixel_pts[3], clicked_pixel_pts[0], (0, 255, 255), 1)

            # Hitung koordinat robot di posisi kursor mouse saat ini
            mx, my = current_mouse
            rx, ry, _ = transformer.pixel_to_robot((mx, my), "koin", frame_width=w, frame_height=h)

            # Gambar crosshair pada kursor mouse
            cv2.drawMarker(frame, (mx, my), (0, 165, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(
                frame,
                f"Mouse: ({mx}, {my}) -> Robot mm: X={rx:.1f}, Y={ry:.1f}",
                (mx + 15, my + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            # Header info status
            mode_calib = "Homography (4-Titik)" if transformer.homography_matrix is not None else "Parametrik Linear"
            status_line = (
                f"Mode: {mode_calib} | "
                f"Berseberangan: {'ON' if transformer.berseberangan else 'OFF'} | "
                f"Flip X: {'ON' if transformer.flip_x else 'OFF'} | "
                f"Flip Y: {'ON' if transformer.flip_y else 'OFF'}"
            )
            cv2.putText(frame, status_line, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)

            instruction = (
                "Klik 4 titik untuk kalibrasi | [b] Berseberangan | [x] Flip X | [y] Flip Y | [r] Reset | [q] Keluar"
            )
            cv2.putText(frame, instruction, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("b"):
                transformer.berseberangan = not transformer.berseberangan
                transformer.flip_x = transformer.berseberangan
                print(f"[Kalibrasi] Berseberangan: {'ON' if transformer.berseberangan else 'OFF'}")
                transformer.save_config(config_path)
            elif key == ord("x"):
                transformer.flip_x = not transformer.flip_x
                print(f"[Kalibrasi] Flip X: {'ON' if transformer.flip_x else 'OFF'}")
                transformer.save_config(config_path)
            elif key == ord("y"):
                transformer.flip_y = not transformer.flip_y
                print(f"[Kalibrasi] Flip Y: {'ON' if transformer.flip_y else 'OFF'}")
                transformer.save_config(config_path)
            elif key == ord("r"):
                clicked_pixel_pts.clear()
                robot_pts.clear()
                transformer.homography_matrix = None
                transformer.calibration_points.clear()
                transformer.save_config(config_path)
                print("[Kalibrasi] Titik kalibrasi di-reset.")
            elif key == ord("s"):
                transformer.save_config(config_path)
                print(f"[Kalibrasi] Konfigurasi disimpan ke {config_path}")

    finally:
        cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
