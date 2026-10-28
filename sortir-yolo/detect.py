#!/usr/bin/env python3
"""Langkah 5 - Deteksi real-time lewat webcam (atau gambar / video).

Benda yang terdeteksi diberi KOTAK MERAH (atau warna per-kelas) + judul (Koin / Tutup Botol)
dan titik tengahnya (cx, cy) dihitung & ditampilkan otomatis.

    python 05_detect.py                              # webcam 0, model models/best.pt
    python 05_detect.py --source 1 --conf 0.4        # kamera lain / threshold lain
    python 05_detect.py --warna-kelas                # bedakan warna kotak koin vs tutup botol
    python 05_detect.py --source foto.jpg            # satu gambar
    python 05_detect.py --source video.mp4

Tombol: q / ESC = keluar | s = simpan screenshot | c = simpan koordinat ke CSV | w = warna per-kelas
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera, CameraError, ThreadedCamera, parse_source
from src.config import (
    BOX_COLOR,
    CLASS_COLORS,
    DEFAULT_CALIB_PATH,
    DEFAULT_TARGET_JSON,
    DEFAULT_WEIGHTS,
    IMAGE_EXTS,
    OUTPUT_DIR,
)
from src.coordinate_transform import (
    RobotCoordinateTransformer,
    choose_best_target,
    export_target_json,
)
from src.detector import Detection, Detector
from src.utils import (
    Penstabil,
    append_csv,
    draw_detection,
    draw_overlay,
    draw_roi,
    save_screenshot,
)

WINDOW = "Deteksi Koin / Tutup Botol"


def parse_roi_arg(val: str | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Parse format 'x1,y1,x2,y2' atau 'auto'."""
    if not val:
        return None
    val = val.strip().lower()
    if val == "auto":
        # Zona tengah default (conveyor area)
        return int(width * 0.15), int(height * 0.20), int(width * 0.85), int(height * 0.90)
    try:
        parts = [int(p.strip()) for p in val.split(",")]
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], parts[3]
    except Exception:
        pass
    print(f"Format ROI tidak valid: {val}. Gunakan 'x1,y1,x2,y2' atau 'auto'.")
    return None


def annotate(
    frame: np.ndarray,
    detections: list[Detection],
    fps: float,
    conf: float | None = None,
    filter_active: bool = True,
    roi: tuple[int, int, int, int] | None = None,
    roi_active: bool = False,
    show_persen: bool = True,
    warna_kelas: bool = False,
    track_active: bool = False,
) -> None:
    if roi_active and roi is not None:
        draw_roi(frame, roi, active=True)
    for det in detections:
        color = CLASS_COLORS.get(det.class_name, BOX_COLOR) if warna_kelas else BOX_COLOR
        draw_detection(frame, det, color=color, show_persen=show_persen)
    draw_overlay(
        frame,
        fps,
        len(detections),
        conf=conf,
        filter_active=filter_active,
        roi_active=roi_active,
        track_active=track_active,
    )


def print_detections(detections: list[Detection]) -> None:
    stamp = time.strftime("%H:%M:%S")
    for d in detections:
        tid = f"#{d.track_id} " if getattr(d, "track_id", None) is not None else ""
        cat = f" [{d.catatan}]" if getattr(d, "catatan", "") else ""
        print(f"[{stamp}] {tid}{d.label}{cat} | conf={d.confidence:.2f} | center={d.center} | "
              f"center_norm={d.center_norm} | bbox={d.bbox}")


def run_image(
    detector: Detector,
    path: Path,
    show: bool,
    show_persen: bool = True,
    warna_kelas: bool = False,
    transformer: RobotCoordinateTransformer | None = None,
) -> int:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Tidak bisa membaca gambar: {path}")
        return 1
    detections = detector.detect(frame)
    print_detections(detections)
    if not detections:
        print("Tidak ada objek terdeteksi (semua disaring atau tidak ada objek).")

    if transformer is not None:
        h, w = frame.shape[:2]
        robot_targets = [
            transformer.to_robot_target(
                class_name=d.class_name,
                label=d.label,
                confidence=d.confidence,
                pixel_center=d.center,
                frame_width=w,
                frame_height=h,
                track_id=d.track_id,
                catatan=d.catatan,
            )
            for d in detections
        ]
        best_t = choose_best_target(robot_targets, roi=detector.roi)
        export_target_json(best_t, robot_targets, DEFAULT_TARGET_JSON)

    annotate(
        frame,
        detections,
        0.0,
        conf=detector.conf,
        filter_active=detector.smart_filter,
        roi=detector.roi,
        roi_active=detector.roi is not None,
        show_persen=show_persen,
        warna_kelas=warna_kelas,
    )
    out = OUTPUT_DIR / f"hasil_{path.stem}.jpg"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"Hasil disimpan: {out}")
    if show:
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_stream(
    detector: Detector,
    source: int | str,
    args: argparse.Namespace,
    transformer: RobotCoordinateTransformer | None = None,
) -> int:
    is_network = isinstance(source, str) and (
        source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://")
    )
    cam_cls = Camera if args.no_threaded else ThreadedCamera
    try:
        cam = cam_cls(source, args.width, args.height).open(auto_fallback=True)
    except CameraError as exc:
        print(exc)
        return 1

    smart_filter_on = not args.no_smart_filter
    warna_kelas_on = bool(args.warna_kelas)
    roi_coords = parse_roi_arg(args.roi, args.width, args.height)
    roi_active = roi_coords is not None
    current_conf = detector.conf
    penstabil = Penstabil() if not args.tanpa_stabil else None

    prev, fps, last_print = time.time(), 0.0, 0.0
    print("\nKontrol Keyboard:")
    print("  [q] / [ESC] : Keluar")
    print("  [s]         : Simpan tangkapan layar (screenshot)")
    print("  [c]         : Simpan koordinat objek ke CSV")
    print("  [t]         : Toggle Object Tracking (ON/OFF)")
    print("  [f]         : Toggle Smart Filter (ON/OFF)")
    print("  [r]         : Toggle ROI Zona Kerja (ON/OFF)")
    print("  [w]         : Toggle Warna per Kelas (Kuning=Koin, Biru=Tutup)")
    print("  [SPACE]     : Manual Trigger IR (Kunci koordinat objek diam seketika)")
    print("  [+] / [-]   : Naikkan / turunkan confidence threshold\n")

    last_center: tuple[int, int] | None = None
    stationary_start: float = 0.0
    settle_delay: float = getattr(args, "settle_delay", 0.6)
    motion_thresh: float = getattr(args, "motion_thresh", 3.5)
    robot_status = "NO_TARGET"
    robot_status_msg = "Konveyor kosong"

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("Frame habis / kamera berhenti.")
                break
            if args.flip:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            # Set default ROI ke zona tengah jika tombol 'r' ditekan tapi belum diset
            if roi_coords is None:
                roi_coords = (int(w * 0.15), int(h * 0.20), int(w * 0.85), int(h * 0.90))

            detector.roi = roi_coords if roi_active else None
            detector.conf = current_conf
            detections = detector.detect(frame, apply_smart_filter=smart_filter_on)
            if penstabil is not None:
                detections = penstabil.update(detections)

            best_t = None
            robot_targets: list[RobotTarget] = []
            if transformer is not None:
                robot_targets = [
                    transformer.to_robot_target(
                        class_name=d.class_name,
                        label=d.label,
                        confidence=d.confidence,
                        pixel_center=d.center,
                        frame_width=w,
                        frame_height=h,
                        track_id=d.track_id,
                        catatan=d.catatan,
                    )
                    for d in detections
                ]
                best_t = choose_best_target(robot_targets, roi=roi_coords if roi_active else None)

                # Logika Stop-and-Go: Menahan data saat objek bergerak, mengunci data saat diam stabil
                if best_t is not None:
                    cx, cy = best_t.pixel_center
                    if last_center is not None:
                        dist = np.hypot(cx - last_center[0], cy - last_center[1])
                        if dist <= motion_thresh:
                            # Benda mulai atau sedang diam di konveyor
                            if stationary_start == 0.0:
                                stationary_start = time.time()
                            elapsed = time.time() - stationary_start
                            if elapsed >= settle_delay:
                                robot_status = "TARGET_READY"
                                robot_status_msg = f"DIAM TOTAL ({elapsed:.1f}s) -> TARGET TERKUNCI"
                                export_target_json(
                                    best_t,
                                    robot_targets,
                                    DEFAULT_TARGET_JSON,
                                    status="TARGET_READY",
                                    message=f"Benda {best_t.label} diam terverifikasi. Koordinat terkunci.",
                                )
                            else:
                                robot_status = "WAITING_STATIONARY"
                                robot_status_msg = f"MEREDAM INERSIA ({elapsed:.1f}s/{settle_delay:.1f}s)..."
                                export_target_json(
                                    None,
                                    robot_targets,
                                    DEFAULT_TARGET_JSON,
                                    status="WAITING_STATIONARY",
                                    message=f"Menunggu inersia reda ({elapsed:.1f}s/{settle_delay:.1f}s)...",
                                )
                        else:
                            # Benda masih meluncur di konveyor -> TAHAN KOORDINAT
                            stationary_start = 0.0
                            robot_status = "CONVEYOR_MOVING"
                            robot_status_msg = "KONVEYOR JALAN (Data ditahan)"
                            export_target_json(
                                None,
                                robot_targets,
                                DEFAULT_TARGET_JSON,
                                status="CONVEYOR_MOVING",
                                message="Objek sedang bergerak di konveyor - data ditahan",
                            )
                    else:
                        stationary_start = time.time()
                    last_center = (cx, cy)
                else:
                    last_center = None
                    stationary_start = 0.0
                    robot_status = "NO_TARGET"
                    robot_status_msg = "KONVEYOR KOSONG"
                    export_target_json(None, [], DEFAULT_TARGET_JSON, status="NO_TARGET", message="Area konveyor kosong")

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6)) if fps else 1.0 / max(now - prev, 1e-6)
            prev = now

            if detections and now - last_print >= args.print_every:
                print_detections(detections)
                last_print = now

            annotate(
                frame,
                detections,
                fps,
                conf=current_conf,
                filter_active=smart_filter_on,
                roi=roi_coords,
                roi_active=roi_active,
                show_persen=not args.tanpa_persen,
                warna_kelas=warna_kelas_on,
                track_active=detector.track,
            )

            # Tampilkan Banner Status Robot di Layar Video
            if transformer is not None:
                color_hud = (0, 255, 0) if robot_status == "TARGET_READY" else (
                    (0, 215, 255) if robot_status == "WAITING_STATIONARY" else (0, 0, 255)
                )
                hud_line = f"ROBOT IPC: [{robot_status}] {robot_status_msg}"
                if robot_status == "TARGET_READY" and best_t is not None:
                    hud_line += f" | X={best_t.x_mm} Y={best_t.y_mm} Z={best_t.z_mm}"
                _text_with_outline(frame, hud_line, (10, 52), scale=0.6, color=color_hud)

            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32:  # Tombol SPACE (Manual Trigger / Kunci Koordinat)
                if best_t is not None:
                    robot_status = "TARGET_READY"
                    export_target_json(
                        best_t,
                        robot_targets,
                        DEFAULT_TARGET_JSON,
                        status="TARGET_READY",
                        message=f"Manual SPACE trigger: {best_t.label} terkunci.",
                    )
                    print(f"\n🎯 [SPACE TRIGGER] Koordinat Terkunci & Dikirim: {best_t.label}")
                    print(f"   Ambil : X={best_t.x_mm} mm, Y={best_t.y_mm} mm, Z={best_t.z_mm} mm")
                    print(f"   Drop  : X={best_t.drop_x_mm} mm, Y={best_t.drop_y_mm} mm, Z={best_t.drop_z_mm} mm\n")
            if key == ord("s"):
                print(f"Screenshot: {save_screenshot(frame, OUTPUT_DIR)}")
            if key == ord("c"):
                n = append_csv(detections, OUTPUT_DIR / "koordinat.csv")
                print(f"{n} koordinat disimpan ke {OUTPUT_DIR / 'koordinat.csv'}")
            if key in (ord("t"), ord("T")):
                detector.track = not detector.track
                print(f"[Object Tracking] -> {'ON' if detector.track else 'OFF'}")
            if key in (ord("f"), ord("F")):
                smart_filter_on = not smart_filter_on
                print(f"[Smart Filter] -> {'ON' if smart_filter_on else 'OFF'}")
            if key in (ord("r"), ord("R")):
                roi_active = not roi_active
                print(f"[ROI Conveyor] -> {'ON' if roi_active else 'OFF'}")
            if key in (ord("w"), ord("W")):
                warna_kelas_on = not warna_kelas_on
                print(f"[Warna per Kelas] -> {'ON' if warna_kelas_on else 'OFF (Merah semua)'}")
            if key in (ord("+"), ord("=")):
                current_conf = min(0.95, round(current_conf + 0.05, 2))
                print(f"[Confidence] -> {current_conf:.2f}")
            if key in (ord("-"), ord("_")):
                current_conf = max(0.10, round(current_conf - 0.05, 2))
                print(f"[Confidence] -> {current_conf:.2f}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--source", default="0", help="nomor kamera (0,1,..) atau path gambar/video")
    ap.add_argument("--conf", type=float, default=0.25, help="threshold confidence (default: 0.25 optimal untuk objek kecil)")
    ap.add_argument("--imgsz", type=int, default=1280, help="resolusi input AI (default: 1280 untuk menjaga detail objek kecil di webcam)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--width", type=int, default=1280, help="lebar kamera yang diminta")
    ap.add_argument("--height", type=int, default=720, help="tinggi kamera yang diminta")
    ap.add_argument("--flip", action="store_true", help="cerminkan gambar kamera")
    ap.add_argument("--print-every", type=float, default=0.5, help="jeda (detik) cetak koordinat di terminal")
    ap.add_argument("--no-show", action="store_true", help="jangan buka jendela (khusus untuk gambar)")
    ap.add_argument("--no-smart-filter", action="store_true", help="matikan filter geometri cerdas")
    ap.add_argument("--edge-margin", type=int, default=0, help="margin piksel batas kamera untuk tolak objek terpotong (default: 0)")
    ap.add_argument("--no-track", action="store_true", help="matikan object tracking real-time (ByteTrack)")
    ap.add_argument("--tanpa-stabil", action="store_true", help="matikan penstabil suara-terbanyak multi-frame")
    ap.add_argument("--tanpa-persen", action="store_true", help="jangan tampilkan persentase confidence di label")
    ap.add_argument("--warna-kelas", action="store_true", help="bedakan warna kotak tiap kelas (koin=emas, tutup_botol=cyan)")
    ap.add_argument("--roi", default=None, help="area aktif 'x1,y1,x2,y2' atau 'auto' (di luar area ini diabaikan)")
    ap.add_argument("--min-ar", type=float, default=0.50, help="aspek rasio w/h minimum (default: 0.50)")
    ap.add_argument("--max-ar", type=float, default=1.85, help="aspek rasio w/h maksimum (default: 1.85)")
    ap.add_argument("--max-area", type=float, default=0.40, help="fraksi luas maksimal terhadap frame (default: 0.40)")
    ap.add_argument("--min-size", type=int, default=10, help="ukuran lebar/tinggi minimum piksel (default: 10)")
    ap.add_argument("--no-threaded", action="store_true", help="matikan background thread capture zero-latency")
    ap.add_argument("--no-robot-export", action="store_true", help="jangan ekspor outputs/target_robot.json secara real-time")
    ap.add_argument("--settle-delay", type=float, default=0.6,
                    help="Jeda inersia konveyor (detik) sebelum koordinat dikunci ke robot (default: 0.6s)")
    ap.add_argument("--motion-thresh", type=float, default=3.5,
                    help="Batas pergeseran piksel benda untuk dianggap diam total (default: 3.5 px)")
    args = ap.parse_args()

    transformer = None
    if not args.no_robot_export:
        transformer = RobotCoordinateTransformer(config_path=DEFAULT_CALIB_PATH)

    try:
        detector = Detector(
            args.weights,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            smart_filter=not args.no_smart_filter,
            track=not args.no_track,
            min_ar=args.min_ar,
            max_ar=args.max_ar,
            max_area_frac=args.max_area,
            min_size_px=args.min_size,
            edge_margin=args.edge_margin,
        )
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print(f"Model: {args.weights} | device: {detector.device} | conf: {args.conf} | "
          f"filter: {'ON' if detector.smart_filter else 'OFF'} | track: {'ON' if detector.track else 'OFF'}")

    source = parse_source(args.source)
    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS:
        return run_image(
            detector,
            Path(source),
            show=not args.no_show,
            show_persen=not args.tanpa_persen,
            warna_kelas=args.warna_kelas,
            transformer=transformer,
        )
    return run_stream(detector, source, args, transformer=transformer)


if __name__ == "__main__":
    sys.exit(main())
