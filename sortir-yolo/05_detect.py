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
from src.config import BOX_COLOR, CLASS_COLORS, DEFAULT_WEIGHTS, IMAGE_EXTS, OUTPUT_DIR
from src.detector import Detection, Detector
from src.middleware_client import MiddlewareSender
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
    middleware_active: bool = True,
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
        middleware_active=middleware_active,
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
    middleware_url: str | None = None,
) -> int:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Tidak bisa membaca gambar: {path}")
        return 1
    detections = detector.detect(frame)
    print_detections(detections)
    if not detections:
        print("Tidak ada objek terdeteksi (semua disaring atau tidak ada objek).")
    elif middleware_url:
        sender = MiddlewareSender(base_url=middleware_url)
        for d in detections:
            z_val = 2.0 if "koin" in d.class_name else 12.0
            sender.send_detection(
                class_name=d.class_name,
                confidence=d.confidence,
                x_mm=float(d.center[0]),
                y_mm=float(d.center[1]),
                z_mm=z_val,
                track_id=d.track_id,
                aksi="AMBIL",
            )
        time.sleep(0.2)  # Beri waktu background thread mengirim payload

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
        middleware_active=bool(middleware_url),
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


def run_stream(detector: Detector, source: int | str, args: argparse.Namespace) -> int:
    is_network = isinstance(source, str) and (
        source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://")
    )
    cam_cls = ThreadedCamera if (args.threaded or is_network) else Camera
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

    middleware_sender = MiddlewareSender(
        base_url=args.middleware,
        enabled=not args.no_middleware,
    )

    prev, fps, last_print = time.time(), 0.0, 0.0
    print("\nKontrol Keyboard:")
    print("  [q] / [ESC] : Keluar")
    print("  [s]         : Simpan tangkapan layar (screenshot)")
    print("  [c]         : Simpan koordinat objek ke CSV")
    print("  [t]         : Toggle Object Tracking (ON/OFF)")
    print("  [m]         : Toggle Pengiriman ke Middleware Lokal (ON/OFF)")
    print("  [f]         : Toggle Smart Filter (ON/OFF)")
    print("  [r]         : Toggle ROI Zona Kerja (ON/OFF)")
    print("  [w]         : Toggle Warna per Kelas (Kuning=Koin, Biru=Tutup)")
    print("  [+] / [-]   : Naikkan / turunkan confidence threshold\n")

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

            # Kirim sinyal ke Local Middleware Server secara real-time
            if detections and middleware_sender.enabled:
                for d in detections:
                    z_val = 2.0 if "koin" in d.class_name else 12.0
                    middleware_sender.send_detection(
                        class_name=d.class_name,
                        confidence=d.confidence,
                        x_mm=float(d.center[0]),
                        y_mm=float(d.center[1]),
                        z_mm=z_val,
                        track_id=d.track_id,
                        aksi="AMBIL",
                    )

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
                middleware_active=middleware_sender.enabled,
            )
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                print(f"Screenshot: {save_screenshot(frame, OUTPUT_DIR)}")
            if key == ord("c"):
                n = append_csv(detections, OUTPUT_DIR / "koordinat.csv")
                print(f"{n} koordinat disimpan ke {OUTPUT_DIR / 'koordinat.csv'}")
            if key in (ord("t"), ord("T")):
                detector.track = not detector.track
                print(f"[Object Tracking] -> {'ON' if detector.track else 'OFF'}")
            if key in (ord("m"), ord("M")):
                middleware_sender.enabled = not middleware_sender.enabled
                print(f"[Middleware Lokal] -> {'ON (Aktif)' if middleware_sender.enabled else 'OFF (Mati)'}")
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
    ap.add_argument("--conf", type=float, default=0.30, help="threshold confidence (default: 0.30)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--width", type=int, default=1280, help="lebar kamera yang diminta")
    ap.add_argument("--height", type=int, default=720, help="tinggi kamera yang diminta")
    ap.add_argument("--flip", action="store_true", help="cerminkan gambar kamera")
    ap.add_argument("--print-every", type=float, default=0.5, help="jeda (detik) cetak koordinat di terminal")
    ap.add_argument("--no-show", action="store_true", help="jangan buka jendela (khusus untuk gambar)")
    ap.add_argument("--no-smart-filter", action="store_true", help="matikan filter geometri cerdas")
    ap.add_argument("--no-track", action="store_true", help="matikan object tracking real-time (ByteTrack)")
    ap.add_argument("--middleware", default="http://localhost:8000", help="URL server middleware lokal (default: http://localhost:8000)")
    ap.add_argument("--no-middleware", action="store_true", help="matikan pengiriman koordinat ke middleware lokal")
    ap.add_argument("--tanpa-stabil", action="store_true", help="matikan penstabil suara-terbanyak multi-frame")
    ap.add_argument("--tanpa-persen", action="store_true", help="jangan tampilkan persentase confidence di label")
    ap.add_argument("--warna-kelas", action="store_true", help="bedakan warna kotak tiap kelas (koin=emas, tutup_botol=cyan)")
    ap.add_argument("--roi", default=None, help="area aktif 'x1,y1,x2,y2' atau 'auto' (di luar area ini diabaikan)")
    ap.add_argument("--min-ar", type=float, default=0.50, help="aspek rasio w/h minimum (default: 0.50)")
    ap.add_argument("--max-ar", type=float, default=1.85, help="aspek rasio w/h maksimum (default: 1.85)")
    ap.add_argument("--max-area", type=float, default=0.40, help="fraksi luas maksimal terhadap frame (default: 0.40)")
    ap.add_argument("--min-size", type=int, default=10, help="ukuran lebar/tinggi minimum piksel (default: 10)")
    ap.add_argument("--threaded", action="store_true", help="gunakan background thread agar capture zero-latency (otomatis aktif untuk stream URL)")
    args = ap.parse_args()

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
        )
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print(f"Model: {args.weights} | device: {detector.device} | conf: {args.conf} | "
          f"filter: {'ON' if detector.smart_filter else 'OFF'} | track: {'ON' if detector.track else 'OFF'} | "
          f"middleware: {'ON (' + args.middleware + ')' if not args.no_middleware else 'OFF'}")

    source = parse_source(args.source)
    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS:
        return run_image(
            detector,
            Path(source),
            show=not args.no_show,
            show_persen=not args.tanpa_persen,
            warna_kelas=args.warna_kelas,
            middleware_url=args.middleware if not args.no_middleware else None,
        )
    return run_stream(detector, source, args)


if __name__ == "__main__":
    sys.exit(main())
