#!/usr/bin/env python3
"""Program Pengujian & Pemantau Pengiriman Koordinat ke Robot.

Program ini dibuat khusus untuk memverifikasi secara langsung apakah data koordinat
benar-benar terkirim atau ditahan sesuai status konveyor & sensor IR:
1. Saat konveyor bergerak  --> Data DITAHAN (target: null, status: CONVEYOR_MOVING).
2. Saat baru berhenti      --> Data DITAHAN (status: WAITING_STATIONARY, jeda inersia).
3. Saat benda diam total   --> Data TERKIRIM (status: TARGET_READY, koordinat X, Y, Z mm terkunci).

CARA PAKAI:
    # 1. Jalankan Simulasi Penuh Siklus Stop-and-Go (Bisa langsung dicoba tanpa kamera/alat):
    python tes_kirim_koordinat.py --mode simulasi

    # 2. Pantau Pengiriman Data Real-Time (Membaca file 'outputs/target_robot.json' secara live):
    python tes_kirim_koordinat.py --mode monitor

    # 3. Tes Langsung dengan Webcam Laptop/USB:
    python tes_kirim_koordinat.py --mode webcam --source 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import DEFAULT_TARGET_JSON
from src.coordinate_transform import RobotTarget, export_target_json
from sortir_vision_api import SortirVisionAPI, read_target_json


def cetak_kotak_status(data: dict | None) -> None:
    """Mencetak kartu informasi status pengiriman koordinat ke terminal."""
    print("\n" + "=" * 65)
    if not data:
        print(" ❌ FILE 'outputs/target_robot.json' BELUM TERSEDIA")
        print("    (Jalankan 'sortir_otomatis.py' atau mode simulasi terlebih dahulu)")
        print("=" * 65)
        return

    status = data.get("status", "UNKNOWN")
    waktu = data.get("waktu_baca", "-")
    pesan = data.get("message", "-")
    target = data.get("target")

    # Indikator apakah data terkirim ke robot atau ditahan
    if status == "TARGET_READY" and target:
        status_banner = " 🟢 [DATA KOORDINAT TERKIRIM KE ROBOT] - STATUS: TARGET_READY "
        data_dikirim = True
    elif status == "WAITING_STATIONARY":
        status_banner = " 🟡 [DATA DITAHAN (JEDA INERSIA)]      - STATUS: WAITING_STATIONARY "
        data_dikirim = False
    elif status == "CONVEYOR_MOVING":
        status_banner = " 🔴 [DATA DITAHAN (KONVEYOR JALAN)]    - STATUS: CONVEYOR_MOVING "
        data_dikirim = False
    else:
        status_banner = f" ⚪ [TIDAK ADA TARGET / DIABAIKAN]     - STATUS: {status} "
        data_dikirim = False

    print(status_banner)
    print("=" * 65)
    print(f" ⏱️ Waktu Update : {waktu}")
    print(f" 📢 Keterangan   : {pesan}")
    print(f" 📡 Status Robot : {'✅ MENERIMA KOORDINAT' if data_dikirim else '⛔ STANDBY (DILARANG JALAN)'}")

    if data_dikirim and target:
        r = target.get("robot_mm", {})
        drop = target.get("drop_mm", {})
        kelas = target.get("label", "-")
        conf = target.get("confidence", 0.0) * 100
        catatan = target.get("catatan", "")

        print("-" * 65)
        print(f" 🎯 Benda Terkunci      : {kelas} (Akurasi AI: {conf:.1f}%)")
        if catatan:
            print(f" 💡 Catatan Khusus      : {catatan}")
        print(f" 📍 Titik Ambil (Pick)  : X={r.get('x')} mm, Y={r.get('y')} mm, Z={r.get('z')} mm")
        print(f" 📦 Titik Wadah (Drop)  : X={drop.get('x')} mm, Y={drop.get('y')} mm, Z={drop.get('z')} mm")
        print(f" 🦾 Perintah G-Code     : G0 X{r.get('x')} Y{r.get('y')} Z{r.get('z')} F100")
    else:
        print("-" * 65)
        print(" 📍 Koordinat Target    : null (Tidak ada koordinat yang boleh diakses robot)")
    print("=" * 65 + "\n")


def mode_monitor() -> None:
    """Mode Pemantau: Membaca file 'outputs/target_robot.json' secara live detik demi detik."""
    print("==========================================================")
    print(" 📡 MODE MONITOR: MEMANTAU PENGIRIMAN KOORDINAT LIVE")
    print("==========================================================")
    print(f"Memantau file: {DEFAULT_TARGET_JSON}")
    print("Buka terminal lain lalu jalankan 'sortir_otomatis.py' atau tes simulasi.")
    print("Tekan Ctrl+C untuk berhenti.\n")

    last_ts = 0.0
    last_status = ""
    try:
        while True:
            data = read_target_json(DEFAULT_TARGET_JSON)
            if data:
                ts = data.get("timestamp", 0.0)
                status = data.get("status", "")
                if ts != last_ts or status != last_status:
                    last_ts = ts
                    last_status = status
                    cetak_kotak_status(data)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Monitor] Berhenti.")


def mode_simulasi(json_path: Path | str = DEFAULT_TARGET_JSON) -> None:
    """Mode Simulasi: Memperagakan siklus konveyor Stop-and-Go lengkap secara otomatis."""
    print("==========================================================")
    print(" 🧪 MODE SIMULASI: SIKLUS PENGIRIMAN KOORDINAT STOP-AND-GO")
    print("==========================================================")
    print("Menyimulasikan alur fisik konveyor -> sensor IR -> jeda inersia -> robot pick.\n")

    p = Path(json_path)

    # 1. Tahap 1: Konveyor sedang berjalan
    print("▶️ [FASE 1/4] Konveyor mulai berjalan membawa objek...")
    export_target_json(
        None,
        [],
        p,
        status="CONVEYOR_MOVING",
        message="Sabuk konveyor sedang berputar aktif. Objek meluncur.",
    )
    cetak_kotak_status(read_target_json(p))
    time.sleep(2.0)

    # 2. Tahap 2: Sensor IR terpicu & konveyor dimatikan (Jeda Inersia)
    print("🛑 [FASE 2/4] SENSOR IR TERPICU! Arduino mematikan relay konveyor.")
    print("   ⏳ Menahan pengiriman data... Meredam inersia mekanis & selip (0.6 detik)...")
    export_target_json(
        None,
        [],
        p,
        status="WAITING_STATIONARY",
        message="Sensor IR terpicu. Menunggu inersia & getaran sabuk reda (0.60s)...",
    )
    cetak_kotak_status(read_target_json(p))
    time.sleep(0.6)  # Settling delay 0.6 detik

    # 3. Tahap 3: Benda diam total terverifikasi -> Koordinat dikunci & dikirim
    print("🎯 [FASE 3/4] Benda telah diam total (Zero-Motion terverifikasi)! AI mengunci koordinat.")
    mock_target = RobotTarget(
        class_name="koin",
        label="Koin",
        confidence=0.96,
        pixel_center=(640, 360),
        x_mm=45.20,
        y_mm=135.00,
        z_mm=2.00,
        drop_x_mm=-150.00,
        drop_y_mm=100.00,
        drop_z_mm=45.00,
        track_id=1,
        catatan="Koin logam 500 perak di pusat konveyor",
    )
    export_target_json(
        mock_target,
        [mock_target],
        p,
        status="TARGET_READY",
        message="Benda Koin diam terverifikasi. Koordinat terkunci siap dieksekusi robot.",
    )
    cetak_kotak_status(read_target_json(p))
    time.sleep(2.5)

    # 4. Tahap 4: Robot selesai mengambil -> Konveyor jalan lagi
    print("✅ [FASE 4/4] Lengan robot selesai memindahkan objek ke wadah.")
    print("   🟢 Mengirim sinyal RESUME: Menyalakan kembali konveyor.")
    export_target_json(
        None,
        [],
        p,
        status="CONVEYOR_MOVING",
        message="Siklus ambil selesai. Konveyor kembali berjalan mencari objek berikutnya.",
    )
    cetak_kotak_status(read_target_json(p))
    print("🎉 SIMULASI SELESAI DENGAN SUKSES!")


def mode_webcam(source: int | str, target_classes: list[str]) -> None:
    """Mode Uji Nyata Webcam: Menggunakan kamera laptop/USB untuk mendeteksi benda riil."""
    print("==========================================================")
    print(f" 📷 MODE UJI LANGSUNG WEBCAM (Source: {source})")
    print("==========================================================")
    print(f"Kelas target: {target_classes}")
    print("Memuat modul Vision API...")

    vision = SortirVisionAPI(source=source)
    print("Vision siap! Letakkan koin / tutup botol / charger di depan webcam.")
    print("👉 Tekan [ENTER] di terminal untuk simulasi 'Sensor IR Terpicu & Konveyor Berhenti'")
    print("Tekan Ctrl+C untuk keluar.\n")

    try:
        no = 1
        while True:
            input(f"[{no}] Tekan ENTER untuk simulasi benda tiba di sensor IR...")
            print("⏳ Menahan data & meredam inersia mekanis (0.6 detik)...")
            target = vision.inspect_stationary_object(
                target_classes=target_classes,
                save_snapshot=True,
                settle_delay=0.6,
                verify_motion=True,
            )
            cetak_kotak_status(read_target_json(DEFAULT_TARGET_JSON))
            no += 1
    except KeyboardInterrupt:
        print("\n[Webcam] Berhenti.")
    finally:
        vision.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pengujian Pengiriman Koordinat Robotik")
    parser.add_argument(
        "--mode",
        choices=["simulasi", "monitor", "webcam"],
        default="simulasi",
        help="Pilihan mode: 'simulasi' (peragaan siklus otomatis), 'monitor' (pantau file JSON live), 'webcam' (uji kamera)",
    )
    parser.add_argument("--source", default="0", help="Indeks kamera (khusus mode webcam)")
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=["koin", "tutup_botol", "charger"],
        help="Daftar kelas target (default: koin tutup_botol charger)",
    )
    args = parser.parse_args()

    if args.mode == "simulasi":
        mode_simulasi()
    elif args.mode == "monitor":
        mode_monitor()
    elif args.mode == "webcam":
        mode_webcam(args.source, args.target_classes)


if __name__ == "__main__":
    main()
