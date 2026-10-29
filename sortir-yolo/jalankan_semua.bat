@echo off
title Sortir YOLO - OpenCV Vision AI
echo ========================================================
echo   SISTEM VISION AI SORTIR 3 KELAS (KOIN, TUTUP BOTOL, ^& CHARGER)
echo ========================================================
echo.

:: ==========================================================
:: PENGATURAN KAMERA (Bisa disesuaikan):
:: Ganti 0 dengan 1 jika webcam USB eksternal di index 1
:: ==========================================================
set INDEX_KAMERA=0

:: 1. Aktifkan Virtual Environment jika ada
if exist ".venv\Scripts\activate.bat" (
    echo Mengaktifkan virtual environment...
    call .venv\Scripts\activate.bat
)

:: 2. Jalankan OpenCV Vision AI langsung
echo Menjalankan OpenCV Vision AI (Webcam USB Index %INDEX_KAMERA%)...
echo --------------------------------------------------------
echo Kontrol Keyboard:
echo   [q] / [ESC] : Keluar dari program
echo   [s]         : Simpan tangkapan layar (screenshot)
echo   [c]         : Simpan koordinat objek ke CSV
echo   [SPACE]     : Manual Trigger IR (Kunci koordinat ke robot)
echo   [t]         : Toggle Object Tracking (ON/OFF)
echo   [f]         : Toggle Smart Filter (ON/OFF)
echo   [r]         : Toggle ROI Conveyor (ON/OFF)
echo   [w]         : Toggle Warna per Kelas
echo --------------------------------------------------------
python detect.py --source %INDEX_KAMERA% --conf 0.25 --imgsz 1280 --edge-margin 0 --warna-kelas

pause
