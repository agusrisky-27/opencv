@echo off
title Sortir YOLO - OpenCV Vision AI
echo ========================================================
echo   SISTEM VISION AI SORTIR KOIN ^& TUTUP BOTOL (OPENCV)
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
echo   [t]         : Toggle Object Tracking (ON/OFF)
echo   [f]         : Toggle Smart Filter (ON/OFF)
echo   [r]         : Toggle ROI Conveyor (ON/OFF)
echo   [w]         : Toggle Warna per Kelas
echo --------------------------------------------------------
python 05_detect.py --source %INDEX_KAMERA%

pause
