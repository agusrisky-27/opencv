@echo off
title Sortir YOLO - Sistem Otomatis
echo ========================================================
echo   MENJALANKAN SISTEM VISION AI + ROBOTIK RNV3 (WINDOWS)
echo ========================================================
echo.

:: ==========================================================
:: PENGATURAN PORT & KAMERA (Bisa disesuaikan):
:: Ganti COM3 dengan port Arduino Anda (cek di Device Manager)
:: Ganti 0 dengan 1 jika webcam USB berada di index 1
:: ==========================================================
set PORT_ROBOT=COM3
set INDEX_KAMERA=0

:: 1. Aktifkan Virtual Environment jika ada
if exist ".venv\Scripts\activate.bat" (
    echo Mengaktifkan virtual environment...
    call .venv\Scripts\activate.bat
)

:: 2. Buka Middleware Server di jendela terminal terpisah
echo [1/3] Menjalankan Middleware Server (Port 8000)...
start "1. Middleware Server" cmd /k "if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && python middleware_server.py"

:: Tunggu 2 detik agar server FastAPI siap
timeout /t 2 /nobreak >nul

:: 3. Buka Klien Robot RNV3 di jendela terminal terpisah
echo [2/3] Menjalankan Pengendali Robot RNV3 (%PORT_ROBOT%)...
start "2. Robot RNV3 Controller" cmd /k "if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && python klien_robot_local.py --port %PORT_ROBOT%"

:: Tunggu 1 detik
timeout /t 1 /nobreak >nul

:: 4. Jalankan OpenCV Vision di jendela utama
echo [3/3] Menjalankan Vision AI (Webcam USB Index %INDEX_KAMERA%)...
echo --------------------------------------------------------
echo Kontrol Kamera: Tekan tombol 'q' di jendela video untuk keluar.
echo --------------------------------------------------------
python 05_detect.py --source %INDEX_KAMERA%

pause
