#!/usr/bin/env python3
"""Local Middleware Server (FastAPI).

Bertindak sebagai HUB / JEMBATAN sentral komunikasi 100% LOKAL (offline) antara:
1. Vision AI (Python YOLOv8 / OpenCV) -> Mengirim hasil deteksi & koordinat (x, y, z)
2. ESP32 (Hardware Conveyor & Robot)  -> Mengambil target koordinat & lapor status
3. Web / Mobile App (Aplikasi Tim)    -> Memantau status, statistik, & kontrol manual

Jalankan server:
    python middleware_server.py
    # Akses dokumentasi interaktif Swagger di browser: http://localhost:8000/docs
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==============================================================================
# // [ISI KOORDINAT WADAH MENARUH BARANG DI SINI] (dalam satuan milimeter - mm)
# ==============================================================================
# Silakan ganti angka X, Y, Z di bawah ini sesuai posisi fisik wadah sortir Anda:
DROP_COORDINATES: dict[str, dict[str, float]] = {
    "koin": {
        "x": -150.0,  # // ISI KOORDINAT X WADAH KOIN DI SINI (misal: -150.0 mm)
        "y": 100.0,   # // ISI KOORDINAT Y WADAH KOIN DI SINI (misal: 100.0 mm)
        "z": 45.0,    # // ISI KOORDINAT Z WADAH KOIN DI SINI (ketinggian aman drop)
    },
    "tutup_botol": {
        "x": 150.0,   # // ISI KOORDINAT X WADAH TUTUP BOTOL DI SINI (misal: 150.0 mm)
        "y": 100.0,   # // ISI KOORDINAT Y WADAH TUTUP BOTOL DI SINI (misal: 100.0 mm)
        "z": 55.0,    # // ISI KOORDINAT Z WADAH TUTUP BOTOL DI SINI (ketinggian aman drop)
    },
}

try:
    from src.config import DROP_COORDINATES as _CONFIG_DROP
    DROP_COORDINATES.update(_CONFIG_DROP)
except ImportError:
    pass

# Inisialisasi FastAPI
app = FastAPI(
    title="Sortir YOLO - Local Middleware Backend",
    description="Jembatan Komunikasi Lokal antara Vision AI, ESP32 Robot, dan Web/Mobile App",
    version="1.0.0",
)

# Aktifkan CORS agar Web/App di port/domain berbeda bisa akses tanpa hambatan
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------- Model Data (Pydantic)
class DetectionPayload(BaseModel):
    kelas: str = Field(..., description="Nama kelas: 'koin', 'tutup_botol', atau 'unknown'")
    confidence: float = Field(0.0, description="Tingkat kepercayaan deteksi (0.0 - 1.0)")
    x_mm: float = Field(..., description="Koordinat X fisik dalam mm")
    y_mm: float = Field(..., description="Koordinat Y fisik dalam mm")
    z_mm: float = Field(..., description="Ketinggian Z fisik dalam mm (koin=2mm, tutup=12mm)")
    track_id: int | None = Field(None, description="ID unik pelacak objek")
    aksi: Literal["AMBIL", "ABAIKAN"] = Field("AMBIL", description="Aksi yang harus dilakukan robot")


class RobotStatusPayload(BaseModel):
    state: Literal["IDLE", "BUSY", "ERROR"] = Field("IDLE", description="Status robot")
    current_action: str = Field("", description="Aktivitas saat ini, misal: 'Mengambil koin'")
    last_picked_class: str | None = None


class ManualControlPayload(BaseModel):
    command: Literal["MOVE", "SUCTION_ON", "SUCTION_OFF", "HOME", "CONVEYOR_START", "CONVEYOR_STOP"]
    x: float | None = None
    y: float | None = None
    z: float | None = None


# ------------------------------------------------------------- State Management Lokal
class SystemState:
    def __init__(self) -> None:
        self.latest_detection: dict[str, Any] | None = None
        self.target_queue: deque[dict[str, Any]] = deque(maxlen=50)
        self.robot_state: str = "IDLE"
        self.robot_action: str = "Menunggu target"
        self.pending_manual_command: dict[str, Any] | None = None
        self.counts = {"koin": 0, "tutup_botol": 0, "diabaikan": 0}
        self.logs: deque[dict[str, Any]] = deque(maxlen=100)

    def log(self, event: str, detail: str) -> None:
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "event": event,
            "detail": detail,
        }
        self.logs.appendleft(entry)


state = SystemState()


# ============================================================= ENDPOINT UNTUK VISION (YOLO)
@app.post("/api/vision/detection", tags=["Vision YOLO"])
def receive_detection(payload: DetectionPayload):
    """Dipanggil oleh Python YOLO (05_detect.py) saat objek terdeteksi."""
    data = payload.model_dump()
    data["timestamp"] = time.time()
    state.latest_detection = data

    if payload.aksi == "AMBIL":
        state.target_queue.append(data)
        state.counts[payload.kelas] = state.counts.get(payload.kelas, 0) + 1
        state.log("DETEKSI", f"Target masuk antrean: {payload.kelas} di ({payload.x_mm}, {payload.y_mm}, {payload.z_mm})")
    else:
        state.counts["diabaikan"] += 1
        state.log("ABAIKAN", f"Benda asing/diluar dataset disaring: {payload.kelas}")

    return {"status": "SUCCESS", "queue_length": len(state.target_queue)}


# ============================================================= ENDPOINT UNTUK ESP32 (ROBOT & CONVEYOR)
@app.get("/api/robot/target", tags=["ESP32 Robot"])
def get_target_for_robot():
    """Dipanggil oleh ESP32 (saat Sensor IR terpotong atau saat robot siap mengambil).

    Mengembalikan target koordinat berikutnya. Jika antrean kosong atau benda asing,
    mengembalikan status 'ABAIKAN' agar conveyor tetap melaju.
    """
    # 1. Cek apakah ada perintah manual dari Web App
    if state.pending_manual_command is not None:
        cmd = state.pending_manual_command
        state.pending_manual_command = None
        return {"type": "MANUAL", "command": cmd}

    # 2. Cek apakah ada target otomatis dari Vision
    if state.target_queue:
        target = state.target_queue.popleft()
        kelas = target["kelas"]
        drop_pos = DROP_COORDINATES.get(kelas, {"x": 0.0, "y": 0.0, "z": 0.0})
        state.log("ROBOT", f"Target diserahkan ke ESP32: {kelas} -> Drop di ({drop_pos['x']}, {drop_pos['y']}, {drop_pos['z']})")
        return {
            "type": "AUTO",
            "available": True,
            "aksi": "AMBIL",
            "kelas": kelas,
            "x": target["x_mm"],
            "y": target["y_mm"],
            "z": target["z_mm"],
            "drop_x": drop_pos["x"],
            "drop_y": drop_pos["y"],
            "drop_z": drop_pos["z"],
            "track_id": target["track_id"],
        }

    # 3. Jika tidak ada target (benda asing / kosong)
    return {
        "type": "AUTO",
        "available": False,
        "aksi": "ABAIKAN",
        "kelas": "unknown",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "drop_x": 0.0,
        "drop_y": 0.0,
        "drop_z": 0.0,
    }


@app.get("/api/robot/drop-zones", tags=["ESP32 Robot"])
def get_drop_zones():
    """Mengembalikan daftar titik koordinat wadah pembuangan (Drop Zones) per kelas."""
    return DROP_COORDINATES


@app.post("/api/robot/status", tags=["ESP32 Robot"])
def update_robot_status(payload: RobotStatusPayload):
    """Dipanggil oleh ESP32 untuk melaporkan kondisinya (IDLE, BUSY, atau SELESAI)."""
    state.robot_state = payload.state
    state.robot_action = payload.current_action
    state.log("STATUS_ROBOT", f"State: {payload.state} | {payload.current_action}")
    return {"status": "OK"}


# ============================================================= ENDPOINT UNTUK WEB / MOBILE APP
@app.get("/api/app/status", tags=["Web & Mobile App"])
def get_app_dashboard():
    """Dipanggil oleh Web/Mobile App untuk menampilkan status live, statistik, dan log."""
    return {
        "system": "ONLINE",
        "robot_state": state.robot_state,
        "robot_action": state.robot_action,
        "counts": state.counts,
        "queue_length": len(state.target_queue),
        "latest_detection": state.latest_detection,
        "recent_logs": list(state.logs)[:15],
    }


@app.post("/api/app/control", tags=["Web & Mobile App"])
def send_manual_control(payload: ManualControlPayload):
    """Dipanggil oleh Web/Mobile App untuk menggerakkan robot secara manual."""
    state.pending_manual_command = payload.model_dump()
    state.log("APP_KONTROL", f"Perintah manual dikirim: {payload.command}")
    return {"status": "COMMAND_QUEUED", "command": payload.command}


@app.post("/api/app/reset-stats", tags=["Web & Mobile App"])
def reset_statistics():
    """Reset hitungan statistik koin dan tutup botol."""
    state.counts = {"koin": 0, "tutup_botol": 0, "diabaikan": 0}
    state.target_queue.clear()
    state.log("RESET", "Statistik dan antrean direset")
    return {"status": "RESET_SUCCESS"}


# ============================================================= ROOT ROUTE
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Sortir YOLO - Local Middleware Backend",
        "status": "RUNNING",
        "docs_url": "/docs",
        "endpoints": {
            "vision_post": "/api/vision/detection",
            "esp32_get_target": "/api/robot/target",
            "esp32_post_status": "/api/robot/status",
            "app_get_status": "/api/app/status",
            "app_post_control": "/api/app/control",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Jalankan Local Middleware Server")
    parser.add_argument("--host", default="0.0.0.0", help="IP host binding (0.0.0.0 agar dapat diakses semua perangkat WiFi lokal)")
    parser.add_argument("--port", type=int, default=8000, help="Port server (default: 8000)")
    args = parser.parse_args()

    print("==================================================")
    print(" 🚀 LOCAL MIDDLEWARE SERVER AKTIF (OFFLINE)")
    print("==================================================")
    print(f" Akses dari Laptop Sendiri : http://localhost:{args.port}")
    print(f" Akses dari ESP32 / Web App: http://<IP_LAPTOP>:{args.port}")
    print(f" Dokumentasi API Interaktif: http://localhost:{args.port}/docs")
    print("==================================================\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
