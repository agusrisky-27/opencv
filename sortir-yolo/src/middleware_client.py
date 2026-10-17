"""Klien pengirim data dari Vision YOLO ke Local Middleware Backend."""
from __future__ import annotations

import threading
import time
from typing import Any

import requests


class MiddlewareSender:
    """Mengirim event deteksi (koin, tutup_botol, x, y, z) ke Middleware FastAPI secara asinkron."""

    def __init__(self, base_url: str | None = None, enabled: bool = True, cooldown: float = 1.5) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.enabled = bool(base_url) and enabled
        self.cooldown = cooldown
        self._last_sent: dict[str, float] = {}
        self._sent_tracks: set[int] = set()

    def send_detection(
        self,
        class_name: str,
        confidence: float,
        x_mm: float = 0.0,
        y_mm: float = 0.0,
        z_mm: float = 2.0,
        track_id: int | None = None,
        aksi: str = "AMBIL",
    ) -> bool:
        if not self.enabled or not self.base_url:
            return False

        now = time.time()
        # Cegah spam pengiriman berkali-kali untuk objek yang sama
        if track_id is not None:
            if track_id in self._sent_tracks:
                if now - self._last_sent.get(class_name, 0.0) < self.cooldown:
                    return False
            self._sent_tracks.add(track_id)
        else:
            if now - self._last_sent.get(class_name, 0.0) < self.cooldown:
                return False

        self._last_sent[class_name] = now

        payload: dict[str, Any] = {
            "kelas": class_name,
            "confidence": round(confidence, 2),
            "x_mm": round(x_mm, 2),
            "y_mm": round(y_mm, 2),
            "z_mm": round(z_mm, 2),
            "track_id": track_id,
            "aksi": aksi,
        }

        # Kirim di background thread agar tidak menghambat FPS OpenCV
        def _post() -> None:
            try:
                requests.post(f"{self.base_url}/api/vision/detection", json=payload, timeout=0.6)
            except Exception:
                pass

        threading.Thread(target=_post, daemon=True).start()
        return True
