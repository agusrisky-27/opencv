"""Pembungkus kamera OpenCV dengan pesan error yang jelas."""
from __future__ import annotations

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Kamera tidak bisa dibuka."""


def parse_source(value: str | int) -> int | str:
    """'0' -> 0 (nomor kamera); selain itu dianggap path file/URL."""
    if isinstance(value, int):
        return value
    return int(value) if value.strip().isdigit() else value


class Camera:
    def __init__(self, source: int | str = 0, width: int | None = None, height: int | None = None) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None

    @staticmethod
    def find_available_cameras(max_index: int = 4) -> list[int]:
        """Cari nomor kamera yang aktif / dapat dibuka."""
        available = []
        for idx in range(max_index):
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                available.append(idx)
                test_cap.release()
        return available

    def open(self, auto_fallback: bool = False) -> "Camera":
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened() and auto_fallback and isinstance(self.source, int):
            # Coba kamera lain yang tersedia
            candidates = self.find_available_cameras()
            for cand in candidates:
                if cand != self.source:
                    self.cap = cv2.VideoCapture(cand)
                    if self.cap.isOpened():
                        self.source = cand
                        break
        if not self.cap or not self.cap.isOpened():
            self.release()
            raise CameraError(
                f"Tidak bisa membuka sumber video: {self.source!r}\n"
                "- Coba nomor kamera lain: --source 1 atau --source 2\n"
                "- Tutup aplikasi lain yang sedang memakai kamera (Zoom, Meet, dll)\n"
                "- macOS: beri izin kamera untuk Terminal/VS Code di "
                "System Settings > Privacy & Security > Camera"
            )
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Warmup kamera (sangat berguna untuk iPhone Continuity Camera agar sensor aktif)
        if isinstance(self.source, int):
            import time
            for _ in range(15):
                ok, _ = self.cap.read()
                if ok:
                    break
                time.sleep(0.15)

        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            raise CameraError("Kamera belum dibuka. Panggil open() dulu.")
        ok, frame = self.cap.read()
        return bool(ok), frame if ok else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()
