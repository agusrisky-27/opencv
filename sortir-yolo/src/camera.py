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

        # Buffer size 1 untuk stream jaringan (HTTP/RTSP ESP32-CAM) agar tidak lag
        if isinstance(self.source, str) and (
            self.source.startswith("http://")
            or self.source.startswith("https://")
            or self.source.startswith("rtsp://")
        ):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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


class ThreadedCamera:
    """Kamera dengan background thread agar zero-latency.

    Sangat direkomendasikan untuk IP Camera / ESP32-CAM (stream HTTP/RTSP)
    dan arsitektur Dual Camera agar pembacaan frame tidak saling memblokir.
    """

    def __init__(self, source: int | str = 0, width: int | None = None, height: int | None = None) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.camera = Camera(source, width, height)
        self.running = False
        self._thread = None
        self.latest_frame: np.ndarray | None = None
        self.latest_ok: bool = False
        import threading
        self.lock = threading.Lock()

    def open(self, auto_fallback: bool = False) -> "ThreadedCamera":
        import threading
        import time

        self.camera.open(auto_fallback=auto_fallback)
        self.running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        # Tunggu frame pertama (maksimal 4 detik)
        t0 = time.time()
        while time.time() - t0 < 4.0:
            with self.lock:
                if self.latest_ok and self.latest_frame is not None:
                    break
            time.sleep(0.05)
        return self

    def _worker(self) -> None:
        import time

        while self.running:
            try:
                ok, frame = self.camera.read()
            except Exception:
                ok, frame = False, None

            with self.lock:
                self.latest_ok = ok
                if ok and frame is not None:
                    self.latest_frame = frame

            if not ok:
                time.sleep(0.01)

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            if not self.running:
                return False, None
            if self.latest_frame is None:
                return self.latest_ok, None
            return self.latest_ok, self.latest_frame.copy()

    def release(self) -> None:
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.camera.release()

    def __enter__(self) -> "ThreadedCamera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()

