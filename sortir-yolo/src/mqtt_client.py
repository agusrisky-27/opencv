"""Pengirim notifikasi deteksi via MQTT untuk ESP32 / IoT."""
from __future__ import annotations

import json
import ssl
import time
from typing import Any

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


class MqttSender:
    """Mengirim sinyal deteksi ke Broker MQTT secara real-time dengan proteksi cooldown."""

    def __init__(
        self,
        broker: str = "p8f1fa01.ala.asia-southeast1.emqxsl.com",
        port: int = 8883,
        username: str | None = "admin_pabrik",
        password: str | None = "rahasia123",
        topic: str = "sortir/koin/agusrisky",
        enabled: bool = True,
        cooldown: float = 2.0,
        use_tls: bool = True,
    ) -> None:
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.topic = topic
        self.enabled = enabled
        self.cooldown = cooldown
        self.use_tls = use_tls
        self.client = None
        self.connected = False
        self._last_sent: dict[str, float] = {}
        self._sent_tracks: set[int] = set()

        if self.enabled:
            if mqtt is None:
                print("[MQTT] Library 'paho-mqtt' belum terpasang. Jalankan: pip install paho-mqtt")
                self.enabled = False
            else:
                self._connect()

    def _connect(self) -> None:
        try:
            if hasattr(mqtt, "CallbackAPIVersion"):
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            else:
                self.client = mqtt.Client()

            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)

            if self.use_tls or self.port == 8883 or "emqxsl.com" in self.broker:
                self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)

            def on_connect(client: Any, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
                rc_code = getattr(rc, "value", rc) if hasattr(rc, "value") else rc
                if str(rc).lower() == "success" or rc_code == 0:
                    self.connected = True
                    print(f"[MQTT] ✅ Terhubung ke broker {self.broker}:{self.port} | Topik: {self.topic}")
                else:
                    self.connected = False
                    print(f"[MQTT] ⚠️ Gagal terhubung (rc={rc})")

            self.client.on_connect = on_connect
            self.client.connect_async(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"[MQTT] Error inisialisasi koneksi: {e}")
            self.connected = False

    def send_detection(self, class_name: str, confidence: float, track_id: int | None = None) -> bool:
        """Kirim notifikasi ke MQTT jika objek baru atau cooldown terpenuhi."""
        if not self.enabled or self.client is None:
            return False

        now = time.time()
        # Jika objek memiliki track_id baru yang belum pernah dikirim:
        if track_id is not None:
            if track_id in self._sent_tracks:
                if now - self._last_sent.get(class_name, 0.0) < self.cooldown:
                    return False
            self._sent_tracks.add(track_id)
        else:
            if now - self._last_sent.get(class_name, 0.0) < self.cooldown:
                return False

        self._last_sent[class_name] = now

        # 1. Payload JSON lengkap (untuk dashboard / PC)
        payload = json.dumps({
            "objek": class_name,
            "id": track_id,
            "confidence": round(confidence, 2),
            "waktu": time.strftime("%H:%M:%S"),
        })

        try:
            self.client.publish(self.topic, payload)
            # 2. String ringkas sederhana untuk kemudahan ESP32 / DFPlayer ("koin" atau "tutup_botol")
            self.client.publish(f"{self.topic}/raw", class_name)
            print(f"[MQTT 📡] Mengirim: {class_name.upper()} (ID: #{track_id if track_id else '-'}, conf: {confidence:.2f}) -> {self.topic}")
            return True
        except Exception as e:
            print(f"[MQTT] Gagal publish: {e}")
            return False

    def close(self) -> None:
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
