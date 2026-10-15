"""Skrip utilitas untuk melakukan tes ping & monitor komunikasi MQTT dengan ESP32 / teman."""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Error: Library 'paho-mqtt' belum terpasang. Jalankan: pip install paho-mqtt")
    sys.exit(1)


BROKER = "p8f1fa01.ala.asia-southeast1.emqxsl.com"
PORT = 8883
USER = "admin_pabrik"
PASS = "rahasia123"
DEFAULT_TOPIC = "sortir/koin/agusrisky"


def main() -> int:
    parser = argparse.ArgumentParser(description="Tes Ping & Pantau Komunikasi MQTT")
    parser.add_argument("--broker", default=BROKER, help=f"Alamat broker MQTT (default: {BROKER})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port MQTT SSL/TLS (default: {PORT})")
    parser.add_argument("--user", default=USER, help=f"Username MQTT (default: {USER})")
    parser.add_argument("--password", default=PASS, help="Password MQTT")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help=f"Topik dasar (default: {DEFAULT_TOPIC})")
    parser.add_argument("--send", choices=["ping", "koin", "tutup_botol"], default=None,
                        help="Kirim pesan simulasi sekali jalan (pilihan: ping, koin, tutup_botol)")
    parser.add_argument("--listen", action="store_true",
                        help="Dengarkan seluruh lalu lintas pesan di topik ini secara real-time")
    args = parser.parse_args()

    print(f"==================================================")
    print(f" TES KONEKSI MQTT KE CLUSTER TEMAN")
    print(f"==================================================")
    print(f" Broker   : {args.broker}")
    print(f" Port     : {args.port} (SSL/TLS)")
    print(f" User     : {args.user}")
    print(f" Topik    : {args.topic}/#")
    print(f"==================================================\n")

    def on_connect(client, userdata, flags, rc, properties=None):
        rc_code = getattr(rc, "value", rc)
        if rc_code == 0 or str(rc) == "Success":
            print(f"[OK] Berhasil terhubung ke Broker EMQX Cloud!\n")
            client.subscribe(f"{args.topic}/#")
            print(f"[*] Berlangganan (Subscribe) ke topik: {args.topic}/#")

            if args.send:
                if args.send == "ping":
                    payload = {"event": "PING", "sender": "agusrisky", "pesan": "Tes Ping Sukses!"}
                    subtopic = f"{args.topic}/ping"
                elif args.send == "koin":
                    payload = {"event": "DETEKSI", "kelas": "koin", "track_id": 99, "confidence": 0.95}
                    subtopic = args.topic
                    client.publish(f"{args.topic}/raw", "KOIN")
                elif args.send == "tutup_botol":
                    payload = {"event": "DETEKSI", "kelas": "tutup_botol", "track_id": 99, "confidence": 0.95}
                    subtopic = args.topic
                    client.publish(f"{args.topic}/raw", "TUTUP_BOTOL")

                client.publish(subtopic, json.dumps(payload))
                print(f"[>] Pesan simulasi '{args.send}' terkirim ke {subtopic}")
        else:
            print(f"[X] Gagal terhubung ke broker! Kode: {rc}")

    def on_message(client, userdata, msg):
        try:
            isi = msg.payload.decode("utf-8")
        except UnicodeDecodeError:
            isi = str(msg.payload)
        waktu = time.strftime("%H:%M:%S")
        print(f"[{waktu}] [< DITERIMA] Topik: {msg.topic}\n    Isi: {isi}\n")

    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    else:
        client = mqtt.Client()

    if args.user and args.password:
        client.username_pw_set(args.user, args.password)

    client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.on_connect = on_connect
    client.on_message = on_message

    print("[*] Menghubungkan ke broker...")
    try:
        client.connect(args.broker, args.port, 60)
    except Exception as exc:
        print(f"[ERROR] Gagal konek ke {args.broker}:{args.port} -> {exc}")
        return 1

    client.loop_start()

    if args.listen:
        print("[*] Mode Pantau Aktif (Tekan Ctrl+C untuk berhenti)...")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nBerhenti memantau.")
    else:
        # Beri waktu 2 detik untuk memastikan pengiriman selesai
        time.sleep(2)

    client.loop_stop()
    client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
