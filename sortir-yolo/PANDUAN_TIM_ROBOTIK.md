# 🤖 Panduan Integrasi Koordinat untuk Tim Robotik (RNV3)
**Sistem Sortir 3 Kelas: Koin, Tutup Botol, & Charger (YOLOv8 + OpenCV Vision)**

Dokumentasi ini dibuat khusus untuk **Rekan Tim Robotik** agar dapat dengan mudah dan cepat membaca data koordinat fisik dari modul Vision AI (OpenCV) dan menggerakkan Robot Lengan Linier RNV3 di laptop Windows.

---

## 📌 Ringkasan Sistem (Architecture Overview)

Sistem penglihatan kamera dan lengan robot dipasang dalam posisi **BERSEBERANGAN (Facing Each Other)** dengan sabuk conveyor berada di tengah-tengahnya:

```
                  ┌───────────────────────────────┐
                  │    KAMERA (Webcam Rexus SW-10) │
                  └───────────────┬───────────────┘
                                  │
    ===============================================================> (Conveyor)
                                  ▲
                                  │
                  ┌───────────────┴───────────────┐
                  │       ROBOT LENGAN RNV3       │
                  └───────────────────────────────┘
```

### ✅ Apa Saja yang Sudah Ditangani Otomatis oleh Modul Vision?
Tim robotik **TIDAK PERLU** lagi menghitung rumus konversi piksel ke milimeter yang rumit. Modul Vision AI sudah otomatis menangani:
1. **Inversi Sumbu X (Flip X):** Sisi kanan pada kamera otomatis dibalik menjadi sisi kiri bagi robot ($X_{robot}$).
2. **Penyesuaian Sumbu Y (Jangkauan Lengan):** Jarak jauh-dekat kamera otomatis diselaraskan dengan jangkauan rel robot ($Y_{robot}$).
3. **Ketinggian Sumbu Z Otomatis per Jenis Benda:**
   - 🪙 **Koin (Benda Tipis):** $Z = 2.0\text{ mm}$
   - 🔵 **Tutup Botol (Benda Sedang):** $Z = 12.0\text{ mm}$
   - 🔌 **Charger (Benda Tebal):** $Z = 28.0\text{ mm}$
4. **Refinement Titik Hisap Charger (Airtight Vacuum Cup Alignment):**
   - Khusus kepala charger, Vision AI secara dinamis menggeser titik hisap vacuum cup dari leher colokan ke **titik tengah permukaan badan charger yang datar** (area logo `iT.`). Hal ini menjamin hisapan karet vacuum cup kedap udara (*airtight*) dan tidak meleset pada dua colokan logam!
5. **Koreksi Sudut Kamera (Homography 4-Titik):** Efek distorsi trapesium akibat kemiringan kamera sudah otomatis dikoreksi.

Semua data yang dikeluarkan oleh Vision AI **sudah dalam satuan Milimeter Fisik Nyata ($X, Y, Z\text{ mm}$)** yang siap langsung dikirim ke motor robot!

---

## 🚀 2 Cara Mudah Tim Robotik Membaca Koordinat

Rekan tim robotik bebas memilih salah satu dari **2 metode integrasi** berikut:

---

### PILIHAN 1 (Paling Direkomendasikan): Membaca File JSON Real-Time
Metode ini adalah yang paling stabil dan terpisah (*Decoupled*).
- **Terminal 1:** Jalankan OpenCV Vision (`python 05_detect.py`) untuk memantau video kamera.
- **Terminal 2:** Jalankan script robot Anda. Program robot membaca target terbaru dari file `outputs/target_robot.json`.

#### 📄 Struktur Isi File `outputs/target_robot.json`:
```json
{
  "status": "TARGET_READY",
  "timestamp": 1729601234.56,
  "target": {
    "class_name": "koin",
    "label": "Koin",
    "confidence": 0.94,
    "pixel_center": { "u": 540, "v": 320 },
    "robot_mm": {
      "x": 45.2,
      "y": 135.0,
      "z": 2.0
    },
    "drop_mm": {
      "x": -150.0,
      "y": 100.0,
      "z": 45.0
    },
    "track_id": 3
  }
}
```

#### 🐍 Contoh Kode Python Pembaca di Skrip Robot Anda:
```python
import json
import time
from pathlib import Path

TARGET_FILE = Path("outputs/target_robot.json")

def ambil_target_terbaru():
    if not TARGET_FILE.exists():
        return None
    try:
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("status") == "TARGET_READY":
                return data["target"]
    except Exception:
        return None
    return None

# Loop pengawasan robot
while True:
    target = ambil_target_terbaru()
    if target:
        kelas = target["label"]               # "Koin" atau "Tutup Botol"
        pos = target["robot_mm"]              # Koordinat titik ambil di conveyor
        wadah = target["drop_mm"]             # Koordinat wadah sortir tujuan
        
        print(f"🎯 Target Siap: {kelas}")
        print(f"   Ambil di: X={pos['x']}, Y={pos['y']}, Z={pos['z']}")
        print(f"   Drop  ke: X={wadah['x']}, Y={wadah['y']}, Z={wadah['z']}")
        
        # Kirim G-Code ke robot RNV3 Anda di sini...
        time.sleep(1.0)
    time.sleep(0.1)
```

---

### PILIHAN 2: Import Modul `SortirVisionAPI` Langsung
Jika Anda ingin menggabungkan seluruh logika dalam 1 script Python tanpa membuka terminal terpisah:

```python
from sortir_vision_api import SortirVisionAPI

# Inisialisasi kamera dan AI (hanya 1 kali di awal)
vision = SortirVisionAPI(source=0)

while True:
    # Ambil objek terbaik yang siap diambil di conveyor
    target = vision.get_target()
    
    if target:
        print(f"Objek: {target.label} (Conf: {target.confidence * 100:.1f}%)")
        print(f"Posisi Pick (mm): X={target.x_mm}, Y={target.y_mm}, Z={target.z_mm}")
        print(f"Posisi Drop (mm): X={target.drop_x_mm}, Y={target.drop_y_mm}, Z={target.drop_z_mm}")
        
        # Eksekusi G-Code...
```

---

## 🕹️ Urutan Gerak Robot yang Direkomendasikan (G-Code Sequence)

Untuk menghindari lengan robot menabrak benda lain di conveyor, gunakan urutan gerak aman berikut:

| Langkah Gerak | Perintah G-Code | Penjelasan |
|---|---|---|
| **1. Meluncur ke Atas Objek** | `G0 X{pos.x} Y{pos.y} Z{pos.z + 25} F100` | Lengan bergerak cepat di posisi melayang aman ($Z+25\text{ mm}$). |
| **2. Turun Menyentuh Objek** | `G0 Z{pos.z} F40` | Ujung karet vakum turun menyentuh koin/tutup botol. |
| **3. Pompa Vakum ON** | `M1` | Nyalakan hisapan pompa vakum (tunggu delay $\approx 0.3$ detik). |
| **4. Angkat Objek** | `G0 Z{pos.z + 30} F80` | Lengan mengangkat objek menjauh dari sabuk conveyor. |
| **5. Geser ke Wadah Drop** | `G0 X{wadah.x} Y{wadah.y} F100` | Bergerak ke koordinat kotak sortir sesuai kelasnya. |
| **6. Turun ke Wadah** | `G0 Z{wadah.z} F50` | Turun ke ketinggian aman pelepasan. |
| **7. Pompa Vakum OFF** | `M2` lalu `M6` | Matikan pompa & buka katup pelepas hisapan. |
| **8. Kembali ke Standby** | `G0 X0 Y150 Z30 F100` | Lengan kembali ke posisi aman menunggu objek berikutnya. |

---

## 🛠️ Alat Kalibrasi Meja / Conveyor (`kalibrasi_robot.py`)

Jika posisi meja, kamera, atau robot dipindahkan ke ruangan baru, jalankan alat kalibrasi interaktif ini:

```bash
python kalibrasi_robot.py --source 0
```

### Cara Pakai:
1. Jendela kamera akan terbuka.
2. Klik **4 titik acuan** di atas conveyor yang bisa dijangkau oleh robot (misal: 4 sudut bidang kerja).
3. Di terminal, ketikkan koordinat fisik robot ($X, Y\text{ mm}$) saat ujung karet vakum menyentuh titik tersebut.
4. Program otomatis menghitung **Matriks Homography** dan langsung menyimpannya ke file `robot_calib.json`.
5. Anda bisa menggerakkan kursor mouse di video untuk melihat koordinat fisik robot secara *live*!

### Tombol Cepat Keyboard:
- **`b`** : Toggle Mode Berseberangan (ON/OFF).
- **`x`** : Toggle Balik Sumbu X (kiri/kanan).
- **`y`** : Toggle Balik Sumbu Y (maju/mundur).
- **`r`** : Reset titik untuk kalibrasi ulang.
- **`s`** : Simpan konfigurasi ke file.
- **`q`** : Keluar dari alat kalibrasi.

---

## ⚙️ Konfigurasi Wadah Sortir (`robot_calib.json`)

Posisi fisik kotak/wadah tempat menaruh koin, tutup botol, dan charger dapat Anda ubah dengan mudah pada file [`robot_calib.json`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/robot_calib.json):

```json
"drop_coordinates": {
  "koin": {
    "x": -150.0,
    "y": 100.0,
    "z": 45.0
  },
  "tutup_botol": {
    "x": 150.0,
    "y": 100.0,
    "z": 55.0
  },
  "charger": {
    "x": 0.0,
    "y": 180.0,
    "z": 70.0
  }
}
```
*Sesuaikan nilai X, Y, Z di atas dengan posisi fisik wadah koin (kiri), tutup botol (kanan), dan charger (depan) di samping robot Anda (dalam satuan mm).*

---

## 📁 File Skrip Contoh Siap Pakai
Rekan tim robotik dapat langsung menguji coba integrasi ini dengan menjalankan:
```bash
python contoh_baca_koordinat_robot.py --mode json
```
atau
```bash
python contoh_baca_koordinat_robot.py --mode api --source 0
```

Jika ada pertanyaan atau butuh penyesuaian sumbu motor stepper, silakan hubungi tim Vision AI! 🚀
