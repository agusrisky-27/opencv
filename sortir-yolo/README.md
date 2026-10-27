# 🤖 SISTEM SORTIR OTOMATIS BERBASIS VISI KOMPUTER (YOLOv8 + OPENCV)
## Integrasi Lengan Robotik (RNV3) & Konveyor Industri 3 Kelas (Koin, Tutup Botol, Charger)

> **Catatan untuk Agen AI & Pengembang:**
> Dokumen ini adalah panduan teknis utama (*Agent Onboarding & System Specification*) yang merinci seluruh arsitektur, parameter inferensi, lapisan filter cerdas, transformasi koordinat robotik, serta tata kelola berkas pada repositori ini.
>
> 📁 *Dokumentasi mendalam lainnya tersedia di folder [DOKUM/](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/DOKUM):*
> * [DOKUM/DOKUMENTASI.md](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/DOKUM/DOKUMENTASI.md) — *Master Engineering Knowledge Base & Deep Technical Manual.*
> * [DOKUM/WORK_DOKUMENTATION.md](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/DOKUM/WORK_DOKUMENTATION.md) — *Development Work Log berkala per minggu (aktivitas, status, kendala, solusi).*
> * [DOKUM/PANDUAN_TIM_ROBOTIK.md](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/DOKUM/PANDUAN_TIM_ROBOTIK.md) — *Panduan integrasi cepat untuk tim robotik di komputer Windows.*

---

## 📑 DAFTAR ISI
1. [Ringkasan Eksekutif & Karakteristik Fisik Objek](#1-ringkasan-eksekutif--karakteristik-fisik-objek)
2. [Spesifikasi Perangkat Keras & Tata Letak Fisik](#2-spesifikasi-perangkat-keras--tata-letak-fisik)
3. [Arsitektur Perangkat Lunak & Peta Direktori](#3-arsitektur-perangkat-lunak--peta-direktori)
4. [Model AI YOLOv8 & Aturan Inferensi Live](#4-model-ai-yolov8--aturan-inferensi-live)
5. [Lapisan Filter Pintar & Koreksi Cerdas (Smart Filters)](#5-lapisan-filter-pintar--koreksi-cerdas-smart-filters)
6. [Sistem Transformasi Koordinat Hand-to-Eye & IPC Robotik](#6-sistem-transformasi-koordinat-hand-to-eye--ipc-robotik)
7. [Alur Operasi Sortir Otomatis Penuh (Stop-and-Go)](#7-alur-operasi-sortir-otomatis-penuh-stop-and-go)
8. [Panduan Siklus Data & Pelatihan Model (Google Colab)](#8-panduan-siklus-data--pelatihan-model-google-colab)
9. [Panduan Eksekusi Program & Cheat-Sheet Kontrol](#9-panduan-eksekusi-program--cheat-sheet-kontrol)
10. [Pengujian Otomatis (Unit Testing) & Standar Keandalan](#10-pengujian-otomatis-unit-testing--standar-keandalan)

---

## 1. RINGKASAN EKSEKUTIF & KARAKTERISTIK FISIK OBJEK

Sistem ini dirancang untuk mendeteksi, mengklasifikasi, dan menentukan koordinat titik hisap vakum presisi milimeter $(X, Y, Z\text{ mm})$ untuk **3 kelas objek** yang berjalan di atas sabuk konveyor industri:

```
┌─────────────────┬───────────────────┬───────────────────────────────────────┬───────────────────────┐
│ Nama Kelas AI   │ Label Tampilan    │ Karakteristik Fisik Nyata             │ Warna Kotak Visual    │
├─────────────────┼───────────────────┼───────────────────────────────────────┼───────────────────────┤
│ `koin` (0)      │ Koin              │ Logam bundar tipis, silau cermin      │ Emas (0, 215, 255)    │
│ `tutup_botol`(1)│ Tutup Botol       │ Plastik silinder warna pekat          │ Biru/Cyan (255, 140, 0│
│ `charger` (2)   │ Charger           │ Balok kubus asimetris + 2 pin colokan │ Hijau (0, 255, 0)     │
└─────────────────┴───────────────────┴───────────────────────────────────────┴───────────────────────┘
```

### Sasaran Wadah Sortir (*Drop Bins*):
* **Wadah Koin (Kiri):** $X = -150.0\text{ mm}, Y = 100.0\text{ mm}, Z = 45.0\text{ mm}$
* **Wadah Tutup Botol (Kanan):** $X = +150.0\text{ mm}, Y = 100.0\text{ mm}, Z = 55.0\text{ mm}$
* **Wadah Charger (Depan):** $X = 0.0\text{ mm}, Y = 180.0\text{ mm}, Z = 70.0\text{ mm}$

---

## 2. SPESIFIKASI PERANGKAT KERAS & TATA LETAK FISIK

```
┌────────────────────────────────────────────────────────────────────────┐
│                        BLUEPRINT TATA LETAK FISIK                      │
│                                                                        │
│                      [ DUDUKAN WEBCAM USB (720p) ]                     │
│                                                                        │
│   ══════════════════════════════════════════════════════════════════   │
│   ░░░░░░░░░░░░░░░░░░  SABUK KONVEYOR HIJAU  ░░░░░░░░░░░░░░░░░░░░░░░   │
│   │                                                                │   │
│   │        (+) Titik A                   (+) Titik B               │   │
│   │        X = -50 mm, Y = 160 mm        X = +50 mm, Y = 160 mm    │   │
│   │                                                                │   │
│   │                  ┌──────────────────────┐                      │   │
│   │                  │    WADAH CHARGER     │                      │   │
│   │                  │ X=0, Y=180, Z=70 mm  │                      │   │
│   │                  └──────────────────────┘                      │   │
│   │                                                                │   │
│   │        (+) Titik D                   (+) Titik C               │   │
│   │        X = -50 mm, Y = 100 mm        X = +50 mm, Y = 100 mm    │   │
│   │                                                                │   │
│   ══════════════════════════════════════════════════════════════════   │
│                                                                        │
│   ┌─────────────┐                                    ┌─────────────┐   │
│   │ WADAH KOIN  │                                    │ WADAH TUTUP │   │
│   │ X = -150 mm │                                    │ X = +150 mm │   │
│   │ Y = 100 mm  │                                    │ Y = 100 mm  │   │
│   │ Z = 45 mm   │                                    │ Z = 55 mm   │   │
│   └─────────────┘                                    └─────────────┘   │
│                                                                        │
│                               ▲ Sumbu Y+ (Maju ke konveyor)            │
│                               │                                        │
│                 ┌─────────────┴─────────────┐                          │
│                 │   LENGAN ROBOT RNV3 (0,0) │                          │
│                 │   (Pangkal Robot Arm)     │ ──► Sumbu X+ (Kanan)     │
│                 └───────────────────────────┘                          │
└────────────────────────────────────────────────────────────────────────┘
```

### Konfigurasi Fisik:
1. **Kamera (Rexus SW-10 USB HD 720p):** Dipasang di atas konveyor dengan sudut kemiringan statis mengarah ke zona kerja robot.
2. **Lengan Robot (RNV3 4-DOF / 6-DOF):** Dipasang **berseberangan (berhadapan)** dengan dudukan kamera.
   * *Konsekuensi Koordinat:* Ketika objek bergerak ke arah kanan pada layar kamera, posisi fisik robot harus bergerak ke arah kiri (`flip_x = True`, `berseberangan = True`).
3. **End-Effector Suction Cup:** Menggunakan pompa hisap vakum dengan karet silikon diameter 15 mm.
4. **Blueprint Kertas Manila:** Garis panduan fisik yang digambar di karton manila memastikan seluruh modul ditaruh pada posisi yang sama persis saat demo berpindah ruangan (*Zero-Setup Time*).

---

## 3. ARSITEKTUR PERANGKAT LUNAK & PETA DIREKTORI

```
sortir-yolo/
├── detect.py                    # Visualisasi deteksi live webcam (imgsz=1280, conf=0.25)
├── sortir_otomatis.py           # Controller otonom Stop-and-Go (100% Hands-Free)
├── sortir_vision_api.py         # Modul API visi mandiri (impor langsung oleh skrip robot)
├── contoh_baca_koordinat_robot.py # Script pengujian baca IPC JSON & demonstrasi robot
├── kalibrasi_robot.py           # GUI kalibrasi 4-titik piksel ke milimeter robot
│
├── capture.py                   # Perekam sampel gambar webcam standar
├── extra_capture_data.py        # Perekam data multi-kelas keyboard 1, 2, 3, b
├── extra_autolabel_own.py       # Engine autolabel adaptif sabuk hijau & alas gelap
├── bersihkan_dataset_blur.py    # Skrip filter otomatis foto blur & oklusi kulit
├── merge_datasets.py            # Penggabung dataset Roboflow + mandiri (70/20/10)
│
├── DOKUM/                       # ARSIP DOKUMENTASI LENGKAP PROYEK
│   ├── DOKUMENTASI.md           # Master Technical Knowledge Base & Guide
│   ├── WORK_DOKUMENTATION.md    # Log Laporan Mingguan & Harian Terstruktur
│   ├── PANDUAN_TIM_ROBOTIK.md   # Panduan integrasi tim robotik di Windows
│   └── README_ARSIP.md          # Arsip README versi sebelumnya
│
├── models/
│   └── best.pt                  # Bobot neural network YOLOv8 utama (6,2 MB)
│
├── outputs/
│   ├── target_robot.json        # File IPC koordinat real-time ke robot
│   └── koordinat.csv            # Riwayat ekspor koordinat deteksi
│
├── robot_calib.json             # Parameter matriks Homography & koordinat wadah drop
├── requirements.txt             # Dependensi pustaka Python
├── pytest.ini                   # Konfigurasi automated testing
│
├── src/                         # MODUL INTERNAL SISTEM
│   ├── autolabel.py             # Algoritma segmentasi konveyor hijau & HSV
│   ├── camera.py                # Wrapper kamera OpenCV + Zero-Latency Thread
│   ├── config.py                # Konfigurasi kelas, warna BGR, & path default
│   ├── coordinate_transform.py  # Logika matematika Homography piksel ke milimeter
│   ├── detector.py              # Engine YOLOv8 + Smart Filters + Refine Suction Center
│   └── utils.py                 # Anotasi visual HUD, Kalman filter, & Penstabil voting
│
└── tests/                       # 49 SKENARIO UNIT TESTING (100% PASSING)
    ├── test_autolabel.py
    ├── test_camera.py
    ├── test_coordinate_transform.py
    ├── test_dataset_tools.py
    ├── test_detector.py
    └── test_utils.py
```

---

## 4. MODEL AI YOLOV8 & ATURAN INFERENSI LIVE

### A. Bobot Model (`models/best.pt`)
* **Arsitektur:** Ultralytics YOLOv8 Nano (`yolov8n.pt`), ukuran file 6.2 MB.
* **Jumlah Kelas:** 3 Kelas (`0: koin`, `1: tutup_botol`, `2: charger`).
* **Kinerja Validasi:** $\text{mAP}_{50} \ge 98.7\%$, recall tinggi pada objek kecil.

### B. Tiga Parameter Kunci Inferensi Live
> [!IMPORTANT]
> Jangan mengubah ketiga parameter default ini tanpa pertimbangan matang:

1. **`--imgsz 1280` (Wajib untuk Webcam 720p):**
   - Kamera beroperasi pada resolusi native $1280 \times 720$. Koin fisik di konveyor selebar 50–70 px.
   - Jika `imgsz` diubah ke 640 saat inferensi live, koin akan menyusut menjadi 25 px dan fitur relief koin pada layer $P_3$ YOLOv8 akan hancur terhapus interpolasi.
   - Menggunakan `imgsz=1280` menjamin deteksi tajam skala 1:1 dengan confidence koin **0.91 – 0.98**.
2. **`--conf 0.25` (Ambang Batas Sensitivitas Optimal):**
   - Objek di konveyor fisik di bawah pencahayaan ruangan difus sering kali menghasilkan confidence antara `0.28` hingga `0.45`.
   - Threshold 0.25 menjamin tidak ada objek valid yang terlewat, sedangkan noise dicegah oleh lapisan filter geometri dan penstabil suara terbanyak.
3. **`--edge-margin 0` (Bebas Potong Tepi Konveyor):**
   - Menghindari pembuangan objek yang melintas dekat dinding atas atau bawah konveyor ($y_1 \le 15$). Pembatasan area kerja sepenuhnya ditangani oleh parameter `ROI`.

---

## 5. LAPISAN FILTER PINTAR & KOREKSI CERDAS (SMART FILTERS)

Modul [`src/detector.py`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py) bertindak sebagai lapisan pertahanan kedua setelah inferensi YOLOv8 untuk menjamin presisi industri:

```
Bounding Box YOLO (x1, y1, x2, y2, conf, cls)
                 │
                 ▼
  [ 1. Filter Geometri Aspek Rasio & Ukuran ] ──► Tolak jika benda lonjong / noise
                 │
                 ▼
  [ 2. Koreksi Koin Silau Cermin (Zero-Retrain) ] ──► Jika tutup_botol tapi perak silau desaturasi -> Ubah jadi Koin!
                 │
                 ▼
  [ 3. Refinement Titik Hisap Suction Charger ] ──► Geser pusat (cx, cy) ke bidang datar logo iT.
                 │
                 ▼
  [ 4. Filter Anti-Wajah & Kulit Manusia ] ──► Tolak jika wajah manusia melintas (skin_frac > 0.48)
                 │
                 ▼
  [ 5. Penstabil Voting Temporal Multi-Frame ] ──► Eliminasi flicker & smooth koordinat
                 │
                 ▼
Daftar Deteksi Final Siap Ekspor ke Robot
```

### 1. Validasi Geometri Fisik ([`validate_geometry`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py#L38-L71))
* **Aspek Rasio ($w / h$):** Koin dan tutup botol dibatasi pada $0.50 \le AR \le 1.85$. Khusus charger balok persegi panjang, rasio aspek diizinkan pada $0.25 \le AR \le 4.00$.
* **Batas Luas Area Layar:** Koin dan tutup botol maksimal $40\%$ layar. Charger maksimal $55\%$ layar.
* **Batas Dimensi Maksimal:** Dimensi charger tegak diizinkan hingga $85\%$ tinggi layar frame ($510\text{ px}$ pada 720p).

### 2. Koreksi Koin Silau Cermin ([`is_specular_metal_coin`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py#L172-L212))
* **Masalah:** Lampu konveyor memantulkan kilap cermin (*specular glare*, $V > 135$) pada koin perak, menghapus kontras cetakan timbul sehingga model AI menduga koin sebagai tutup botol putih.
* **Solusi Zero-Retrain:** Koin perak silau bersifat monokromatik desaturasi ($S < 0.20$ dan $frac\_pekat < 0.18$), sedangkan tutup botol plastik riil selalu memiliki saturasi tinggi ($S > 0.55$). Sistem secara cerdas **mengoreksi kembali kelasnya menjadi Koin** tanpa mengubah file `best.pt`.

### 3. Refinement Titik Hisap Vacuum Charger ([`refine_charger_vacuum_center`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py#L282-L322))
* **Masalah:** Titik tengah bounding box charger jatuh di leher pin colokan logam berundak, menyebabkan karet hisap vakum bocor udara.
* **Solusi:** Algoritma menganalisis rasio $w / h$ dan densitas massa biner Otsu, lalu menggeser koordinat hisap sejauh offset dinamis ke **pusat bidang datar plastik kepala charger (area logo `iT.`)** untuk hisapan kedap udara (*airtight seal*).

### 4. Filter Anti-Wajah & Kulit Manusia ([`is_human_skin_face`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py#L92-L108))
* Menggunakan kombinasi warna YCrCb ($Cr \in [133, 173], Cb \in [77, 127]$) dan HSV ($H \in [0, 25]$).
* Ambang batas diatur pada $\text{skin\_frac} > 0.48$ dan $\min(w, h) \ge 90\text{ px}$. Pengaturan ini memastikan charger di atas alas bersuhu hangat tidak terbuang, sementara wajah manusia asli yang melintas tetap ditolak.

### 5. Penstabil Multi-Frame ([`Penstabil`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/utils.py#L107-L230))
* Menerapkan voting suara terbanyak dari 5 frame terakhir untuk mengunci stabilitas label (*anti-flicker*).
* Menerapkan penghalusan koordinat kotak (*Exponential Moving Average*) untuk meredam getaran kamera.

---

## 6. SISTEM TRANSFORMASI KOORDINAT HAND-TO-EYE & IPC ROBOTIK

### A. Teori Homography 2D
Kamera terpasang miring mengawasi konveyor sehingga menghasilkan distorsi perspektif. Transformasi dihitung via matriks $\mathbf{H} \in \mathbb{R}^{3 \times 3}$:

$$\begin{bmatrix} X_{robot} \\ Y_{robot} \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} u_{kamera} \\ v_{kamera} \\ 1 \end{bmatrix}$$

Nilai matriks ini dihasilkan dari GUI `kalibrasi_robot.py` dan disimpan di `robot_calib.json`.

### B. Konfigurasi Ketinggian Sumbu Z & Inversi Sumbu
* **Orientasi Berseberangan:** `flip_x = True`, `berseberangan = True` (membalik arah sumbu X kamera ke fisik robot).
* **Ketinggian Pick Sumbu Z:**
  - Koin logam tipis: $Z = 2.0\text{ mm}$
  - Tutup botol silinder: $Z = 12.0\text{ mm}$
  - Kepala charger kubus: $Z = 28.0\text{ mm}$

### C. Protokol Pertukaran Data (Atomic JSON IPC)
Data ditulis secara atomik ke `outputs/target_robot.json` (menulis ke file temporer `.tmp` lalu seketika me-rename) untuk mencegah *permission error / file-locking* pada komputer Windows tim robotik:

```json
{
  "status": "TARGET_READY",
  "timestamp": 1793068150.56,
  "waktu_baca": "2026-10-27 09:29:10",
  "target": {
    "class_name": "charger",
    "label": "Charger",
    "confidence": 0.81,
    "pixel_center": [681, 569],
    "robot_mm": {
      "x": 12.5,
      "y": 145.0,
      "z": 28.0
    },
    "drop_mm": {
      "x": 0.0,
      "y": 180.0,
      "z": 70.0
    },
    "track_id": 3,
    "catatan": "titik hisap vacuum di badan datar logo iT."
  },
  "total_objek": 3,
  "daftar_objek": [...]
}
```

---

## 7. ALUR OPERASI SORTIR OTOMATIS PENUH (STOP-AND-GO)

Untuk implementasi demonstrasi atau pabrik tanpa operator, sistem dijalankan melalui skrip [`sortir_otomatis.py`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/sortir_otomatis.py):

```
       [ Konveyor Berjalan Membawa Objek ]
                        │
                        ▼
       [ Sinyal Kedatangan Objek ]
       ├─ Opsi A: Sensor Infrared Hardware (Serial Arduino USB)
       └─ Opsi B: Kamera Auto-Vision (Mendeteksi objek diam > 0.3 detik)
                        │
                        ▼
       [ Konveyor Berhenti Sempurna (Motor OFF) ]
                        │
                        ▼
       [ Kamera Mengambil 1 Snapshot Statis (Single-Shot) ]
       - Menghitung koordinat milimeter fisik bebas getaran (Anti-Jitter)
                        │
                        ▼
       [ Evaluasi Kelas Target Sortir ]
        /                             \
    [ Target Valid ]              [ Bukan Target ]
           │                              │
           ▼                              ▼
 [ Robot Ambil & Taruh ]       [ Lengan Robot Diam Saja ]
           │                              │
           └──────────────┬───────────────┘
                          │
                          ▼
       [ Konveyor Dinyalakan Kembali (Motor ON) ]
```

---

## 8. PANDUAN SIKLUS DATA & PELATIHAN MODEL (GOOGLE COLAB)

### A. Trik Fine-Tuning Cepat Tanpa Mengulang dari Epoch 0
Jika Anda menambahkan data objek baru (misal charger) atau ingin mengadaptasi warna konveyor baru pada model yang sudah terlatih 80 epoch:
```python
from ultralytics import YOLO

# 1. Muat bobot model best.pt yang sudah pintar koin & tutup botol
model = YOLO('/content/sortir-yolo/models/best.pt')

# 2. Latih HANYA 15 - 20 epoch tambahan! (~5 menit di GPU Tesla T4)
results = model.train(
    data='/content/sortir-yolo/dataset_gabungan/data.yaml',
    epochs=20,
    imgsz=640,          # Kompresi efisien saat backprop GPU
    batch=16,
    lr0=0.001,          # Learning rate kecil menjaga bobot lama tidak rusak
    device=0,
    name='sortir_finetune'
)
```

### B. Melanjutkan Training yang Terputus (*Resume*)
Jika koneksi Google Colab terputus di tengah jalan, jangan ulangi dari nol:
```python
from ultralytics import YOLO
model = YOLO('/content/sortir-yolo/runs/detect/sortir_scratch/weights/last.pt')
model.train(resume=True)
```

---

## 9. PANDUAN EKSEKUSI PROGRAM & CHEAT-SHEET KONTROL

### Aktivasi Virtual Environment:
```bash
cd ~/Documents/opencv_3class/sortir-yolo
source .venv/bin/activate
```

### Perintah Operasional Utama:
```bash
# 1. Deteksi Live Real-Time Standar (Webcam Rexus Index 0)
python detect.py --source 0 --warna-kelas

# 2. Deteksi pada Kamera Laptop FaceTime (Index 1)
python detect.py --source 1 --warna-kelas

# 3. Mode Sortir Otomatis Penuh (100% Hands-Free Auto-Vision Trigger)
python sortir_otomatis.py --mode auto-vision --source 0

# 4. Mode Sortir Hardware Sensor Infrared Serial Arduino
python sortir_otomatis.py --mode serial --port /dev/ttyUSB0 --baud 9600

# 5. Kalibrasi Homography 4-Titik Konveyor
python kalibrasi_robot.py --source 0

# 6. Menjalankan Seluruh Automated Unit Testing (49 Tests)
pytest -v
```

### Tombol Cepat Keyboard saat Jendela Video Terbuka:
* **`q` / `ESC`** : Keluar dari program.
* **`w`** : Mengaktifkan palet warna visual per kelas (Emas = Koin, Biru = Tutup Botol, Hijau = Charger).
* **`s`** : Menyimpan tangkapan layar (*screenshot*) ke `outputs/`.
* **`c`** : Menyimpan koordinat deteksi saat itu ke file CSV.
* **`t`** : Toggle pelacakan objek bergerak (*ByteTrack Object Tracking*).
* **`f`** : Toggle lapisan filter geometri cerdas (ON/OFF).
* **`r`** : Toggle area kerja aktif konveyor (*Region of Interest / ROI*).
* **`+` / `-`** : Menaikkan / menurunkan threshold confidence secara langsung di layar.

---

## 10. PENGUJIAN OTOMATIS (UNIT TESTING) & STANDAR KEANDALAN

Sistem ini memiliki **49 skenario pengujian unit otomatis** yang memverifikasi seluruh komponen tanpa membutuhkan kehadiran kamera fisik nyata:

```bash
pytest -v
```

### Cakupan Pengujian:
1. **`test_autolabel.py` (7 tests):** Memvalidasi segmentasi konveyor hijau, batas HSV, dan bounding box charger.
2. **`test_camera.py` (3 tests):** Memvalidasi threaded capture dan fallback kamera.
3. **`test_coordinate_transform.py` (5 tests):** Memvalidasi transformasi Homography, inversi sumbu X, dan ekspor JSON atomic.
4. **`test_dataset_tools.py` (9 tests):** Memvalidasi pembersih foto blur Laplacian dan integrasi dataset.
5. **`test_detector.py` (15 tests):** Memvalidasi geometri lingkaran/balok, filter anti-silau koin, refinement titik hisap charger, toleransi charger tegak, dan filter anti-wajah.
6. **`test_utils.py` (10 tests):** Memvalidasi voting suara terbanyak `Penstabil`, penghalusan EMA, dan anotasi HUD.

**Status Saat Ini:** Seluruh 49 unit tests **100% Lulus (Passing)**.
