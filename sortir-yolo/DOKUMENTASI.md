# 📘 DOKUMENTASI LENGKAP & CATATAN PENTING PROYEK
## Sistem Sortir Otomatis Berbasis Visi Komputer (YOLOv8 + OpenCV) & Integrasi Robot Lengan (RNV3)

> Dokumen ini disusun sebagai panduan menyeluruh (*knowledge base*) yang merangkum seluruh perjalanan pengembangan sistem dari awal hingga saat ini. Dokumen ini dirancang agar mudah dipahami, sistematis, dan dapat dijadikan rujukan utama ketika menghadapi proyek atau kasus serupa di masa mendatang.

---

## 📑 DAFTAR ISI
1. [Latar Belakang & Arsitektur Sistem](#1-latar-belakang--arsitektur-sistem)
2. [Kronologi & Perjalanan Pengembangan (Tahap demi Tahap)](#2-kronologi--perjalanan-pengembangan)
3. [Tantangan Utama, Masalah (Bugs), & Solusi Teknis](#3-tantangan-utama-masalah-bugs--solusi-teknis)
4. [Logika & Filter Cerdas (Smart Filter)](#4-logika--filter-cerdas-smart-filter)
5. [Sistem Kalibrasi Hand-to-Eye & Integrasi Robotik](#5-sistem-kalibrasi-hand-to-eye--integrasi-robotik)
6. [Inovasi Blueprint / Alas Acuan di Kertas Manila](#6-inovasi-blueprint--alas-acuan-di-kertas-manila)
7. [Struktur File & Penjelasan Modul](#7-struktur-file--penjelasan-modul)
8. [Ringkasan Perintah Cepat (Cheat-Sheet)](#8-ringkasan-perintah-cepat-cheat-sheet)
9. [Best Practices & Tips Menghadapi Kasus Serupa](#9-best-practices--tips-menghadapi-kasus-serupa)

---

## 1. LATAR BELAKANG & ARSITEKTUR SISTEM

### A. Tujuan Proyek
Membangun sistem visi komputer cerdas yang mampu:
1. **Mendeteksi dan mengenali** dua jenis benda pada sabuk konveyor: **Koin** (logam tipis) dan **Tutup Botol** (plastik berwarna tebal).
2. **Menentukan titik pusat fisik** objek dalam satuan milimeter nyata ($X, Y, Z\text{ mm}$).
3. **Mengirimkan koordinat target** secara instan ke sistem kontroler Robot Lengan (RNV3) untuk dilakukan penyortiran (*pick and place*) ke wadah masing-masing.

### B. Diagram Alur Sistem (End-to-End Pipeline)

```
       [ Kamera Webcam USB ] (Rexus SW-10)
                 │
                 ▼
       [ Frame Video OpenCV ] (1920x1080 / 1280x720)
                 │
                 ▼
       [ Model AI YOLOv8 ] (models/best.pt @ imgsz=1280)
                 │  - Menghasilkan Bounding Box & Class (Koin / Tutup)
                 │  - Confidence Score (0.00 - 1.00)
                 ▼
       [ Smart Geometric & Color Filters ]
                 │  - Filter Rasio Aspek (0.50 <= w/h <= 1.85)
                 │  - Filter Luas Area & Batas Ukuran
                 │  - Bypass Koin Otomatis (Conf >= 0.65)
                 │  - Verifikasi Saturasi HSV Tutup Botol
                 ▼
       [ Hand-to-Eye Coordinate Transformation ]
                 │  - Mengubah titik piksel (u, v) ke Milimeter (X, Y mm)
                 │  - Inversi Sumbu X (Orientasi Berseberangan)
                 │  - Ketinggian Z Otomatis (Koin = 2 mm, Tutup = 12 mm)
                 ▼
       [ IPC Data Export ] ──► outputs/target_robot.json (Atomic Write)
                 │
                 ▼
       [ Pengontrol Lengan Robot (RNV3) ]
                 │  - Meluncur ke atas objek (Approach Z+25)
                 │  - Turun & Pompa Vakum ON (Suction)
                 │  - Angkat & Bawa ke Wadah Koin / Tutup Botol
                 │  - Pompa Vakum OFF & Kembali Standby
```

---

## 2. KRONOLOGI & PERJALANAN PENGEMBANGAN

### Tahap 1: Inisialisasi & Pengumpulan Dataset Awal
* **Dataset Publik (Roboflow):** Mengunduh dataset awal koin dan tutup botol via Roboflow API.
* **Auto-labeling:** Membuat script otomatisasi pelabelan untuk mempercepat persiapan data latih.
* **Eksperimen Kelas:** Sempat menguji penambahan kelas lain (seperti *charger*), namun akhirnya dievaluasi dan diputuskan untuk **fokus pada 2 kelas inti (Koin & Tutup Botol)** agar model memiliki spesialisasi tinggi dan tidak membingungkan robot.

### Tahap 2: Pengumpulan Dataset Mandiri & Background Negatif
* **Masalah:** Model yang dilatih hanya dengan dataset publik kerap mendeteksi bayangan, meja, atau tangan sebagai koin.
* **Solusi (`capture.py`):** Membuat modul pengambilan foto langsung dari webcam di meja konveyor asli. 
* **Data Negatif (Latar Kosong):** Mengambil foto meja kosong, tangan, dan ruangan tanpa objek sebagai contoh negatif (*background images*), sehingga AI belajar membedakan latar belakang dari objek asli.

### Tahap 3: Pelatihan Ulang (Retraining) di Google Colab
* **Ekspor Dataset Gabungan:** Menggabungkan dataset Roboflow + foto mandiri ke format YOLOv8 standar (`dataset_gabungan/`).
* **Notebook Colab:** Melatih model `yolov8n.pt` selama puluhan epoch dengan akselerasi GPU Tesla T4 di Google Colab.
* **Bobot Model Baru:** Menghasilkan model terbaik [`models/best.pt`](file:///Users/agusrisky/Documents/opencv/sortir-yolo/models/best.pt) serta mencadangkan model sebelumnya [`models/best-cadangan.pt`](file:///Users/agusrisky/Documents/opencv/sortir-yolo/models/best-cadangan.pt).

### Tahap 4: Investigasi & Penyempurnaan False Positive Koin
* **Masalah:** Saat diuji coba pada ratusan foto uji, koin sering tidak terdeteksi (hanya ~50% lolos).
* **Akar Masalah:** Filter wajah/kulit manusia (`is_human_skin_face`) mendeteksi meja berwarna krem atau pantulan jari di dekat koin sebagai "wajah", lalu membuang koin tersebut.
* **Solusi:** Menambahkan aturan prioritas: jika confidence YOLO $\ge 0.65$, koin langsung divalidasi sebagai koin asli tanpa terhalang warna meja. Rasio kelolosan uji koin melonjak ke **80.6% (238/295 gambar lolos presisi)**.

### Tahap 5: Perampingan Arsitektur (Menghapus Middleware)
* **Awalnya:** Sempat dibuat server perantara HTTP middleware (`middleware_server.py`) untuk mengirim data via REST API.
* **Evaluasi:** Menambah beban CPU, memperkenalkan latensi jaringan, rentan error port/koneksi, dan memperumit operasional tim robotik.
* **Keputusan:** Middleware dihapus secara menyeluruh. Sistem beralih ke arsitektur **OpenCV Standalone**. Komunikasi data ke robot dialihkan ke metode file JSON real-time (`target_robot.json`) yang jauh lebih ringan, instan, dan anti-lag.

### Tahap 6: Standarisasi Kode & Pembersihan Folder
* Mengubah nama skrip berawalan angka (`01_`, `02_`, `03_`, `05_`) menjadi nama baku (`detect.py`, `capture.py`, dll).
* Menghapus skrip `detect_dual.py` karena sistem terkonfirmasi hanya menggunakan 1 webcam USB.
* Memindahkan file arsip lama dan screenshot ke folder [arsip_luar/](file:///Users/agusrisky/Documents/opencv/arsip_luar/) agar root workspace bersih dan rapi.

### Tahap 7: Modul Kalibrasi Robot & Desain Blueprint
* Membangun modul transformasi affine / homography 4-titik (`src/coordinate_transform.py` dan `kalibrasi_robot.py`).
* Menyusun ide inovatif: **Blueprint alas kerja di kertas manila** untuk memastikan akurasi jarak cm fisik dan mempercepat setup saat demo berpindah ruangan.

---

## 3. TANTANGAN UTAMA, MASALAH (BUGS), & SOLUSI TEKNIS

Berikut daftar kendala nyata yang ditemukan selama pengerjaan beserta solusinya:

| No | Masalah yang Ditemukan | Penyebab Utama | Solusi Teknis yang Diterapkan |
| :--- | :--- | :--- | :--- |
| **1** | Koin kecil di konveyor tidak terdeteksi saat kamera agak tinggi | Resolusi input default YOLO (640x640) mengecilkan piksel koin hingga detailnya hilang | Mengubah default `imgsz=1280` pada detector sehingga objek koin kecil tetap tajam |
| **2** | Koin ditolak / hilang saat di atas meja kayu/krem | Filter deteksi kulit manusia salah mengenali warna meja sebagai wajah | Menambahkan bypass: `if confidence >= 0.65: return True` sebelum filter kulit dieksekusi |
| **3** | Baut dan ring bulat ikut terdeteksi saat confidence rendah (0.30 - 0.45) | Bentuk lingkaran baut mirip koin sehingga AI ragu-ragu di skor ~40% | Mengatur sweet spot confidence ke `0.55 - 0.60`. Baut otomatis gugur, koin asli tetap terdeteksi |
| **4** | Kamera macbook yang terbuka saat dicolokkan webcam USB | Di macOS, saat webcam USB dicolok, OpenCV menetapkan webcam USB sebagai Index 0 dan kamera laptop bergeser ke Index 1 | Menjalankan dengan `--source 0` (bukan 1) |
| **5** | Koordinat robot terbalik arah kiri dan kanan | Kamera dan robot dipasang berseberangan (saling berhadapan) | Menerapkan pembalikan sumbu X otomatis (`flip_x = True`) pada modul koordinat |
| **6** | Bentuk konveyor terlihat trapesium (sudut kamera miring) | Distorsi perspektif kamera miring | Menggunakan matriks **Homography 4-titik** (`cv2.getPerspectiveTransform`) |

---

## 4. LOGIKA & FILTER CERDAS (SMART FILTER)

AI Deep Learning (YOLO) sangat handal mengenali pola, namun terkadang bisa menghasilkan *false positive*. Oleh karena itu, modul [`src/detector.py`](file:///Users/agusrisky/Documents/opencv/sortir-yolo/src/detector.py) dilengkapi lapisan filter cerdas:

### A. Validasi Geometri (Bentuk & Ukuran)
1. **Aspect Ratio ($w / h$):** Koin dan tutup botol dari atas berbentuk lingkaran simetris. Benda yang terlalu lonjong (rasio $< 0.50$ atau $> 1.85$, seperti pulpen, obeng, atau jendela) otomatis dibuang.
2. **Batas Ukuran Maksimal (`max_area_frac = 0.40`):** Benda yang memenuhi lebih dari 40% layar (seperti wajah manusia mendekat atau kardus besar) otomatis diabaikan.
3. **Batas Ukuran Minimal (`min_size_px = 10`):** Menghilangkan bintik noise kamera.

### B. Verifikasi Warna HSV (Tutup Botol)
* Tutup botol umumnya terbuat dari plastik dengan warna pekat (biru, hijau, merah, kuning).
* Sistem mengonversi area bounding box ke ruang warna HSV dan memeriksa nilai saturasi ($S$). Logam koin memiliki saturasi sangat rendah (keabu-abuan/perak/emas kusam), sedangkan tutup botol memiliki saturasi tinggi.

### C. Jalur Bypass Koin Presisi Tinggi
* Koin asli memiliki relief angka, garuda, dan tekstur logam. Jika model menghasilkan skor keyakinan $\ge 0.65$, sistem mempercayai hasil AI dan meloloskannya langsung tanpa terhalang warna permukaan meja.

---

## 5. SISTEM KALIBRASI HAND-TO-EYE & INTEGRASI ROBOTIK

### A. Konsep Dasar
* **Hand-to-Eye (Eye-to-Hand):** Posisi kamera dipasang statis mengawasi meja kerja conveyor, sedangkan lengan robot bergerak di seberang kamera.
* Kalibrasi ini mencari matriks transformasi $\mathbf{H}$ sehingga:
  $$\begin{bmatrix} X_{robot} \\ Y_{robot} \\ 1 \end{bmatrix} = \mathbf{H} \times \begin{bmatrix} u_{kamera} \\ v_{kamera} \\ 1 \end{bmatrix}$$

### B. Langkah Kerja `kalibrasi_robot.py`
1. Tentukan 4 titik di atas meja conveyor yang dapat dijangkau oleh robot.
2. Catat koordinat milimeter $(X, Y)$ masing-masing titik dari kontroler robot.
3. Jalankan `python kalibrasi_robot.py --source 0`, lalu klik ke-4 titik tersebut di jendela video.
4. Masukkan koordinat milimeternya $\rightarrow$ Tekan **`s`** untuk menyimpan ke [`robot_calib.json`](file:///Users/agusrisky/Documents/opencv/sortir-yolo/robot_calib.json).

### C. Alur Pengiriman Target ke Robot (Atomic JSON IPC)
Data deteksi diekspor ke file `outputs/target_robot.json` menggunakan teknik *atomic rename* (menulis ke file temporer lalu me-rename seketika) agar skrip robot tidak membaca file yang sedang setengah tertulis.

**Contoh Payload JSON:**
```json
{
  "status": "TARGET_READY",
  "timestamp": 1729601234.56,
  "target": {
    "class_name": "koin",
    "label": "Koin",
    "confidence": 0.94,
    "pixel_center": { "u": 540, "v": 320 },
    "robot_mm": { "x": 45.2, "y": 135.0, "z": 2.0 },
    "drop_mm": { "x": -150.0, "y": 100.0, "z": 45.0 },
    "track_id": 3
  }
}
```

---

## 6. INOVASI BLUEPRINT / ALAS ACUAN DI KERTAS MANILA

### A. Konsep "Calibration Base Mat"
Untuk mengatasi masalah **tempat dan meja yang sering berpindah-pindah**, dibuat alas kerja presisi menggunakan kertas manila tebal yang digambari garis-garis acuan fisik (satuan cm/mm):

```
┌────────────────────────────────────────────────────────────────────────┐
│                        KERTAS MANILA / KARTON ALAS                     │
│                                                                        │
│                      [ TITIK KAKI DUDUKAN WEBCAM ]                     │
│                                                                        │
│   ══════════════════════════════════════════════════════════════════   │
│   ░░░░░░░░░░░░░░░░░░  AREA JALUR CONVEYOR  ░░░░░░░░░░░░░░░░░░░░░░░░░   │
│   │                                                                │   │
│   │        (+) Titik A                   (+) Titik B               │   │
│   │        X = -50 mm, Y = 160 mm        X = +50 mm, Y = 160 mm    │   │
│   │                                                                │   │
│   │        (+) Titik D                   (+) Titik C               │   │
│   │        X = -50 mm, Y = 100 mm        X = +50 mm, Y = 100 mm    │   │
│   │                                                                │   │
│   ══════════════════════════════════════════════════════════════════   │
│                                                                        │
│   ┌─────────────┐                    ┌─────────────┐                   │
│   │ WADAH KOIN  │                    │ WADAH TUTUP │                   │
│   │ X = -150 mm │                    │ X = +150 mm │                   │
│   └─────────────┘                    └─────────────┘                   │
│                                                                        │
│                               ▲ Sumbu Y+ (Maju ke conveyor)            │
│                               │                                        │
│                 ┌─────────────┴─────────────┐                          │
│                 │     KOTAK DUDUKAN ROBOT   │                          │
│                 │   (Titik Pusat Home = 0,0)│ ──► Sumbu X+ (Kanan)     │
│                 └───────────────────────────┘                          │
└────────────────────────────────────────────────────────────────────────┘
```

### B. Manfaat Utama Blueprint:
1. **Zero-Setup Time:** Saat demo di tempat baru, cukup gelar kertas manila di meja, posisikan alat tepat pada garis blueprint, dan sistem langsung siap digunakan.
2. **Tidak Perlu Kalibrasi Berulang:** Nilai `robot_calib.json` yang sudah tersimpan tetap valid karena geometri fisik terjaga 100% konsisten.
3. **Koordinat Nyata:** Titik kalibrasi $A, B, C, D$ terukur pasti dengan penggaris fisik nyata dari titik `(0, 0)` basis robot.

---

## 7. STRUKTUR FILE & PENJELASAN MODUL

```
opencv/
├── .gitignore
├── sortir-yolo.zip                  # Arsip zip untuk Google Colab
├── arsip_luar/                      # Backup dokumentasi & screenshot lama
└── sortir-yolo/                     # DIREKTORI UTAMA PROYEK
    ├── README.md                    # Dokumentasi operasional & cheat sheet
    ├── DOKUMENTASI.md               # Knowledge base & catatan teknis lengkap (file ini)
    ├── PANDUAN_TIM_ROBOTIK.md       # Panduan khusus untuk rekan tim robotik
    ├── robot_calib.json             # Konfigurasi matriks kalibrasi & wadah drop
    ├── requirements.txt             # Daftar library Python
    ├── pytest.ini                   # Konfigurasi pengujian otomatis
    │
    ├── detect.py                    # Program utama deteksi real-time webcam/gambar
    ├── capture.py                   # Perekam sampel gambar baru / dataset tambahan
    ├── kalibrasi_robot.py           # GUI interaktif kalibrasi kamera ke koordinat robot
    ├── sortir_vision_api.py         # Modul API visi mandiri siap panggil
    ├── contoh_baca_koordinat_robot.py # Script pengujian baca koordinat untuk tim robot
    ├── autolabel_tutup_botol.py     # Script auto-labeling dataset
    ├── download_datasets.py         # Script pengunduh dataset Roboflow
    ├── merge_datasets.py            # Script penggabung dataset latih/validasi
    │
    ├── models/
    │   ├── best.pt                  # Model terlatih utama (YOLOv8)
    │   └── best-cadangan.pt         # Cadangan model sebelumnya
    ├── src/
    │   ├── camera.py                # Wrapper kamera OpenCV + auto fallback index
    │   ├── config.py                # Parameter konfigurasi global
    │   ├── coordinate_transform.py  # Logika matematika konversi piksel ke milimeter
    │   ├── detector.py              # Engine deteksi YOLOv8 + Smart Geometric Filter
    │   └── utils.py                 # Utilitas anotasi visual & CSV logger
    └── tests/                       # 41 skenario pengujian unit otomatis
```

---

## 8. RINGKASAN PERINTAH CEPAT (CHEAT-SHEET)

### Persiapan Lingkungan:
```bash
cd ~/Documents/opencv/sortir-yolo
source .venv/bin/activate
```

### Menjalankan Program:
```bash
# 1. Deteksi Real-Time Standar (Webcam USB)
python detect.py --source 0

# 2. Deteksi dengan Warna Kotak Berbeda (Koin=Emas, Tutup=Biru)
python detect.py --source 0 --warna-kelas

# 3. Deteksi dengan Confidence Optimal (Sweet Spot Anti-Baut)
python detect.py --source 0 --conf 0.55 --warna-kelas

# 4. Ambil Foto Sampel Baru / Dataset Tambahan
python capture.py --source 0

# 5. Kalibrasi Koordinat Kamera ke Meja Robot
python kalibrasi_robot.py --source 0

# 6. Menjalankan Seluruh Unit Test (41/41 Tests)
pytest -q
```

### Tombol Cepat Keyboard saat Jendela Kamera Terbuka:
* **`q` / `ESC`** : Keluar dari program
* **`w`** : Toggle warna kotak tiap kelas (kuning koin vs biru tutup)
* **`s`** : Simpan screenshot ke folder `outputs/`
* **`c`** : Simpan data koordinat saat itu ke file CSV
* **`+` / `-`** : Naikkan / turunkan confidence threshold secara langsung
* **`t`** : Toggle object tracking (ByteTrack)
* **`f`** : Toggle filter geometri cerdas (ON/OFF)

---

## 9. BEST PRACTICES & TIPS MENGHADAPI KASUS SERUPA

Jika Anda membuat proyek *Computer Vision + Robotika* serupa di masa depan, terapkan prinsip-prinsip ini:

1. **Gunakan `imgsz` Sesuai Ukuran Objek:**
   Objek kecil (seperti koin, mur, baut) membutuhkan resolusi inferensi lebih tinggi (misal `imgsz=1280`). Jangan paksakan 640x640 jika objek fisik di kamera berukuran kecil.
2. **Kombinasikan AI Deep Learning dengan Rule-Based Filter:**
   Model AI hebat dalam mengenali objek, tetapi filter matematika sederhana (seperti batasan rasio aspek, rentang luas, atau nilai warna HSV) sangat ampuh menjadi lapisan pertahanan terakhir dari *false positive*.
3. **Cari "Sweet Spot" Nilai Confidence:**
   Jangan menyetel confidence terlalu rendah ($< 0.40$) karena benda bulat asing akan mudah ikut terdeteksi. Setel di rentang $0.50 - 0.60$ agar benda asing otomatis tertolak sementara objek asli tetap terdeteksi stabil.
4. **Hindari Cahaya Sorot Tegak Lurus pada Logam:**
   Benda logam memantulkan cahaya lampu seperti cermin (*specular glare*), yang bisa menghapus detail permukaan koin. Gunakan pencahayaan menyebar (*diffused lighting*) atau arahkan lampu dari sudut samping.
5. **Gunakan Alas Blueprint (Base Mat) untuk Demo:**
   Dalam robotika fisik, konsistensi posisi adalah kunci. Membuat blueprint di karton manila menghemat waktu setup hingga 90% saat harus melakukan demonstrasi di lokasi yang berpindah-pindah.
6. **Desain Komunikasi yang Sederhana (Decoupled IPC):**
   Hindari arsitektur jaringan yang terlalu rumit jika kamera dan pengendali robot berada di lingkungan lokal. Pertukaran data lewat file JSON atomik atau modul API lokal jauh lebih tangguh, minim bug, dan tanpa latensi jaringan.
