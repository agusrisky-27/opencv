# 🎯 Sistem Vision Sortir 3 Kelas: Koin, Tutup Botol, & Charger (YOLOv8 + OpenCV)

Program Computer Vision berbasis **YOLOv8 & OpenCV** yang dirancang untuk mendeteksi benda (**Koin**, **Tutup Botol**, dan **Charger**) secara real-time pada sabuk conveyor industri:
- Memberikan kotak pembatas (*bounding box*) dan label nama kelas yang presisi untuk 3 kelas.
- Menghitung **titik koordinat hisap vakum $(cx, cy)$** secara otomatis (khusus charger: menggeser titik hisap ke bidang datar kedap udara di area logo `iT.`).
- Dilengkapi sistem pelacakan objek (*ByteTrack*) dan penstabil Kalman agar ID objek tidak berkedip (*flicker*) saat bergerak di atas conveyor.
- Menulis koordinat robot fisik ($X, Y, Z\text{ mm}$) secara atomik ke `outputs/target_robot.json`.

---

## ⚡ PANDUAN CEPAT: Perintah yang Sering Dijalankan (Cheat Sheet)

Simpan dan gunakan kumpulan perintah praktis ini untuk operasional sehari-hari:

### 1. Masuk ke Folder Proyek
Buka aplikasi **Terminal** (atau Terminal di VS Code / Cursor), lalu arahkan ke direktori proyek:
```bash
cd sortir-yolo
```

### 2. Aktifkan Virtual Environment (.venv)
Wajib dijalankan setiap kali baru membuka jendela terminal baru:
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (Command Prompt / PowerShell):**
  ```cmd
  .venv\Scripts\activate
  ```
*(Tanda aktif: muncul tulisan `(.venv)` di sebelah kiri baris perintah terminal).*

---

### 3. Menjalankan Deteksi Kamera (`detect.py`)

- **Webcam Eksternal USB Konveyor (Rexus SW-10, Index 0):** 👈 *(Paling Sering Dipakai)*
  ```bash
  python detect.py --source 0 --warna-kelas
  ```
- **Kamera Bawaan Laptop / Mac (Index 1):**
  ```bash
  python detect.py --source 1
  ```
- **Mode Warna Kotak Berbeda Tiap Kelas (Koin = Emas, Tutup Botol = Cyan, Charger = Merah):**
  ```bash
  python detect.py --source 0 --warna-kelas
  ```
- **Uji Coba Deteksi pada Satu File Gambar:**
  ```bash
  python detect.py --source outputs/preview_charger_on_conveyor.jpg --warna-kelas
  ```

#### 🎮 Tombol Pintas Saat Jendela Kamera Terbuka:
| Tombol | Fungsi |
|---|---|
| **`q`** atau **`ESC`** | Keluar dari program kamera |
| **`w`** | Mengubah warna kotak (warna per-kelas: emas, cyan, merah) |
| **`s`** | Menyimpan tangkapan layar (*screenshot*) ke folder `outputs/` |
| **`c`** | Menyimpan titik koordinat objek ke file `outputs/koordinat.csv` |
| **`t`** | Mengaktifkan / mematikan pelacak objek (*Object Tracking*) |
| **`r`** | Mengaktifkan zona fokus conveyor (*Region of Interest / ROI*) |
| **`f`** | Menyalakan / mematikan Filter Cerdas Geometri (*Smart Filter*) |
| **`+` / `-`** | Menaikkan / menurunkan batas sensitivitas deteksi (*Confidence*) |

---

### 4. Mengambil Foto Dataset Sendiri (`capture.py`)

Gunakan fitur ini untuk mengambil foto objek secara mandiri lewat kamera:
```bash
python capture.py --source 0   # Webcam MacBook
python capture.py --source 1   # Webcam USB eksternal / iPhone
```

#### 🎮 Tombol Pintas & Struktur Folder Otomatis:
1. Tekan **`1`** $\to$ `Target: Koin` (tersimpan ke `datasets_raw/koin_capture_manual/`)
2. Tekan **`2`** $\to$ `Target: Tutup Botol` (tersimpan ke `datasets_raw/tutup_botol_capture_manual/`)
3. Tekan **`3`** $\to$ `Target: Charger` (tersimpan ke `datasets_raw/charger_capture_manual/`)
4. Tekan **`0`** $\to$ `Target: Latar Negatif` (tersimpan ke `datasets_raw/latar_negatif_manual/`)
5. Tekan **`SPASI`** $\to$ Ambil 1 foto manual
6. Tekan **`a`** $\to$ **Auto-Capture ON/OFF** (memotret tiap 0.4 detik otomatis)
7. Tekan **`q`** atau **`ESC`** $\to$ Selesai dan keluar (menampilkan rekap total foto)

---

### 5. Pembersih Otomatis Foto Blur / Buram (`bersihkan_dataset_blur.py`)

Setelah selesai mengambil foto, jalankan filter kualitas ini untuk menyingkirkan foto yang buram akibat gerakan tangan cepat (*motion blur*):

```bash
# 1. Cek dulu berapa foto blur tanpa memindahkan file (Simulasi):
python bersihkan_dataset_blur.py --dry-run

# 2. Pindahkan foto blur ke folder karantina 'datasets_raw/sampah_blur/' (Aman):
python bersihkan_dataset_blur.py

# 3. Atau langsung hapus permanen foto blur:
python bersihkan_dataset_blur.py --delete
```

---

## 🔄 Alur Kerja Sistem (Pipeline Vision)

Sistem pengolahan visi ini bekerja dalam 5 tahapan terpadu:

```
[Koleksi Data]           [Pelabelan Otomatis]         [Penggabungan Dataset]       [Training Model]          [Deteksi Real-Time]
download_datasets.py ──► autolabel_tutup_botol.py ──► merge_datasets.py        ──► Google Colab (GPU)    ──► detect.py
        ▲                         ▲                           ▲
        └────── capture.py ───────┴─ extra_autolabel_own.py ──┘
```

1. **Pengumpulan Dataset:**
   - Dataset koin rupiah dan tutup botol publik diunduh otomatis via Roboflow API (`download_datasets.py`).
   - Foto kondisi nyata meja/conveyor diambil langsung menggunakan kamera (`capture.py`).
2. **Pelabelan Otomatis (Auto-labeling):**
   - Foto tutup botol dan foto kamera sendiri otomatis diberi kotak koordinat format YOLO tanpa perlu klik manual satu per satu (`autolabel_tutup_botol.py` & `extra_autolabel_own.py`).
3. **Penyatuan Dataset & Pencegahan Kebocoran Data:**
   - Semua sumber data digabungkan ke subset `train`, `valid`, dan `test` (`merge_datasets.py`).
   - Sistem secara cerdas memfilter foto duplikat agar tidak terjadi *data leakage* (kebocoran data validasi).
4. **Pelatihan Model AI (Training):**
   - Dilakukan di Google Colab menggunakan GPU T4 gratis selama 80 Epochs (`colab/Training_YOLO_Colab.ipynb`).
   - Menghasilkan bobot model terbaik yang disimpan di `models/best.pt`.
5. **Inferensi & Ekstraksi Titik Target:**
   - Program `detect.py` memuat model `best.pt`, membuka webcam, mendeteksi objek, dan mengekstrak koordinat tengah $(cx, cy)$ untuk sistem sortir robot.

---

## 📂 Struktur Direktori Proyek

```text
sortir-yolo/
├── detect.py                    # Program utama deteksi webcam real-time
├── capture.py                   # Program ambil foto objek kamera (Auto-capture)
├── download_datasets.py         # Skrip unduh dataset Roboflow
├── autolabel_tutup_botol.py     # Skrip label kotak otomatis tutup botol
├── extra_autolabel_own.py       # Skrip label kotak otomatis foto kamera sendiri
├── merge_datasets.py            # Skrip penyatuan seluruh dataset ke dataset final
│
├── colab/
│   └── Training_YOLO_Colab.ipynb # Notebook pelatihan model YOLOv8 di Google Colab
│
├── models/
│   ├── best.pt                  # Model AI aktif hasil training (siap pakai)
│   └── best-cadangan.pt         # Cadangan model stabil sebelumnya
│
├── datasets_raw/                # Arsip foto mentah (Roboflow & foto kamera sendiri)
├── dataset_gabungan/            # Dataset final berformat standar YOLOv8 (train/valid/test)
├── outputs/                     # Tempat penyimpanan screenshot, preview, & CSV
│
├── src/                         # Modul inti pustaka (Core Library)
│   ├── camera.py                # Wrapper kamera OpenCV (USB, Mac, & Threaded stream)
│   ├── config.py                # Konfigurasi pusat (nama kelas, warna, & path)
│   ├── detector.py              # Engine deteksi YOLOv8, pelacakan ByteTrack, & Smart Filter
│   ├── utils.py                 # Perhitungan titik tengah, visual overlay, & gambar kotak
│   ├── autolabel.py             # Algoritma penentu bounding box otomatis
│   └── dataset_tools.py         # Utilitas merging, konversi label, & split data
│
├── requirements.txt             # Daftar dependensi library Python
└── tests/                       # Kumpulan pengujian unit otomatis (pytest)
```

---

## 🛡️ Logika Filter Cerdas (Smart Filter)

Untuk mencegah AI salah mendeteksi benda asing (misal tangan, wajah, atau bayangan meja), modul `src/detector.py` menerapkan filter berlapis:
1. **Validasi Geometri (Aspect Ratio & Ukuran):**  
   Benda yang terlalu lonjong (aspek rasio $> 1.85$) atau terlalu raksasa memenuhi layar ($> 40\%$ layar) otomatis disaring.
2. **Koreksi Warna Tutup Botol:**  
   Jika ada benda plastik berwarna pekat (biru, hijau, merah, oranye), sistem memvalidasi bahwa itu adalah tutup botol.
3. **Verifikasi Koin Asli:**  
   Mendeteksi fitur relief koin logam dan mengabaikan benda berlubang besar tunggal (seperti kunci rumah). Jika deteksi YOLO memiliki confidence tinggi ($\ge 0.65$), koin langsung diloloskan tanpa terhalang warna meja atau jari tangan.

---

## 🧪 Pengujian Sistem (Automated Unit Tests)

Seluruh fungsi matematika, kamera, dan modul deteksi dapat diuji secara otomatis dengan menjalankan:

```bash
pytest -q
```
*(Seluruh 41 unit test dirancang untuk lulus 100% sebelum sistem dioperasikan).*

---

## ❓ Masalah Umum & Solusi Cepat (Troubleshooting)

- **Kamera tidak mau terbuka (di macOS):**  
  Buka **System Settings** > **Privacy & Security** > **Camera**, pastikan izin untuk *Terminal* atau *VS Code* dalam posisi **ON**.
- **Kamera tiba-tiba nge-zoom sendiri (di macOS):**  
  Lihat pojok kanan atas layar Mac (ikon kamera hijau 🎥). Klik **Video Effects** > Matikan fitur **Center Stage (OFF)** dan pastikan slider zoom di posisi **1x**.
- **Koin sulit terdeteksi karena silau:**  
  Koin logam memantulkan cahaya seperti cermin. Hindari lampu sorot yang mengarah tepat tegak lurus di atas koin. Gunakan pencahayaan menyebar (*diffused light*).

============================================================
# 🚀 PROGRES MINGGU INI (Jumat Lalu – Sekarang)
============================================================

Berikut adalah rangkuman pencapaian dan progres pengembangan proyek sistem sortir visi komputer (YOLOv8 + OpenCV) dari hari **Jumat pekan lalu hingga sekarang**:

### 1. Pelatihan Model Baru & Peningkatan Dataset
- **Retraining Model YOLOv8 di Colab:** Melatih ulang model dengan dataset gabungan yang diperkaya (termasuk variasi tutup botol baru dari Roboflow).
- **Hasil Model Terbaru (`models/best.pt`):** Bobot model baru memiliki akurasi deteksi yang lebih stabil untuk koin dan tutup botol di berbagai kondisi pencahayaan.
- **Resolusi Inferensi Optimal (`imgsz=1280`):** Mengatur resolusi bawaan deteksi ke 1280 piksel sehingga objek kecil (seperti koin receh) tetap terdeteksi dengan tajam meskipun kamera dipasang cukup tinggi di atas meja kerja.

### 2. Investigasi & Penyempurnaan Deteksi Koin
- **Solusi Masalah Koin Sulit Terdeteksi:** Dari evaluasi pengujian sampel gambar, ditemukan bahwa filter warna kulit (`is_human_skin_face`) sebelumnya kerap salah mengira meja berwarna krem atau jari tangan di dekat koin sebagai wajah manusia, sehingga koin ikut terbuang.
- **Optimasi Smart Filter:** Menambahkan jalur *bypass* otomatis: jika confidence model $\ge 0.65$, objek koin langsung divalidasi sebagai koin asli tanpa terhalang warna meja/tangan.
- **Hasil Pengujian:** Rasio keberhasilan deteksi pada kumpulan gambar uji koin melonjak signifikan dari ~50% menjadi **80.6% (238 dari 295 gambar uji lolos deteksi presisi)**.

### 3. Penyederhanaan Arsitektur (Bebas Lag & Ringan)
- **Penghapusan Server Middleware HTTP:** Menghilangkan perantara server HTTP/klien web yang rumit dan memakan resource. Sistem visi sekarang berjalan langsung (*standalone*) menggunakan OpenCV lokal, sehingga respons deteksi menjadi instan tanpa jeda waktu jaringan (*zero network latency*).
- **Penghapusan Skrip Dual Kamera:** Menghapus skrip `detect_dual.py` karena sistem sudah dipastikan hanya menggunakan 1 kamera webcam tunggal yang dipasang di atas meja sortir.

### 4. Standarisasi Nama File & Pembersihan Workspace
- **Penamaan Skrip Bersih (Clean Code):** Mengubah penamaan file skrip lama yang bernomor kaku (`01_`, `02_`, `03_`, `05_`) menjadi nama skrip standar yang jelas dan intuitif:
  - `detect.py`: Program deteksi real-time utama.
  - `capture.py`: Perekam sampel gambar baru / dataset tambahan.
  - `download_datasets.py`, `autolabel_tutup_botol.py`, `merge_datasets.py`: Pipeline dataset.
- **Pembersihan Root Workspace:** Membuat folder baru `arsip_luar/` dan memindahkan file dokumentasi lama, tangkapan layar lama, dan runner duplikat ke dalamnya agar folder utama tetap bersih. File arsip Colab `sortir-yolo.zip` tetap terjaga di root.

### 5. Persiapan Integrasi Koordinat untuk Tim Robotik
- **Modul Transformasi Koordinat Meja (Affine Transform):** Membangun modul perhitungan matriks kalibrasi (`src/coordinate_transform.py` dan `kalibrasi_robot.py`) untuk mengubah titik piksel kamera $(X_{cam}, Y_{cam})$ langsung menjadi koordinat fisik milimeter robot $(X_{robot}, Y_{robot}, Z_{robot})$.
- **API Visi Siap Pakai:** Menyediakan modul `sortir_vision_api.py` dan skrip contoh `contoh_baca_koordinat_robot.py` sehingga tim robotik dapat langsung mengimpor fungsi `dapatkan_target_sortir()` tanpa perlu memikirkan detail OpenCV di dalamnya.
- **Dokumentasi Khusus Robotik:** Menyusun buku panduan lengkap di `PANDUAN_TIM_ROBOTIK.md` yang menjelaskan alur kerja, rumus kalibrasi 3-titik, dan cara membaca data koordinat.

### 6. Pengujian Mutu Otomatis (Unit Tests)
- **Konfigurasi Mandiri:** Menambahkan `pytest.ini` di dalam direktori `sortir-yolo` agar pengujian dapat dijalankan secara independen.
- **100% Lulus Uji:** Seluruh **41 skenario pengujian otomatis** (`tests/`) terverifikasi berhasil (*PASSED*) tanpa kegagalan (mencakup modul deteksi, filter geometri, kalkulasi koordinat, autolabeling, dan kamera).

---

### 📊 Tabel Rangkuman Status Progres

| Komponen / Fitur | Status Sebelumnya | Status Saat Ini | Keterangan |
| :--- | :--- | :--- | :--- |
| **Model YOLOv8** | Model lama (terlatih dasar) | **Model baru (`best.pt`)** | Deteksi koin & tutup botol lebih stabil |
| **Filter Koin** | Sering tertolak warna meja | **Smart Bypass ($\ge 0.65$)** | Lolos 238/295 gambar uji |
| **Arsitektur Sistem**| Bergantung Middleware HTTP | **Standalone OpenCV** | Cepat, ringan, bebas kendala port/jaringan |
| **Struktur File** | Awalan `01_` s/d `05_` | **Nama bersih & modular** | `detect.py`, `capture.py`, dll |
| **Koordinat Robot** | Belum ada | **Siap & terdokumentasi** | Affine transform piksel ke milimeter |
| **Kerapian Root** | Banyak file loose campur aduk | **Rapi & Terarsip** | Dipindahkan ke `arsip_luar/` |
| **Unit Testing** | Sebagian modul | **41/41 Passed (100%)** | Seluruh test suite hijau |

