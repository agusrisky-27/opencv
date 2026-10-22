# 🎯 Sistem Vision Sortir Koin & Tutup Botol (YOLOv8 + OpenCV)

Program Computer Vision berbasis **YOLOv8 & OpenCV** yang dirancang untuk mendeteksi benda (**Koin Rupiah** dan **Tutup Botol**) secara real-time pada sabuk conveyor:
- Memberikan kotak pembatas (*bounding box*) dan label nama kelas yang presisi.
- Menghitung **titik koordinat tengah $(cx, cy)$** secara otomatis sebagai target hisap bagi robot lengan vakum.
- Dilengkapi sistem pelacakan objek (*ByteTrack*) agar ID objek tidak berkedip (*flicker*) saat bergerak di atas conveyor.

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

- **Kamera Bawaan Laptop / Mac (Index 0):**
  ```bash
  python detect.py
  ```
- **Webcam Eksternal USB (Rexus SW-10) atau Kamera iPhone (Index 1):** 👈 *(Paling Sering Dipakai)*
  ```bash
  python detect.py --source 1
  ```
- **Mode Warna Kotak Berbeda Tiap Kelas (Koin = Emas, Tutup Botol = Cyan):**
  ```bash
  python detect.py --source 1 --warna-kelas
  ```
- **Uji Coba Deteksi pada Satu File Gambar:**
  ```bash
  python detect.py --source foto.jpg
  ```

#### 🎮 Tombol Pintas Saat Jendela Kamera Terbuka:
| Tombol | Fungsi |
|---|---|
| **`q`** atau **`ESC`** | Keluar dari program kamera |
| **`w`** | Mengubah warna kotak (warna per-kelas: emas & cyan) |
| **`s`** | Menyimpan tangkapan layar (*screenshot*) ke folder `outputs/` |
| **`c`** | Menyimpan titik koordinat objek ke file `outputs/koordinat.csv` |
| **`t`** | Mengaktifkan / mematikan pelacak objek (*Object Tracking*) |
| **`r`** | Mengaktifkan zona fokus conveyor (*Region of Interest / ROI*) |
| **`f`** | Menyalakan / mematikan Filter Cerdas Geometri (*Smart Filter*) |
| **`+` / `-`** | Menaikkan / menurunkan batas sensitivitas deteksi (*Confidence*) |

---

### 4. Mengambil Foto Dataset Sendiri (`capture.py`)

Gunakan fitur ini jika Anda ingin menambah foto koin atau tutup botol baru dari kamera Anda untuk meningkatkan akurasi:
```bash
python capture.py
```
*(Otomatis menggunakan webcam index 1 secara default)*.

#### 🎮 Tombol Pintas Auto-Capture:
1. Tekan **`1`** pada keyboard $\to$ Layar menampilkan `Target: Koin`.
2. Tekan **`2`** pada keyboard $\to$ Layar menampilkan `Target: Tutup Botol`.
3. Tekan **`a`** pada keyboard $\to$ **Mulai Auto-Capture** (kamera akan otomatis memotret tiap 0.4 detik). Gerakkan objek/kamera perlahan untuk memvariasikan sudut dan jarak.
4. Tekan **`a`** lagi $\to$ Menghentikan pemotretan otomatis.
5. Tekan **`SPASI`** $\to$ Ambil 1 foto manual.
6. Tekan **`q`** $\to$ Selesai dan keluar. Foto otomatis tersimpan rapi di folder `datasets_raw/foto_sendiri_mentah/`.

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
