# Deteksi & Sortir Koin / Tutup Botol (YOLOv8 + OpenCV)

Sistem visi komputer real-time berbasis **YOLOv8** dan **OpenCV** untuk mendeteksi dan menyortir **Koin** dan **Tutup Botol**. Dilengkapi dengan **Smart Filter** fisik/geometri (rasio aspek, batas ukuran, verifikasi warna pekat, dan ROI area kerja) untuk mengeliminasi *false positives* (seperti wajah manusia atau jendela ruangan).

Dirancang untuk sistem konveyor dan integrasi lengan robot vakum.

---

## Fitur Utama

- **Real-Time Detection:** Deteksi cepat via webcam dengan akselerasi GPU (Apple Silicon `MPS` / NVIDIA `CUDA`).
- **Kotak Merah & Titik Tengah (cx, cy):** Menampilkan bounding box, label benda, titik pusat objek, dan koordinat normalisasi (0.0 - 1.0).
- **Smart Geometric Filter:**
  - Menyaring objek lonjong/persegi panjang (seperti jendela/pintu) dengan batas rasio aspek ($0.65 \le w/h \le 1.50$).
  - Menyaring objek terlalu besar (wajah manusia) atau terlalu kecil (noise).
- **Verifikasi Warna HSV:** Membedakan tutup botol berwarna pekat (biru, merah, hijau) dari koin logam secara akurat.
- **Zona Kerja (ROI Conveyor):** Membatasi area deteksi hanya pada sabuk konveyor/meja kerja.
- **Penstabil Multi-Frame:** Menghilangkan kedipan label (*flicker*) antar-frame.

---

## Struktur Folder

```
├── README.md
├── Training_YOLO_Colab.ipynb          # Notebook training Google Colab
├── sortir-yolo/
│   ├── 01_download_datasets.py        # Unduh dataset Roboflow
│   ├── 02_autolabel_tutup_botol.py    # Auto-label tutup botol
│   ├── 03_merge_datasets.py           # Gabung dataset train/val
│   ├── 04_train.py                    # Training YOLO lokal
│   ├── 05_detect.py                   # Program deteksi real-time webcam
│   ├── extra_capture_data.py          # Ambil foto sendiri & background negatif
│   ├── extra_autolabel_own.py         # Auto-label foto sendiri
│   ├── requirements.txt               # Daftar dependensi Python
│   ├── .env.example                   # Format API key Roboflow
│   ├── models/
│   │   └── best.pt                    # Model hasil latih YOLOv8 (siap pakai)
│   ├── src/                           # Modul detector, camera, utils, geometry
│   └── tests/                         # Unit tests
└── claude_extracted/                  # Versi single-file modular (sortir.py)
```

---

## Cara Instalasi & Menjalankan

### 1. Buat Virtual Environment & Install Dependensi

```bash
cd sortir-yolo
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Jalankan Deteksi Webcam Langsung

Model hasil latih (`models/best.pt`) sudah tersedia, sehingga bisa langsung dijalankan:

```bash
python 05_detect.py
```

Jika menggunakan kamera eksternal / USB webcam:
```bash
python 05_detect.py --source 1
```

### 3. Kontrol Keyboard Saat Kamera Aktif

- `f` : Nyalakan/Matikan **Smart Filter** (ON/OFF).
- `r` : Nyalakan/Matikan **ROI Zona Conveyor** (ON/OFF).
- `+` / `-` : Naikkan / turunkan threshold confidence (default: 0.65).
- `s` : Ambil foto tangkapan layar (tersimpan di `outputs/`).
- `c` : Simpan koordinat objek ke file CSV (`outputs/koordinat.csv`).
- `q` / `ESC` : Keluar dari program.
