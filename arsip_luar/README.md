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
├── README.md                          # Dokumentasi utama proyek
├── detect.py                          # Shortcut runner untuk deteksi langsung dari root
├── .gitignore                         # Konfigurasi file yang diabaikan Git
└── sortir-yolo/                       # Folder kerja utama
    ├── requirements.txt               # Daftar dependensi Python
    ├── .env.example                   # Format API key Roboflow
    ├── 01_download_datasets.py        # Unduh dataset Roboflow
    ├── 02_autolabel_tutup_botol.py    # Auto-label tutup botol
    ├── 03_merge_datasets.py           # Gabung dataset train/val
    ├── 05_detect.py                   # Program deteksi real-time webcam
    ├── extra_capture_data.py          # Ambil foto sendiri & background negatif
    ├── extra_autolabel_own.py         # Auto-label foto sendiri
    ├── colab/
    │   └── Training_YOLO_Colab.ipynb  # Notebook training Google Colab
    ├── models/
    │   └── best.pt                    # Model hasil latih YOLOv8 (siap pakai)
    ├── src/                           # Modul detector, camera, utils, config
    └── tests/                         # Unit tests
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

**Opsi A — Dari root workspace:**
```bash
python detect.py
```

**Opsi B — Dari dalam folder `sortir-yolo`:**
```bash
cd sortir-yolo
python 05_detect.py
```

Jika menggunakan kamera eksternal / USB webcam:
```bash
python detect.py --source 1
```

Untuk membedakan warna kotak visual tiap kelas (Koin = Kuning Emas, Tutup Botol = Cyan):
```bash
python detect.py --warna-kelas
```

### 3. Kontrol Keyboard Saat Kamera Aktif

- `f` : Nyalakan/Matikan **Smart Filter** (ON/OFF).
- `r` : Nyalakan/Matikan **ROI Zona Conveyor** (ON/OFF).
- `w` : Nyalakan/Matikan **Warna per Kelas** (Kuning=Koin, Biru=Tutup Botol / Default=Merah).
- `+` / `-` : Naikkan / turunkan threshold confidence (default: 0.65).
- `s` : Ambil foto tangkapan layar (tersimpan di `outputs/`).
- `c` : Simpan koordinat objek ke file CSV (`outputs/koordinat.csv`).
- `q` / `ESC` : Keluar dari program.
