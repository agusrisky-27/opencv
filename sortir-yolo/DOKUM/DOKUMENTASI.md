# 📘 DOKUMENTASI LENGKAP & MASTER KNOWLEDGE BASE PROYEK
## Sistem Sortir Otomatis Berbasis Visi Komputer (YOLOv8 + OpenCV) & Integrasi Robot Lengan (RNV3)

> Dokumen ini disusun sebagai panduan menyeluruh (*Master Engineering Guide & Knowledge Base*) yang merangkum arsitektur sistem, alur pembuatan proyek dari nol hingga produksi, panduan komprehensif pelatihan AI (Google Colab, Fine-Tuning, & Resume Epoch), lapisan filter cerdas, kalibrasi koordinat robot, serta praktik terbaik rekayasa visi komputer.
>
> 📌 *Untuk log kronologis progres kerja harian dan mingguan secara terperinci, silakan merujuk ke file [WORK_DOKUMENTATION.md](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/WORK_DOKUMENTATION.md).*

---

## 📑 DAFTAR ISI
1. [Latar Belakang & Arsitektur Global Sistem](#1-latar-belakang--arsitektur-global-sistem)
2. [Prasyarat & Cara Menjalankan Sistem (Getting Started)](#2-prasyarat--cara-menjalankan-sistem-getting-started)
3. [Alur Pembuatan Proyek dari Nol hingga Selesai (Life-Cycle)](#3-alur-pembuatan-proyek-dari-nol-hingga-selesai-life-cycle)
4. [Panduan Lengkap Pelatihan AI YOLOv8 (Colab, Fine-Tuning, & Resume)](#4-panduan-lengkap-pelatihan-ai-yolov8-colab-fine-tuning--resume)
5. [Lapisan Logika & Filter Cerdas (Smart Filters)](#5-lapisan-logika--filter-cerdas-smart-filters)
6. [Sistem Kalibrasi Hand-to-Eye & Inovasi Blueprint Manila](#6-sistem-kalibrasi-hand-to-eye--inovasi-blueprint-manila)
7. [Sistem Sortir Otomatis Penuh (100% Hands-Free / Stop-and-Go)](#7-sistem-sortir-otomatis-penuh-100-hands-free--stop-and-go)
8. [Struktur File & Penjelasan Modul Proyek](#8-struktur-file--penjelasan-modul-proyek)
9. [Ringkasan Perintah Cepat (Cheat-Sheet) Operasional](#9-ringkasan-perintah-cepat-cheat-sheet-operasional)
10. [Best Practices & Tips Menghadapi Kasus Serupa](#10-best-practices--tips-menghadapi-kasus-serupa)

---

## 1. LATAR BELAKANG & ARSITEKTUR GLOBAL SISTEM

### A. Tujuan Proyek
Membangun sistem visi komputer industri yang mandiri (*standalone*), cepat (*real-time 30 FPS*), dan memiliki keandalan tinggi untuk menyortir **3 jenis objek** pada sabuk konveyor berjalan:
1. 🪙 **Koin Logam:** Objek bulat tipis berbahan logam perak/kuningan dengan pantulan cahaya tinggi (*specular reflectance*).
2. 🔵 **Tutup Botol Plastik:** Objek silinder plastik berdiameter sedang dengan variasi warna pekat (oranye, biru, merah, hijau).
3. 🔌 **Kepala Charger HP:** Objek elektronik persegi panjang/kubus asimetris (badan plastik datar + dua pin colokan logam sempit).

Sistem bertugas mendeteksi benda, menghitung titik koordinat fisik presisi milimeter $(X, Y, Z\text{ mm})$, menggeser titik hisap vakum ke permukaan datar kedap udara, dan mengekspor koordinat target ke kontroler Lengan Robot (RNV3) untuk dipindahkan ke wadah sortir masing-masing:
* **Wadah Kiri:** Koin ($X = -150.0\text{ mm}, Y = 100.0\text{ mm}, Z = 45.0\text{ mm}$)
* **Wadah Kanan:** Tutup Botol ($X = +150.0\text{ mm}, Y = 100.0\text{ mm}, Z = 55.0\text{ mm}$)
* **Wadah Depan:** Charger ($X = 0.0\text{ mm}, Y = 180.0\text{ mm}, Z = 70.0\text{ mm}$)

---

### B. Diagram Alur Sistem (End-to-End Pipeline)

```
       [ Kamera Webcam USB ] (Rexus SW-10 @ 1280x720 HD)
                 │
                 ▼
       [ Threaded Frame Capture ] (Zero-Latency Background Thread Queue @ 30 FPS)
                 │
                 ▼
       [ Model AI YOLOv8 ] (models/best.pt @ imgsz=1280, conf=0.25)
                 │  - Deteksi 3 Kelas: Koin (0), Tutup Botol (1), Charger (2)
                 │  - Menghasilkan Bounding Box & Class Confidence Score
                 ▼
       [ Smart Geometric & Optical Filters ] (src/detector.py)
                 │  - Validasi Aspek Rasio (Koin/Tutup: 0.50..1.85 | Charger: 0.25..4.00)
                 │  - Proteksi Wajah / Kulit Manusia (is_human_skin_face, skin_frac > 0.48)
                 │  - Koreksi Koin Silau Cermin (is_specular_metal_coin: saturasi < 0.20)
                 │  - Verifikasi Saturasi Plastik Tutup Botol (tutup_berwarna: sat > 0.55)
                 ▼
       [ Vacuum Suction Point Refinement ] (Khusus Charger)
                 │  - Analisis orientasi kaki colokan logam (horizontal vs vertikal)
                 │  - Geser titik hisap dari leher colokan ke pusat bidang datar logo iT.
                 │  - Mencegah kebocoran hisap vakum pada celah pin colokan
                 ▼
       [ Penstabil Pelacakan Temporal ] (src/utils.py -> Penstabil)
                 │  - Voting suara terbanyak multi-frame (anti-flicker)
                 │  - Smoothing posisi kotak bergerak (Exponential Moving Average)
                 ▼
       [ Hand-to-Eye Coordinate Transformation ] (src/coordinate_transform.py)
                 │  - Mengubah piksel (u, v) ke Milimeter Fisik Robot (X, Y mm) via Homography
                 │  - Inversi Sumbu X otomatis (kamera & robot saling berhadapan)
                 │  - Ketinggian Z spesifik per benda (Koin=2mm, Tutup=12mm, Charger=28mm)
                 ▼
       [ Atomic JSON IPC Export ] ──► outputs/target_robot.json
                 │
                 ▼
       [ Pengontrol Lengan Robot (RNV3) ]
                 │  - Approach melayang di atas objek (Z + 25 mm)
                 │  - Turun ke Z_pick & Pompa Vakum ON
                 │  - Angkat objek & Geser ke Wadah Drop (Koin / Tutup / Charger)
                 │  - Pompa Vakum OFF & Kembali Standby
```

---

## 2. PRASYARAT & CARA MENJALANKAN SISTEM (GETTING STARTED)

### A. Kebutuhan Sistem & Perangkat Keras
* **Sistem Operasi:** macOS (Apple Silicon / Intel) atau Windows 10/11 (64-bit).
* **Python:** Versi 3.10 hingga 3.12 (disarankan Python 3.12).
* **Kamera:** Webcam USB eksternal (disarankan kamera dengan resolusi minimal 720p 30 FPS, misal Rexus SW-10).
* **Konveyor:** Sabuk konveyor industri (lebar sabuk $\approx 150 - 200\text{ mm}$).
* **Lengan Robot:** Robot arm 4-DOF / 6-DOF dengan suction cup pompa vakum (misal RNV3).

### B. Instalasi Lingkungan Virtual (Virtual Environment)
```bash
# 1. Masuk ke direktori proyek
cd ~/Documents/opencv_3class/sortir-yolo

# 2. Buat virtual environment (jika belum ada)
python3 -m venv .venv

# 3. Aktifkan virtual environment
# Pada macOS / Linux:
source .venv/bin/activate
# Pada Windows PowerShell:
# .venv\Scripts\Activate.ps1

# 4. Pasang seluruh dependensi resmi
pip install --upgrade pip
pip install -r requirements.txt
```

### C. Pemetaan Indeks Kamera USB (`--source`)
* **Pada macOS:**
  - `--source 0` : Webcam USB Eksternal Rexus SW-10 (kamera utama menghadap konveyor).
  - `--source 1` : Kamera Internal FaceTime MacBook Air.
  - `--source 2` : Kamera Continuity iPhone (jika terhubung).
* **Pada Windows:**
  - `--source 0` : Kamera bawaan laptop (atau webcam USB jika hanya ada 1 kamera).
  - `--source 1` : Webcam USB eksternal kedua.

### D. Menjalankan Program Utama

```bash
# 1. Menjalankan Deteksi Real-Time Standar (Warna per Kelas)
python detect.py --source 0 --warna-kelas

# 2. Menjalankan Mode Otomatis Penuh Tanpa Sentuh Laptop (Auto-Vision Stop-and-Go)
python sortir_otomatis.py --mode auto-vision --source 0

# 3. Menjalankan Mode Sensor Infrared Serial (Arduino USB)
python sortir_otomatis.py --mode serial --port /dev/ttyUSB0 --baud 9600

# 4. Pengujian Baca Koordinat untuk Tim Robotik
python contoh_baca_koordinat_robot.py --mode json

# 5. Menjalankan Seluruh Pengujian Otomatis (Unit Testing)
pytest -v
```

---

## 3. ALUR PEMBUATAN PROYEK DARI NOL HINGGA SELESAI (LIFE-CYCLE)

Jika Anda ingin mereplikasi atau membangun proyek computer vision berbasis konveyor dari nol, ikuti 7 tahapan standar industri berikut:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ TAHAP 1:        │     │ TAHAP 2:        │     │ TAHAP 3:        │     │ TAHAP 4:        │
│ Dataset Publik  │────►│ Capture Mandiri │────►│ Autolabel &     │────►│ Audit & Filter  │
│ (Roboflow)      │     │ di Konveyor     │     │ Segmentasi Mask │     │ Foto Blur       │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                 │
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐              ▼
│ TAHAP 7:        │     │ TAHAP 6:        │     │ TAHAP 5:        │     ┌─────────────────┐
│ Integrasi Robot │◄────│ Smart Filters & │◄────│ Pelatihan AI    │◄────│ Penggabungan    │
│ & Stop-and-Go   │     │ Suction Refine  │     │ di Google Colab │     │ Dataset Bersih  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Tahap 1: Eksplorasi & Pengunduhan Dataset Publik Roboflow
1. Kunjungi [Roboflow Universe](https://universe.roboflow.com/) dan cari kata kunci objek (misal `coin-detection`, `bottle-cap`).
2. Pilih dataset yang memiliki variasi pencahayaan dan latar belakang berbeda.
3. Gunakan modul `download_datasets.py` untuk mengunduh dataset berformat YOLOv8 TXT.
4. Lakukan pemetaan (*mapping*) kelas seragam agar indeks kelas teratur (`0: koin, 1: tutup_botol, 2: charger`).

### Tahap 2: Pengambilan Dataset Mandiri & Background Negatif
Dataset publik tidak memuat kondisi pencahayaan dan warna konveyor spesifik di pabrik/kantor Anda. Oleh karena itu, wajib mengambil foto mandiri:
1. Jalankan `python extra_capture_data.py --source 0`.
2. Gunakan hotkey keyboard untuk memisahkan folder penyimpanan foto otomatis:
   * Tekan tombol **`1`** : Menyimpan foto Koin ke `datasets_raw/koin_capture_manual/`.
   * Tekan tombol **`2`** : Menyimpan foto Tutup Botol ke `datasets_raw/tutup_botol_capture_manual/`.
   * Tekan tombol **`3`** : Menyimpan foto Charger ke `datasets_raw/charger_capture_manual/`.
   * Tekan tombol **`b`** : Mengambil **Latar Negatif** (konveyor kosong tanpa benda, tangan melintas, bayangan operator) ke `datasets_raw/latar_negatif_manual/`.
   > [!IMPORTANT]
   > Gambar latar negatif (*background images*) sangat krusial agar model AI tidak mengira pantulan meja atau permukaan sabuk konveyor sebagai objek sortir.

### Tahap 3: Segmentasi & Autolabeling Adaptif (`src/autolabel.py`)
Melabeli ribuan foto satu per satu dengan mouse (*labelImg*) memakan waktu berhari-hari. Sistem ini dilengkapi autolabel adaptif berbasis penglihatan komputer klasik:
* **Mode Konveyor Hijau:** Algoritma mengisolasi pita sabuk konveyor hijau ($H \in [30, 95]$), memotong laptop atau meja di luar konveyor, dan membungkus objek kontras di atas sabuk hijau.
* **Mode Alas Gelap:** Untuk pemotretan di atas mousepad hitam/alas gelap menggunakan segmentasi Otsu Thresholding.
* Jalankan `python extra_autolabel_own.py` untuk menghasilkan anotasi koordinat ternormalisasi YOLO `(class_id, x_center, y_center, width, height)`.

### Tahap 4: Pembersihan Foto Blur & Oklusi Tangan
* Foto yang kabur akibat benda bergerak cepat atau getaran kamera akan menurunkan kualitas bobot model AI.
* Jalankan `python bersihkan_dataset_blur.py`:
  - Algoritma menghitung variansi Laplacian: $\text{Var}(\nabla^2 I)$. Foto dengan skor $< 10.0$ otomatis dipindahkan ke folder `sampah_blur/`.
  - Algoritma mengecek warna kulit ($YCrCb$). Foto di mana tangan operator menutupi objek $> 25\%$ otomatis dibuang.

### Tahap 5: Penggabungan Dataset (`merge_datasets.py`)
* Menggabungkan dataset Roboflow + dataset foto mandiri bersih ke dalam folder `dataset_gabungan/` dengan rasio standar industri:
  - **70% Training Set:** Digunakan model untuk mempelajari fitur visual.
  - **20% Validation Set:** Digunakan untuk evaluasi berkala di setiap epoch.
  - **10% Test Set:** Digunakan untuk audit independen setelah pelatihan usai.

---

## 4. PANDUAN LENGKAP PELATIHAN AI YOLOV8 (COLAB, FINE-TUNING, & RESUME)

Pelatihan model deep learning membutuhkan akselerasi kartu grafis (GPU). Jika komputer lokal Anda tidak memiliki GPU NVIDIA bertenaga, gunakan **Google Colab** yang menyediakan GPU Tesla T4 (16 GB VRAM) secara gratis.

### A. Struktur File Konfigurasi Data (`dataset_gabungan/data.yaml`)
Pastikan file `data.yaml` mendefinisikan path dan nama kelas dengan benar:
```yaml
path: /content/sortir-yolo/dataset_gabungan
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['koin', 'tutup_botol', 'charger']
```

---

### B. Trik Penting: Training dari Nol (*Scratch*) vs Melanjutkan (*Fine-Tuning*)

Banyak pemula melakukan kesalahan dengan **selalu melatih model dari awal (epoch 0)** setiap kali menambahkan sedikit data baru. Ini membuang waktu dan berisiko merusak pola visual yang sudah dipelajari sebelumnya.

#### OPSI 1: Melatih dari Nol (*Training from Scratch*)
* **Kapan digunakan:** Ketika arsitektur kelas dirombak total dari nol atau dataset baru berbeda drastis.
* **Base Model:** `yolov8n.pt` (bobot umum dari COCO dataset).
* **Kode Training:**
  ```python
  from ultralytics import YOLO

  # Memuat bobot dasar umum
  model = YOLO('yolov8n.pt')

  # Melatih selama 80 - 100 epoch
  results = model.train(
      data='/content/sortir-yolo/dataset_gabungan/data.yaml',
      epochs=80,
      imgsz=640,
      batch=16,
      device=0,
      name='sortir_scratch'
  )
  ```
* **Kelemahan:** Membutuhkan waktu lama (35–50 menit di GPU T4).

---

#### OPSI 2: Melanjutkan Training dari Model Pintar (*Adaptive Fine-Tuning*) 🌟
* **Kapan digunakan:** Anda sudah memiliki model `models/best.pt` yang sudah sangat pintar mengenali koin dan tutup botol (misal sudah dilatih 80 epoch), lalu Anda ingin menambahkan objek baru (charger) atau mengadaptasi pencahayaan konveyor kantor yang baru.
* **Base Model:** Gunakan file `best.pt` Anda sendiri sebagai titik awal!
* **Kode Training:**
  ```python
  from ultralytics import YOLO

  # 1. Muat model best.pt hasil training sebelumnya (sudah pintar 2 kelas)
  model = YOLO('/content/sortir-yolo/models/best.pt')

  # 2. Latih HANYA 15 - 20 epoch tambahan!
  results = model.train(
      data='/content/sortir-yolo/dataset_gabungan/data.yaml',
      epochs=20,          # Cukup 20 epoch (~5-6 menit!)
      imgsz=640,          # Ukuran standar kompresi training
      batch=16,
      lr0=0.001,          # Learning rate awal lebih kecil agar bobot lama tidak rusak
      lrf=0.01,
      device=0,
      name='sortir_finetune'
  )
  ```
* **Keunggulan Besar:**
  1. **Super Cepat:** Hanya membutuhkan waktu **~5 menit** di GPU Tesla T4.
  2. **Anti-Lupa (*Preserve Feature Representations*):** Pengetahuan lama tentang tekstur relief koin dan warna tutup botol tetap melekat, sementara fitur charger langsung terintegrasi secara mulus.
  3. **Hemat Kuota GPU Colab:** Tidak menghabiskan batas kuota komputasi harian Google Colab.

---

#### OPSI 3: Melanjutkan Training yang Terputus (*Resume Training*)
Jika saat Anda sedang melatih model tiba-tiba koneksi internet terputus di epoch 50 (dari target 80 epoch), **jangan ulangi dari epoch 0!**
Ultralytics YOLO menyimpan checkpoint berkala di file `last.pt`. Cukup jalankan perintah:

```python
from ultralytics import YOLO

# Melanjutkan langsung dari epoch terakhir yang tersimpan
model = YOLO('/content/sortir-yolo/runs/detect/sortir_scratch/weights/last.pt')
model.train(resume=True)
```
Model akan otomatis melanjutkan dari epoch ke-51 hingga selesai 80 epoch tanpa kehilangan data apa pun.

---

### C. Mengapa Training Menggunakan `imgsz=640` tapi Inferensi Webcam Menggunakan `imgsz=1280`?
> [!NOTE]
> Ini adalah prinsip optimasi paling penting yang membedakan sistem amatir dari sistem industri:
* **Saat Training (`imgsz=640`):** Dataset berisi ribuan gambar. Menggunakan resolusi 640 menghemat alokasi memori VRAM GPU, mempercepat komputasi backpropagation hingga 400%, dan mencegah overfitting pada noise detail kamera.
* **Saat Inferensi Live di Webcam (`imgsz=1280`):** Kamera konveyor beresolusi 720p ($1280 \times 720$). Koin di konveyor fisik hanya selebar 50–70 piksel. Jika saat deteksi live gambar dipaksa susut ke 640, koin akan mengecil menjadi 25 piksel dan detail cetakan koin hancur. Dengan menetapkan `--imgsz 1280` saat inferensi live di [detect.py](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/detect.py), koin diproses pada skala asli 1:1 sehingga confidence deteksi mencapai **0.91 – 0.98**.

---

## 5. LAPISAN LOGIKA & FILTER CERDAS (SMART FILTERS)

Model Deep Learning (YOLO) adalah pendeteksi berbasis probabilitas statistik. Dalam skenario nyata di pabrik, model bisa saja dipicu oleh pantulan lampu atau benda asing. Modul [`src/detector.py`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py) dilengkapi lapisan filter cerdas matematika & optik (*Rule-Based Post Processing*):

### A. Validasi Geometri Fisik ([`validate_geometry`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py#L38-L71))
1. **Aspek Rasio ($w / h$):**
   * Koin dan tutup botol berbentuk bundar/elips simetris. Benda dengan rasio $< 0.50$ atau $> 1.85$ (seperti pulpen, penggaris, obeng) **otomatis ditolak**.
   * Khusus kelas **Charger**, rasio aspek dilonggarkan ke $0.25 \le AR \le 4.00$ karena bentuk fisik charger berupa balok persegi panjang.
2. **Batas Luas Area Layar (`max_area_frac`):**
   * Objek koin/tutup botol yang menutupi $> 40\%$ layar (seperti wajah manusia yang mendekat atau kardus besar) otomatis diabaikan.
   * Khusus charger, batas luas diizinkan hingga $55\%$.
3. **Batas Dimensi Maksimal Adaptif:**
   * Batas dimensi objek dibatasi hingga $65\%$ frame (atau $85\%$ untuk charger tegak), mencegah deteksi palsu pada dinding konveyor.

---

### B. Logika Pembeda Koin Silau vs Tutup Botol Putih ([`is_specular_metal_coin`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py#L174-L215) & [`is_white_bottle_cap`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py#L217-L255)) 🪙 🔵
* **Masalah Lapangan:** 
  1. Koin perak di bawah sorotan lampu memantulkan cahaya cermin (*specular glare* dengan $V > 135$), menghapus kontras cetakan relief koin menjadi lingkaran putih/abu-abu polos sehingga model AI keliru menduganya sebagai tutup botol.
  2. Tutup botol plastik berwarna putih (seperti tutup botol air mineral Aqua/Le Minerale) juga memiliki warna putih polos desaturasi ($S \approx 0.02, V > 180$). Jika filter hanya melihat saturasi, tutup botol putih akan keliru diubah menjadi koin!
* **Solusi Rekayasa Dua Arah (Dual Geometric & Photometric Discriminator):**
  - **Pembeda Dimensi Fisik:** Diameter fisik tutup botol ($30 - 45\text{ mm}$) jauh lebih besar dari koin logam ($24 - 27\text{ mm}$). Pada resolusi kamera 1280x720, tutup botol memiliki dimensi $\min(W, H) \ge 135\text{ px}$ dan luas area $\ge 20.000\text{ px}^2$.
  - **Restriksi Dimensi Koin Silau:** Fungsi `is_specular_metal_coin()` dibatasi hanya berlaku untuk objek berdimensi koin ($\min(W, H) \le 135\text{ px}$ dan luas $\le 18.500\text{ px}^2$).
  - **Koreksi Tutup Botol Putih:** Fungsi `is_white_bottle_cap()` mendeteksi jika objek berukuran besar ($\ge 135\text{ px}$), memiliki kecerahan putih solid merata ($V \ge 165$), dan rasio bundar. Jika AI keliru menduga koin, sistem otomatis mengoreksinya kembali menjadi **`tutup_botol`** dengan catatan `[koreksi tutup putih]`.

---

### C. Refinement Titik Hisap Vacuum Charger ([`refine_charger_vacuum_center`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py#L382-L425)) 🔌
* **Tujuan:** Menjamin karet hisap hisapan pompa vakum robot arm RNV3 mendarat pada bidang datar plastik kedap udara tanpa melompat-lompat (*anti-jitter*).
* **Mekanisme Kerja:**
  1. Memotong crop bounding box charger.
  2. Menerapkan segmentasi biner Otsu + operasi morfologi pembukaan (*morphological opening*).
  3. Menganalisis rasio $w / h$ untuk mengetahui orientasi pin colokan (vertikal atau horizontal).
  4. Menghitung selisih densitas massa bodi plastik vs kaki steker logam ($\Delta \text{density} = \text{density}_{\text{sisi } 1} - \text{density}_{\text{sisi } 2}$).
  5. Menerapkan formulasi proporsional kontinu mulus tanpa loncatan diskrit:
     $$\text{bias} = 0.50 - 0.14 \times \text{clip}\left(\frac{\Delta \text{density}}{120.0}, -1.0, 1.0\right)$$
  6. **Anti-Jitter:** Menghilangkan loncatan biner 28 piksel ($\pm 14\text{ mm}$) yang sebelumnya terjadi saat fluktuasi cahaya melewati batas 10.0. Titik hisap kini bergeser secara halus dan stabil mengunci di bidang datar plastik charger.

---

### D. Filter Anti-Wajah & Kulit Manusia ([`is_human_skin_face`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py#L92-L108))
* Menggunakan perpaduan ruang warna YCrCb ($Cr \in [133, 173], Cb \in [77, 127]$) dan HSV ($H \in [0, 25], S \in [35, 170], V \ge 50$).
* Objek hanya ditolak sebagai wajah manusia jika memenuhi ambang batas ketat:
  $$\text{skin\_frac} > 0.48 \quad \text{DAN} \quad \text{area\_frac} > 0.020 \quad \text{DAN} \quad \min(w, h) \ge 90\text{ px}$$
* Ambang batas `0.48` memastikan charger di atas meja bersuhu warna hangat tidak tertolak, sementara wajah manusia asli yang melintas tetap tereliminasi.

---

### E. Verifikasi Charger Asli vs Benda Acak Penguji ([`verifikasi_charger_asli`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/detector.py#L325-L380)) 🛡️
* **Masalah Pengujian:** Penguji sering menguji sistem dengan meletakkan benda acak yang tidak terduga di atas konveyor, seperti peraut pensil, karet penghapus, kartu, atau pulpen. Model AI YOLOv8 yang bekerja pada `--conf 0.25` dapat keliru menduga benda balok kecil (seperti penghapus/peraut) sebagai charger dengan confidence $0.26 - 0.35$.
* **Lapisan Verifikasi Ketat:**
  1. **Dimensi Fisik Minimum:** Adaptor charger HP nyata memiliki ukuran fisik yang jauh lebih besar dari peraut/penghapus mini. Diterapkan batas keras: $\min(w, h) \ge 50\text{ px}$ dan luas area $\ge 4000\text{ px}^2$. Peraut pensil mini ($< 50\text{ px}$) langsung **dieliminasi 100%**.
  2. **Aspek Rasio Balok Proporsional:** Dibatasi $0.35 \le AR \le 2.85$ (menolak pulpen, pensil panjang, atau kabel lurus).
  3. **Analisis Tekstur Kontras (Canny & Deviasi Standar):**
     - Karet penghapus polos memiliki permukaan seragam tanpa variasi warna ($\text{std} < 11.0$ dan kerapatan Canny edges $< 0.012$). Objek datar polos ini otomatis ditolak jika confidence $< 0.60$.
     - Kepala charger asli memiliki kontras tinggi antara bodi plastik polimer dan kaki steker logam hitam/perak atau celah rongga port USB ($\text{std} \ge 14.0$, $\text{edges} \ge 0.020$).
  4. **Confidence Floor Gating:** Objek kotak yang diragukan dengan confidence $< 0.38$ otomatis dibuang.

---

### F. Penstabil Suara Terbanyak Multi-Frame ([`Penstabil`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/src/utils.py#L107-L230))
* Mencegah kedipan (*flicker*) label pada video live dengan sistem voting riwayat 5 frame terakhir (*majority voting buffer*).
* Dilengkapi *Exponential Moving Average (EMA)* untuk menghaluskan koordinat bounding box saat kamera terkena getaran mesin konveyor.

---

## 6. SISTEM KALIBRASI HAND-TO-EYE & INOVASI BLUEPRINT MANILA

### A. Teori Transformasi Homography 2D
Kamera terpasang miring (*angled perspective*) di atas meja konveyor. Transformasi linier sederhana ($X = a \cdot u + b$) tidak akurat karena adanya distorsi perspektif trapesium.
Sistem menggunakan **Matriks Homography $3 \times 3$**:

$$\begin{bmatrix} X_{robot} \\ Y_{robot} \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} u_{kamera} \\ v_{kamera} \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

Matriks $\mathbf{H}$ dihitung menggunakan algoritma RANSAC dari 4 pasang titik kalibrasi yang dipilih lewat program GUI `kalibrasi_robot.py`.

---

### B. Inovasi "Calibration Base Mat" di Kertas Manila

Untuk mengatasi masalah meja yang sering berpindah saat pameran atau demonstrasi, dibuat alas kerja presisi menggunakan kertas manila tebal:

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
│                               ▲ Sumbu Y+ (Maju ke conveyor)            │
│                               │                                        │
│                 ┌─────────────┴─────────────┐                          │
│                 │     KOTAK DUDUKAN ROBOT   │                          │
│                 │   (Titik Pusat Home = 0,0)│ ──► Sumbu X+ (Kanan)     │
│                 └───────────────────────────┘                          │
└────────────────────────────────────────────────────────────────────────┘
```

**Keuntungan Operasional Blueprint:**
1. **Zero-Setup Time:** Saat pindah ruangan, cukup bentangkan kertas manila di meja dan letakkan dudukan konveyor, robot, dan webcam tepat di atas garis pola.
2. **Tanpa Perlu Kalibrasi Ulang:** Nilai matriks pada `robot_calib.json` tetap valid 100% karena posisi geometris fisik terkunci sempurna.

---

### C. Alur Pengiriman Target ke Robot (Atomic JSON IPC)
Data target diekspor ke file `outputs/target_robot.json` menggunakan teknik *atomic write* (menulis ke file temporer `.tmp` lalu me-rename seketika) agar kontroler robot di sistem operasi Windows tidak mengalami konflik *file-locking*.

**Contoh Payload JSON Target Robot:**
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

## 7. SISTEM SORTIR OTOMATIS PENUH (100% HANDS-FREE / STOP-AND-GO)

Untuk implementasi demonstrasi atau operasional pabrik nyata, **tidak boleh ada operator yang menekan tombol keyboard laptop**. Seluruh siklus dijalankan secara otonom oleh modul [`sortir_otomatis.py`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/sortir_otomatis.py):

### A. Masalah Fisik Nyata: Inersia Sabuk Konveyor & Jeda Berhenti (Settling Time)
Secara mekanis, saat motor konveyor dimatikan oleh sensor Infrared (IR), sabuk konveyor dan objek di atasnya **tidak pernah berhenti beku seketika (0 mm/s)** dalam 0 milidetik. Ada faktor deselerasi motor, selip sabuk (*belt slippage*), dan getaran inersia (*mechanical vibration / jitter*):
1. **Risiko Jika Jepret Instan:** Jika kamera langsung mengambil snapshot di milidetik sensor IR terputus, kamera akan menangkap foto dengan *motion blur* atau benda yang masih bergeser 3–8 mm dari titik final. Akibatnya, gripper/suction cup robot akan meleset saat turun mengambil!
2. **Risiko Buffer Kamera Usang:** Webcam USB memiliki buffer frame hardware (3–5 frame). Jika kode hanya melakukan `sleep` tanpa membersihkan buffer (*flush*), frame yang dibaca adalah foto lama saat konveyor masih meluncur kencang!

### B. Solusi Rekayasa: Triple-Stage Settling Filter
Sistem menerapkan 3 lapisan pengamanan di [`sortir_vision_api.py`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/sortir_vision_api.py) dan [`sortir_otomatis.py`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/sortir_otomatis.py):
1. **Lapisan 1: Settling Delay (0.5 – 0.8 Detik, Default 0.6s):**
   Saat sensor IR aktif, sistem menahan pengiriman koordinat ke robot (status JSON: `WAITING_STATIONARY`, `target: null`). Waktu 0.6 detik adalah *sweet spot* optimal: cukup untuk meredam inersia mekanis tanpa memperlambat siklus sortir (*cycle time*).
2. **Lapisan 2: Camera Buffer Flush (`flush_frames=4`):**
   Setelah jeda inersia selesai, sistem membuang 4 frame antrian lama dari memori webcam (`self.read_frame()`), memastikan frame berikutnya adalah citra *real-time* terbaru.
3. **Lapisan 3: Zero-Motion Verification ($\Delta d \le 3.5\text{ px}$):**
   AI membandingkan posisi sentroid objek antara 2 frame berurutan:
   $$\Delta d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
   Jika $\Delta d \le 3.5\text{ piksel}$, objek dikonfirmasi **100% DIAM TOTAL & STABIL**. Jika masih ada sisa selip, sistem memberi toleransi jeda mikro (0.2 detik) sebelum mengunci koordinat.

### C. Protokol Status JSON IPC (`outputs/target_robot.json`)
Agar robot tidak bergerak sembarangan saat objek meluncur di konveyor:
* `status: "CONVEYOR_MOVING"` $\rightarrow$ Konveyor sedang jalan, robot WAJIB diam di posisi HOME (`target: null`).
* `status: "WAITING_STATIONARY"` $\rightarrow$ Sensor IR terpicu, konveyor deselerasi, robot tetap standby.
* `status: "TARGET_READY"` $\rightarrow$ Objek terbukti diam total, koordinat $(X, Y, Z)$ terkunci permanen, robot mengeksekusi Pick & Place.
* `status: "NO_TARGET"` $\rightarrow$ Benda bukan target sortir atau area kosong, konveyor langsung dinyalakan kembali (`RESUME`).

### D. Penguncian Koordinat Statis (Coordinate Latch & Spatial Deadzone Hysteresis)
* **Masalah Lapangan:** Bounding box YOLO alami bergetar tipis $\pm 2 - 4\text{ px}$ akibat derau pencahayaan kamera (*sensor noise*). Jika koordinat $(X, Y, Z)$ selalu dihitung ulang setiap 33 ms, koordinat robot akan bergoyang $\pm 1 - 2\text{ mm}$ dan dapat mereset timer stabilitas.
* **Mekanisme Latch & Histeresis ($14\text{ px}$):**
  1. Begitu objek terkonfirmasi diam (`TARGET_READY`), sistem mengunci (*latches*) seluruh parameter: `locked_target = best_t` dan `locked_center = (cx, cy)`.
  2. Selama pergeseran sentroid frame-ke-frame masih berada di dalam radius deadband histeresis ($\le 14\text{ px}$), sistem **membekukan nilai koordinat** yang sama persis.
  3. Kunci hanya terlepas jika objek berpindah jauh ($> 14\text{ px}$) saat konveyor kembali dinyalakan atau objek berhasil dipindahkan oleh lengan robot.

### E. Anti-Banjir Output Terminal (Single-Shot Target Key Debounce)
* Di program [`contoh_baca_koordinat_robot.py`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/contoh_baca_koordinat_robot.py), diterapkan sistem debouncing berbasis kunci unik target:
  $$\text{target\_key} = (\text{track\_id}, \text{round}(X, 1), \text{round}(Y, 1), \text{label})$$
* Simulasi urutan perintah gerak robot (G-Code: meluncur di atas objek $\rightarrow$ vakum ON $\rightarrow$ angkat $\rightarrow$ lepas ke wadah) dieksekusi **tepat 1 KALI** per objek unik yang tiba. Terminal tetap bersih, tenang, dan tidak dibanjiri teks berulang saat menunggu benda dipindahkan.

### F. Firmware Arduino Siap Pakai
Tersedia firmware siap flash di [`arduino/conveyor_ir_controller.ino`](file:///Users/agusrisky/Documents/opencv_3class/sortir-yolo/arduino/conveyor_ir_controller.ino) yang menghubungkan sensor IR (Pin 2), Relay Motor (Pin 7), dan komunikasi serial USB dua arah (`IR_TRIGGER` $\leftrightarrow$ `PICK` / `RESUME`).

---

## 8. STRUKTUR FILE & PENJELASAN MODUL PROYEK

```
opencv_3class/
├── .gitignore                       # Proteksi kredensial API & file cache lokal
├── README.md                        # Ringkasan cepat & petunjuk bagi AI agent
├── DOKUM/                           # DIREKTORI DOKUMENTASI PROYEK
│   ├── DOKUMENTASI.md               # Master Technical Knowledge Base (file ini)
│   ├── WORK_DOKUMENTATION.md        # Log kronologis laporan kerja harian/mingguan
│   ├── PANDUAN_TIM_ROBOTIK.md       # Panduan integrasi khusus rekan tim robotik Windows
│   └── README_ARSIP.md              # Arsip dokumentasi awal
│
└── sortir-yolo/                     # DIREKTORI KODE UTAMA
    ├── 05_detect.py                 # Pembungkus backward-compatible untuk tim robotik
    ├── detect.py                    # Program visualisasi real-time (imgsz=1280, conf=0.25, Latch)
    ├── sortir_otomatis.py           # Controller sortir otonom 100% hands-free (Stop-and-Go)
    ├── sortir_vision_api.py         # Modul API visi mandiri siap panggil oleh skrip robot
    ├── contoh_baca_koordinat_robot.py # Demo baca koordinat robot (Debounced Single-Shot)
    ├── tes_kirim_koordinat.py       # Alat uji pengiriman koordinat & simulasi IR interaktif
    ├── kalibrasi_robot.py           # GUI interaktif kalibrasi 4-titik kamera ke robot mm
    ├── jalankan_semua.bat           # Batch launcher 1-klik untuk rekan tim Windows
    ├── robot_calib.json             # Parameter matriks Homography & koordinat wadah drop
    ├── outputs/
    │   └── target_robot.json        # Berkas atomic JSON IPC real-time untuk robot
    │
    ├── capture.py                   # Perekam sampel gambar baru standar
    ├── extra_capture_data.py        # Perekam data multi-kelas keyboard 1, 2, 3, b
    ├── extra_autolabel_own.py       # Engine autolabel adaptif konveyor hijau & alas gelap
    ├── bersihkan_dataset_blur.py    # Skrip filter otomatis foto blur & oklusi kulit
    ├── merge_datasets.py            # Penggabung dataset Roboflow + mandiri (70/20/10)
    │
    ├── colab/
    │   └── Training_YOLO_Colab.ipynb # Notebook pelatihan Google Colab (Scratch & Finetune)
    │
    ├── models/
    │   └── best.pt                  # Bobot neural network YOLOv8 utama (6,2 MB)
    │
    ├── outputs/
    │   ├── target_robot.json        # File IPC koordinat real-time ke robot
    │   ├── simulasi_hisap_charger.jpg # Visualisasi titik hisap vacuum cup pada charger
    │   └── preview_charger_on_conveyor.jpg # Bukti validasi bounding box autolabel
    │
    ├── src/
    │   ├── autolabel.py             # Algoritma segmentasi sabuk hijau konveyor & HSV
    │   ├── camera.py                # Wrapper kamera OpenCV + Threaded Capture anti-lag
    │   ├── config.py                # Parameter konfigurasi global & palet warna kelas
    │   ├── coordinate_transform.py  # Logika matematika Homography piksel ke milimeter
    │   ├── detector.py              # Engine YOLOv8 + Smart Filters + Refine Vacuum Center
    │   └── utils.py                 # Anotasi visual HUD, Kalman filter, & Penstabil voting
    │
    └── tests/                       # 56 skenario pengujian unit otomatis (100% lolos)
```

---

## 9. RINGKASAN PERINTAH CEPAT (CHEAT-SHEET) OPERASIONAL

### Persiapan Lingkungan Terminal:
```bash
cd ~/Documents/opencv_3class/sortir-yolo
source .venv/bin/activate
```

### Eksekusi Program:
```bash
# 1. Deteksi Real-Time Standar (Koin=Emas, Tutup=Biru, Charger=Hijau)
python detect.py --source 0 --warna-kelas

# 2. Deteksi pada Kamera FaceTime Laptop (Source 1)
python detect.py --source 1 --warna-kelas

# 3. Mode Sortir Otomatis Penuh (Auto-Vision Trigger)
python sortir_otomatis.py --mode auto-vision --source 0

# 4. Mode Sortir Sensor Hardware Infrared Serial Arduino
python sortir_otomatis.py --mode serial --port /dev/ttyUSB0 --baud 9600

# 5. Kalibrasi Homography 4-Titik Meja Konveyor
python kalibrasi_robot.py --source 0

# 6. Menjalankan Seluruh Automated Unit Testing (56 Tests)
pytest -v
```

### Tombol Kontrol Keyboard di Jendela Kamera:
* **`q` / `ESC`** : Keluar dari program.
* **`w`** : Mengaktifkan warna kotak pembeda per kelas (Emas = Koin, Biru/Cyan = Tutup Botol, Hijau = Charger).
* **`s`** : Menyimpan tangkapan layar (*screenshot*) ke folder `outputs/`.
* **`c`** : Mengekspor seluruh data koordinat saat itu ke `outputs/koordinat.csv`.
* **`t`** : Mengaktifkan / mematikan pelacak objek bergerak (*ByteTrack Object Tracking*).
* **`f`** : Mengaktifkan / mematikan filter geometri cerdas (*Smart Geometric Filters*).
* **`r`** : Mengaktifkan zona batas kerja konveyor (*Conveyor Region of Interest / ROI*).
* **`+` / `-`** : Menaikkan / menurunkan ambang batas confidence secara live di layar.

---

## 10. BEST PRACTICES & TIPS MENGHADAPI KASUS SERUPA

1. **Selalu Gunakan Resolusi Inferensi Sesuai Karakteristik Objek:**
   Untuk objek berukuran kecil di kamera (koin, baut, tutup botol), selalu gunakan resolusi inferensi tinggi (`imgsz=1280`). Menggunakan 640 saat inferensi live akan mengikis kontras objek kecil akibat downsampling.
2. **Manfaatkan Fine-Tuning Ketimbang Melatih dari Awal:**
   Jika model lama sudah akurat pada 2 kelas dan Anda hanya ingin menambahkan 1 kelas baru atau beradaptasi pada pencahayaan konveyor baru, lakukan **fine-tuning 20 epoch dari `best.pt` lama**. Ini menghemat waktu dari 45 menit menjadi hanya 5 menit.
3. **Pencahayaan Menyebar (*Diffused Lighting*) untuk Logam:**
   Permukaan logam koin bertindak seperti cermin di bawah lampu sorot tajam. Selalu pasang diffuser tipis di depan lampu kerja atau arahkan lampu secara miring agar relief koin tampak jelas tanpa silau overexposure.
4. **Gunakan Blueprint Fisik pada Lomba / Demonstrasi:**
   Menggambar pola tata letak konveyor dan dudukan robot di atas kertas manila menghemat waktu setup hingga 90% dan mengeliminasi kebutuhan kalibrasi ulang saat berpindah tempat.
5. **Pisahkan Titik Pusat Geometri dari Titik Hisap Vakum:**
   Pada benda asimetris seperti charger HP, jangan gunakan titik tengah bounding box untuk titik penjemputan robot. Selalu geser titik hisap ke bidang datar terluas agar karet vakum menempel kedap udara.
6. **Desain Komunikasi IPC yang Tangguh (*Decoupled IPC*):**
   Hindari middleware jaringan yang rumit untuk komunikasi antar-program lokal. Penggunaan file JSON berformat atomic write (`target_robot.tmp` $\rightarrow$ `target_robot.json`) terbukti 100% bebas lag, ringan di CPU, dan tidak memicu konflik *file-locking* di Windows.
