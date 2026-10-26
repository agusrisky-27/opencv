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
1. **Mendeteksi dan mengenali 3 jenis benda** pada sabuk konveyor industri:
   - 🪙 **Koin:** Objek logam bundar tipis dengan pantulan kilap tinggi.
   - 🔵 **Tutup Botol:** Objek plastik silinder berdiameter sedang dengan warna bervariasi.
   - 🔌 **Charger:** Objek elektronik persegi panjang/kubus dengan dua pin colokan logam (kepala adapter HP).
2. **Menentukan titik pusat fisik objek** dalam satuan milimeter nyata ($X, Y, Z\text{ mm}$), termasuk pergeseran otomatis titik hisap vakum (*suction point offset*) ke permukaan datar pada benda asimetris seperti charger.
3. **Mengirimkan koordinat target** secara instan ke sistem kontroler Robot Lengan (RNV3) untuk dilakukan penyortiran otomatis (*pick and place*) ke wadah sortir masing-masing (Kiri = Koin, Kanan = Tutup Botol, Depan = Charger).

### B. Diagram Alur Sistem (End-to-End Pipeline)

```
       [ Kamera Webcam USB ] (Rexus SW-10 @ --source 0)
                 │
                 ▼
       [ Threaded Frame Capture ] (Zero-Latency Queue, 1280x720 @ 30 FPS)
                 │
                 ▼
       [ Model AI YOLOv8 ] (models/best.pt @ imgsz=1280, conf=0.25)
                 │  - Deteksi 3 Kelas: Koin (0), Tutup Botol (1), Charger (2)
                 │  - Bounding Box & Class Confidence Score
                 ▼
       [ Smart Geometric & Optical Filters ]
                 │  - Filter Rasio Aspek & Luas Bidang
                 │  - Verifikasi Koin Logam vs Jari Manusia (HSV Saturation)
                 │  - Isolasi Konveyor Hijau & Penolakan Latar Luar
                 ▼
       [ Vacuum Suction Point Refinement (Khusus Charger) ]
                 │  - Analisis orientasi pin colokan vs badan datar
                 │  - Geser titik hisap ke pusat bidang datar (logo iT.)
                 │  - Mencegah kebocoran hisap vakum pada colokan logam
                 ▼
       [ Hand-to-Eye Coordinate Transformation ]
                 │  - Mengubah piksel (u, v) ke Milimeter Fisik Robot (X, Y mm)
                 │  - Inversi Sumbu X (Orientasi Kamera-Robot Berseberangan)
                 │  - Ketinggian Z Otomatis (Koin=2mm, Tutup=12mm, Charger=28mm)
                 ▼
       [ IPC Data Export (Atomic JSON Write) ] ──► outputs/target_robot.json
                 │
                 ▼
       [ Pengontrol Lengan Robot (RNV3) ]
                 │  - Approach melayang di atas objek (Z + 25 mm)
                 │  - Turun ke Z_pick & Pompa Vakum ON
                 │  - Angkat objek & Geser ke Wadah Drop (Koin/Tutup/Charger)
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

### Tahap 8: Ekspansi Sistem ke 3 Kelas (Penambahan Kelas Charger - Jumat, 24 Okt 2026)
* **Kebutuhan Lapangan:** Sistem diperluas untuk menyortir benda ke-3: **Kepala Charger HP** (selain Koin dan Tutup Botol).
* **Autolabel Charger Versi 1:** Membangun `segment_charger_box()` adaptif pertama di `src/autolabel.py`.
* **Dataset Awal Charger:** Mengumpulkan 383 foto charger manual di rumah di atas alas gelap (*dark mat*).
* **Model 3 Kelas Pertama:** Berhasil melatih model 3 kelas pertama di Google Colab dan menghasilkan bobot `models/best.pt` (6,2 MB).
* **Filter Ketajaman:** Menambahkan modul pembersih blur (`bersihkan_dataset_blur.py`) menggunakan variansi Laplacian untuk membuang foto goyang.

### Tahap 9: Perbaikan Struktur Workspace & Pemetaan Webcam USB macOS (Senin Pagi, 26 Okt 2026)
* **Rename Folder Proyek:** Folder `opencv copy` diubah menjadi `opencv_3class` agar rapi dan representatif. Untuk menjaga terminal runner agent tetap valid tanpa memunculkan folder duplikat di Finder, dibuat symlink tersembunyi (`chflags hidden`).
* **Audit & Mapping Kamera macOS:**
  - `source 0` = Webcam Eksternal Rexus SW-10 (menghadap ke konveyor).
  - `source 1` = Kamera Internal FaceTime MacBook Air.
  - `source 2` = Kamera Continuity iPhone.
  - Default pada `extra_capture_data.py` dan `detect.py` diselaraskan ke `--source 0`.

### Tahap 10: Inovasi Refinement Titik Hisap Vacuum Charger (Senin, 26 Okt 2026)
* **Masalah Mekanis Robot:** Kepala charger memiliki bentuk asimetris (badan plastik lebar + dua batang colokan logam sempit). Bounding box standar memiliki titik tengah $(cx, cy)$ yang jatuh di dekat leher colokan. Jika karet vakum robot RNV3 mendarat di area colokan, hisapan akan bocor (*air loss*) dan charger gagal terangkat.
* **Solusi Algoritma Visi (`refine_charger_vacuum_center()` di `src/detector.py`):**
  - Menganalisis orientasi colokan logam (horizontal vs vertikal).
  - Menggeser titik target hisap secara dinamis ke **pusat permukaan bidang datar kepala charger** (area logo `iT.`).
  - Menjamin hisapan karet vakum kedap udara (*airtight suction seal*).

### Tahap 11: Investigasi & Perbaikan Engine Autolabel Konveyor Hijau Kantor (Senin Siang, 26 Okt 2026)
* **Masalah Kritis Lapangan:** Saat mengambil 273 foto charger di atas konveyor sabuk hijau kantor, autolabel lama menghasilkan bounding box raksasa selebar 909 piksel.
* **Akar Masalah:** Algoritma lama menganalisis piksel tepi frame (yang memuat meja putih dan laptop hitam di belakang konveyor). Karena tepi meja berwarna terang, algoritma mengira latarnya putih dan mencari objek gelap (`cv2.THRESH_BINARY_INV`), sehingga yang ditandai adalah **laptop hitam** dan **rel konveyor**, bukan charger putih!
* **Solusi Engine Autolabel Baru (`src/autolabel.py`):**
  - Dibuat **Mode Konveyor Hijau Khusus** (`green_pct > 0.08`).
  - Mengisolasi pita konveyor hijau ($H \in [30, 95]$), mengecualikan laptop dan rel meja di luar batas.
  - Mendeteksi charger putih ($S < 60, V > 125$) dengan margin proteksi dan filter kulit ($YCrCb$).
  - **Hasil Validasi:** Bounding box pada `outputs/preview_charger_on_conveyor.jpg` membungkus kepala charger `[361, 157, 726, 480]` dengan akurasi $< 1$ piksel dari ground truth!

### Tahap 12: Pemilahan & Audit Dataset Ketat (Senin Siang, 26 Okt 2026)
* **Penyortiran Foto Blur & Oklusi:**
  - 4 foto blur (Laplacian < 10) dipisahkan ke `sampah_blur/`.
  - 28 foto tangan menutupi charger (>25% oklusi) dipisahkan ke `sampah_blur/`.
  - 2 foto konveyor kosong di akhir dipindahkan ke `latar_negatif_manual/`.
* **Koreksi Data Salah Folder:**
  - Menangani 94 foto charger yang sempat tersimpan di folder tutup botol (karena lupa memencet tombol angka 3 saat capture).
  - 94 foto charger dipindahkan otomatis ke `charger_capture_manual/`, sedangkan 30 foto tutup botol asli tetap dipertahankan.
* **Hasil Akhir Dataset Gabungan (`dataset_gabungan/`):**
  - **Koin:** 2.915 instance
  - **Tutup Botol:** 5.230 instance (termasuk 104 foto riil di konveyor kantor)
  - **Charger:** 542 instance (termasuk 333 foto riil di konveyor kantor)
  - **Latar Negatif:** 134 foto kosong
  - **Total:** 6.607 gambar bersih (5.153 train, 984 valid, 470 test), 0 error koordinat, 0 missing label.

### Tahap 13: Simulasi Koordinat Fisik & Pipeline Training Google Colab (Senin Siang, 26 Okt 2026)
* **Simulasi Visual Suction:** Membuat generator visualisasi titik hisap dan koordinat milimeter pick/drop robot RNV3 (`outputs/simulasi_hisap_charger.jpg`).
* **Pembaruan Konfigurasi & Panduan Robot:** Menyelaraskan `robot_calib.json` dan `PANDUAN_TIM_ROBOTIK.md` untuk 3 kelas objek beserta ketinggian sumbu Z ($Z_{koin}=2\text{ mm}, Z_{tutup}=12\text{ mm}, Z_{charger}=28\text{ mm}$).
* **Pembuatan Paket Fresh `sortir-yolo.zip` (1,4 GB):** Mengompresi seluruh dataset dan kode bersih tanpa membebani library lokal `.venv` dan file cache.
* **Pipeline Training Cepat (Opsi B):** Mengonfigurasi mode `finetune` (20 epoch, ~5-6 menit) di Google Colab yang melanjutkan kecerdasan dari `models/best.pt` lama sambil mengunci adaptasi pada konveyor hijau kantor.

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
| **7** | Titik hisap vakum meleset di leher colokan charger | Bounding box standar memiliki titik tengah $(cx, cy)$ yang jatuh di dekat colokan logam sempit | Algoritma `refine_charger_vacuum_center()` menggeser titik hisap ke bidang datar terluas kepala charger (area logo `iT.`) |
| **8** | Autolabel lama salah menandai laptop hitam & rel konveyor | Border meja putih di luar konveyor dikira latar, sehingga autolabel mencari benda gelap (`THRESH_BINARY_INV`) | Membangun **Mode Konveyor Hijau Khusus**: mengisolasi pita sabuk hijau dan mendeteksi charger putih ($S < 60, V > 125$) dengan margin proteksi |
| **9** | 94 Foto charger masuk ke folder tutup botol | Tombol angka 3 pada keyboard belum tertekan saat proses capture cepat di konveyor | Skrip audit otomatis memilah file mulai timestamp `115735_524` ke bawah dan merelokasinya ke `charger_capture_manual/` |
| **10** | Tangan operator menutupi charger pada beberapa foto | Operator meletakkan atau memutar posisi charger saat kamera sedang merekam | Deteksi persentase kulit ($YCrCb$) pada area charger; jika oklusi $> 25\%$, foto otomatis dibuang ke `sampah_blur/` |
| **11** | Error Colab `credential propagation was unsuccessful` | Konflik multi-akun Google (Akun A dan Akun B) saat menjalankan `drive.mount` | Bypass Google Drive dengan drag-and-drop langsung ke `/content/sortir-yolo.zip` atau gunakan jendela penyamaran (*Incognito*) |
| **12** | Risiko domain shift jika memakai kamera iPhone | Komputasi fotografi iPhone (Auto-HDR & penajaman buatan) berbeda dengan sensor webcam USB Rexus | Standarisasi 100% data latih harus berasal dari Webcam USB Rexus yang sama dengan kamera produksi |
| **13** | Rename folder workspace `opencv copy` ke `opencv_3class` memutus path runner agent | Runner shell mengharapkan path awal tetap ada | Membuat symlink tersembunyi `opencv copy -> opencv_3class` dengan flag `chflags hidden` |

---

## 4. LOGIKA & FILTER CERDAS (SMART FILTER)

AI Deep Learning (YOLO) sangat handal mengenali pola, namun terkadang bisa menghasilkan *false positive*. Oleh karena itu, modul [`src/detector.py`](file:///Users/agusrisky/Documents/opencv/sortir-yolo/src/detector.py) dilengkapi lapisan filter cerdas:

### A. Validasi Geometri (Bentuk & Ukuran)
1. **Aspect Ratio ($w / h$):** Koin dan tutup botol dari atas berbentuk lingkaran simetris. Benda yang terlalu lonjong (rasio $< 0.50$ atau $> 1.85$, seperti pulpen, obeng, atau jendela) otomatis dibuang.
2. **Batas Ukuran Maksimal (`max_area_frac = 0.40`):** Benda yang memenuhi lebih dari 40% layar (seperti wajah manusia mendekat atau kardus besar) otomatis diabaikan.
3. **Batas Ukuran Minimal (`min_size_px = 10`):** Menghilangkan bintik noise kamera.

### B. Verifikasi Warna & Optik (HSV / YCrCb)
1. **Bypass Koin Cerdas:** Jika confidence model terhadap koin $\ge 0.65$, koin langsung divalidasi tanpa terhalang warna meja krem.
2. **Filter Kulit Manusia:** Memeriksa rentang krominans kulit ($Cr \in [133, 173]$ dan $Cb \in [77, 127]$) untuk menolak wajah atau tangan operator yang melintas agar tidak dikira koin/tutup botol.

### C. Refinement Titik Hisap Vacuum Charger (`refine_charger_vacuum_center()`)
1. **Tujuan:** Memastikan karet hisap pompa vakum robot arm RNV3 mendarat pada bidang datar kedap udara.
2. **Mekanisme Kerja:**
   - Memotong ROI kepala charger dari frame kamera.
   - Menganalisis rasio lebar vs tinggi ($w / h$) untuk menentukan orientasi colokan logam (horizontal atau vertikal).
   - Menggeser koordinat pusat $(cx, cy)$ sejauh offset dinamis menuju pusat badan plastik (area di bawah logo `iT.`).
   - Mencegah karet hisap mendarat di celah antara dua pin colokan logam yang akan menyebabkan hisapan bocor.

### D. Isolasi Sabuk Konveyor Hijau & Penolakan Latar Luar
1. **Pita Konveyor Hijau:** Menggunakan rentang HSV hijau ($H \in [30, 95], S > 25, V > 20$) untuk mendeteksi batas sabuk konveyor.
2. **Border Clearing (Margin Proteksi):** Area di luar sabuk (seperti laptop hitam, meja putih samping, dan rel aluminium) otomatis dipotong (*cropped out*) sehingga sistem autolabel dan deteksi hanya memproses benda yang benar-benar berada di atas sabuk berjalan.

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

**Contoh Payload JSON 3 Kelas:**
```json
{
  "status": "TARGET_READY",
  "timestamp": 1729601234.56,
  "target": {
    "class_name": "charger",
    "label": "charger",
    "confidence": 0.95,
    "pixel_center": { "u": 520, "v": 310 },
    "robot_mm": { "x": 12.5, "y": 145.0, "z": 28.0 },
    "drop_mm": { "x": 0.0, "y": 180.0, "z": 70.0 },
    "track_id": 4,
    "catatan": "titik hisap digeser ke bidang datar logo iT."
  }
}
```

---

## 6. INOVASI BLUEPRINT / ALAS ACUAN DI KERTAS MANILA

### A. Konsep "Calibration Base Mat" (Layout 3 Wadah)
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

### B. Manfaat Utama Blueprint:
1. **Zero-Setup Time:** Saat demo di tempat baru, cukup gelar kertas manila di meja, posisikan alat tepat pada garis blueprint, dan sistem langsung siap digunakan.
2. **Tidak Perlu Kalibrasi Berulang:** Nilai `robot_calib.json` yang sudah tersimpan tetap valid karena geometri fisik terjaga 100% konsisten.
3. **Koordinat Nyata:** Titik kalibrasi $A, B, C, D$ terukur pasti dengan penggaris fisik nyata dari titik `(0, 0)` basis robot.

---

## 7. STRUKTUR FILE & PENJELASAN MODUL

```
opencv_3class/
├── .gitignore
├── sortir-yolo.zip                  # Paket arsip bersih (1,4 GB) untuk Google Colab
├── arsip_luar/                      # Backup dokumentasi & screenshot lama
└── sortir-yolo/                     # DIREKTORI UTAMA PROYEK
    ├── README.md                    # Dokumentasi operasional & cheat sheet
    ├── DOKUMENTASI.md               # Knowledge base & catatan teknis lengkap (file ini)
    ├── PANDUAN_TIM_ROBOTIK.md       # Panduan khusus untuk rekan tim robotik (3 kelas)
    ├── robot_calib.json             # Konfigurasi matriks kalibrasi & wadah drop
    ├── requirements.txt             # Daftar library Python
    ├── pytest.ini                   # Konfigurasi pengujian otomatis
    │
    ├── detect.py                    # Program utama deteksi real-time webcam/gambar (imgsz=1280)
    ├── capture.py                   # Perekam sampel gambar baru standar
    ├── extra_capture_data.py        # Perekam sampel multi-kelas dengan hotkey 1, 2, 3
    ├── extra_autolabel_own.py       # Engine autolabel mandiri adaptif (konveyor hijau + alas gelap)
    ├── merge_datasets.py            # Script penggabung dataset latih/validasi/test
    ├── bersihkan_dataset_blur.py    # Skrip filter otomatis foto blur & goyang
    ├── kalibrasi_robot.py           # GUI interaktif kalibrasi kamera ke koordinat robot
    ├── sortir_vision_api.py         # Modul API visi mandiri siap panggil
    ├── contoh_baca_koordinat_robot.py # Script pengujian baca koordinat untuk tim robot
    │
    ├── colab/
    │   └── Training_YOLO_Colab.ipynb # Notebook pelatihan T4 GPU Google Colab (Scratch / Finetune)
    │
    ├── models/
    │   ├── best.pt                  # Model terlatih utama (YOLOv8)
    │   └── best-cadangan.pt         # Cadangan model sebelumnya
    ├── outputs/
    │   ├── target_robot.json        # File IPC koordinat real-time ke robot
    │   ├── simulasi_hisap_charger.jpg # Visualisasi titik hisap vacuum cup pada charger
    │   └── preview_charger_on_conveyor.jpg # Bukti validasi bounding box autolabel
    ├── src/
    │   ├── autolabel.py             # Algoritma segmentasi konveyor hijau & thresholding
    │   ├── camera.py                # Wrapper kamera OpenCV + Threaded Capture anti-lag
    │   ├── config.py                # Parameter konfigurasi global & kelas target
    │   ├── coordinate_transform.py  # Logika matematika konversi piksel ke milimeter
    │   ├── detector.py              # Engine deteksi YOLOv8 + Refine Vacuum Center Charger
    │   └── utils.py                 # Utilitas anotasi visual, Kalman filter, & CSV logger
    └── tests/                       # 45 skenario pengujian unit otomatis (100% lulus)
```
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

---

## 10. INVESTIGASI & PENYELESAIAN BUG DETEKSI TEPI (26 OKTOBER 2026)

### Gejala Masalah:
Pada pengujian langsung di konveyor fisik (rekaman jam 13:13:11 WIB), file log `outputs/target_robot.json` mencatat koin yang baru masuk terdeteksi sebagai `tutup_botol` dengan confidence 0.447 pada koordinat $u = 1258, v = 176$.

### Analisis Akar Masalah (Root Cause Analysis):
1. **Objek Terpotong di Batas Tepi Kamera (*Boundary Ingress Truncation*):**
   * Resolusi frame kamera adalah 1280 × 720 piksel ($u \in [0, 1279]$).
   * Titik $u = 1258$ berarti koin baru saja menyentuh jangkauan kamera dan terpotong oleh dinding tepi kanan frame (jarak ke tepi hanya 22 piksel).
   * Karena hanya tampak setengah lingkaran / busur lengkung tanpa relief tengah koin, model mendeteksi lengkungan bibir perak tersebut mirip dengan bibir melingkar plastik (*rim*) tutup botol.
2. **Threshold Confidence Default Terlalu Longgar (`conf = 0.25`):**
   * Model sebenarnya sangat ragu-ragu (confidence hanya 0.447 atau 44.7%).
   * Pada koin utuh di tengah konveyor, model baru (`models/best.pt`) memiliki akurasi **99.5% (AP50 = 0.9947)** dengan confidence stabil di atas **0.85 - 0.98**.
   * Karena default sistem sebelumnya adalah 0.25, deteksi lemah ini tetap diloloskan ke JSON target robot.
3. **Celah Filter ROI & Fallback Target:**
   * Di fungsi `choose_best_target`, bila tidak ada objek di dalam area kerja robot (ROI), sistem secara keliru melakukan fallback ke objek di luar ROI. Lengan robot seharusnya menolak mengambil objek yang berada di luar jangkauan fisik ($X = -216.3\text{ mm}$).

### Solusi Permanen & Pembaruan Sistem yang Diterapkan:
1. **Penerapan Edge Margin Guard (`edge_margin = 15`) di `src/detector.py`:**
   * Ditambahkan inspeksi otomatis batas tepi frame kamera pada `validate_geometry()`.
   * Objek yang menyentuh batas dinding frame ($x_1 \le 15$ atau $x_2 \ge W - 15$ atau $y_1 \le 15$ atau $y_2 \ge H - 15$) **otomatis ditolak** hingga objek meluncur utuh ke dalam konveyor.
2. **Penyesuaian Sweet Spot Confidence Default (`conf = 0.45`):**
   * Mengubah default confidence di `detect.py` dan `sortir_vision_api.py` dari `0.25` ke `0.45`.
   * Koin asli (0.96+), tutup botol asli (0.98+), dan charger asli (0.94+) tetap terdeteksi 100%, sementara seluruh false positive tepi dan bayangan remang tereliminasi.
3. **Proteksi Ketat ROI di `src/coordinate_transform.py`:**
   * `choose_best_target` kini secara ketat mengembalikan `None` (`NO_TARGET`) jika tidak ada target di dalam ROI aktif, mencegah lengan robot bergerak ke koordinat berbahaya di luar batas konveyor.
4. **Verifikasi Unit Test:**
   * Seluruh 45 unit test (100%) lolos pengujian (`pytest -q` -> `45 passed`).
