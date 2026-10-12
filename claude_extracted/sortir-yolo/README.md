# Deteksi Koin / Tutup Botol (YOLOv8) — satu file

Tunjukkan benda ke kamera → kotak merah + judul (**Koin** / **Tutup Botol**) + titik tengah (cx, cy).
Ini tahap vision untuk proyek conveyor + robot lengan vakum.

Seluruh program ada di **`sortir.py`** (sebelumnya 18 file Python). File lain: `requirements.txt`,
`test_sortir.py`, `Training_YOLO_Colab.ipynb`, dan README ini.

## Akar masalah (kenapa tutup botol biru terbaca "Koin 89%")

1. **Latar = kelas (shortcut learning).** Dataset koin berisi foto koin di berbagai latar nyata (tangan, meja).
   Dataset tutup botol bertipe *klasifikasi*: foto close-up di latar polos, kotaknya dibuat otomatis, dan bendanya
   sering memenuhi seluruh gambar. Model akhirnya belajar "latar ramai/tangan = koin, latar polos & besar = tutup botol",
   bukan melihat bentuk dan warna benda. Di kamera kamu (tangan + dinding), tutup botol jatuh ke kelas koin
   dengan keyakinan tinggi — jadi menaikkan `--conf` tidak menolong.
2. **Tidak ada satu gambar pun yang berisi koin DAN tutup botol bersamaan**, dan tidak ada contoh negatif
   (tangan/meja kosong), sehingga model tidak pernah dipaksa membedakan keduanya.
3. **Label dataset kotor:** koin berlabel ganda (`500` dan `KOIN-RUPIAH-MVPR` untuk koin yang sama),
   dan kotak otomatis tutup botol tidak diperiksa kewajarannya.
4. **Pasca-proses lemah:** NMS per kelas (satu benda bisa dapat dua label) dan keputusan per-frame (label berkedip).
5. Training bawaan memakai `yolov8n` dan augmentasi default.

## Perbaikan di versi ini

| Masalah | Perbaikan |
|---|---|
| Latar = kelas | **Data sintetis** (`gabung`): koin & tutup botol dipotong lalu ditempel ke SEMUA jenis latar (latar kosong milikmu, latar dataset yang bendanya dihapus, permukaan polos), 1–4 benda per gambar, kedua kelas bisa dalam satu gambar. Hanya masuk ke *train*; valid tetap data asli. |
| Tidak ada negatif | Foto **latar kosong** (`foto`, tombol `0`) dipakai sebagai contoh negatif dan sebagai latar sintetis. |
| Label kotor | Kotak ganda dibuang; kotak otomatis yang tidak bulat/terlalu besar ditolak (`kotak_wajar`). |
| Label foto sendiri di latar ramai | `label sendiri --metode model`: `best.pt` lama hanya dipakai mencari **lokasi** benda; nama kelas diambil dari nama folder. |
| NMS per kelas | `agnostic_nms=True`: satu benda = satu kotak = satu label. |
| Label berkedip | **Penstabil**: suara terbanyak beberapa frame terakhir per benda (koordinat tidak diubah). |
| Salah yakin | **Pemeriksa warna**: benda yang bagian tengahnya jelas berwarna pekat (biru/merah/hijau/ungu/pink) tidak mungkin koin logam → dikoreksi jadi Tutup Botol. Kuning/oranye tidak dikoreksi (mirip koin emas). Hanya mengubah Koin→Tutup Botol, tidak pernah sebaliknya. |
| Model kecil | Default training `yolov8s`, 100 epoch, augmentasi disesuaikan (flip atas-bawah, warna tidak digeser banyak). |

Persentase tidak lagi tampil di judul (tambah `--persen` jika mau). Confidence tetap tercetak di terminal.

## Alur

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python sortir.py unduh                  # dataset Roboflow (butuh ROBOFLOW_API_KEY, lihat bawah)
python sortir.py foto                   # SANGAT disarankan, lihat "Cara ambil foto"
python sortir.py label tutup-botol      # kotak otomatis; CEK outputs/preview_autolabel_tutup_botol.jpg
python sortir.py label sendiri          # kotak otomatis foto sendiri; CEK outputs/preview_autolabel_foto_sendiri.jpg
python sortir.py gabung                 # gabung + sintetis; CEK outputs/preview_sintetis.jpg
python sortir.py latih                  # training -> models/best.pt (di laptop lambat: pakai Colab)
python sortir.py deteksi                # webcam
```

Di Colab cukup jalankan `Training_YOLO_Colab.ipynb` dari atas ke bawah.

### Cara ambil foto (`python sortir.py foto`)
Tombol: `1` koin, `2` tutup botol, `0` **latar kosong**, SPASI foto, `a` otomatis, `q` selesai.
- Pakai kamera dan latar yang sama dengan kondisi akhir (tangan, dinding, atau conveyor).
- Koin/tutup botol: **1 benda per foto**, 80–150 foto per kelas. Variasikan posisi, jarak, cahaya, putaran.
  Tutup botol: ambil berbagai warna (biru, merah, putih, hitam, hijau).
- Latar kosong: 60–120 foto tangan/meja/conveyor **tanpa benda** (tangan terbuka, tangan memegang kosong, dll).
- Latar polos → `label sendiri` (metode `cv`, default sudah cukup). Latar ramai (tangan + dinding) →
  `label sendiri --metode model` (pakai `best.pt` lama) atau `--metode world` (YOLO-World, di Colab).

### Deteksi
```bash
python sortir.py deteksi                               # webcam 0
python sortir.py deteksi --source 1 --conf 0.4         # kamera lain / threshold lain
python sortir.py deteksi --source foto.jpg             # satu gambar (hasil di outputs/)
python sortir.py deteksi --persen                      # tampilkan persentase
python sortir.py deteksi --tanpa-cek-warna --tanpa-stabil   # matikan kedua pasca-proses (untuk uji model murni)
```
Tombol: `q`/`ESC` keluar, `s` screenshot, `c` simpan koordinat ke `outputs/koordinat.csv`.

### API key Roboflow
Login di app.roboflow.com → Settings → API Keys → salin *Private API Key*. Simpan sebagai variabel
`ROBOFLOW_API_KEY`, atau file `.env` berisi `ROBOFLOW_API_KEY=kunci`, atau Colab Secrets. Jangan di-upload ke GitHub.

## Pemetaan dari versi lama

| Lama | Baru |
|---|---|
| `01_download_datasets.py` | `sortir.py unduh` |
| `extra_capture_data.py` | `sortir.py foto` |
| `02_autolabel_tutup_botol.py` / `extra_autolabel_own.py` | `sortir.py label tutup-botol` / `label sendiri` |
| `03_merge_datasets.py` | `sortir.py gabung` |
| `04_train.py` | `sortir.py latih` |
| `05_detect.py` | `sortir.py deteksi` |
| `src/*`, `tests/*` | di dalam `sortir.py` dan `test_sortir.py` |

## Masalah umum
- **Kamera tidak terbuka (Mac):** beri izin kamera ke Terminal/VS Code di System Settings → Privacy & Security → Camera; coba `--source 1`.
- **Masih tertukar:** cek `outputs/preview_*.jpg` (kotak harus pas), tambah foto sendiri di kondisi nyata, lalu `gabung` dan `latih` ulang.
- **Terlalu lambat (FPS rendah):** `python sortir.py latih --model yolov8n.pt`.
- **CUDA out of memory:** `--batch 8`.
- **Tes:** `python -m pytest -q`.
