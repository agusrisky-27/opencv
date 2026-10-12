# Deteksi Koin / Korek Api / Tutup Botol (YOLOv8)

Program deteksi real-time: tunjukkan benda ke kamera, lalu muncul **kotak merah** dengan **judul**
(`Koin`, `Korek Api`, atau `Tutup Botol`) dan **titik tengah (cx, cy)** dihitung otomatis.
Ini tahap vision untuk proyek conveyor + robot lengan vakum.

```
Dataset Roboflow --(01 unduh)--> --(02 beri kotak tutup botol)--> --(03 gabung 3 kelas)-->
--(04 training di Colab)--> models/best.pt --(05 deteksi webcam)--> kotak merah + titik tengah
```

## Penting: kondisi 3 dataset (sudah dicek di Roboflow Universe)

| Dataset | Tipe | Isi | Konsekuensi |
|---|---|---|---|
| `arfan-faris-pramatya/koin-rupiah-mvpr` | Object Detection | 905 gambar, kelas `100`, `200`, `500`, `KOIN-RUPIAH-MVPR` | Siap pakai. Semua kelas digabung jadi `koin`. |
| `korek-api-pwx7k/deteksi-korek-api-mf1ww` | Object Detection | 50 gambar (foto stok korek gas, latar putih) | **Belum punya versi** -> tidak bisa diunduh lewat API sebelum di-fork & di-generate. Datanya sedikit. |
| `bottlecapsorting/tutup-botol` | **Classification** | 1678 gambar, 14 kelas (warna tutup) | **Tidak punya kotak**, jadi tidak bisa langsung dipakai YOLO deteksi. Program ini membuat kotak otomatis (langkah 2). |

Akibatnya: hasil dari dataset ini saja **belum tentu akurat di kamera kamu** (latar, sudut, dan cahaya
berbeda; korek hanya 50 foto stok). Untuk hasil yang bagus, tambahkan **foto sendiri** (langkah EXTRA).

## Cara mendapatkan API key Roboflow

1. Buka https://app.roboflow.com dan login (gratis).
2. Masuk **Settings** workspace > **API Keys** (kadang bernama *Roboflow API*).
3. Salin **Private API Key**. Jangan dibagikan / jangan di-upload ke GitHub.
4. Simpan:
   - Lokal: salin `.env.example` jadi `.env`, isi `ROBOFLOW_API_KEY=kunci_kamu`
   - Colab: menu **Secrets** (ikon kunci) > tambah `ROBOFLOW_API_KEY` > aktifkan *Notebook access*

## Cara pakai (urutan)

Training dilakukan di **Google Colab** (lihat `colab/Training_YOLO_Colab.ipynb`, semua langkah sudah ada di sana).
Untuk menjalankan di laptop:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python 01_download_datasets.py        # unduh lewat API
python 02_autolabel_tutup_botol.py    # beri kotak otomatis (cek outputs/preview_autolabel_tutup_botol.jpg!)
python 03_merge_datasets.py           # gabung jadi 3 kelas + laporan
python 04_train.py                    # training (di laptop lambat -> pakai Colab)
python 05_detect.py                   # deteksi webcam dengan models/best.pt
```

### Dataset korek api (sekali saja)
1. Buka halaman dataset korek di Universe > **Fork Dataset**.
2. Di workspace kamu buka project hasil fork > **Generate** (buat versi).
3. **Download Dataset** > format **YOLOv8** > unduh zip.
4. `python 01_download_datasets.py --only korek_api --zip korek_api=path/ke/korek_api.zip`

### Deteksi (`05_detect.py`)
```bash
python 05_detect.py                          # webcam 0
python 05_detect.py --source 1 --conf 0.4    # kamera lain / threshold lain
python 05_detect.py --source foto.jpg        # satu gambar (hasil di outputs/)
```
Tombol: `q`/`ESC` keluar, `s` screenshot, `c` simpan koordinat ke `outputs/koordinat.csv`.
Terminal mencetak: `[12:30:01] Koin | conf=0.92 | center=(320, 240) | center_norm=(0.5, 0.5) | bbox=(...)`.

## EXTRA: tambah foto sendiri (sangat disarankan)

```bash
python extra_capture_data.py     # tombol 1/2/3 pilih kelas, SPASI foto, a = otomatis
python extra_autolabel_own.py    # beri kotak otomatis (latar polos), cek preview-nya
python 03_merge_datasets.py      # gabung ulang, lalu training lagi
```
Tips: ambil foto dengan kamera & latar yang sama dengan kondisi akhir (di atas conveyor), 1 benda per foto,
variasikan posisi/putaran/jarak/cahaya, target 100-200 foto per kelas.

## Struktur

```
src/config.py            nama kelas, label tampilan, sumber dataset  (ubah di sini)
src/detector.py          Detector.detect(frame) -> list Detection (kelas, conf, bbox, center)
src/utils.py             hitung titik tengah, gambar kotak merah/judul, simpan CSV/screenshot
src/camera.py            pembungkus kamera + pesan error jelas
src/dataset_tools.py     gabung dataset, konversi label, cek kebocoran data
src/autolabel.py         kotak otomatis (metode 'cv' dan YOLO-World)
tests/                   unit test:  python -m pytest -q
```

## Masalah umum

- **Kamera tidak terbuka (Mac):** beri izin kamera ke Terminal/VS Code di System Settings > Privacy & Security > Camera; coba `--source 1`.
- **Tidak terdeteksi / salah deteksi:** turunkan/naikkan `--conf`; tambah foto sendiri; cek confusion matrix hasil training.
- **CUDA out of memory:** `--batch 8`.
- **Nama kelas model beda:** model lama pakai kelas lain? Latih ulang dengan program ini; label tampilan diatur di `src/config.py`.
