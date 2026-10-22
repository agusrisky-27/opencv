#!/usr/bin/env python3
"""Langkah 1 - Unduh dataset dari Roboflow Universe lewat API.

Contoh:
    python 01_download_datasets.py                  # unduh semua
    python 01_download_datasets.py --only koin      # satu dataset saja
    python 01_download_datasets.py --zip koin=/path/koin.zip         # pakai zip hasil unduh manual

API key dibaca dari: variabel lingkungan ROBOFLOW_API_KEY, file .env, Colab Secrets,
atau ditanya langsung kalau dijalankan di terminal.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

from src.config import DATASETS, IMAGE_EXTS, RAW_DIR, ROOT
from src.dataset_tools import find_split_dirs, read_class_names

API_KEY_HELP = """
API key Roboflow belum ditemukan. Cara mendapatkannya:
  1. Login di https://app.roboflow.com (gratis)
  2. Buka Settings workspace > menu "API Keys" (atau "Roboflow API")
  3. Salin "Private API Key"
Lalu pilih salah satu:
  - Buat file .env di folder proyek berisi:  ROBOFLOW_API_KEY=kunci_kamu
  - atau jalankan:  export ROBOFLOW_API_KEY=kunci_kamu   (Mac/Linux)
  - atau di Colab: simpan di menu Secrets (ikon kunci) dengan nama ROBOFLOW_API_KEY
"""


def get_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    env_file = ROOT / ".env"
    if not key and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        try:  # Google Colab Secrets
            from google.colab import userdata  # type: ignore

            key = (userdata.get("ROBOFLOW_API_KEY") or "").strip()
        except Exception:
            pass
    if not key and sys.stdin.isatty():
        import getpass

        key = getpass.getpass("Tempel API key Roboflow (tidak akan terlihat): ").strip()
    if not key or key.startswith("isi_api_key"):
        raise SystemExit(API_KEY_HELP)
    return key


def has_data(folder: Path) -> bool:
    """True kalau folder ada dan berisi minimal satu gambar."""
    if not folder.is_dir():
        return False
    return any(p.suffix.lower() in IMAGE_EXTS for p in folder.rglob("*"))


def latest_version(project) -> int | None:
    """Nomor versi terbesar dari sebuah project (None kalau project belum punya versi)."""
    try:
        versions = project.versions()
    except Exception:
        return None
    numbers: list[int] = []
    for v in versions or []:
        for attr in ("version", "id", "name"):
            raw = getattr(v, attr, None)
            m = re.search(r"(\d+)$", str(raw).rstrip("/").split("/")[-1]) if raw is not None else None
            if m:
                numbers.append(int(m.group(1)))
                break
    return max(numbers) if numbers else None


def extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    # kalau isinya terbungkus satu folder, naikkan satu tingkat
    children = [c for c in dest.iterdir() if c.is_dir()]
    if not (dest / "train").exists() and len(children) == 1 and (children[0] / "train").exists():
        for item in children[0].iterdir():
            shutil.move(str(item), str(dest / item.name))
        children[0].rmdir()


def download_one(rf, spec: dict, dest: Path, force: bool) -> None:
    if has_data(dest) and not force:
        print(f"  sudah ada di {dest} (pakai --force untuk unduh ulang)")
        return
    if dest.exists():
        shutil.rmtree(dest)
    project = rf.workspace(spec["workspace"]).project(spec["project"])
    version_no = spec["version"] or latest_version(project)
    if version_no is None:
        raise RuntimeError(
            "project ini belum punya versi yang bisa diunduh lewat API.\n"
            "    Solusi: buka halaman Universe-nya > klik 'Fork Dataset' > di workspace kamu klik 'Generate'\n"
            "    (buat versi) > 'Download Dataset' > pilih format YOLOv8 > unduh zip, lalu jalankan:\n"
            f"    python 01_download_datasets.py --zip {spec['key']}=<path/zip>"
        )
    fmt = "yolov8" if spec["kind"] == "detection" else "folder"
    print(f"  versi {version_no}, format {fmt} -> {dest}")
    project.version(version_no).download(fmt, location=str(dest))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=[d["key"] for d in DATASETS], help="unduh satu dataset saja")
    ap.add_argument("--force", action="store_true", help="unduh ulang walau sudah ada")
    ap.add_argument("--zip", action="append", default=[], metavar="KEY=PATH",
                    help="ekstrak zip hasil unduh manual ke datasets_raw/KEY (boleh berulang)")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for item in args.zip:
        key, _, path = item.partition("=")
        if key not in {d["key"] for d in DATASETS} or not Path(path).is_file():
            print(f"[zip] lewati '{item}': key harus salah satu dari {[d['key'] for d in DATASETS]} dan file harus ada")
            continue
        dest = RAW_DIR / key
        if dest.exists():
            shutil.rmtree(dest)
        extract_zip(Path(path), dest)
        print(f"[zip] {key}: diekstrak ke {dest}")

    specs = [d for d in DATASETS if not args.only or d["key"] == args.only]
    pending = [d for d in specs if not has_data(RAW_DIR / d["key"]) or args.force]
    rf = None
    if pending:
        from roboflow import Roboflow

        rf = Roboflow(api_key=get_api_key())

    failed: list[str] = []
    for spec in specs:
        dest = RAW_DIR / spec["key"]
        print(f"\n[{spec['key']}] {spec['workspace']}/{spec['project']} ({spec['kind']})")
        try:
            if rf is not None and (spec in pending):
                download_one(rf, spec, dest, args.force)
            else:
                print(f"  sudah ada di {dest}")
            if spec["kind"] == "detection":
                names = read_class_names(dest)
                splits = sorted(find_split_dirs(dest))
                print(f"  kelas asli: {names} -> semua digabung jadi '{spec['target_class']}'; split: {splits}")
        except Exception as exc:  # satu dataset gagal tidak menghentikan yang lain
            failed.append(spec["key"])
            print(f"  GAGAL: {exc}")

    print("\n" + ("Selesai tanpa error." if not failed else f"Selesai, tetapi gagal: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
