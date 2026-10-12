#!/usr/bin/env python3
"""Langkah 4 - Training YOLOv8 pada dataset gabungan (jalankan di Google Colab dengan GPU).

    python 04_train.py                                  # default: yolov8n, 80 epoch
    python 04_train.py --epochs 120 --model yolov8s.pt
    python 04_train.py --project /content/drive/MyDrive/yolo_runs   # simpan hasil ke Drive
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.config import CLASS_NAMES, MERGED_DIR, MODELS_DIR, RUNS_DIR
from src.utils import pick_device


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=MERGED_DIR / "data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt (cepat) / yolov8s.pt (lebih akurat)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="turunkan ke 8 kalau CUDA out of memory")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--project", type=Path, default=RUNS_DIR)
    ap.add_argument("--name", default="sortir")
    ap.add_argument("--patience", type=int, default=25, help="berhenti dini kalau tidak membaik")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="lanjutkan training yang terputus (last.pt)")
    args = ap.parse_args()

    if not args.data.is_file():
        print(f"{args.data} tidak ditemukan. Jalankan dulu: python 03_merge_datasets.py")
        return 1

    from ultralytics import YOLO

    device = pick_device(args.device)
    print(f"Device: {device} | data: {args.data}")
    if device == "cpu":
        print("PERINGATAN: training di CPU sangat lambat. Pakai Google Colab (Runtime > T4 GPU).")

    project = args.project.resolve()  # path absolut -> tidak ada folder 'runs/detect/runs/detect'
    if args.resume:
        last = project / args.name / "weights" / "last.pt"
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=device, project=str(project), name=args.name, exist_ok=True,
            patience=args.patience, workers=args.workers, save_period=10, plots=True,
        )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        print("best.pt tidak ditemukan, training mungkin gagal.")
        return 1

    print(f"\nEvaluasi model terbaik: {best}")
    metrics = YOLO(str(best)).val(data=str(args.data), device=device, plots=True)
    print(f"\nmAP50 = {metrics.box.map50:.3f} | mAP50-95 = {metrics.box.map:.3f}")
    print(f"{'kelas':<14}{'precision':>10}{'recall':>9}{'mAP50':>8}{'mAP50-95':>10}")
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"{CLASS_NAMES[int(cls_idx)]:<14}{p:>10.3f}{r:>9.3f}{ap50:>8.3f}{ap:>10.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, MODELS_DIR / "best.pt")
    print(f"\nModel disalin ke: {MODELS_DIR / 'best.pt'}")
    print("Metrik tinggi di data valid BUKAN jaminan akurat di kamera nyata -> tes langsung dengan 05_detect.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
