"""Alat dataset format YOLO: baca split, konversi label, gabung beberapa dataset,
dan cek kebocoran data (gambar yang sama muncul di train DAN valid/test)."""
from __future__ import annotations

import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import IMAGE_EXTS

SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("valid", "val", "validation"),
    "test": ("test",),
}


# ------------------------------------------------------------- baca dataset
def find_split_dirs(dataset_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Cari folder images/labels tiap split. Return {'train': (images, labels), ...}."""
    found: dict[str, tuple[Path, Path]] = {}
    for split, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            images = dataset_dir / alias / "images"
            if images.is_dir():
                found[split] = (images, dataset_dir / alias / "labels")
                break
    return found


def read_class_names(dataset_dir: Path) -> list[str]:
    """Baca daftar nama kelas dari data.yaml (format list maupun dict)."""
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.is_file():
        return []
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    return [str(n) for n in names]


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def group_key(filename: str) -> str:
    """Kunci 'foto asli'. Roboflow menamai file hasil augmentasi `<nama>_jpg.rf.<hash>.jpg`,
    jadi semua turunan dari satu foto asli punya awalan yang sama sebelum '.rf.'."""
    name = Path(filename).name
    if ".rf." in name:
        return name.split(".rf.")[0]
    return Path(name).stem


# ----------------------------------------------------------- konversi label
def convert_label_line(line: str, target_id: int) -> str | None:
    """Ubah satu baris label YOLO menjadi 'target_id cx cy w h' (dinormalisasi & di-clamp).

    Mendukung baris kotak (5 nilai) dan poligon segmentasi (diubah jadi kotak pembungkus).
    Return None kalau baris rusak / kotak kosong.
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        nums = [float(p) for p in parts[1:]]
    except ValueError:
        return None

    if len(nums) == 4:
        cx, cy, w, h = nums
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    elif len(nums) >= 6 and len(nums) % 2 == 0:  # poligon
        xs, ys = nums[0::2], nums[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    else:
        return None

    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    w, h = x2 - x1, y2 - y1
    if w <= 1e-4 or h <= 1e-4:
        return None
    return f"{target_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}"


# -------------------------------------------------------------- gabung data
@dataclass
class Source:
    key: str                      # nama singkat, jadi awalan nama file
    path: Path                    # folder dataset format YOLO
    target_class: str | None      # semua kelas asli -> 1 kelas final; None = cocokkan lewat nama


@dataclass
class _Entry:
    key: str
    split: str
    image: Path
    lines: list[str]


@dataclass
class MergeReport:
    instances: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    images: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    leak_dropped: int = 0
    skipped_lines: int = 0
    warnings: list[str] = field(default_factory=list)


def _holdout_valid(entries: list[_Entry], key: str, fraction: float, seed: int) -> None:
    """Dataset tanpa folder valid: ambil sebagian 'foto asli' (bukan gambar acak) dari train."""
    train = [e for e in entries if e.key == key and e.split == "train"]
    groups = sorted({group_key(e.image.name) for e in train})
    if len(groups) < 5:
        return
    rng = random.Random(seed)
    rng.shuffle(groups)
    held = set(groups[: max(1, int(len(groups) * fraction))])
    for e in train:
        if group_key(e.image.name) in held:
            e.split = "valid"


def merge_datasets(
    sources: list[Source],
    out_dir: Path,
    class_names: list[str],
    seed: int = 0,
    drop_leak: bool = True,
    valid_fraction: float = 0.15,
) -> MergeReport:
    """Gabungkan beberapa dataset YOLO menjadi satu dataset dengan kelas final `class_names`."""
    report = MergeReport()
    entries: list[_Entry] = []

    for src in sources:
        splits = find_split_dirs(src.path)
        if not splits:
            report.warnings.append(f"[{src.key}] tidak ada folder train/valid/test di {src.path} -> dilewati")
            continue
        src_names = read_class_names(src.path)

        for split, (img_dir, lbl_dir) in splits.items():
            for img in list_images(img_dir):
                lines: list[str] = []
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.is_file():
                    for raw in lbl.read_text(encoding="utf-8").splitlines():
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            src_id = int(float(raw.split()[0]))
                        except (ValueError, IndexError):
                            report.skipped_lines += 1
                            continue
                        if src.target_class is not None:
                            target = src.target_class
                        else:
                            target = src_names[src_id] if 0 <= src_id < len(src_names) else None
                        if target not in class_names:
                            report.skipped_lines += 1
                            continue
                        converted = convert_label_line(raw, class_names.index(target))
                        if converted is None:
                            report.skipped_lines += 1
                            continue
                        lines.append(converted)
                entries.append(_Entry(src.key, split, img, lines))

        if "valid" not in splits and "train" in splits:
            _holdout_valid(entries, src.key, valid_fraction, seed)
            report.warnings.append(
                f"[{src.key}] tidak punya split valid -> {int(valid_fraction * 100)}% foto asli diambil dari train"
            )

    # --- cek kebocoran: foto asli yang sama ada di train DAN valid/test
    held_groups = {(e.key, group_key(e.image.name)) for e in entries if e.split in ("valid", "test")}
    leaked = [e for e in entries if e.split == "train" and (e.key, group_key(e.image.name)) in held_groups]
    if leaked:
        if drop_leak:
            leaked_ids = {id(e) for e in leaked}
            entries = [e for e in entries if id(e) not in leaked_ids]
            report.leak_dropped = len(leaked)
            report.warnings.append(
                f"Kebocoran data: {len(leaked)} gambar train dibuang karena foto aslinya ada di valid/test"
            )
        else:
            report.warnings.append(f"PERINGATAN kebocoran data: {len(leaked)} gambar train sefoto dengan valid/test")

    # --- tulis hasil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "valid", "test"):
        if any(e.split == split for e in entries):
            (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    import hashlib

    for e in entries:
        raw_stem = e.image.stem
        ext = e.image.suffix
        # Batasi panjang nama file agar aman dari batas 255 karakter OS
        max_stem_len = 160 - len(e.key) - len(ext) - 2
        if len(raw_stem) > max_stem_len:
            h = hashlib.md5(raw_stem.encode("utf-8")).hexdigest()[:8]
            raw_stem = f"{raw_stem[:max_stem_len - 9]}_{h}"
        new_name = f"{e.key}__{raw_stem}{ext}"
        stem = Path(new_name).stem
        shutil.copy2(e.image, out_dir / e.split / "images" / new_name)
        (out_dir / e.split / "labels" / f"{stem}.txt").write_text(
            "\n".join(e.lines) + ("\n" if e.lines else ""), encoding="utf-8"
        )
        report.images[e.split][e.key] += 1
        for line in e.lines:
            report.instances[e.split][class_names[int(line.split()[0])]] += 1

    write_data_yaml(out_dir, class_names, has_test=any(e.split == "test" for e in entries))
    report.warnings.extend(balance_warnings(report, class_names))
    return report


def write_data_yaml(out_dir: Path, class_names: list[str], has_test: bool = False) -> Path:
    cfg: dict = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    if has_test:
        cfg["test"] = "test/images"
    path = out_dir / "data.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def balance_warnings(report: MergeReport, class_names: list[str]) -> list[str]:
    """Peringatan kalau ada kelas yang datanya terlalu sedikit / tidak seimbang."""
    msgs: list[str] = []
    train = report.instances.get("train", {})
    valid = report.instances.get("valid", {})
    for name in class_names:
        n_train, n_valid = train.get(name, 0), valid.get(name, 0)
        if n_train == 0:
            msgs.append(f"Kelas '{name}' TIDAK punya data train sama sekali -> tidak akan terdeteksi.")
        elif n_train < 100:
            msgs.append(f"Kelas '{name}' hanya {n_train} instance di train (disarankan >= 100-300). "
                        "Tambahkan foto sendiri (extra_capture_data.py).")
        if n_train > 0 and n_valid == 0:
            msgs.append(f"Kelas '{name}' tidak punya data valid -> metrik kelas ini tidak bisa dihitung.")
    counts = [train.get(n, 0) for n in class_names if train.get(n, 0) > 0]
    if len(counts) >= 2 and max(counts) / min(counts) > 5:
        msgs.append(f"Data tidak seimbang (rasio terbesar:terkecil = {max(counts) / min(counts):.1f}x). "
                    "Kelas dengan data sedikit cenderung kurang akurat.")
    return msgs


def format_report(report: MergeReport, class_names: list[str]) -> str:
    """Tabel teks ringkas untuk dicetak di terminal."""
    lines = ["", "Jumlah INSTANCE (kotak) per kelas:", f"{'kelas':<14}{'train':>8}{'valid':>8}{'test':>8}"]
    for name in class_names:
        row = [report.instances.get(s, {}).get(name, 0) for s in ("train", "valid", "test")]
        lines.append(f"{name:<14}{row[0]:>8}{row[1]:>8}{row[2]:>8}")
    lines.append("")
    lines.append("Jumlah GAMBAR per split / sumber:")
    for split in ("train", "valid", "test"):
        if report.images.get(split):
            detail = ", ".join(f"{k}={v}" for k, v in sorted(report.images[split].items()))
            lines.append(f"  {split:<6} total={sum(report.images[split].values())}  ({detail})")
    if report.skipped_lines:
        lines.append(f"\nBaris label rusak/tidak dikenal yang dilewati: {report.skipped_lines}")
    if report.warnings:
        lines.append("\nCATATAN:")
        lines.extend(f"  - {w}" for w in report.warnings)
    return "\n".join(lines)
