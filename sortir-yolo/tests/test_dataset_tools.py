from pathlib import Path

import cv2
import numpy as np
import yaml

from src.dataset_tools import (Source, convert_label_line, find_split_dirs, group_key,
                               merge_datasets, read_class_names)

CLASSES = ["koin", "korek_api", "tutup_botol"]


def test_convert_label_line_kotak_dan_clamp():
    assert convert_label_line("5 0.5 0.5 0.2 0.4", 0) == "0 0.500000 0.500000 0.200000 0.400000"
    # kotak melebihi tepi -> dipotong ke 0..1
    out = convert_label_line("1 0.95 0.5 0.3 0.2", 2).split()
    assert out[0] == "2" and abs(float(out[3]) - 0.2) < 1e-6  # lebar 0.3 -> terpotong jadi 0.2


def test_convert_label_line_poligon_jadi_kotak():
    out = convert_label_line("0 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8", 1).split()
    assert out[0] == "1"
    assert [round(float(v), 3) for v in out[1:]] == [0.4, 0.5, 0.4, 0.6]


def test_convert_label_line_rusak():
    assert convert_label_line("", 0) is None
    assert convert_label_line("0 0.5 0.5", 0) is None
    assert convert_label_line("0 a b c d", 0) is None
    assert convert_label_line("0 0.5 0.5 0 0.2", 0) is None  # lebar nol


def test_group_key_roboflow():
    assert group_key("IMG_1_jpg.rf.abc123.jpg") == "IMG_1_jpg"
    assert group_key("IMG_1_jpg.rf.zzz999.jpg") == "IMG_1_jpg"
    assert group_key("foto.jpg") == "foto"


def make_dataset(root: Path, names, splits: dict[str, list[tuple[str, list[str]]]]):
    """splits = {'train': [(nama_file, [baris_label,...]), ...]}"""
    for split, items in splits.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for fname, lines in items:
            cv2.imwrite(str(root / split / "images" / fname), np.full((32, 32, 3), 128, np.uint8))
            (root / split / "labels" / (Path(fname).stem + ".txt")).write_text("\n".join(lines))
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}))


def test_find_split_dirs_dan_nama_kelas(tmp_path):
    make_dataset(tmp_path, ["a", "b"], {"train": [("x.jpg", [])], "val": [("y.jpg", [])]})
    assert set(find_split_dirs(tmp_path)) == {"train", "valid"}
    assert read_class_names(tmp_path) == ["a", "b"]


def test_merge_semua_kelas_asli_jadi_satu_kelas(tmp_path):
    koin = tmp_path / "koin"
    make_dataset(koin, ["100", "200", "500", "KOIN-RUPIAH-MVPR"], {
        "train": [("a_jpg.rf.1.jpg", ["0 0.5 0.5 0.2 0.2"]), ("b_jpg.rf.1.jpg", ["2 0.4 0.4 0.2 0.2", "3 0.8 0.8 0.1 0.1"])],
        "valid": [("c_jpg.rf.1.jpg", ["1 0.5 0.5 0.2 0.2"])],
    })
    korek = tmp_path / "korek"
    make_dataset(korek, ["korek"], {"train": [("k1.jpg", ["0 0.5 0.5 0.3 0.3"])], "valid": [("k2.jpg", ["0 0.5 0.5 0.3 0.3"])]})
    out = tmp_path / "gabungan"
    report = merge_datasets([Source("koin", koin, "koin"), Source("korek_api", korek, "korek_api")], out, CLASSES)

    assert report.instances["train"]["koin"] == 3
    assert report.instances["valid"]["koin"] == 1
    assert report.instances["train"]["korek_api"] == 1
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["names"] == {0: "koin", 1: "korek_api", 2: "tutup_botol"}
    assert (out / "train" / "images" / "koin__a_jpg.rf.1.jpg").is_file()
    # semua ID label harus valid (0..2)
    for lbl in out.rglob("*.txt"):
        for line in lbl.read_text().splitlines():
            assert int(line.split()[0]) in (0, 1, 2)
    # kelas tutup_botol kosong -> harus ada peringatan
    assert any("tutup_botol" in w for w in report.warnings)


def test_merge_membuang_kebocoran_data(tmp_path):
    ds = tmp_path / "ds"
    make_dataset(ds, ["x"], {
        "train": [("foto1_jpg.rf.AAA.jpg", ["0 0.5 0.5 0.2 0.2"]),
                  ("foto1_jpg.rf.BBB.jpg", ["0 0.5 0.5 0.2 0.2"]),   # turunan foto1, ikut bocor
                  ("foto2_jpg.rf.CCC.jpg", ["0 0.5 0.5 0.2 0.2"])],
        "valid": [("foto1_jpg.rf.DDD.jpg", ["0 0.5 0.5 0.2 0.2"])],
    })
    out = tmp_path / "out"
    report = merge_datasets([Source("ds", ds, "koin")], out, CLASSES)
    assert report.leak_dropped == 2
    assert report.images["train"]["ds"] == 1 and report.images["valid"]["ds"] == 1

    out2 = tmp_path / "out2"
    report2 = merge_datasets([Source("ds", ds, "koin")], out2, CLASSES, drop_leak=False)
    assert report2.images["train"]["ds"] == 3


def test_merge_dataset_tanpa_valid_dibuatkan_valid(tmp_path):
    ds = tmp_path / "ds"
    items = [(f"f{i}.jpg", ["0 0.5 0.5 0.2 0.2"]) for i in range(20)]
    make_dataset(ds, ["x"], {"train": items})
    report = merge_datasets([Source("ds", ds, "korek_api")], tmp_path / "out", CLASSES)
    assert report.images["valid"]["ds"] >= 1
    assert report.images["train"]["ds"] + report.images["valid"]["ds"] == 20


def test_merge_kelas_dicocokkan_lewat_nama(tmp_path):
    ds = tmp_path / "own"
    make_dataset(ds, CLASSES, {"train": [("a.jpg", ["2 0.5 0.5 0.2 0.2"])], "valid": [("b.jpg", ["1 0.5 0.5 0.2 0.2"])]})
    report = merge_datasets([Source("own", ds, None)], tmp_path / "out", CLASSES)
    assert report.instances["train"]["tutup_botol"] == 1
    assert report.instances["valid"]["korek_api"] == 1
