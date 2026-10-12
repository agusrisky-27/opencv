"""Unit test. Jalankan:  python -m pytest -q"""
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import sortir as S


# ------------------------------------------------------------------ bantu
def bgr_hsv(h_deg, s, v):
    px = np.uint8([[[h_deg / 2, s * 255, v * 255]]])
    return tuple(int(x) for x in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


def frame_dengan_bulatan(warna, bbox=(200, 150, 320, 270), bg=(150, 160, 170)):
    f = np.full((480, 640, 3), bg, np.uint8)
    x1, y1, x2, y2 = bbox
    cv2.circle(f, ((x1 + x2) // 2, (y1 + y2) // 2), (x2 - x1) // 2, warna, -1)
    return f


class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy, self.cls, self.conf = np.array([xyxy], float), np.array([cls]), np.array([conf])


class FakeResult:
    names = {0: "koin", 1: "tutup_botol"}

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, boxes):
        self._boxes, self.kwargs = boxes, None

    def predict(self, frame, **kw):
        self.kwargs = kw
        return [FakeResult(self._boxes)]


def det(cid, name, bbox=(100, 100, 200, 200), conf=0.8):
    return S.Detection(cid, name, S.display_label(name), conf, bbox, S.compute_center(bbox), (0.0, 0.0))


# ------------------------------------------------------------------ utils
def test_geometri():
    assert S.compute_center((100, 100, 200, 160)) == (150, 130)
    assert S.normalize_point((320, 240), 640, 480) == (0.5, 0.5)
    with pytest.raises(ValueError):
        S.normalize_point((1, 1), 0, 480)
    assert S.clamp_bbox((-5, -5, 700, 500), 640, 480) == (0, 0, 639, 479)
    assert S.display_label("tutup_botol") == "Tutup Botol"


def test_draw_detection_persen_opsional():
    a = np.zeros((300, 400, 3), np.uint8)
    b = a.copy()
    S.draw_detection(a, det(0, "koin"), persen=False)
    S.draw_detection(b, det(0, "koin"), persen=True)
    red = (a[:, :, 2] == 255) & (a[:, :, 0] == 0) & (a[:, :, 1] == 0)
    assert red.sum() > 100
    assert (a != b).any()          # label dengan persen lebih lebar


def test_append_csv(tmp_path):
    p = tmp_path / "k.csv"
    assert S.append_csv([det(0, "koin")] * 2, p) == 2
    assert S.append_csv([det(0, "koin")], p) == 1
    assert len(p.read_text().splitlines()) == 4


# ------------------------------------------------------------- cek warna
def test_cek_warna_biru_adalah_tutup_botol():
    assert S.tutup_berwarna(frame_dengan_bulatan((200, 120, 30)), (200, 150, 320, 270))        # biru (BGR)


@pytest.mark.parametrize("hue", [0, 130, 280, 330])   # merah, hijau, ungu, pink
def test_cek_warna_warna_pekat_lain(hue):
    assert S.tutup_berwarna(frame_dengan_bulatan(bgr_hsv(hue, 0.8, 0.8)), (200, 150, 320, 270))


def test_cek_warna_koin_perak_dan_emas_tidak_dikoreksi():
    assert not S.tutup_berwarna(frame_dengan_bulatan((175, 178, 180)), (200, 150, 320, 270))   # perak
    assert not S.tutup_berwarna(frame_dengan_bulatan(bgr_hsv(35, 0.55, 0.8)), (200, 150, 320, 270))  # emas


def test_cek_warna_tangan_di_tepi_tidak_ikut():
    f = frame_dengan_bulatan((175, 178, 180), bg=bgr_hsv(15, 0.5, 0.8))     # latar kulit pekat, koin perak
    assert not S.tutup_berwarna(f, (200, 150, 320, 270))


# ---------------------------------------------------------------- detektor
def test_detector_agnostic_nms_dan_titik_tengah():
    f = frame_dengan_bulatan((175, 178, 180))
    m = FakeModel([FakeBox((200, 150, 320, 270), 0, 0.87)])
    d = S.Detector(model=m, device="cpu").detect(f)[0]
    assert m.kwargs["agnostic_nms"] is True
    assert d.label == "Koin" and d.center == (260, 210)


def test_detector_koreksi_koin_biru_jadi_tutup_botol():
    f = frame_dengan_bulatan((200, 120, 30))                  # benda biru, model salah bilang koin 89%
    m = FakeModel([FakeBox((200, 150, 320, 270), 0, 0.89)])
    d = S.Detector(model=m, device="cpu").detect(f)[0]
    assert d.class_name == "tutup_botol" and d.catatan == "koreksi warna"
    d2 = S.Detector(model=m, device="cpu", cek_warna=False).detect(f)[0]
    assert d2.class_name == "koin"


def test_detector_urut_confidence_dan_kosong():
    f = frame_dengan_bulatan((175, 178, 180))
    m = FakeModel([FakeBox((10, 10, 60, 60), 0, 0.6), FakeBox((200, 150, 320, 270), 0, 0.9)])
    assert [d.confidence for d in S.Detector(model=m, device="cpu").detect(f)] == [0.9, 0.6]
    assert S.Detector(model=FakeModel([]), device="cpu").detect(f) == []


def test_detector_model_tidak_ada():
    with pytest.raises(FileNotFoundError):
        S.Detector(Path("tidak_ada.pt"))


# --------------------------------------------------------------- penstabil
def test_penstabil_suara_terbanyak():
    st = S.Penstabil(jendela=5)
    hasil = [st.update([det(1, "tutup_botol", conf=0.8)])[0] for _ in range(3)]
    d = st.update([det(0, "koin", conf=0.6)])[0]               # 1 frame salah tebak
    assert d.class_name == "tutup_botol" and d.catatan == "suara terbanyak"
    assert all(h.class_name == "tutup_botol" for h in hasil)


def test_penstabil_dua_benda_terpisah_dan_koordinat_tak_berubah():
    st = S.Penstabil()
    a, b = det(0, "koin", (10, 10, 60, 60)), det(1, "tutup_botol", (300, 300, 380, 380))
    for _ in range(4):
        out = st.update([a, b])
    assert [o.class_name for o in out] == ["koin", "tutup_botol"]
    assert out[0].bbox == a.bbox and out[0].center == a.center


def test_penstabil_benda_baru_dan_hilang():
    st = S.Penstabil(umur_maks=2)
    st.update([det(0, "koin")])
    for _ in range(4):
        st.update([])
    assert st._tracks == []
    assert st.update([det(1, "tutup_botol")])[0].class_name == "tutup_botol"


# ------------------------------------------------------------------ dataset
def test_convert_label_line():
    assert S.convert_label_line("5 0.5 0.5 0.2 0.4", 0) == "0 0.500000 0.500000 0.200000 0.400000"
    out = S.convert_label_line("0 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8", 1).split()
    assert [round(float(v), 3) for v in out[1:]] == [0.4, 0.5, 0.4, 0.6]
    for bad in ("", "0 0.5 0.5", "0 a b c d", "0 0.5 0.5 0 0.2"):
        assert S.convert_label_line(bad, 0) is None


def test_group_key():
    assert S.group_key("IMG_1_jpg.rf.abc123.jpg") == "IMG_1_jpg" and S.group_key("foto.jpg") == "foto"


def test_dedup_kotak_ganda():
    lines = ["0 0.5 0.5 0.2 0.2", "0 0.502 0.5 0.2 0.2", "0 0.8 0.8 0.1 0.1", "1 0.5 0.5 0.2 0.2"]
    out = S.dedup_lines(lines)
    assert len(out) == 3 and sum(l.startswith("0 ") for l in out) == 2    # kelas beda tidak dianggap ganda


def make_ds(root: Path, names, splits):
    for split, items in splits.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for fname, lines in items:
            cv2.imwrite(str(root / split / "images" / fname), np.full((32, 32, 3), 128, np.uint8))
            (root / split / "labels" / (Path(fname).stem + ".txt")).write_text("\n".join(lines))
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}))


def test_merge_semua_kelas_jadi_satu_dan_kotak_ganda_dibuang(tmp_path):
    koin = tmp_path / "koin"
    make_ds(koin, ["100", "200", "500", "KOIN-RUPIAH-MVPR"], {
        "train": [("a_jpg.rf.1.jpg", ["0 0.5 0.5 0.2 0.2"]),
                  ("b_jpg.rf.1.jpg", ["2 0.4 0.4 0.2 0.2", "3 0.401 0.4 0.2 0.2", "3 0.8 0.8 0.1 0.1"])],
        "valid": [("c_jpg.rf.1.jpg", ["1 0.5 0.5 0.2 0.2"])]})
    out = tmp_path / "gab"
    r = S.merge_datasets([S.Source("koin", koin, "koin")], out, S.CLASS_NAMES)
    assert r.instances["train"]["koin"] == 3 and r.dup_dropped == 1
    assert yaml.safe_load((out / "data.yaml").read_text())["names"] == {0: "koin", 1: "tutup_botol"}


def test_merge_buang_kebocoran(tmp_path):
    ds = tmp_path / "ds"
    make_ds(ds, ["x"], {"train": [("f1_jpg.rf.A.jpg", ["0 0.5 0.5 0.2 0.2"]), ("f1_jpg.rf.B.jpg", ["0 0.5 0.5 0.2 0.2"]),
                                  ("f2_jpg.rf.C.jpg", ["0 0.5 0.5 0.2 0.2"])],
                        "valid": [("f1_jpg.rf.D.jpg", ["0 0.5 0.5 0.2 0.2"])]})
    r = S.merge_datasets([S.Source("ds", ds, "koin")], tmp_path / "o", S.CLASS_NAMES)
    assert r.leak_dropped == 2 and r.images["train"]["ds"] == 1


def test_merge_kelas_lewat_nama_dan_holdout(tmp_path):
    own = tmp_path / "own"
    make_ds(own, S.CLASS_NAMES, {"train": [("a.jpg", ["1 0.5 0.5 0.2 0.2"])], "valid": [("b.jpg", ["0 0.5 0.5 0.2 0.2"])]})
    r = S.merge_datasets([S.Source("own", own, None)], tmp_path / "o", S.CLASS_NAMES)
    assert r.instances["train"]["tutup_botol"] == 1 and r.instances["valid"]["koin"] == 1
    ds = tmp_path / "ds"
    make_ds(ds, ["x"], {"train": [(f"f{i}.jpg", ["0 0.5 0.5 0.2 0.2"]) for i in range(20)]})
    r = S.merge_datasets([S.Source("ds", ds, "koin")], tmp_path / "o2", S.CLASS_NAMES)
    assert r.images["valid"]["ds"] >= 1 and r.images["train"]["ds"] + r.images["valid"]["ds"] == 20


# ----------------------------------------------------------------- label otomatis
def plain(bg=(235, 235, 235)):
    rng = np.random.default_rng(0)
    return cv2.add(np.full((480, 640, 3), bg, np.uint8), rng.integers(0, 8, (480, 640, 3), dtype=np.uint8))


def test_box_from_background():
    img = plain()
    cv2.circle(img, (400, 200), 70, (200, 120, 30), -1)
    cx, cy, w, h = S.box_from_background(img)
    assert abs(cx - 400 / 640) < 0.02 and abs(cy - 200 / 480) < 0.02 and abs(w - 140 / 640) < 0.03
    assert S.box_from_background(plain()) is None
    assert S.box_from_background(np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)) is None


def test_kotak_wajar():
    assert S.kotak_wajar((0.5, 0.5, 0.2, 0.2 * 640 / 480), 640, 480)
    assert not S.kotak_wajar(None, 640, 480)
    assert not S.kotak_wajar((0.5, 0.5, 0.97, 0.97), 640, 480)       # menutupi seluruh gambar
    assert not S.kotak_wajar((0.5, 0.5, 0.6, 0.05), 640, 480)        # terlalu lonjong


# -------------------------------------------------------------------- sintetis
def dataset_palsu(root: Path, n=24):
    """koin = bulatan abu-abu (latar kulit), tutup_botol = bulatan biru (latar putih polos)."""
    rng = np.random.default_rng(0)
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    for i in range(n):
        cid = i % 2
        bg = (120, 150, 190) if cid == 0 else (240, 240, 240)
        img = np.full((320, 320, 3), bg, np.uint8)
        img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
        cx, cy, r = int(rng.integers(120, 200)), int(rng.integers(120, 200)), 60
        cv2.circle(img, (cx, cy), r, (170, 172, 175) if cid == 0 else (200, 120, 30), -1)
        split = "valid" if i < 4 else "train"
        cv2.imwrite(str(root / split / "images" / f"x{i}.jpg"), img)
        (root / split / "labels" / f"x{i}.txt").write_text(
            f"{cid} {cx / 320:.5f} {cy / 320:.5f} {2 * r / 320:.5f} {2 * r / 320:.5f}\n")
    S.write_data_yaml(root, S.CLASS_NAMES)


def test_sintetis_label_pas_dan_dua_kelas_sekaligus(tmp_path):
    dataset_palsu(tmp_path)
    info = S.buat_sintetis(tmp_path, S.CLASS_NAMES, 60, latar_dir=None, seed=1, preview=tmp_path / "pv.jpg")
    assert info["dibuat"] == 60 and (tmp_path / "pv.jpg").is_file()
    assert (tmp_path / "valid" / "images").glob("*") and not list((tmp_path / "valid").rglob("sint__*"))  # valid utuh
    kelas_per_gambar, total = [], 0
    for lbl in (tmp_path / "train" / "labels").glob("sint__*.txt"):
        rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        kelas_per_gambar.append({r[0] for r in rows})
        img = cv2.imread(str(tmp_path / "train" / "images" / f"{lbl.stem}.jpg"))
        for cid, cx, cy, w, h in rows:
            cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
            assert 0 <= cx - w / 2 and cx + w / 2 <= 1.0001 and 0 <= cy - h / 2 and cy + h / 2 <= 1.0001
            assert 0.8 < w / h < 1.25                                  # benda bulat -> kotak ~persegi
            total += 1
            # isi kotak harus sesuai warna kelasnya: biru (tutup) vs abu-abu (koin)
            S_ = int(img.shape[0])
            patch = img[int((cy - h / 6) * S_):int((cy + h / 6) * S_), int((cx - w / 6) * S_):int((cx + w / 6) * S_)]
            if patch.size:
                b, g, r = patch.reshape(-1, 3).mean(0)
                if cid == "1":
                    assert b - r > 40
                else:
                    assert abs(b - r) < 40
    assert total > 60
    assert any(len(k) == 2 for k in kelas_per_gambar)                  # ada gambar berisi koin DAN tutup botol


def test_sintetis_tanpa_data_tidak_error(tmp_path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    assert S.buat_sintetis(tmp_path, S.CLASS_NAMES, 10, None)["dibuat"] == 0


def test_latar_kosong_jadi_negatif(tmp_path):
    latar = tmp_path / "latar"
    latar.mkdir()
    for i in range(3):
        cv2.imwrite(str(latar / f"{i}.jpg"), np.full((64, 64, 3), 100, np.uint8))
    out = tmp_path / "out"
    assert S.tambah_latar_kosong(out, latar) == 3
    assert all(p.read_text() == "" for p in (out / "train" / "labels").glob("latar__*.txt"))


def test_cli_parser():
    a = S.build_parser().parse_args(["deteksi", "--persen", "--conf", "0.4"])
    assert a.persen and a.conf == 0.4 and a.fn is S.cmd_deteksi
    a = S.build_parser().parse_args(["label", "sendiri", "--metode", "model"])
    assert a.target == "sendiri" and a.metode == "model"
