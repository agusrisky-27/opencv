import numpy as np
import pytest

from src.detector import Detection
from src.utils import (append_csv, clamp_bbox, compute_center, display_label, draw_detection,
                       normalize_point)


def make_det(bbox=(100, 100, 200, 160), label="Koin", conf=0.91):
    c = compute_center(bbox)
    return Detection(0, "koin", label, conf, bbox, c, (0.0, 0.0))


def test_compute_center():
    assert compute_center((100, 100, 200, 160)) == (150, 130)
    assert compute_center((0, 0, 1, 1)) == (0, 0) or compute_center((0, 0, 1, 1)) == (1, 1)


def test_normalize_point():
    assert normalize_point((320, 240), 640, 480) == (0.5, 0.5)
    with pytest.raises(ValueError):
        normalize_point((1, 1), 0, 480)


def test_clamp_bbox():
    assert clamp_bbox((-5, -5, 700, 500), 640, 480) == (0, 0, 639, 479)


def test_display_label():
    assert display_label("tutup_botol") == "Tutup Botol"
    assert display_label("korek_api") == "Korek Api"
    assert display_label("koin") == "Koin"
    assert display_label("benda_lain") == "Benda Lain"  # kelas tak dikenal tetap rapi


def test_draw_detection_menggambar_merah():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    draw_detection(frame, make_det())
    red = (frame[:, :, 2] == 255) & (frame[:, :, 0] == 0) & (frame[:, :, 1] == 0)
    assert red.sum() > 100          # kotak merah (BGR 0,0,255) benar-benar tergambar
    assert frame[100, 150, 2] == 255  # garis atas kotak


def test_draw_detection_kotak_mepet_atas_tidak_error():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    draw_detection(frame, make_det(bbox=(10, 0, 80, 50)))


def test_append_csv(tmp_path):
    path = tmp_path / "out" / "k.csv"
    assert append_csv([make_det(), make_det()], path) == 2
    assert append_csv([make_det()], path) == 1
    lines = path.read_text().splitlines()
    assert lines[0].startswith("waktu,kelas")
    assert len(lines) == 4
