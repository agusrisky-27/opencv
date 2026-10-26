from pathlib import Path
import numpy as np
import pytest

from src.coordinate_transform import (
    RobotCoordinateTransformer,
    RobotTarget,
    choose_best_target,
    export_target_json,
)
from sortir_vision_api import read_target_json


def test_linear_opposite_transformation():
    # Setting: kamera & robot berseberangan
    # frame 1280x720, center piksel = (640, 360)
    # origin robot = (0.0, 200.0) mm
    # flip_x = True (kamera kanan = robot kiri)
    t = RobotCoordinateTransformer(
        berseberangan=True,
        flip_x=True,
        flip_y=False,
        scale_x_mm_per_px=0.5,
        scale_y_mm_per_px=0.5,
        origin_robot_x_mm=0.0,
        origin_robot_y_mm=200.0,
        z_heights={"koin": 2.0, "tutup_botol": 12.0},
    )

    # 1. Titik tepat di tengah frame (640, 360) harus sama dengan origin robot
    xr, yr, zr = t.pixel_to_robot((640, 360), "koin", frame_width=1280, frame_height=720)
    assert xr == 0.0
    assert yr == 200.0
    assert zr == 2.0  # Koin Z = 2.0 mm

    # 2. Objek bergerak ke KANAN di kamera (u = 740, delta_u = +100 px)
    # Karena berseberangan & flip_x=True, di Robot harus bergerak ke KIRI (delta_X = -50 mm)
    xr_right, yr_right, _ = t.pixel_to_robot((740, 360), "koin", frame_width=1280, frame_height=720)
    assert xr_right == -50.0

    # 3. Ketinggian Z tutup botol harus 12.0 mm
    _, _, z_tutup = t.pixel_to_robot((640, 360), "tutup_botol", frame_width=1280, frame_height=720)
    assert z_tutup == 12.0


def test_homography_4point_calibration(tmp_path):
    t = RobotCoordinateTransformer()

    # 4 titik acuan piksel (persegi di frame)
    pixel_pts = [(100.0, 100.0), (500.0, 100.0), (500.0, 400.0), (100.0, 400.0)]
    # 4 titik acuan robot mm (misal diputar atau dibalik karena berseberangan)
    robot_pts = [(100.0, 300.0), (-100.0, 300.0), (-100.0, 150.0), (100.0, 150.0)]

    ok = t.set_homography_points(pixel_pts, robot_pts)
    assert ok is True
    assert t.homography_matrix is not None

    # Verifikasi titik pertama kembali ke koordinat robot yang tepat
    xr, yr, _ = t.pixel_to_robot((100, 100), "koin")
    assert pytest.approx(xr, abs=1.0) == 100.0
    assert pytest.approx(yr, abs=1.0) == 300.0

    # Verifikasi simpan dan muat kembali file config
    cfg_file = tmp_path / "test_calib.json"
    assert t.save_config(cfg_file) is True

    t2 = RobotCoordinateTransformer(config_path=cfg_file)
    assert t2.homography_matrix is not None
    xr2, yr2, _ = t2.pixel_to_robot((100, 100), "koin")
    assert pytest.approx(xr2, abs=1.0) == 100.0


def test_choose_best_target():
    t1 = RobotTarget(
        class_name="koin",
        label="Koin",
        confidence=0.85,
        pixel_center=(200, 200),
        x_mm=10.0,
        y_mm=150.0,
        z_mm=2.0,
        drop_x_mm=-150.0,
        drop_y_mm=100.0,
        drop_z_mm=45.0,
    )
    t2 = RobotTarget(
        class_name="tutup_botol",
        label="Tutup Botol",
        confidence=0.95,
        pixel_center=(800, 800),  # di luar ROI
        x_mm=-20.0,
        y_mm=160.0,
        z_mm=12.0,
        drop_x_mm=150.0,
        drop_y_mm=100.0,
        drop_z_mm=55.0,
    )

    # Tanpa ROI: t2 menang karena confidence lebih tinggi (0.95 > 0.85)
    best_no_roi = choose_best_target([t1, t2], roi=None)
    assert best_no_roi == t2

    # Dengan ROI conveyor (100, 100, 500, 500): t1 berada di dalam ROI sehingga terpilih
    roi_conveyor = (100, 100, 500, 500)
    best_in_roi = choose_best_target([t1, t2], roi=roi_conveyor)
    assert best_in_roi == t1

    # Jika semua target berada di luar ROI, robot tidak boleh menargetkan sembarang benda
    assert choose_best_target([t2], roi=roi_conveyor) is None


def test_export_and_read_target_json(tmp_path):
    json_path = tmp_path / "target_robot.json"
    t = RobotTarget(
        class_name="koin",
        label="Koin",
        confidence=0.92,
        pixel_center=(640, 360),
        x_mm=12.5,
        y_mm=180.0,
        z_mm=2.0,
        drop_x_mm=-150.0,
        drop_y_mm=100.0,
        drop_z_mm=45.0,
        track_id=1,
    )

    ok = export_target_json(t, [t], json_path)
    assert ok is True
    assert json_path.exists()

    data = read_target_json(json_path)
    assert data is not None
    assert data["status"] == "TARGET_READY"
    assert data["target"]["label"] == "Koin"
    assert data["target"]["robot_mm"]["x"] == 12.5
    assert data["target"]["robot_mm"]["y"] == 180.0
    assert data["target"]["robot_mm"]["z"] == 2.0
