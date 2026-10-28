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


def test_inspect_stationary_object(monkeypatch, tmp_path):
    from sortir_vision_api import SortirVisionAPI
    from src.detector import Detection

    # Buat dummy API tanpa membuka kamera fisik
    monkeypatch.setattr("src.camera.Camera.open", lambda self, auto_fallback=True: self)
    frame_dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    monkeypatch.setattr("src.camera.Camera.read", lambda self: (True, frame_dummy))

    vision = SortirVisionAPI(source=0)
    # Mock detector output: koin di (300, 300)
    dummy_det = Detection(
        class_id=0,
        class_name="koin",
        label="Koin",
        confidence=0.92,
        bbox=(250, 250, 350, 350),
        center=(300, 300),
        center_norm=(300/1280, 300/720),
    )
    monkeypatch.setattr(vision.detector, "detect", lambda f, apply_smart_filter=True: [dummy_det])

    # 1. Jika target_classes mencakup "koin", harus dikembalikan sebagai RobotTarget
    target = vision.inspect_stationary_object(target_classes=["koin", "charger"], save_snapshot=False)
    assert target is not None
    assert target.class_name == "koin"
    assert target.pixel_center == (300, 300)

    # 2. Jika target_classes hanya "charger", koin harus diabaikan (mengembalikan None)
    ignored = vision.inspect_stationary_object(target_classes=["charger"], save_snapshot=False)
    assert ignored is None


def test_export_target_json_custom_status(tmp_path):
    json_path = tmp_path / "target_status.json"

    # Status konveyor bergerak
    export_target_json(None, [], json_path, status="CONVEYOR_MOVING", message="Konveyor jalan")
    d1 = read_target_json(json_path)
    assert d1["status"] == "CONVEYOR_MOVING"
    assert d1["target"] is None
    assert d1["message"] == "Konveyor jalan"

    # Status jeda inersia
    export_target_json(None, [], json_path, status="WAITING_STATIONARY", message="Menunggu diam")
    d2 = read_target_json(json_path)
    assert d2["status"] == "WAITING_STATIONARY"
    assert d2["message"] == "Menunggu diam"


def test_inspect_stationary_with_motion_verification(monkeypatch, tmp_path):
    from sortir_vision_api import SortirVisionAPI
    from src.detector import Detection

    monkeypatch.setattr("src.camera.Camera.open", lambda self, auto_fallback=True: self)
    frame_dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    monkeypatch.setattr("src.camera.Camera.read", lambda self: (True, frame_dummy))

    target_json = tmp_path / "target_robot.json"
    monkeypatch.setattr("sortir_vision_api.TARGET_JSON_PATH", target_json)

    vision = SortirVisionAPI(source=0)
    det = Detection(
        class_id=1,
        class_name="tutup_botol",
        label="Tutup Botol",
        confidence=0.88,
        bbox=(100, 100, 200, 200),
        center=(150, 150),
        center_norm=(150/1280, 150/720),
    )
    monkeypatch.setattr(vision.detector, "detect", lambda f, apply_smart_filter=True: [det])

    # Jalankan inspect dengan settling delay pendek dan verifikasi gerak
    target = vision.inspect_stationary_object(
        target_classes=["tutup_botol"],
        settle_delay=0.01,
        verify_motion=True,
        save_snapshot=False,
    )
    assert target is not None
    assert target.class_name == "tutup_botol"

    # Verifikasi file JSON terakhir berstatus TARGET_READY
    data = read_target_json(target_json)
    assert data is not None
    assert data["status"] == "TARGET_READY"
    assert data["target"]["label"] == "Tutup Botol"

    # Test set_status
    vision.set_status("CONVEYOR_MOVING", "Konveyor jalan lagi")
    data_moving = read_target_json(target_json)
    assert data_moving["status"] == "CONVEYOR_MOVING"
    assert data_moving["target"] is None
