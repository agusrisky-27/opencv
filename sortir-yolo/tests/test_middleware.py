import pytest
from fastapi.testclient import TestClient
from middleware_server import app, state


@pytest.fixture(autouse=True)
def reset_state():
    state.counts = {"koin": 0, "tutup_botol": 0, "diabaikan": 0}
    state.target_queue.clear()
    state.latest_detection = None
    state.pending_manual_command = None


client = TestClient(app)


def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "RUNNING"


def test_vision_post_and_robot_get():
    # 1. Vision kirim deteksi koin
    payload = {
        "kelas": "koin",
        "confidence": 0.94,
        "x_mm": 130.5,
        "y_mm": 55.0,
        "z_mm": 2.0,
        "track_id": 10,
        "aksi": "AMBIL",
    }
    post_res = client.post("/api/vision/detection", json=payload)
    assert post_res.status_code == 200
    assert post_res.json()["queue_length"] == 1

    # 2. ESP32 panggil target
    get_res = client.get("/api/robot/target")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["available"] is True
    assert data["aksi"] == "AMBIL"
    assert data["kelas"] == "koin"
    assert data["x"] == 130.5
    assert "drop_x" in data
    assert "drop_y" in data
    assert "drop_z" in data

    # Cek endpoint drop-zones
    drop_res = client.get("/api/robot/drop-zones")
    assert drop_res.status_code == 200
    assert "koin" in drop_res.json()
    assert "tutup_botol" in drop_res.json()

    # 3. Panggilan kedua ESP32 (antrean sudah kosong)
    get_res_empty = client.get("/api/robot/target")
    assert get_res_empty.json()["available"] is False
    assert get_res_empty.json()["aksi"] == "ABAIKAN"


def test_app_dashboard_and_control():
    # 1. Dashboard cek status
    res = client.get("/api/app/status")
    assert res.status_code == 200
    assert "counts" in res.json()

    # 2. App kirim kontrol manual
    ctrl_res = client.post("/api/app/control", json={"command": "HOME"})
    assert ctrl_res.status_code == 200

    # 3. ESP32 baca perintah manual
    robot_res = client.get("/api/robot/target")
    assert robot_res.json()["type"] == "MANUAL"
    assert robot_res.json()["command"]["command"] == "HOME"
