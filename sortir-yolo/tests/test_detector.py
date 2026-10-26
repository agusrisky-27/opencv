import numpy as np
import pytest

from src.detector import Detector


class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = np.array([xyxy], dtype=float)
        self.cls = np.array([cls])
        self.conf = np.array([conf])


class FakeResult:
    names = {0: "koin", 1: "korek_api", 2: "tutup_botol"}

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, boxes):
        self._boxes = boxes
        self.kwargs = None

    def predict(self, frame, **kwargs):
        self.kwargs = kwargs
        return [FakeResult(self._boxes)]


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_detect_menghitung_titik_tengah_dan_label():
    model = FakeModel([FakeBox((100, 50, 300, 150), 2, 0.87)])
    dets = Detector(model=model, device="cpu").detect(FRAME)
    assert len(dets) == 1
    d = dets[0]
    assert d.class_name == "tutup_botol" and d.label == "Tutup Botol"
    assert d.center == (200, 100)
    assert d.center_norm == (round(200 / 640, 4), round(100 / 480, 4))
    assert d.bbox == (100, 50, 300, 150)
    assert d.to_dict()["label"] == "Tutup Botol"


def test_detect_urut_confidence_dan_banyak_objek():
    model = FakeModel([FakeBox((0, 0, 50, 50), 0, 0.6), FakeBox((200, 200, 300, 300), 1, 0.95)])
    dets = Detector(model=model, device="cpu").detect(FRAME)
    assert [d.label for d in dets] == ["Korek Api", "Koin"]


def test_detect_tanpa_objek():
    assert Detector(model=FakeModel([]), device="cpu").detect(FRAME) == []


def test_detect_clamp_di_tepi_frame():
    model = FakeModel([FakeBox((-10, -10, 700, 500), 0, 0.9)])
    d = Detector(model=model, device="cpu").detect(FRAME)[0]
    assert d.bbox == (0, 0, 639, 479)


def test_parameter_diteruskan_ke_model():
    model = FakeModel([])
    Detector(model=model, conf=0.33, imgsz=320, device="cpu").detect(FRAME)
    assert model.kwargs["conf"] == 0.33 and model.kwargs["imgsz"] == 320


def test_weights_tidak_ada_memberi_error_jelas(tmp_path):
    with pytest.raises(FileNotFoundError, match="training"):
        Detector(tmp_path / "tidak_ada.pt")


def test_validate_geometry():
    from src.detector import validate_geometry
    # Bundar / kotak wajar
    assert validate_geometry((100, 100, 200, 200), 1280, 720) is True
    # Jendela lonjong (aspek rasio 3.0)
    assert validate_geometry((100, 100, 400, 200), 1280, 720) is False
    # Terlalu besar (wajah memenuhi layar: 600x600 pada 720p)
    assert validate_geometry((100, 50, 700, 650), 1280, 720) is False
    # Terlalu kecil (noise 5x5 px)
    assert validate_geometry((10, 10, 15, 15), 1280, 720) is False
    # Menempel di batas tepi frame kamera (edge_margin proteksi objek terpotong)
    assert validate_geometry((5, 100, 60, 160), 1280, 720, edge_margin=15) is False
    assert validate_geometry((1220, 100, 1275, 160), 1280, 720, edge_margin=15) is False
    assert validate_geometry((30, 100, 90, 160), 1280, 720, edge_margin=15) is True


def test_smart_filter_menolak_jendela_dan_wajah():
    # Box 1: koin wajar (100x100)
    # Box 2: jendela lonjong (300x100)
    # Box 3: objek terlalu besar (600x600)
    boxes = [
        FakeBox((100, 100, 200, 200), 0, 0.95),
        FakeBox((50, 50, 350, 150), 0, 0.90),
        FakeBox((10, 10, 610, 610), 0, 0.85),
    ]
    det_smart = Detector(model=FakeModel(boxes), device="cpu", smart_filter=True)
    dets = det_smart.detect(FRAME)
    assert len(dets) == 1
    assert dets[0].bbox == (100, 100, 200, 200)


def test_filter_roi():
    boxes = [
        FakeBox((150, 150, 250, 250), 0, 0.95),  # center=(200, 200) -> di dalam ROI
        FakeBox((500, 400, 600, 460), 0, 0.90),  # center=(550, 430) -> di luar ROI
    ]
    roi = (100, 100, 300, 300)
    det_roi = Detector(model=FakeModel(boxes), device="cpu", roi=roi)
    dets = det_roi.detect(FRAME)
    assert len(dets) == 1
    assert dets[0].bbox == (150, 150, 250, 250)


def test_verifikasi_koin_asli_tolak_kunci_dan_plastik():
    from pathlib import Path
    import cv2
    from src.detector import verifikasi_koin_asli
    img_path = Path("hasil pengujian.png")
    if not img_path.is_file():
        img_path = Path("../hasil pengujian.png")
    if img_path.is_file():
        img = cv2.imread(str(img_path))
        # Koin 1000 asli harus lolos
        assert verifikasi_koin_asli(img, (1370, 84, 1370+362, 84+332)) is True
        # Kunci harus ditolak
        assert verifikasi_koin_asli(img, (526, 556, 526+350, 556+396)) is False
        # Plastik abu polos harus ditolak
        assert verifikasi_koin_asli(img, (794, 63, 794+442, 63+349)) is False


def test_verifikasi_tutup_botol_asli_tolak_wajah():
    from src.detector import verifikasi_tutup_botol_asli
    # Simulasi frame ukuran 1280x720
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 1. Benda wajah manusia (warna kulit sawo matang / tan, ukuran besar 150x150)
    # BGR untuk warna kulit: (110, 150, 195)
    frame[100:250, 100:250] = (110, 150, 195)
    assert verifikasi_tutup_botol_asli(frame, (100, 100, 250, 250), 1280, 720) is False

    # 2. Tutup botol asli berwarna biru plastik pekat (ukuran 60x60)
    # BGR untuk biru pekat: (220, 120, 20)
    frame[300:360, 300:360] = (220, 120, 20)
    assert verifikasi_tutup_botol_asli(frame, (300, 300, 360, 360), 1280, 720) is True

    # 3. Tutup botol asli berwarna merah plastik pekat (ukuran 60x60)
    # BGR untuk merah pekat: (20, 30, 230)
    frame[400:460, 400:460] = (20, 30, 230)
    assert verifikasi_tutup_botol_asli(frame, (400, 400, 460, 460), 1280, 720) is True


def test_refine_charger_vacuum_center():
    from src.detector import refine_charger_vacuum_center
    # Buat charger sintetis: kepala di atas (putih pekat), dua kaki colokan di bawah
    crop = np.zeros((200, 100, 3), dtype=np.uint8)
    # Kepala (y: 0..130): balok lebar putih
    crop[0:130, 10:90] = 240
    # Steker/pin logam (y: 130..200): dua kaki tipis
    crop[130:200, 25:35] = 200
    crop[130:200, 65:75] = 200

    bbox = (100, 100, 200, 300)
    # Titik tengah standar adalah 100 + 100 = 200
    std_cy = 200
    cx, cy = refine_charger_vacuum_center(crop, bbox)
    # Titik hisap vacum harus bergeser ke atas menuju kepala charger (cy < std_cy)
    assert cx == 150
    assert cy < std_cy
    assert 160 <= cy <= 180


def test_is_specular_metal_coin_dan_koreksi_silau():
    from src.detector import is_specular_metal_coin

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 1. Koin logam dengan pantulan silau putih (achromatic / specular glare: BGR 220, 220, 220)
    frame[200:300, 200:300] = (220, 220, 220)
    assert is_specular_metal_coin(frame, (200, 200, 300, 300)) is True

    # 2. Tutup botol plastik berwarna pekat (merah pekat: BGR 20, 30, 230)
    frame[400:500, 400:500] = (20, 30, 230)
    assert is_specular_metal_coin(frame, (400, 400, 500, 500)) is False

    # 3. Uji koreksi deteksi di Detector: jika AI salah tebak tutup_botol pada koin silau
    box_silau = FakeBox((200, 200, 300, 300), 2, 0.95)  # class 2 = tutup_botol di FakeModel
    det = Detector(model=FakeModel([box_silau]), device="cpu", smart_filter=True, cek_warna=True)
    dets = det.detect(frame)
    assert len(dets) == 1
    assert dets[0].class_name == "koin"
    assert dets[0].catatan == "koreksi silau"




