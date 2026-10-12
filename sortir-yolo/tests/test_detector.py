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

