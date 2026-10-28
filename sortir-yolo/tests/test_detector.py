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


def test_validate_geometry_charger_tinggi():
    from src.detector import validate_geometry
    # Charger berdiri tinggi (h=510 pada frame 720p, area_frac < 0.55)
    bbox = (339, 127, 861, 638)
    # Harus lolos validasi geometri charger (max_dim_frac fleksibel)
    assert validate_geometry(
        bbox, 1280, 720, min_ar=0.25, max_ar=4.00, min_size_px=10, max_area_frac=0.55, edge_margin=0
    ) is True


def test_edge_margin_default_nol():
    from src.detector import Detector
    # Box menempel dekat batas atas frame 640x480 (y1 = 6)
    box = FakeBox((100, 6, 200, 106), 2, 0.93)
    det = Detector(model=FakeModel([box]), device="cpu", smart_filter=True)
    dets = det.detect(FRAME)
    # Default edge_margin=0 tidak boleh menolak objek ini
    assert len(dets) == 1
    assert dets[0].bbox == (100, 6, 200, 106)


def test_white_bottle_cap_correction():
    from src.detector import is_white_bottle_cap

    # 1. Objek tutup botol plastik putih besar di frame 720p: w=160, h=160 (area 25600 px)
    frame = np.full((720, 1280, 3), (30, 90, 40), dtype=np.uint8)  # alas konveyor hijau
    frame[250:410, 500:660] = (215, 215, 215)  # putih cerah merata khas plastik
    bbox_white_cap = (500, 250, 660, 410)

    assert is_white_bottle_cap(frame, bbox_white_cap) is True

    # 2. Uji koreksi deteksi di Detector: jika AI salah memprediksi koin (class 0) pada tutup botol putih
    box_putih = FakeBox(bbox_white_cap, 0, 0.92)  # class 0 = koin di FakeModel
    det = Detector(model=FakeModel([box_putih]), device="cpu", smart_filter=True, cek_warna=True)
    dets = det.detect(frame)
    assert len(dets) == 1
    assert dets[0].class_name == "tutup_botol"
    assert dets[0].catatan == "koreksi tutup putih"


def test_verifikasi_charger_asli_tolak_penghapus_dan_peraut():
    from src.detector import verifikasi_charger_asli
    frame = np.full((720, 1280, 3), (35, 95, 45), dtype=np.uint8)  # alas konveyor hijau

    # 1. Peraut pensil mini (ukuran kecil 40x30 px, area 1200 px^2) -> HARUS DITOLAK
    bbox_peraut = (200, 200, 240, 230)
    assert verifikasi_charger_asli(frame, bbox_peraut, confidence=0.75) is False

    # 2. Karet penghapus polos (ukuran 80x55 px, tapi tekstur datar monokromatik tanpa pin/kontras)
    # BGR abu-abu datar seragam
    frame[300:360, 400:480] = (180, 180, 180)
    bbox_penghapus = (400, 300, 480, 360)
    # Jika model ragu/confidence moderat (<0.60) pada penghapus polos -> HARUS DITOLAK
    assert verifikasi_charger_asli(frame, bbox_penghapus, confidence=0.42) is False

    # 3. Kepala charger asli (ukuran 120x100 px, bodi putih dengan pin steker gelap bertekstur kontras tinggi)
    # Kepala bodi charger
    frame[500:580, 600:720] = (245, 245, 245)
    # Pin steker logam hitam pekat
    frame[580:620, 620:640] = (15, 15, 15)
    frame[580:620, 680:700] = (15, 15, 15)
    bbox_charger = (600, 500, 720, 620)
    assert verifikasi_charger_asli(frame, bbox_charger, confidence=0.88) is True


def test_detector_smart_filter_tolak_penghapus_sebagai_charger():
    # Simulasi penguji meletakkan penghapus mini (40x35 px) dan AI salah memprediksi sebagai charger (conf 0.32)
    class FakeResultCharger:
        names = {0: "koin", 1: "charger", 2: "tutup_botol"}
        def __init__(self, boxes):
            self.boxes = boxes

    class FakeModelCharger:
        def __init__(self, boxes):
            self._boxes = boxes
        def predict(self, frame, **kwargs):
            return [FakeResultCharger(self._boxes)]

    frame = np.full((720, 1280, 3), (35, 95, 45), dtype=np.uint8)
    # Box peraut/penghapus mini
    box_penghapus = FakeBox((300, 300, 345, 335), 1, 0.32)  # class 1 = charger
    det = Detector(model=FakeModelCharger([box_penghapus]), device="cpu", smart_filter=True)
    dets = det.detect(frame)
    # Benda acak penguji harus ditolak 100% (tidak terbaca sama sekali)
    assert len(dets) == 0


def test_refine_charger_vacuum_center_kontinu_tanpa_loncatan():
    from src.detector import refine_charger_vacuum_center
    crop = np.zeros((200, 100, 3), dtype=np.uint8)
    crop[0:120, 10:90] = 230
    crop[120:200, 25:35] = 180
    crop[120:200, 65:75] = 180
    bbox = (100, 100, 200, 300)

    # Periksa kontinuitas: variasi kecil pada kecerahan piksel
    cx1, cy1 = refine_charger_vacuum_center(crop, bbox)
    crop_sedikit_berbeda = crop.copy()
    crop_sedikit_berbeda[10:30, 20:40] = 220
    cx2, cy2 = refine_charger_vacuum_center(crop_sedikit_berbeda, bbox)

    assert cx1 == 150 and cx2 == 150
    # Jarak pergeseran harus sangat kecil (halus / anti-jitter, bukan loncatan 28 px)
    assert abs(cy1 - cy2) <= 3





