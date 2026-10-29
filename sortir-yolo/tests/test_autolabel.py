import cv2
import numpy as np

from src.autolabel import box_from_background, segment_charger_box


def plain_image(bg=(235, 235, 235)):
    rng = np.random.default_rng(0)
    img = np.full((480, 640, 3), bg, np.uint8)
    return cv2.add(img, rng.integers(0, 8, img.shape, dtype=np.uint8))


def test_benda_di_latar_polos_terdeteksi():
    img = plain_image()
    cv2.circle(img, (400, 200), 70, (200, 120, 30), -1)
    cx, cy, w, h = box_from_background(img)
    assert abs(cx - 400 / 640) < 0.02 and abs(cy - 200 / 480) < 0.02
    assert abs(w - 140 / 640) < 0.03 and abs(h - 140 / 480) < 0.03


def test_latar_gelap_conveyor():
    img = plain_image((60, 60, 60))
    cv2.circle(img, (150, 300), 45, (190, 200, 205), -1)
    cx, cy, _, _ = box_from_background(img)
    assert abs(cx - 150 / 640) < 0.02 and abs(cy - 300 / 480) < 0.02


def test_latar_ramai_ditolak():
    noisy = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert box_from_background(noisy) is None


def test_gambar_kosong_tanpa_benda():
    assert box_from_background(plain_image()) is None


def test_segment_charger_box_putih_di_latar_gelap():
    # Latar gelap (conveyor)
    img = np.full((720, 1280, 3), 35, dtype=np.uint8)
    # Charger putih kotak di tengah
    cv2.rectangle(img, (500, 200), (780, 520), (220, 220, 220), -1)
    box = segment_charger_box(img)
    assert box is not None
    cx, cy, w, h = box
    assert abs(cx - 640 / 1280) < 0.05
    assert abs(cy - 360 / 720) < 0.05
    assert abs(w - 280 / 1280) < 0.05
    assert abs(h - 320 / 720) < 0.05


def test_segment_charger_box_hitam_di_latar_terang():
    # Latar terang (meja/kertas putih)
    img = np.full((720, 1280, 3), 210, dtype=np.uint8)
    # Charger hitam di tengah
    cv2.rectangle(img, (450, 250), (830, 470), (30, 30, 30), -1)
    box = segment_charger_box(img)
    assert box is not None
    cx, cy, w, h = box
    assert abs(cx - 640 / 1280) < 0.05
    assert abs(cy - 360 / 720) < 0.05
    assert abs(w - 380 / 1280) < 0.05
    assert abs(h - 220 / 720) < 0.05


def test_segment_charger_box_conveyor_hijau():
    # Simulasi frame conveyor kantor dengan meja terang di atas & bawah
    img = np.full((720, 1280, 3), (220, 220, 220), dtype=np.uint8)
    # Belt konveyor hijau di tengah
    img[120:580, :] = (100, 145, 110)
    # Charger putih kotak di atas conveyor
    cv2.rectangle(img, (480, 220), (820, 480), (240, 240, 240), -1)
    box = segment_charger_box(img)
    assert box is not None
    cx, cy, w, h = box
    assert abs(cx - 650 / 1280) < 0.05
    assert abs(cy - 350 / 720) < 0.05
    assert abs(w - 340 / 1280) < 0.05
    assert abs(h - 260 / 720) < 0.05


def test_segment_charger_box_memotong_pin_logam_fokus_kepala():
    # Simulasi conveyor hijau dengan charger putih berlogo IT dan 2 pin colokan logam abu-abu
    img = np.full((720, 1280, 3), (220, 220, 220), dtype=np.uint8)
    img[120:580, :] = (100, 145, 110)
    # 1. Bodi kepala charger putih (x: 520 s.d. 800 -> w=280, y: 220 s.d. 480 -> h=260)
    cv2.rectangle(img, (520, 220), (800, 480), (245, 245, 245), -1)
    # 2. Dua pin logam steker abu-abu gelap mencuat di sisi kiri (x: 440 s.d. 520 -> w=80)
    cv2.rectangle(img, (440, 270), (520, 290), (120, 120, 120), -1)
    cv2.rectangle(img, (440, 410), (520, 430), (120, 120, 120), -1)
    
    box = segment_charger_box(img)
    assert box is not None
    cx, cy, w, h = box
    # Titik tengah harus fokus di bodi putih (cx sekitar 660, cy sekitar 350)
    # Lebar kotak harus mencakup bodi kepala (w ~ 280), BUKAN total bersama pin (w ~ 360)
    assert abs(cx - 660 / 1280) < 0.05
    assert abs(cy - 350 / 720) < 0.05
    assert abs(w - 280 / 1280) < 0.05
    assert abs(h - 260 / 720) < 0.05



