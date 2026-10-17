import pytest
from src.camera import Camera, ThreadedCamera, parse_source


def test_parse_source():
    assert parse_source(0) == 0
    assert parse_source("0") == 0
    assert parse_source("1") == 1
    assert parse_source("http://192.168.1.50:81/stream") == "http://192.168.1.50:81/stream"
    assert parse_source("video.mp4") == "video.mp4"


def test_camera_instantiation():
    cam = Camera(0, 640, 480)
    assert cam.source == 0
    assert cam.width == 640
    assert cam.height == 480
    assert cam.cap is None


def test_threaded_camera_instantiation():
    tcam = ThreadedCamera("http://192.168.1.50:81/stream", 640, 480)
    assert tcam.source == "http://192.168.1.50:81/stream"
    assert tcam.running is False
    assert tcam.latest_frame is None
