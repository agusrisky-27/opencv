import time
from src.mqtt_client import MqttSender


def test_mqtt_disabled():
    sender = MqttSender(enabled=False)
    assert sender.send_detection("koin", 0.95, track_id=1) is False
    sender.close()


def test_mqtt_cooldown_logic():
    sender = MqttSender(enabled=False, cooldown=1.0)
    # Mocking internal state
    sender.enabled = True
    sender.client = None  # No actual network call without client
    assert sender.send_detection("koin", 0.9) is False
