import json
import paho.mqtt.client as mqtt
from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_KEEPALIVE


class MQTTService:
    def __init__(self):
        self.client = None

    def connect(self):
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            self.client.loop_start()
            print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            self.client = None
            print(f"[MQTT] Connection failed: {e}")

    def publish(self, payload: dict):
        if self.client is None:
            return
        try:
            self.client.publish(MQTT_TOPIC, json.dumps(payload))
        except Exception as e:
            print(f"[MQTT] Publish failed: {e}")

    def disconnect(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None