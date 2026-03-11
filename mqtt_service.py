import json
import paho.mqtt.client as mqtt
from config import (
    MQTT_ENABLED,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_KEEPALIVE,
    MQTT_TOPIC_ACCESS,
    MQTT_TOPIC_ALERT,
)


class MQTTService:

    def __init__(self):
        self.client = None
        self.enabled = MQTT_ENABLED

    def connect(self):
        if not self.enabled:
            print("[MQTT] Disabled in config")
            return

        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            self.client.loop_start()
            print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            self.client = None
            print(f"[MQTT] Connection failed: {e}")

    def publish(self, payload: dict, topic: str | None = None):
        if not self.enabled:
            return

        if self.client is None:
            return

        try:
            if topic is None:
                topic = MQTT_TOPIC_ACCESS

            self.client.publish(topic, json.dumps(payload))

        except Exception as e:
            print(f"[MQTT] Publish failed: {e}")

    def publish_alert(self, payload: dict):
        self.publish(payload, MQTT_TOPIC_ALERT)

    def disconnect(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None