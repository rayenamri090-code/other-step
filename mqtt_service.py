import json
import paho.mqtt.client as mqtt

from config import (
    MQTT_ENABLED,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_KEEPALIVE,
    MQTT_TOPIC_ACCESS,
    MQTT_TOPIC_ALERT,
    MQTT_TOPIC_SYSTEM,
)


class MQTTService:

    def __init__(self):
        self.client = None
        self.enabled = MQTT_ENABLED
        self.connected = False

    def connect(self):
        if not self.enabled:
            print("[MQTT] Disabled in config")
            return

        try:
            self.client = mqtt.Client()
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect

            self.client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            self.client.loop_start()

        except Exception as e:
            self.client = None
            self.connected = False
            print(f"[MQTT] Connection failed: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT}")
        else:
            print(f"[MQTT] Connect error code: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        print("[MQTT] Disconnected")

    def _publish_raw(self, topic, payload):
        if not self.enabled:
            return

        if self.client is None or not self.connected:
            return

        try:
            self.client.publish(topic, json.dumps(payload), qos=0, retain=False)
        except Exception as e:
            print(f"[MQTT] Publish failed: {e}")

    def publish_access(self, payload: dict):
        self._publish_raw(MQTT_TOPIC_ACCESS, payload)

    def publish_alert(self, payload: dict):
        self._publish_raw(MQTT_TOPIC_ALERT, payload)

    def publish_system(self, payload: dict):
        self._publish_raw(MQTT_TOPIC_SYSTEM, payload)

    def publish(self, payload: dict):
        # fallback generic publisher → system topic
        self.publish_system(payload)

    def disconnect(self):
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass

            self.client = None
            self.connected = False