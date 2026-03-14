import json
from datetime import datetime

import paho.mqtt.client as mqtt

from config import (
    MQTT_ENABLED,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_KEEPALIVE,
    MQTT_TOPIC_ACCESS,
    MQTT_TOPIC_ALERT,
    MQTT_TOPIC_SYSTEM,
    CAMERA_ID,
)


class MQTTService:
    def __init__(self):
        self.client = None
        self.enabled = MQTT_ENABLED
        self.connected = False

    # =========================================================
    # Connection
    # =========================================================

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

    def disconnect(self):
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass

            self.client = None
            self.connected = False

    # =========================================================
    # Internal Helpers
    # =========================================================

    def _now_str(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _normalize_payload(self, payload: dict | None):
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            return {"value": payload}
        return payload

    def _build_envelope(self, topic_type: str, payload: dict):
        """
        Standard event envelope for later Odoo consumption.
        Keeps backward compatibility by preserving all incoming fields
        while also adding metadata if missing.
        """
        payload = self._normalize_payload(payload)

        event = dict(payload)

        event.setdefault("timestamp", self._now_str())
        event.setdefault("camera_id", CAMERA_ID)
        event.setdefault("event_family", topic_type)
        event.setdefault("source", "face_recognition_pipeline")
        event.setdefault("schema_version", "1.0")

        return event

    def _publish_raw(self, topic, payload):
        if not self.enabled:
            return

        if self.client is None or not self.connected:
            return

        try:
            self.client.publish(
                topic,
                json.dumps(payload, ensure_ascii=False),
                qos=0,
                retain=False,
            )
        except Exception as e:
            print(f"[MQTT] Publish failed: {e}")

    def _publish_structured(self, topic: str, topic_type: str, payload: dict):
        event = self._build_envelope(topic_type, payload)
        self._publish_raw(topic, event)

    # =========================================================
    # Public Publishers
    # =========================================================

    def publish_access(self, payload: dict):
        self._publish_structured(MQTT_TOPIC_ACCESS, "access", payload)

    def publish_alert(self, payload: dict):
        self._publish_structured(MQTT_TOPIC_ALERT, "alert", payload)

    def publish_system(self, payload: dict):
        self._publish_structured(MQTT_TOPIC_SYSTEM, "system", payload)

    def publish(self, payload: dict):
        # fallback generic publisher → system topic
        self.publish_system(payload)

    # =========================================================
    # Odoo-Friendly Explicit Publishers
    # Optional helpers for cleaner future usage
    # =========================================================

    def publish_access_decision(
        self,
        person_id: str | None,
        person_type: str | None,
        decision: str,
        reason: str,
        zone_id: str | None = None,
        zone_name: str | None = None,
        track_id: str | None = None,
        identity_score: float | None = None,
        display_name: str | None = None,
        predicted_gender: str | None = None,
        predicted_age_range: str | None = None,
        attributes_locked: int | None = None,
        extra: dict | None = None,
    ):
        payload = {
            "event_type": "access_decision",
            "person_id": person_id,
            "person_type": person_type,
            "display_name": display_name,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "track_id": track_id,
            "identity_score": identity_score,
            "decision": decision,
            "reason": reason,
            "predicted_gender": predicted_gender,
            "predicted_age_range": predicted_age_range,
            "attributes_locked": attributes_locked,
            "data": extra or {},
        }
        self.publish_access(payload)

    def publish_alert_event(
        self,
        alert_type: str,
        person_id: str | None = None,
        person_type: str | None = None,
        zone_id: str | None = None,
        zone_name: str | None = None,
        track_id: str | None = None,
        status: str = "open",
        reason: str | None = None,
        display_name: str | None = None,
        predicted_gender: str | None = None,
        predicted_age_range: str | None = None,
        attributes_locked: int | None = None,
        extra: dict | None = None,
    ):
        payload = {
            "event_type": "alert_created",
            "alert_type": alert_type,
            "status": status,
            "reason": reason,
            "person_id": person_id,
            "person_type": person_type,
            "display_name": display_name,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "track_id": track_id,
            "predicted_gender": predicted_gender,
            "predicted_age_range": predicted_age_range,
            "attributes_locked": attributes_locked,
            "data": extra or {},
        }
        self.publish_alert(payload)

    def publish_system_event(
        self,
        event_type: str,
        person_id: str | None = None,
        person_type: str | None = None,
        zone_id: str | None = None,
        zone_name: str | None = None,
        track_id: str | None = None,
        identity_score: float | None = None,
        display_name: str | None = None,
        predicted_gender: str | None = None,
        predicted_age_range: str | None = None,
        attributes_locked: int | None = None,
        extra: dict | None = None,
    ):
        payload = {
            "event_type": event_type,
            "person_id": person_id,
            "person_type": person_type,
            "display_name": display_name,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "track_id": track_id,
            "identity_score": identity_score,
            "predicted_gender": predicted_gender,
            "predicted_age_range": predicted_age_range,
            "attributes_locked": attributes_locked,
            "data": extra or {},
        }
        self.publish_system(payload)