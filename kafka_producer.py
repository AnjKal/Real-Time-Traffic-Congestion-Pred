"""Kafka producers for the live traffic pipeline."""
from __future__ import absolute_import

import json
import logging
import time

from kafka_topics import RAW_TOPIC, PREDICTIONS_TOPIC, READING_RECORD, POLL_COMPLETE_RECORD

LOG = logging.getLogger(__name__)


def _json_bytes(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _key_bytes(value):
    if value is None:
        return None
    return str(value).encode("utf-8")


class JsonProducer(object):
    """Synchronous JSON producer. A send returns only after Kafka acknowledges it."""

    def __init__(self, bootstrap_servers, topic, retries=30):
        from kafka import KafkaProducer
        self.topic = topic
        self._retries = int(retries)
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=_json_bytes,
            key_serializer=_key_bytes,
            acks="all",
            max_in_flight_requests_per_connection=1,
            retries=max(1, self._retries))

    def send(self, value, key=None):
        last_error = None
        for attempt in range(self._retries):
            try:
                return self._producer.send(self.topic, key=key, value=value).get(timeout=30)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self._retries:
                    LOG.warning("Kafka send to %s failed; retry %d/%d", self.topic,
                                attempt + 1, self._retries)
                    time.sleep(min(2 ** attempt, 10))
        raise last_error

    def send_poll(self, poll_id, timestamp, rows, expected_count, failed_count):
        """Publish readings followed by a marker so consumers can process one poll atomically."""
        sent = 0
        for row in rows:
            message = dict(row)
            message.update({"record_type": READING_RECORD, "poll_id": str(poll_id),
                            "timestamp": timestamp})
            self.send(message, key=poll_id)
            sent += 1
        self.send({"record_type": POLL_COMPLETE_RECORD, "poll_id": str(poll_id),
                   "timestamp": timestamp, "expected_count": int(expected_count),
                   "successful_count": sent, "failed_count": int(failed_count)}, key=poll_id)
        return sent

    def close(self):
        self._producer.flush(timeout=30)
        self._producer.close(timeout=30)


def create_pipeline_producers(bootstrap_servers):
    return (JsonProducer(bootstrap_servers, RAW_TOPIC),
            JsonProducer(bootstrap_servers, PREDICTIONS_TOPIC))
