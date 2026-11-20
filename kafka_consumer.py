"""Kafka consumers and poll-batch assembly for the live traffic pipeline."""
from __future__ import absolute_import

import json
import logging
import threading

from kafka_topics import READING_RECORD, POLL_COMPLETE_RECORD

LOG = logging.getLogger(__name__)


class JsonConsumer(threading.Thread):
    """Consume JSON records and commit only after the handler succeeds.

    When ``commit_when`` is supplied, the handler returns whether the current record completes
    a processable unit. This lets the raw consumer commit only after a poll marker has been
    fully handled, so a restart replays an interrupted poll instead of losing its earlier rows.
    """

    def __init__(self, bootstrap_servers, topic, group_id, on_message,
                 stop_event=None, auto_offset_reset="latest", commit_when=None):
        super(JsonConsumer, self).__init__(daemon=True)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.on_message = on_message
        self.stop_event = stop_event or threading.Event()
        self.auto_offset_reset = auto_offset_reset
        self.commit_when = commit_when
        self.consumer = None

    def run(self):
        from kafka import KafkaConsumer
        while not self.stop_event.is_set():
            try:
                self.consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    auto_offset_reset=self.auto_offset_reset,
                    enable_auto_commit=False,
                    value_deserializer=lambda value: json.loads(value.decode("utf-8")),
                    consumer_timeout_ms=1000)
                LOG.info("Consuming %s as group %s", self.topic, self.group_id)
                while not self.stop_event.is_set():
                    for record in self.consumer:
                        if self.stop_event.is_set():
                            break
                        try:
                            should_commit = self.on_message(record.value)
                            if self.commit_when is None or should_commit:
                                self.consumer.commit()
                        except Exception:
                            LOG.exception("Failed to process %s offset %s; retrying",
                                          self.topic, record.offset)
                            from kafka.structs import TopicPartition
                            self.consumer.seek(TopicPartition(record.topic, record.partition),
                                               record.offset)
                            self.stop_event.wait(2)
                            break
            except Exception:
                if not self.stop_event.is_set():
                    LOG.exception("Kafka consumer for %s disconnected; reconnecting", self.topic)
                    self.stop_event.wait(2)
            finally:
                if self.consumer is not None:
                    try:
                        self.consumer.close()
                    except Exception:
                        LOG.debug("Error while closing consumer for %s", self.topic,
                                  exc_info=True)
                    self.consumer = None

    def close(self):
        self.stop_event.set()


class RawPollAssembler(object):
    """Collect raw records by poll ID and call ``on_batch`` at the completion marker."""

    def __init__(self, on_batch):
        self.on_batch = on_batch
        self._polls = {}
        self._lock = threading.Lock()

    def handle(self, message):
        record_type = message.get("record_type")
        poll_id = str(message.get("poll_id", ""))
        if not poll_id:
            LOG.warning("Ignoring Kafka raw record without a poll_id")
            return True

        if record_type == READING_RECORD:
            junction_id = str(message.get("junction_id", ""))
            if not junction_id:
                LOG.warning("Ignoring raw reading without a junction_id in poll %s", poll_id)
                return False
            with self._lock:
                self._polls.setdefault(poll_id, {})[junction_id] = message
            return False

        if record_type == POLL_COMPLETE_RECORD:
            with self._lock:
                readings = list(self._polls.get(poll_id, {}).values())
            self.on_batch(poll_id, message.get("timestamp"), readings, message)
            with self._lock:
                self._polls.pop(poll_id, None)
            return True

        LOG.warning("Ignoring unknown raw record type %r", record_type)
        return True
