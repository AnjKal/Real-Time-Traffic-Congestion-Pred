"""Wiring that feeds the journey agent and drives the simulated car.

The agent itself knows nothing about Kafka or HTTP. This module gives it forecasts and congestion
events, publishes its decisions, and owns the simulator thread.

Two ways in, one decision path:

* With ``KAFKA_BOOTSTRAP_SERVERS`` set (how docker compose runs it) the agent consumes
  ``traffic.predictions`` as group ``traffic-journey`` and ``traffic.congestion_events`` as group
  ``traffic-journey-incidents``, and publishes to ``traffic.journey_events``. Those groups are its
  own, so the model, dashboard, feature and incident groups keep their offsets untouched.
* With no broker (plain ``python run_webapp.py``) the same agent is fed by polling the model
  service over HTTP, so the decision logic has exactly one code path either way.

Either way the runtime fetches a full book of forecasts over HTTP at startup, so a journey started
in the first two minutes is judged against real predictions instead of waiting for the next poll.
"""
from __future__ import absolute_import

import logging
import os
import threading

import requests

from kafka_topics import (CONGESTION_EVENTS_TOPIC, FORECAST_RECORD, JOURNEY_EVENTS_TOPIC,
                          PREDICTIONS_TOPIC)
from webapp.journey_agent import ForecastBook
from webapp.journey_simulator import JourneySimulator

LOG = logging.getLogger(__name__)

HTTP_REFRESH_SECONDS = 120.0   # the model's own poll cadence
HTTP_RETRY_SECONDS = 10.0      # while the model service is still warming up
REQUEST_TIMEOUT = 20


class JourneyRuntime(object):
    """Feeds the agent, publishes its decisions, and runs the car."""

    def __init__(self, agent, model_service_url, bootstrap_servers=None,
                 simulator=None):
        self.agent = agent
        self.model_service_url = (model_service_url or "").rstrip("/")
        self.bootstrap_servers = (bootstrap_servers
                                  or os.environ.get("KAFKA_BOOTSTRAP_SERVERS") or "").strip()
        self.simulator = simulator or JourneySimulator(agent)
        self._stop = threading.Event()
        self._poller = None
        self._consumers = []
        self._producer = None
        self._started = False

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if self._started:
            return
        self._started = True
        if self.bootstrap_servers:
            self._start_kafka()
        else:
            LOG.info("No KAFKA_BOOTSTRAP_SERVERS set; the journey agent will poll %s",
                     self.model_service_url)
        self._poller = threading.Thread(target=self._poll_loop, name="journey-forecasts",
                                        daemon=True)
        self._poller.start()
        self.simulator.start()

    def stop(self):
        self._stop.set()
        self.simulator.close()
        for consumer in self._consumers:
            try:
                consumer.close()
            except Exception:
                LOG.debug("Error closing a journey consumer", exc_info=True)
        for consumer in self._consumers:
            try:
                consumer.join(timeout=5)
            except Exception:
                LOG.debug("Error joining a journey consumer", exc_info=True)
        self._consumers = []
        if self._producer is not None:
            try:
                self._producer.close()
            except Exception:
                LOG.debug("Error closing the journey producer", exc_info=True)
            self._producer = None
        self._started = False

    # ------------------------------------------------------------------ Kafka
    def _start_kafka(self):
        from kafka_consumer import JsonConsumer
        from kafka_producer import JsonProducer
        try:
            self._producer = JsonProducer(self.bootstrap_servers, JOURNEY_EVENTS_TOPIC,
                                          retries=5)
            self.agent.set_producer(self._producer)
        except Exception:
            LOG.exception("Could not open the %s producer; decisions will not be published",
                          JOURNEY_EVENTS_TOPIC)
        # traffic.predictions from 'latest': an old forecast is of no use to a live journey.
        self._consumers.append(JsonConsumer(
            self.bootstrap_servers, PREDICTIONS_TOPIC, "traffic-journey",
            self._on_prediction, auto_offset_reset="latest"))
        # traffic.congestion_events from the beginning: the topic is compacted, so replaying it
        # rebuilds the current state of every open event.
        self._consumers.append(JsonConsumer(
            self.bootstrap_servers, CONGESTION_EVENTS_TOPIC, "traffic-journey-incidents",
            self._on_congestion_event, auto_offset_reset="earliest"))
        for consumer in self._consumers:
            consumer.start()
        LOG.info("Journey agent consuming %s and %s; publishing %s", PREDICTIONS_TOPIC,
                 CONGESTION_EVENTS_TOPIC, JOURNEY_EVENTS_TOPIC)

    def _on_prediction(self, message):
        if message.get("record_type") != FORECAST_RECORD:
            return True
        try:
            self.agent.on_forecast(message)
        except Exception:
            LOG.exception("Journey agent failed on a forecast; the offset is still committed")
        return True

    def _on_congestion_event(self, message):
        try:
            self.agent.on_congestion_event(message)
        except Exception:
            LOG.exception("Journey agent failed on a congestion event")
        return True

    # ------------------------------------------------------------------ HTTP forecasts
    def _get(self, path, params=None):
        response = requests.get(self.model_service_url + path, params=params,
                                timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def horizons(self):
        health = self._get("/health")
        return [int(h) for h in (health.get("horizons_min") or [])]

    def refresh_book(self):
        """Build a full multi-horizon book over HTTP. Returns the book, or None if not ready.

        The book is what lets the simulated car travel through time, so every horizon the model
        publishes is fetched, not just the one the page is showing.
        """
        horizons = self.horizons()
        if not horizons:
            return None
        book = None
        for horizon in horizons:
            try:
                payload = self._get("/forecast", {"horizon_min": horizon})
            except Exception:
                continue
            if book is None:
                book = ForecastBook.from_single_forecast(
                    payload, poll_id=payload.get("input_last_slot"))
                if book is not None:
                    book.merge_horizon(horizon, payload)
            else:
                book.merge_horizon(horizon, payload)
        return book

    def _poll_loop(self):
        """Keep a book of forecasts in the agent's hands, whatever else is or is not running."""
        wait = 0.0
        while not self._stop.is_set():
            if wait:
                self._stop.wait(wait)
                if self._stop.is_set():
                    break
            try:
                book = self.refresh_book()
            except Exception as error:
                LOG.info("Model service not ready for the journey agent yet (%s)", error)
                wait = HTTP_RETRY_SECONDS
                continue
            if book is None:
                wait = HTTP_RETRY_SECONDS
                continue
            previous = self.agent.book
            is_new = previous is None or previous.poll_id != book.poll_id
            # A Kafka message carries every horizon already, so only replace the book when it is
            # genuinely newer than what the agent is holding.
            if is_new:
                self.agent.set_book(book)
                LOG.info("Journey agent has forecasts for %s (horizons %s)",
                         book.poll_id, book.horizons)
                try:
                    self.agent.evaluate_all()
                except Exception:
                    LOG.exception("Re-deciding journeys on a new forecast failed")
            elif previous is not None and len(previous.horizons) < len(book.horizons):
                self.agent.set_book(book)
            wait = HTTP_REFRESH_SECONDS

    # ------------------------------------------------------------------ debugging
    def status(self):
        book = self.agent.book
        return {"kafka": bool(self.bootstrap_servers),
                "bootstrap_servers": self.bootstrap_servers or None,
                "model_service_url": self.model_service_url,
                "publishing": self._producer is not None,
                "consumers": [consumer.topic for consumer in self._consumers],
                "simulator_running": self.simulator.is_alive(),
                "forecast_poll_id": book.poll_id if book else None,
                "forecast_horizons": book.horizons if book else []}
