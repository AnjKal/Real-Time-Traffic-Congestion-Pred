"""Background runtime for proactive notifications (Phase 4).

Phase 3 built the engine (``process_notifications``) as a plain, independently callable function
that never starts anything. Phase 4 gives it a controlled home: one daemon thread that runs the
engine on the configured interval, started and stopped by the web application's own lifespan.

Design rules, all of them load-bearing:

* **Started by the lifecycle, never by import.** Constructing :class:`NotificationRuntime` starts
  nothing. ``start()`` is called from ``webapp/server.py``'s lifespan, so a test or a CLI that
  only imports this module gets no thread.
* **One worker, ever.** ``start()`` is idempotent: a second call while running does nothing, so a
  double lifespan or a reload cannot spawn a second poller.
* **No overlapping cycles.** The thread runs cycles one at a time, and ``run_once`` takes a
  non-blocking lock so a manual trigger that lands mid-cycle is skipped rather than interleaved.
* **Never fatal.** A failing cycle is logged and recorded; the thread keeps going, and the app is
  never brought down by a bad poll of a data file or a locked database.
* **No LLM, no AWS, no Kafka.** The engine it drives is deterministic and reads the JSON the other
  agents already published. This module adds no broker client and no model call.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from webapp.interaction_agent.config import InteractionConfig
from webapp.interaction_agent.notifications import process_notifications
from webapp.journey_memory import iso, utc_now

LOG = logging.getLogger(__name__)

DEFAULT_JOIN_TIMEOUT_S = 5.0
MIN_POLL_INTERVAL_S = 1.0


def _summarise(result) -> Dict[str, Any]:
    """A small, non-sensitive summary of a cycle, safe to report on the health endpoint."""
    if not isinstance(result, dict):
        return {}
    return {"enabled": bool(result.get("enabled")), "generated": int(result.get("generated") or 0),
            "suppressed": int(result.get("suppressed") or 0),
            "skipped": int(result.get("skipped") or 0),
            "scanned_journeys": int(result.get("scanned_journeys") or 0),
            "incidents_fresh": bool(result.get("incidents_fresh")),
            "journeys_fresh": int(result.get("journeys_fresh") or 0)}


class NotificationRuntime:
    """Runs :func:`process_notifications` on a timer until told to stop."""

    def __init__(self, sources, store, config: Optional[InteractionConfig] = None,
                 processor=None):
        self.sources = sources
        self.store = store
        self.config = config or InteractionConfig.from_env()
        # Injectable so a test can drive the loop without a real store or data files.
        self._processor = processor or process_notifications
        self._lock = threading.RLock()
        self._cycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._state: Dict[str, Any] = {"cycles": 0, "last_cycle_at": None, "last_result": None,
                                       "last_error": None, "last_started_at": None}

    # -- lifecycle -------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.config.notifications_enabled)

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._started and self._thread is not None and self._thread.is_alive())

    def start(self) -> bool:
        """Start the poller. Returns True if this call started it.

        Idempotent: a second call while running returns False and starts nothing, which is how a
        duplicate worker is prevented. With notifications disabled the call is a harmless no-op
        that still marks the runtime as started so it is not retried on every request.
        """
        with self._lock:
            if self._started:
                return False
            self._started = True
            if not self.enabled:
                LOG.info("Interaction notifications are disabled; no worker started")
                return False
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="interaction-notifications",
                                            daemon=True)
            self._thread.start()
            LOG.info("Interaction notification worker started (interval %.0fs)",
                     float(self.config.notification_poll_interval_s))
            return True

    def stop(self, timeout: float = DEFAULT_JOIN_TIMEOUT_S) -> None:
        """Stop the poller and wait for the current cycle to finish. Safe to call repeatedly."""
        with self._lock:
            if not self._started:
                return
            self._stop.set()
            thread = self._thread
            self._thread = None
            self._started = False
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    # -- one cycle -------------------------------------------------------------
    def run_once(self) -> Dict[str, Any]:
        """Run a single cycle synchronously. Never raises; overlapping calls are skipped."""
        if not self._cycle_lock.acquire(blocking=False):
            return {"skipped": "overlap"}
        try:
            return self._process()
        finally:
            self._cycle_lock.release()

    def _process(self) -> Dict[str, Any]:
        started = iso(utc_now())
        with self._lock:
            self._state["last_started_at"] = started
        try:
            result = self._processor(self.sources, self.store, config=self.config)
        except Exception as exc:                          # a bad cycle must never stop the worker
            LOG.exception("interaction notification cycle failed: %s", exc)
            with self._lock:
                self._state["cycles"] += 1
                self._state["last_cycle_at"] = started
                self._state["last_error"] = str(exc)
            return {"error": str(exc)}
        with self._lock:
            self._state["cycles"] += 1
            self._state["last_cycle_at"] = started
            self._state["last_result"] = _summarise(result)
            self._state["last_error"] = None
        return result

    def _loop(self) -> None:
        interval = max(MIN_POLL_INTERVAL_S, float(self.config.notification_poll_interval_s))
        while not self._stop.is_set():
            self.run_once()
            if self._stop.wait(interval):                 # wakes immediately on stop()
                break

    # -- reporting -------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """Runtime state for the health endpoint. Counts only; no notification text, no secrets."""
        with self._lock:
            return {"enabled": self.enabled, "running": self.running,
                    "poll_interval_s": float(self.config.notification_poll_interval_s),
                    "cycles": int(self._state["cycles"]),
                    "last_cycle_at": self._state["last_cycle_at"],
                    "last_started_at": self._state["last_started_at"],
                    "last_error": self._state["last_error"],
                    "last_result": self._state["last_result"]}


def build_notification_runtime(sources, store, config=None, processor=None) -> NotificationRuntime:
    """Build the runtime from configuration (or the environment when no config is given)."""
    return NotificationRuntime(sources, store, config=config, processor=processor)
