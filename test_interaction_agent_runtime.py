"""Isolated runtime tests for Phase 4: the notification worker and the Phase 3 corrections that
make it safe to run concurrently.

    python -m unittest test_interaction_agent_runtime -v

Two things are covered here that the API tests deliberately do not exercise:

  * the background runtime -- startup, shutdown, duplicate-worker prevention, overlapping-cycle
    skipping, exception recovery and the disabled-mode no-op -- driven through the *app's* lifespan
    rather than started by hand;
  * the engine-level corrections made for Phase 4 -- atomic deduplication under real concurrency,
    congestion alerts limited to junctions still ahead of the traveller, no alerts for a finished
    journey, and per-event freshness for both open and recently resolved incidents.

Every test uses a temporary SQLite database and an in-memory fake data source; nothing touches the
runtime database, AWS, Kafka, Docker, the network or the DCRNN model.
"""
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

from webapp import server as srv
from webapp.interaction_agent import (InteractionConfig, Notification, NotificationRuntime,
                                      NotificationStore, build_notification_runtime,
                                      process_notifications)
from webapp.interaction_agent.notifications import NOTIF_INCIDENT, NOTIF_INCIDENT_RESOLVED
from webapp.interaction_api import InteractionServices
from webapp.journey_memory import iso, utc_now

JOURNEY_ID = "jr-ff466f71c8"
WORKER_NAME = "interaction-notifications"


def stamp(minutes=0):
    """An ISO timestamp ``minutes`` in the past (negative = in the future)."""
    return iso(utc_now() - timedelta(minutes=minutes))


# --------------------------------------------------------------------------- fixtures

def incident_event(event_id="ce-J001-aaaa1111", event_type="observed_slowdown", status="active",
                   severity="moderate", area="Silk Board", junctions=("J001", "J002"),
                   age_min=0):
    """One incident event in the incident agent's schema."""
    return {"event_id": event_id, "event_type": event_type, "status": status, "severity": severity,
            "area": area, "anchor_junction_id": junctions[0], "anchor_name": area + " Junction",
            "affected_junction_ids": list(junctions),
            "affected_junctions": [{"junction_id": j, "name": j} for j in junctions],
            "affected_junction_count": len(junctions),
            "avg_current_speed_kmh": 8.5, "avg_free_flow_speed_kmh": 45.0,
            "avg_predicted_speed_kmh": 7.0, "forecast_horizon_min": 15,
            "first_seen": stamp(age_min + 5), "last_seen": stamp(age_min),
            "updated_at": stamp(age_min)}


def snapshot(active=(), resolved=(), age_min=0):
    """An incidents.json snapshot carrying only what the engine reads."""
    return {"made_at": stamp(age_min), "active_events": list(active),
            "recent_resolved": list(resolved)}


def journey(journey_id=JOURNEY_ID, status="active", position_cum_m=0.0,
            spine=(("J001", 0.0), ("J002", 1200.0), ("J003", 2400.0)), action="hold",
            incident_ids=()):
    return {"journey_id": journey_id, "label": "Home to Office", "status": status,
            "updated_at": stamp(0), "position_cum_m": position_cum_m,
            "spine": [{"junction_id": junction, "cum_m": cum} for junction, cum in spine],
            "route_nodes": [junction for junction, _ in spine],
            "decision": {"journey_id": journey_id, "action": action, "progress_pct": 50.0,
                         "remaining_km": 5.0, "eta_min": 20.0, "baseline_eta_min": 22.0,
                         "eta_delta_min": 2.0, "incident_ids": list(incident_ids),
                         "confidence": "medium", "updated_at": stamp(0)}}


class FakeSources(object):
    """In-memory stand-in for :class:`LiveDataSources`. No files, no HTTP, no AWS."""

    def __init__(self, incidents=None, journeys=None, stale_after_min=10.0):
        self._incidents = incidents
        self._journeys = list(journeys or [])
        self.stale_after_min = stale_after_min

    def incidents(self):
        return self._incidents

    def features(self):
        return {"made_at": stamp(0), "areas": []}

    def journeys(self):
        return [dict(item) for item in self._journeys]

    def journey(self, journey_id):
        for item in self._journeys:
            if item.get("journey_id") == journey_id:
                return dict(item)
        return None

    def switches(self, journey_id=None):
        return []

    def system_health(self):
        return {"status": "ok"}


# --------------------------------------------------------------------------- base case

class RuntimeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = os.path.join(self.tmp.name, "runtime.sqlite")

    def make_store(self, enabled=True):
        store = NotificationStore(db_path=self.db_path, enabled=enabled)
        self.addCleanup(store.close)
        store.connect()
        return store

    def config(self, **overrides):
        overrides.setdefault("memory_db_path", self.db_path)
        overrides.setdefault("llm_provider", "mock")
        overrides.setdefault("notifications_enabled", True)
        overrides.setdefault("notification_cooldown_min", 0.0)
        overrides.setdefault("notification_poll_interval_s", 1.0)
        return InteractionConfig(**overrides)

    @staticmethod
    def wait_until(predicate, timeout=6.0, interval=0.02):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(interval)
        return predicate()


# --------------------------------------------------------------------------- 1. worker lifecycle

class TestNotificationWorker(RuntimeTestCase):
    def test_construction_starts_nothing(self):
        runtime = NotificationRuntime(FakeSources(), self.make_store(), config=self.config())
        self.assertFalse(runtime.running)
        self.assertFalse(any(t.name == WORKER_NAME and t.is_alive()
                             for t in threading.enumerate()))

    def test_worker_starts_runs_and_stops(self):
        calls = []

        def processor(sources, store, config=None):
            calls.append(1)
            return {"enabled": True, "generated": 0}

        runtime = NotificationRuntime(FakeSources(), self.make_store(), config=self.config(),
                                      processor=processor)
        self.assertTrue(runtime.start())
        self.assertTrue(runtime.running)
        self.assertTrue(self.wait_until(lambda: calls), "worker never ran a cycle")
        runtime.stop()
        self.assertFalse(runtime.running)
        self.assertFalse(any(t.name == WORKER_NAME and t.is_alive()
                             for t in threading.enumerate()))

    def test_second_start_does_not_spawn_a_second_worker(self):
        runtime = NotificationRuntime(FakeSources(), self.make_store(),
                                      config=self.config(notification_poll_interval_s=60.0),
                                      processor=lambda *a, **k: {"enabled": True})
        self.assertTrue(runtime.start())
        try:
            self.assertFalse(runtime.start())                 # idempotent
            workers = [t for t in threading.enumerate()
                       if t.name == WORKER_NAME and t.is_alive()]
            self.assertEqual(len(workers), 1)
        finally:
            runtime.stop()

    def test_disabled_runtime_is_a_noop(self):
        runtime = NotificationRuntime(FakeSources(), self.make_store(),
                                      config=self.config(notifications_enabled=False),
                                      processor=lambda *a, **k: {"enabled": True})
        self.assertFalse(runtime.enabled)
        self.assertFalse(runtime.start())                     # starts nothing
        self.assertFalse(runtime.running)
        self.assertFalse(any(t.name == WORKER_NAME and t.is_alive()
                             for t in threading.enumerate()))
        runtime.stop()                                        # safe when never started

    def test_stop_is_safe_to_call_repeatedly(self):
        runtime = NotificationRuntime(FakeSources(), self.make_store(),
                                      config=self.config(notification_poll_interval_s=60.0),
                                      processor=lambda *a, **k: {"enabled": True})
        runtime.start()
        runtime.stop()
        runtime.stop()
        self.assertFalse(runtime.running)

    def test_runtime_recovers_from_a_failing_cycle(self):
        calls = []

        def processor(sources, store, config=None):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("simulated upstream failure")
            return {"enabled": True, "generated": 1}

        runtime = NotificationRuntime(FakeSources(), self.make_store(),
                                      config=self.config(notification_poll_interval_s=1.0),
                                      processor=processor)
        runtime.start()
        try:
            self.assertTrue(self.wait_until(lambda: len(calls) >= 2, timeout=8.0),
                            "worker stopped after a failing cycle")
        finally:
            runtime.stop()
        status = runtime.status()
        self.assertGreaterEqual(status["cycles"], 2)
        self.assertIsNone(status["last_error"])               # the last cycle succeeded

    def test_overlapping_cycles_are_skipped(self):
        entered = threading.Event()
        release = threading.Event()

        def processor(sources, store, config=None):
            entered.set()
            release.wait(5)
            return {"enabled": True, "generated": 0}

        runtime = NotificationRuntime(FakeSources(), self.make_store(), config=self.config(),
                                      processor=processor)
        first = {}
        thread = threading.Thread(target=lambda: first.update(runtime.run_once()))
        thread.start()
        self.assertTrue(entered.wait(5), "first cycle never started")
        overlap = runtime.run_once()                          # lands mid-cycle
        self.assertEqual(overlap.get("skipped"), "overlap")
        release.set()
        thread.join(5)
        self.assertNotIn("skipped", first)

    def test_status_reports_counts_only(self):
        runtime = NotificationRuntime(FakeSources(), self.make_store(), config=self.config())
        runtime.run_once()
        status = runtime.status()
        self.assertTrue(status["enabled"])
        self.assertIn("cycles", status)
        self.assertIsNotNone(status["last_cycle_at"])
        self.assertNotIn("system_prompt", status)

    def test_run_once_never_raises(self):
        def boom(sources, store, config=None):
            raise ValueError("nope")

        runtime = NotificationRuntime(FakeSources(), self.make_store(), config=self.config(),
                                      processor=boom)
        result = runtime.run_once()                           # must not propagate
        self.assertIn("error", result)
        self.assertEqual(runtime.status()["last_error"], "nope")

    def test_build_notification_runtime(self):
        runtime = build_notification_runtime(FakeSources(), self.make_store(), config=self.config())
        self.assertIsInstance(runtime, NotificationRuntime)


# --------------------------------------------------------------------------- 2. app lifespan

class TestLifespanIntegration(RuntimeTestCase):
    def test_lifespan_starts_and_stops_the_worker(self):
        config = self.config(notification_poll_interval_s=60.0)
        services = InteractionServices(config=config, sources=FakeSources())
        app = srv.create_app(model_service_url="http://fake", graph=None, spots=None,
                             interaction=services)
        self.assertFalse(services.runtime.running)
        with TestClient(app) as client:                       # entering runs the lifespan
            self.assertTrue(services.started)
            self.assertTrue(services.runtime.running)
            self.assertEqual(client.get("/api/interaction/status").json()
                             ["notification_runtime"]["running"], True)
            # A second lifecycle-style start must not create a second worker.
            self.assertFalse(services.runtime.start())
            workers = [t for t in threading.enumerate()
                       if t.name == WORKER_NAME and t.is_alive()]
            self.assertEqual(len(workers), 1)
        self.assertFalse(services.runtime.running)            # exiting stopped it
        self.assertFalse(services.started)

    def test_lifespan_with_notifications_disabled_starts_no_worker(self):
        config = self.config(notifications_enabled=False)
        services = InteractionServices(config=config, sources=FakeSources())
        app = srv.create_app(model_service_url="http://fake", graph=None, spots=None,
                             interaction=services)
        with TestClient(app) as client:
            self.assertFalse(services.runtime.running)
            self.assertFalse(client.get("/api/interaction/status").json()
                             ["notification_runtime"]["running"])
        self.assertFalse(services.runtime.running)


# --------------------------------------------------------------------------- 3. atomic dedup

class TestAtomicDedup(RuntimeTestCase):
    def test_only_one_of_many_racing_writers_wins(self):
        store = self.make_store()
        workers = 8
        barrier = threading.Barrier(workers)
        results = []
        lock = threading.Lock()

        def racer():
            barrier.wait()
            notification = Notification(session_id="cs-race", notification_type=NOTIF_INCIDENT,
                                        title="t", message="m", severity="moderate")
            outcome = store.add_guarded(notification, dedup_key="key-race", signature="sig-1",
                                        cooldown_min=0.0)
            with lock:
                results.append(outcome)

        threads = [threading.Thread(target=racer) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        written = [item for item in results if item[0] is not None]
        self.assertEqual(len(written), 1, results)            # exactly one winner
        self.assertEqual(store.notification_count("cs-race"), 1)
        losers = [item for item in results if item[0] is None]
        self.assertTrue(all(item[1] == "duplicate" for item in losers), results)

    def test_guarded_write_reports_unavailable_when_offline(self):
        store = NotificationStore(db_path=self.db_path, enabled=False)
        self.addCleanup(store.close)
        notification = Notification(session_id="cs-x", notification_type=NOTIF_INCIDENT,
                                    title="t", message="m", severity="moderate")
        notification_id, reason = store.add_guarded(notification, dedup_key="k", signature="s")
        self.assertIsNone(notification_id)
        self.assertEqual(reason, "unavailable")

    def test_two_engine_passes_emit_once(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        sources = FakeSources(incidents=snapshot(active=[incident_event(junctions=("J002",))]),
                              journeys=[journey(position_cum_m=0.0)])
        first = process_notifications(sources, store, config=self.config())
        second = process_notifications(sources, store, config=self.config())
        self.assertEqual(first["generated"], 1)
        self.assertEqual(second["generated"], 0)
        self.assertGreaterEqual(second["suppressed"], 1)
        self.assertEqual(store.notification_count("cs-1"), 1)


# --------------------------------------------------------------------------- 4. engine corrections

class TestEngineRelevance(RuntimeTestCase):
    def subscribe(self, store, session_id="cs-1"):
        store.subscribe(session_id, JOURNEY_ID)

    def test_incident_ahead_on_route_is_alerted(self):
        store = self.make_store()
        self.subscribe(store)
        sources = FakeSources(incidents=snapshot(active=[incident_event(junctions=("J003",))]),
                              journeys=[journey(position_cum_m=1500.0)])
        result = process_notifications(sources, store, config=self.config())
        self.assertEqual(result["generated"], 1)
        rows = store.list_notifications("cs-1")
        self.assertEqual(rows[0]["notification_type"], NOTIF_INCIDENT)

    def test_incident_on_a_passed_junction_is_not_alerted(self):
        store = self.make_store()
        self.subscribe(store)
        # position 1500 m along the spine: J001 (0 m) and J002 (1200 m) are behind the traveller.
        sources = FakeSources(incidents=snapshot(active=[incident_event(junctions=("J002",))]),
                              journeys=[journey(position_cum_m=1500.0)])
        result = process_notifications(sources, store, config=self.config())
        self.assertEqual(result["generated"], 0)
        self.assertEqual(store.notification_count("cs-1"), 0)

    def test_incident_matched_by_decision_id_is_still_alerted(self):
        store = self.make_store()
        self.subscribe(store)
        event = incident_event(event_id="ce-xyz", junctions=("J999",))
        sources = FakeSources(incidents=snapshot(active=[event]),
                              journeys=[journey(position_cum_m=1500.0,
                                                incident_ids=["ce-xyz"])])
        result = process_notifications(sources, store, config=self.config())
        self.assertEqual(result["generated"], 1)

    def test_completed_journey_gets_no_congestion_alert(self):
        store = self.make_store()
        self.subscribe(store)
        sources = FakeSources(incidents=snapshot(active=[incident_event(junctions=("J003",))]),
                              journeys=[journey(status="arrived", position_cum_m=0.0)])
        result = process_notifications(sources, store, config=self.config())
        self.assertEqual(result["generated"], 0)

    def test_incident_on_unrelated_junctions_is_not_alerted(self):
        store = self.make_store()
        self.subscribe(store)
        sources = FakeSources(incidents=snapshot(active=[incident_event(junctions=("J500",))]),
                              journeys=[journey(position_cum_m=0.0)])
        result = process_notifications(sources, store, config=self.config())
        self.assertEqual(result["generated"], 0)

    def test_stale_incident_event_is_not_alerted(self):
        store = self.make_store()
        self.subscribe(store)
        # The snapshot is fresh, but this one event inside it is an hour old.
        sources = FakeSources(incidents=snapshot(active=[incident_event(junctions=("J003",),
                                                                        age_min=60)],
                                                 age_min=0),
                              journeys=[journey(position_cum_m=0.0)])
        result = process_notifications(sources, store, config=self.config())
        self.assertEqual(result["generated"], 0)

    def test_fresh_resolved_incident_is_alerted(self):
        store = self.make_store()
        self.subscribe(store)
        resolved = incident_event(event_id="ce-done", status="resolved", junctions=("J003",),
                                  age_min=1)
        sources = FakeSources(incidents=snapshot(resolved=[resolved]),
                              journeys=[journey(position_cum_m=0.0)])
        result = process_notifications(sources, store, config=self.config())
        self.assertEqual(result["generated"], 1)
        self.assertEqual(store.list_notifications("cs-1")[0]["notification_type"],
                         NOTIF_INCIDENT_RESOLVED)

    def test_stale_resolved_incident_is_not_alerted(self):
        store = self.make_store()
        self.subscribe(store)
        resolved = incident_event(event_id="ce-old", status="resolved", junctions=("J003",),
                                  age_min=90)
        sources = FakeSources(incidents=snapshot(resolved=[resolved]),
                              journeys=[journey(position_cum_m=0.0)])
        result = process_notifications(sources, store, config=self.config())
        self.assertEqual(result["generated"], 0)

    def test_stale_snapshot_produces_no_incident_alert(self):
        store = self.make_store()
        self.subscribe(store)
        sources = FakeSources(incidents=snapshot(active=[incident_event(junctions=("J003",))],
                                                 age_min=90),
                              journeys=[journey(position_cum_m=0.0)])
        result = process_notifications(sources, store, config=self.config())
        self.assertFalse(result["incidents_fresh"])
        self.assertEqual(result["generated"], 0)

    def test_disabled_config_produces_nothing(self):
        store = self.make_store()
        self.subscribe(store)
        sources = FakeSources(incidents=snapshot(active=[incident_event(junctions=("J003",))]),
                              journeys=[journey(position_cum_m=0.0)])
        result = process_notifications(sources, store,
                                       config=self.config(notifications_enabled=False))
        self.assertFalse(result["enabled"])
        self.assertEqual(result["generated"], 0)

    def test_disabled_store_produces_nothing(self):
        store = NotificationStore(db_path=self.db_path, enabled=False)
        self.addCleanup(store.close)
        sources = FakeSources(incidents=snapshot(active=[incident_event(junctions=("J003",))]),
                              journeys=[journey(position_cum_m=0.0)])
        result = process_notifications(sources, store, config=self.config())
        self.assertFalse(result["enabled"])
        self.assertEqual(result["generated"], 0)

    def test_no_subscriptions_means_no_notifications(self):
        store = self.make_store()
        sources = FakeSources(incidents=snapshot(active=[incident_event(junctions=("J003",))]),
                              journeys=[journey(position_cum_m=0.0)])
        result = process_notifications(sources, store, config=self.config())
        self.assertEqual(result["generated"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
