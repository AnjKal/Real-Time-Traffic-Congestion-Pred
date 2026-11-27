"""Isolated unit tests for Phase 3: proactive traffic notifications.

    python -m unittest test_interaction_agent_notifications -v

Every test uses a temporary SQLite database; nothing touches the runtime database, AWS, Kafka,
Docker or the network. The upstream data the engine consumes is identical in shape to what the
incident monitoring agent (``data/live/incidents.json``) and the journey intelligence agent
(``data/live/journey_state.json``) actually publish, so the fixtures are based on the real schemas
rather than invented ones.
"""
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.journey_memory import iso, utc_now
from webapp.interaction_agent import (ConfigError, ConversationMemory, InteractionConfig,
                                      Notification, NotificationStore, build_notification_store,
                                      evaluate_dedup, incident_relevant_to_journey,
                                      process_notifications, severity_rank)
from webapp.interaction_agent.memory import MESSAGE_TABLE, SESSION_TABLE
from webapp.interaction_agent.notifications import (NOTIF_INCIDENT, NOTIF_INCIDENT_RESOLVED,
                                                    NOTIF_JOURNEY_COMPLETED, NOTIF_JOURNEY_DELAY,
                                                    NOTIF_ROUTE_ALTERNATIVE, NOTIFICATION_TABLE,
                                                    NOTIFICATION_META_TABLE, NOTIFICATION_STATE_TABLE,
                                                    NOTIFICATION_TYPES, SUBSCRIPTION_TABLE)

# A fixed instant so cooldown windows can be asserted exactly, with no wall-clock flakiness.
BASE = utc_now()


def at(minutes):
    return BASE + timedelta(minutes=minutes)


def at_iso(minutes):
    return iso(at(minutes))


# --------------------------------------------------------------------------- fixtures
#
# Incident event: the exact field set ``incident_agent.py`` writes (``_build_event``).
INCIDENT_EVENT = {
    "event_id": "ce-J001-aaaa1111",
    "event_type": "observed_slowdown",
    "status": "active",
    "severity": "moderate",
    "area": "Silk Board",
    "anchor_junction_id": "J001",
    "anchor_name": "Silk Board Junction",
    "affected_junction_ids": ["J001", "J002"],
    "affected_junctions": [{"junction_id": "J001", "name": "Silk Board Junction"},
                           {"junction_id": "J002", "name": "Jayanagar"}],
    "affected_junction_count": 2,
    "avg_current_speed_kmh": 8.5,
    "avg_free_flow_speed_kmh": 45.0,
    "avg_predicted_speed_kmh": 7.0,
    "forecast_horizon_min": 15,
    "first_seen": at_iso(-5),
    "last_seen": at_iso(0),
    "updated_at": at_iso(0),
}

JOURNEY_ID = "jr-ff466f71c8"


def incident_event(event_id=INCIDENT_EVENT["event_id"], event_type="observed_slowdown",
                   status="active", severity="moderate", area="Silk Board",
                   junctions=("J001", "J002"), **overrides):
    """One incident event in the incident agent's schema."""
    event = dict(INCIDENT_EVENT)
    event.update({
        "event_id": event_id, "event_type": event_type, "status": status, "severity": severity,
        "area": area, "anchor_junction_id": junctions[0], "anchor_name": area + " Junction",
        "affected_junction_ids": list(junctions),
        "affected_junctions": [{"junction_id": junction, "name": junction} for junction in junctions],
        "affected_junction_count": len(junctions),
    })
    event.update(overrides)
    return event


def incident_snapshot(active=None, resolved=None, made_at=None):
    """The incidents.json snapshot: ``made_at`` plus ``active_events`` / ``recent_resolved``."""
    return {
        "made_at": made_at if made_at is not None else at_iso(0),
        "poll_id": "poll-9",
        "timestamp": at_iso(0),
        "active_events": list(active or []),
        "recent_resolved": list(resolved or []),
        "sensor_coverage_pct": 88.0,
        "model_mode": "pipeline_test",
    }


def journey(journey_id=JOURNEY_ID, label="Home to Office", status="active", updated_at=None,
            spine=None, decision=None):
    """A watched journey with its stored route (``spine`` / ``route_nodes``) and last decision."""
    return {
        "journey_id": journey_id,
        "label": label,
        "status": status,
        "total_length_km": 14.2,
        "updated_at": updated_at if updated_at is not None else at_iso(0),
        "spine": spine if spine is not None else [{"junction_id": "J001", "cum_m": 0.0},
                                                  {"junction_id": "J002", "cum_m": 1200.0}],
        "route_nodes": ["J001", "J002"],
        "decision": decision or {},
    }


def decision(action="hold", journey_id=JOURNEY_ID, updated_at=None, **overrides):
    """A journey decision in the journey agent's schema (``_finish``)."""
    base = {
        "record_type": "journey_event", "schema_version": 1, "journey_id": journey_id,
        "action": action, "status": "active", "label": "Home to Office",
        "eta_min": 27.0, "baseline_eta_min": 26.0, "eta_delta_min": 0.0,
        "remaining_eta_min": 20.0, "remaining_km": 8.1, "progress_pct": 40.0,
        "incidents_ahead": 0, "incident_ids": [], "confidence": "medium",
        "rationale": "the route holds", "alternative": None,
        "updated_at": updated_at if updated_at is not None else at_iso(0),
    }
    base.update(overrides)
    return base


class FakeSources(object):
    """In-memory stand-in for :class:`LiveDataSources`; the engine needs only these three."""

    def __init__(self, incidents=None, journeys=None, stale_after_min=10.0):
        self._incidents = incidents
        self._journeys = journeys or []
        self.stale_after_min = stale_after_min

    def incidents(self):
        return self._incidents

    def journeys(self):
        return [dict(item) for item in self._journeys]


def sources_at(now, active=None, resolved=None, journeys=None, made_at=None):
    """Build sources whose snapshot is stamped at ``now``, so it is never stale by accident."""
    return FakeSources(
        incidents=incident_snapshot(active=active, resolved=resolved,
                                    made_at=made_at if made_at is not None else iso(now)),
        journeys=journeys if journeys is not None else [journey(updated_at=iso(now))])


def table_names(path):
    con = sqlite3.connect(path)
    try:
        return {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


class NotificationTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = os.path.join(self.tmp.name, "notifications.sqlite")

    def make_store(self, **kwargs):
        kwargs.setdefault("db_path", self.db_path)
        store = NotificationStore(**kwargs)
        self.addCleanup(store.close)
        return store

    def make_config(self, **kwargs):
        kwargs.setdefault("memory_db_path", self.db_path)
        kwargs.setdefault("notifications_enabled", True)
        kwargs.setdefault("notification_cooldown_min", 0.0)      # deterministic unless overridden
        return InteractionConfig(**kwargs)

    def process(self, store, config, now, **kwargs):
        return process_notifications(sources_at(now, **kwargs), store, config=config, now=now)


# --------------------------------------------------------------------------- 1. configuration

class TestPhase3Config(unittest.TestCase):
    def test_notification_defaults(self):
        config = InteractionConfig.from_env({})
        self.assertTrue(config.notifications_enabled)
        self.assertEqual(config.notification_cooldown_min, 10.0)
        self.assertEqual(config.notification_poll_interval_s, 120.0)

    def test_notification_env_overrides(self):
        config = InteractionConfig.from_env({
            "INTERACTION_NOTIFICATIONS_ENABLED": "false",
            "INTERACTION_NOTIFICATION_COOLDOWN_MIN": "0",
            "INTERACTION_NOTIFICATION_POLL_INTERVAL_S": "30"})
        self.assertFalse(config.notifications_enabled)
        self.assertEqual(config.notification_cooldown_min, 0.0)
        self.assertEqual(config.notification_poll_interval_s, 30.0)

    def test_notifications_enabled_accepts_truthy_words(self):
        for value in ("1", "true", "YES", "on"):
            self.assertTrue(InteractionConfig.from_env(
                {"INTERACTION_NOTIFICATIONS_ENABLED": value}).notifications_enabled, value)

    def test_zero_cooldown_is_allowed(self):
        self.assertEqual(InteractionConfig.from_env(
            {"INTERACTION_NOTIFICATION_COOLDOWN_MIN": "0"}).notification_cooldown_min, 0.0)

    def test_bad_notification_values_are_rejected(self):
        with self.assertRaises(ConfigError):
            InteractionConfig.from_env({"INTERACTION_NOTIFICATIONS_ENABLED": "maybe"})
        with self.assertRaises(ConfigError):
            InteractionConfig.from_env({"INTERACTION_NOTIFICATION_COOLDOWN_MIN": "-1"})
        with self.assertRaises(ConfigError):
            InteractionConfig.from_env({"INTERACTION_NOTIFICATION_POLL_INTERVAL_S": "0"})
        with self.assertRaises(ConfigError):
            InteractionConfig(notification_poll_interval_s=0)


# --------------------------------------------------------------------------- 2. database schema

class TestNotificationSchema(NotificationTestCase):
    def test_initialization_creates_namespaced_tables(self):
        store = self.make_store()
        self.assertTrue(store.connect())
        names = table_names(self.db_path)
        self.assertLessEqual({NOTIFICATION_TABLE, NOTIFICATION_STATE_TABLE, SUBSCRIPTION_TABLE,
                              NOTIFICATION_META_TABLE}, names)
        con = sqlite3.connect(self.db_path)
        try:
            version = con.execute("SELECT value FROM %s WHERE key='schema_version'"
                                  % NOTIFICATION_META_TABLE).fetchone()[0]
        finally:
            con.close()
        self.assertEqual(version, "1")

    def test_opening_is_lazy(self):
        store = self.make_store()
        self.assertFalse(os.path.exists(self.db_path))
        self.assertFalse(store.available)
        store.subscribe("cs-1", JOURNEY_ID)
        self.assertTrue(os.path.exists(self.db_path))
        self.assertTrue(store.available)

    def test_existing_database_is_preserved(self):
        # Stand in for the LangGraph checkpoint store and the Phase 2 conversation tables that
        # already live in the runtime file.
        con = sqlite3.connect(self.db_path)
        con.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT, data BLOB)")
        con.execute("INSERT INTO checkpoints VALUES ('t1', 'c1', X'00')")
        con.commit()
        con.close()
        memory = ConversationMemory(db_path=self.db_path)
        memory.create_session("cs-keep")
        memory.add_message("cs-keep", "user", "hello")
        memory.close()

        store = self.make_store()
        self.assertTrue(store.connect())
        store.subscribe("cs-keep", JOURNEY_ID)

        names = table_names(self.db_path)
        self.assertIn("checkpoints", names)
        self.assertIn(SESSION_TABLE, names)
        self.assertIn(MESSAGE_TABLE, names)
        con = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(con.execute("SELECT thread_id, checkpoint_id FROM checkpoints")
                             .fetchall(), [("t1", "c1")])
            self.assertEqual(con.execute("SELECT COUNT(*) FROM %s" % MESSAGE_TABLE)
                             .fetchone()[0], 1)
        finally:
            con.close()

    def test_reopening_an_existing_schema_is_idempotent(self):
        first = self.make_store()
        first.subscribe("cs-1", JOURNEY_ID)
        first.close()
        second = self.make_store()
        self.assertTrue(second.connect())
        self.assertEqual(second.notification_count("cs-1"), 0)

    def test_concurrent_writes_do_not_lose_notifications(self):
        store = self.make_store()
        errors = []

        def writer(worker):
            try:
                for index in range(10):
                    store.add(Notification(session_id="cs-1", notification_type=NOTIF_INCIDENT,
                                           title="w%d-%d" % (worker, index), message="m"))
            except Exception as exc:                                    # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(worker,)) for worker in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(store.notification_count("cs-1"), 40)


# --------------------------------------------------------------------------- 3. subscriptions

class TestSubscriptions(NotificationTestCase):
    def test_subscribe_and_list(self):
        store = self.make_store()
        self.assertTrue(store.subscribe("cs-1", JOURNEY_ID))
        self.assertTrue(store.is_subscribed("cs-1", JOURNEY_ID))
        self.assertEqual(store.subscribed_journeys("cs-1"), [JOURNEY_ID])

    def test_subscribe_is_idempotent(self):
        store = self.make_store()
        self.assertTrue(store.subscribe("cs-1", JOURNEY_ID))
        self.assertFalse(store.subscribe("cs-1", JOURNEY_ID))
        self.assertEqual(len(store.subscriptions("cs-1")), 1)

    def test_unsubscribe(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        self.assertTrue(store.unsubscribe("cs-1", JOURNEY_ID))
        self.assertFalse(store.is_subscribed("cs-1", JOURNEY_ID))
        self.assertFalse(store.unsubscribe("cs-1", JOURNEY_ID))

    def test_subscribe_rejects_blank_ids(self):
        store = self.make_store()
        self.assertFalse(store.subscribe("", JOURNEY_ID))
        self.assertFalse(store.subscribe("cs-1", ""))

    def test_subscriptions_are_session_scoped(self):
        store = self.make_store()
        store.subscribe("cs-a", JOURNEY_ID)
        store.subscribe("cs-b", "jr-other")
        self.assertEqual(store.subscribed_journeys("cs-a"), [JOURNEY_ID])
        self.assertEqual(store.subscribed_journeys("cs-b"), ["jr-other"])
        self.assertEqual(len(store.subscriptions()), 2)


# --------------------------------------------------------------------------- 4. generation

class TestNotificationGeneration(NotificationTestCase):
    def test_incident_on_a_monitored_journey_notifies(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        result = self.process(store, self.make_config(), BASE, active=[incident_event()])
        self.assertEqual(result["generated"], 1)
        self.assertTrue(result["incidents_fresh"])

        row = store.list_notifications("cs-1")[0]
        self.assertEqual(row["notification_type"], NOTIF_INCIDENT)
        self.assertEqual(row["journey_id"], JOURNEY_ID)
        self.assertEqual(row["source_event_id"], "ce-J001-aaaa1111")
        self.assertEqual(row["severity"], "moderate")
        self.assertIn("Silk Board", row["message"])
        self.assertFalse(row["read"])

    def test_journey_delay_notifies(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        target = journey(updated_at=iso(BASE),
                         decision=decision("notify_delay", eta_delta_min=12.0, eta_min=38.0))
        result = self.process(store, self.make_config(), BASE, journeys=[target])
        self.assertEqual(result["generated"], 1)
        row = store.list_notifications("cs-1")[0]
        self.assertEqual(row["notification_type"], NOTIF_JOURNEY_DELAY)
        self.assertEqual(row["severity"], "moderate")

    def test_alternative_route_notifies(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        target = journey(updated_at=iso(BASE),
                         decision=decision("recommend_switch",
                                           alternative={"rank": 1, "eta_min": 18.0, "gain_min": 8.0,
                                                        "gain_pct": 20.0, "total_length_km": 12.9,
                                                        "turn_off_after_km": 1.2}))
        result = self.process(store, self.make_config(), BASE, journeys=[target])
        self.assertEqual(result["generated"], 1)
        row = store.list_notifications("cs-1")[0]
        self.assertEqual(row["notification_type"], NOTIF_ROUTE_ALTERNATIVE)
        self.assertEqual(row["severity"], "moderate")               # gain 8.0 -> moderate

    def test_resolved_incident_notifies(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        cleared = incident_event(event_id="ce-J001-done", status="resolved")
        result = self.process(store, self.make_config(), BASE, resolved=[cleared])
        self.assertEqual(result["generated"], 1)
        row = store.list_notifications("cs-1")[0]
        self.assertEqual(row["notification_type"], NOTIF_INCIDENT_RESOLVED)
        self.assertEqual(row["severity"], "info")
        self.assertEqual(row["source_event_id"], "ce-J001-done")

    def test_journey_completed_notifies(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        target = journey(status="arrived", updated_at=iso(BASE),
                         decision=decision("arrived", status="arrived", achieved_total_min=24.5))
        result = self.process(store, self.make_config(), BASE, journeys=[target])
        self.assertEqual(result["generated"], 1)
        row = store.list_notifications("cs-1")[0]
        self.assertEqual(row["notification_type"], NOTIF_JOURNEY_COMPLETED)

    def test_no_notification_without_a_subscription(self):
        store = self.make_store()
        target = journey(updated_at=iso(BASE),
                         decision=decision("notify_delay", eta_delta_min=12.0))
        result = self.process(store, self.make_config(), BASE,
                              active=[incident_event()], journeys=[target])
        self.assertEqual(result["generated"], 0)
        self.assertEqual(result["scanned_journeys"], 0)
        self.assertTrue(result["incidents_fresh"])
        self.assertEqual(store.notification_count("cs-1"), 0)

    def test_a_tool_result_alone_does_not_subscribe(self):
        """Reading a journey (as a question would) must not opt the user into notifications."""
        store = self.make_store()
        self.process(store, self.make_config(), BASE,
                     active=[incident_event()],
                     journeys=[journey(updated_at=iso(BASE),
                                       decision=decision("notify_delay", eta_delta_min=12.0))])
        self.assertFalse(store.is_subscribed("cs-1", JOURNEY_ID))
        self.assertEqual(store.notification_count("cs-1"), 0)


# --------------------------------------------------------------------------- 5. relevance

class TestRelevance(NotificationTestCase):
    def test_irrelevant_incident_is_filtered(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        elsewhere = incident_event(event_id="ce-J999-zzz", junctions=("J900", "J901"))
        result = self.process(store, self.make_config(), BASE, active=[elsewhere])
        self.assertEqual(result["generated"], 0)
        self.assertEqual(store.notification_count("cs-1"), 0)

    def test_incident_relevant_through_the_decision_ids(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        elsewhere = incident_event(event_id="ce-J999-zzz", junctions=("J900", "J901"))
        target = journey(updated_at=iso(BASE),
                         decision=decision("hold", incident_ids=["ce-J999-zzz"]))
        result = self.process(store, self.make_config(), BASE, active=[elsewhere], journeys=[target])
        self.assertEqual(result["generated"], 1)

    def test_helper_matches_junctions_or_decision_ids(self):
        on_route = {"event_id": "e1", "affected_junction_ids": ["J001"]}
        off_route = {"event_id": "e2", "affected_junction_ids": ["J900"]}
        self.assertTrue(incident_relevant_to_journey(on_route, {"J001", "J002"}, set()))
        self.assertFalse(incident_relevant_to_journey(off_route, {"J001", "J002"}, set()))
        self.assertTrue(incident_relevant_to_journey(off_route, {"J001"}, {"e2"}))
        self.assertFalse(incident_relevant_to_journey(off_route, set(), set()))

    def test_only_the_subscribed_session_is_notified(self):
        store = self.make_store()
        store.subscribe("cs-a", JOURNEY_ID)
        self.process(store, self.make_config(), BASE, active=[incident_event()])
        self.assertEqual(store.notification_count("cs-a"), 1)
        self.assertEqual(store.notification_count("cs-b"), 0)


# --------------------------------------------------------------------------- 6. session isolation

class TestSessionIsolation(NotificationTestCase):
    def test_notifications_are_session_scoped(self):
        store = self.make_store()
        store.subscribe("cs-a", JOURNEY_ID)
        self.process(store, self.make_config(), BASE, active=[incident_event()])
        self.assertEqual(store.notification_count("cs-a"), 1)
        self.assertEqual(store.list_notifications("cs-b"), [])

    def test_each_subscribed_session_gets_its_own_copy(self):
        store = self.make_store()
        store.subscribe("cs-a", JOURNEY_ID)
        store.subscribe("cs-b", JOURNEY_ID)
        result = self.process(store, self.make_config(), BASE, active=[incident_event()])
        self.assertEqual(result["generated"], 2)
        self.assertEqual(store.notification_count("cs-a"), 1)
        self.assertEqual(store.notification_count("cs-b"), 1)


# --------------------------------------------------------------------------- 7. dedup + cooldown

class TestDeduplication(NotificationTestCase):
    def test_duplicate_event_is_suppressed(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        config = self.make_config()
        first = self.process(store, config, BASE, active=[incident_event()])
        second = self.process(store, config, at(1), active=[incident_event()])
        self.assertEqual(first["generated"], 1)
        self.assertEqual(second["generated"], 0)
        self.assertEqual(second["reasons"].get("duplicate"), 1)
        self.assertEqual(store.notification_count("cs-1"), 1)

    def test_dedup_persists_across_a_restart(self):
        first_store = self.make_store()
        first_store.subscribe("cs-1", JOURNEY_ID)
        config = self.make_config()
        self.assertEqual(self.process(first_store, config, BASE,
                                      active=[incident_event()])["generated"], 1)
        first_store.close()

        second_store = self.make_store()
        again = self.process(second_store, config, at(1), active=[incident_event()])
        self.assertEqual(again["generated"], 0)
        self.assertEqual(again["reasons"].get("duplicate"), 1)

    def test_cooldown_suppresses_a_material_change(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        config = self.make_config(notification_cooldown_min=10.0)
        first = self.process(store, config, BASE, active=[incident_event(status="active")])
        second = self.process(store, config, at(1), active=[incident_event(status="recovering")])
        self.assertEqual(first["generated"], 1)
        self.assertEqual(second["generated"], 0)
        self.assertEqual(second["reasons"].get("cooldown"), 1)

    def test_a_change_after_the_cooldown_notifies(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        config = self.make_config(notification_cooldown_min=10.0)
        first = self.process(store, config, BASE, active=[incident_event(status="active")])
        # The event is stamped at the pass that observes it: the incident agent refreshes
        # last_seen/updated_at on every poll, so a still-open incident at +11 min carries a
        # current timestamp. (Freshness is now checked per event, not just per snapshot.)
        later = self.process(store, config, at(11),
                             active=[incident_event(status="recovering", updated_at=at_iso(11),
                                                    last_seen=at_iso(11))])
        self.assertEqual(first["generated"], 1)
        self.assertEqual(later["generated"], 1)
        self.assertEqual(later["reasons"].get("changed"), 1)

    def test_severity_escalation_breaks_through_the_cooldown(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        config = self.make_config(notification_cooldown_min=60.0)
        first = self.process(store, config, BASE, active=[incident_event(severity="moderate")])
        worse = self.process(store, config, at(1), active=[incident_event(severity="high")])
        self.assertEqual(first["generated"], 1)
        self.assertEqual(worse["generated"], 1)
        self.assertEqual(worse["reasons"].get("severity_increased"), 1)

    def test_a_lower_severity_does_not_break_through(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        config = self.make_config(notification_cooldown_min=60.0)
        self.process(store, config, BASE, active=[incident_event(severity="high")])
        calmer = self.process(store, config, at(1), active=[incident_event(severity="low")])
        self.assertEqual(calmer["generated"], 0)
        self.assertEqual(calmer["reasons"].get("cooldown"), 1)


class TestEvaluateDedup(unittest.TestCase):
    def test_new_key_notifies(self):
        self.assertEqual(evaluate_dedup(None, "sig", "moderate"), (True, "new"))

    def test_same_signature_is_a_duplicate(self):
        existing = {"signature": "sig", "severity": "moderate", "last_notified_at": iso(BASE)}
        self.assertEqual(evaluate_dedup(existing, "sig", "moderate", now=BASE),
                         (False, "duplicate"))

    def test_severity_increase_breaks_through(self):
        existing = {"signature": "sig", "severity": "moderate", "last_notified_at": iso(BASE)}
        self.assertEqual(evaluate_dedup(existing, "sig2", "high", now=BASE, cooldown_min=60),
                         (True, "severity_increased"))

    def test_cooldown_blocks_a_change(self):
        existing = {"signature": "sig", "severity": "moderate", "last_notified_at": iso(BASE)}
        self.assertEqual(evaluate_dedup(existing, "sig2", "moderate", now=at(1), cooldown_min=10),
                         (False, "cooldown"))

    def test_change_after_the_cooldown_passes(self):
        existing = {"signature": "sig", "severity": "moderate", "last_notified_at": iso(BASE)}
        self.assertEqual(evaluate_dedup(existing, "sig2", "moderate", now=at(11), cooldown_min=10),
                         (True, "changed"))

    def test_zero_cooldown_lets_a_change_through_immediately(self):
        existing = {"signature": "sig", "severity": "moderate", "last_notified_at": iso(BASE)}
        self.assertEqual(evaluate_dedup(existing, "sig2", "moderate", now=BASE, cooldown_min=0),
                         (True, "changed"))


# --------------------------------------------------------------------------- 8. freshness

class TestFreshness(NotificationTestCase):
    def test_stale_incidents_are_not_notified(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        result = self.process(store, self.make_config(), BASE, active=[incident_event()],
                              made_at=at_iso(-60))
        self.assertEqual(result["generated"], 0)
        self.assertFalse(result["incidents_fresh"])

    def test_stale_decision_is_not_notified(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        target = journey(updated_at=at_iso(-60),
                         decision=decision("notify_delay", eta_delta_min=12.0, updated_at=at_iso(-60)))
        result = self.process(store, self.make_config(), BASE, journeys=[target])
        self.assertEqual(result["generated"], 0)
        self.assertEqual(result["journeys_fresh"], 0)

    def test_missing_incident_snapshot_is_handled(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        result = process_notifications(FakeSources(incidents=None, journeys=[journey()]),
                                       store, config=self.make_config(), now=BASE)
        self.assertEqual(result["generated"], 0)
        self.assertFalse(result["incidents_fresh"])

    def test_a_fresh_decision_still_notifies_when_incidents_are_missing(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        target = journey(updated_at=iso(BASE),
                         decision=decision("notify_delay", eta_delta_min=12.0))
        result = process_notifications(FakeSources(incidents=None, journeys=[target]),
                                       store, config=self.make_config(), now=BASE)
        self.assertEqual(result["generated"], 1)
        self.assertEqual(store.list_notifications("cs-1")[0]["notification_type"],
                         NOTIF_JOURNEY_DELAY)

    def test_a_journey_no_longer_watched_is_skipped(self):
        store = self.make_store()
        store.subscribe("cs-1", "jr-gone")
        result = self.process(store, self.make_config(), BASE, active=[incident_event()])
        self.assertEqual(result["generated"], 0)
        self.assertEqual(result["skipped"], 1)

    def test_disabled_sources_do_not_produce_current_alerts(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        sources = FakeSources(incidents=None, journeys=[journey(updated_at=at_iso(-60),
                                                                decision={})])
        result = process_notifications(sources, store, config=self.make_config(), now=BASE)
        self.assertEqual(result["generated"], 0)
        self.assertEqual(result["journeys_fresh"], 0)


# --------------------------------------------------------------------------- 9. read / unread

class TestReadUnread(NotificationTestCase):
    def seed(self, store, session="cs-1", count=3):
        ids = []
        for index in range(count):
            ids.append(store.add(Notification(session_id=session, notification_type=NOTIF_INCIDENT,
                                              title="t%d" % index, message="m%d" % index,
                                              journey_id=JOURNEY_ID)))
        return ids

    def test_add_returns_a_unique_notification_id(self):
        store = self.make_store()
        first = store.add(Notification(session_id="cs-1", notification_type=NOTIF_INCIDENT,
                                       title="a", message="a"))
        second = store.add(Notification(session_id="cs-1", notification_type=NOTIF_INCIDENT,
                                        title="b", message="b"))
        self.assertTrue(first.startswith("nt-"))
        self.assertNotEqual(first, second)

    def test_unread_count_and_unread_only_listing(self):
        store = self.make_store()
        ids = self.seed(store)
        self.assertEqual(store.unread_count("cs-1"), 3)
        self.assertEqual(len(store.list_notifications("cs-1")), 3)
        self.assertEqual(len(store.list_notifications("cs-1", unread_only=True)), 3)
        store.mark_read("cs-1", ids[0])
        self.assertEqual(store.unread_count("cs-1"), 2)
        self.assertEqual(len(store.list_notifications("cs-1", unread_only=True)), 2)

    def test_mark_all_read(self):
        store = self.make_store()
        self.seed(store)
        self.assertEqual(store.mark_read("cs-1"), 3)
        self.assertEqual(store.unread_count("cs-1"), 0)
        self.assertEqual(len(store.list_notifications("cs-1", unread_only=True)), 0)

    def test_mark_read_is_session_scoped(self):
        store = self.make_store()
        self.seed(store, session="cs-1")
        self.seed(store, session="cs-2")
        self.assertEqual(store.mark_read("cs-1"), 3)
        self.assertEqual(store.unread_count("cs-2"), 3)

    def test_notifications_are_listed_newest_first(self):
        store = self.make_store()
        store.add(Notification(session_id="cs-1", notification_type=NOTIF_INCIDENT,
                               title="first", message="1"))
        store.add(Notification(session_id="cs-1", notification_type=NOTIF_INCIDENT,
                               title="second", message="2"))
        self.assertEqual([row["title"] for row in store.list_notifications("cs-1")],
                         ["second", "first"])

    def test_limit(self):
        store = self.make_store()
        self.seed(store)
        self.assertEqual(len(store.list_notifications("cs-1", limit=2)), 2)

    def test_notification_count(self):
        store = self.make_store()
        self.assertEqual(store.notification_count("cs-1"), 0)
        self.seed(store, count=2)
        self.assertEqual(store.notification_count("cs-1"), 2)


# --------------------------------------------------------------------------- 10. persistence

class TestPersistence(NotificationTestCase):
    def test_notifications_and_subscriptions_survive_a_reopen(self):
        store = self.make_store()
        store.subscribe("cs-1", JOURNEY_ID)
        store.add(Notification(session_id="cs-1", notification_type=NOTIF_INCIDENT,
                               title="kept", message="kept message", journey_id=JOURNEY_ID))
        store.close()

        reopened = self.make_store()
        rows = reopened.list_notifications("cs-1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "kept")
        self.assertTrue(reopened.is_subscribed("cs-1", JOURNEY_ID))

    def test_a_second_handle_sees_the_same_rows(self):
        first = self.make_store()
        first.add(Notification(session_id="cs-1", notification_type=NOTIF_INCIDENT,
                               title="x", message="x"))
        second = self.make_store()
        self.assertEqual(second.notification_count("cs-1"), 1)

    def test_read_state_survives_a_reopen(self):
        store = self.make_store()
        notification_id = store.add(Notification(session_id="cs-1",
                                                 notification_type=NOTIF_INCIDENT,
                                                 title="a", message="a"))
        store.mark_read("cs-1", notification_id)
        store.close()
        reopened = self.make_store()
        self.assertEqual(reopened.unread_count("cs-1"), 0)
        self.assertTrue(reopened.list_notifications("cs-1")[0]["read"])


# --------------------------------------------------------------------------- 11. disabled / broken

class TestDisabledAndBroken(NotificationTestCase):
    def test_disabled_store_is_a_no_op(self):
        store = self.make_store(enabled=False)
        self.assertFalse(store.enabled)
        self.assertFalse(store.connect())
        self.assertFalse(store.available)
        self.assertFalse(store.subscribe("cs-1", JOURNEY_ID))
        self.assertFalse(store.is_subscribed("cs-1", JOURNEY_ID))
        self.assertEqual(store.list_notifications("cs-1"), [])
        self.assertIsNone(store.add(Notification(session_id="cs-1",
                                                 notification_type=NOTIF_INCIDENT,
                                                 title="a", message="a")))
        self.assertEqual(store.unread_count("cs-1"), 0)
        self.assertFalse(os.path.exists(self.db_path))

    def test_process_notifications_is_disabled_by_config(self):
        store = self.make_store()
        result = self.process(store, self.make_config(notifications_enabled=False), BASE,
                              active=[incident_event()])
        self.assertFalse(result["enabled"])
        self.assertEqual(result["generated"], 0)
        self.assertEqual(store.notification_count("cs-1"), 0)

    def test_unusable_path_degrades_instead_of_raising(self):
        store = NotificationStore(db_path=self.tmp.name)          # a directory, not a file
        self.addCleanup(store.close)
        self.assertFalse(store.connect())
        self.assertFalse(store.available)
        self.assertFalse(store.subscribe("cs-1", JOURNEY_ID))
        self.assertEqual(store.list_notifications("cs-1"), [])
        self.assertIsNone(store.add(Notification(session_id="cs-1",
                                                 notification_type=NOTIF_INCIDENT,
                                                 title="a", message="a")))

    def test_process_notifications_with_a_broken_store(self):
        store = NotificationStore(db_path=self.tmp.name)
        self.addCleanup(store.close)
        result = self.process(store, self.make_config(), BASE, active=[incident_event()])
        self.assertFalse(result["enabled"])
        self.assertEqual(result["generated"], 0)


# --------------------------------------------------------------------------- 12. factory / api

class TestFactoryAndApi(NotificationTestCase):
    def test_build_notification_store_uses_the_config(self):
        config = self.make_config(notifications_enabled=False)
        store = build_notification_store(config)
        self.addCleanup(store.close)
        self.assertEqual(store.db_path, self.db_path)
        self.assertFalse(store.enabled)

    def test_severity_rank_orders_escalations(self):
        self.assertLess(severity_rank("info"), severity_rank("low"))
        self.assertLess(severity_rank("low"), severity_rank("moderate"))
        self.assertLess(severity_rank("moderate"), severity_rank("high"))
        self.assertEqual(severity_rank(None), severity_rank("info"))

    def test_notification_types_are_the_documented_five(self):
        self.assertEqual(set(NOTIFICATION_TYPES),
                         {NOTIF_INCIDENT, NOTIF_INCIDENT_RESOLVED, NOTIF_JOURNEY_DELAY,
                          NOTIF_ROUTE_ALTERNATIVE, NOTIF_JOURNEY_COMPLETED})

    def test_notification_as_dict_round_trips(self):
        notification = Notification(session_id="cs-1", notification_type=NOTIF_INCIDENT,
                                    title="t", message="m", journey_id=JOURNEY_ID,
                                    severity="high")
        payload = notification.as_dict()
        self.assertEqual(payload["session_id"], "cs-1")
        self.assertEqual(payload["journey_id"], JOURNEY_ID)
        self.assertEqual(payload["severity"], "high")
        self.assertIsNone(payload["notification_id"])
        self.assertFalse(payload["read"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
