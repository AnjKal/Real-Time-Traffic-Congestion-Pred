"""Isolated API tests for Phase 4: the interaction agent over the web app's HTTP surface.

    python -m unittest test_interaction_agent_api -v

Every test uses a temporary SQLite database and an in-memory fake data source; nothing touches the
runtime database, AWS, Kafka, Docker or the network. The web app is built with ``create_app`` and a
TestClient, exactly as ``test_webapp_offline.py`` does, so the routes under test are the real ones.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

from webapp import server as srv
from webapp.interaction_agent import (InteractionConfig, Notification, NotificationStore)
from webapp.interaction_api import InteractionServices
from webapp.journey_memory import iso, utc_now

JOURNEY_ID = "jr-ff466f71c8"
OTHER_JOURNEY_ID = "jr-0011223344"


def fresh():
    return iso(utc_now())


def old(minutes=90):
    return iso(utc_now() - timedelta(minutes=minutes))


INCIDENT_EVENT = {
    "event_id": "ce-J001-aaaa1111", "event_type": "observed_slowdown", "status": "active",
    "severity": "moderate", "area": "Silk Board", "anchor_junction_id": "J001",
    "anchor_name": "Silk Board Junction", "affected_junction_ids": ["J001", "J002"],
    "affected_junction_count": 2, "avg_current_speed_kmh": 8.5, "avg_free_flow_speed_kmh": 45.0,
    "avg_predicted_speed_kmh": 7.0, "forecast_horizon_min": 15, "first_seen": old(5),
    "last_seen": fresh(), "updated_at": fresh(),
}


class FakeSources(object):
    """In-memory stand-in for :class:`LiveDataSources`. No files, no HTTP, no AWS."""

    def __init__(self, incidents=None, features=None, journeys=None, switches=None, health=None,
                 fail=()):
        self._incidents = incidents
        self._features = features
        self._journeys = dict(journeys or {})
        self._switches = list(switches or [])
        self._health = health
        self._fail = set(fail)
        self.stale_after_min = 10.0
        self.model_service_url = "http://fake"
        self.incidents_path = "incidents.json"
        self.features_path = "features.json"
        self.state_path = "journey_state.json"
        self.switch_path = "journey_switches.json"

    def incidents(self):
        if "incidents" in self._fail:
            raise RuntimeError("incidents file is unreadable")
        return self._incidents

    def features(self):
        if "features" in self._fail:
            raise RuntimeError("features file is unreadable")
        return self._features

    def journeys(self):
        if "journeys" in self._fail:
            raise RuntimeError("journey state is unreadable")
        return [dict(journey) for journey in self._journeys.values()]

    def journey(self, journey_id):
        if "journeys" in self._fail:
            raise RuntimeError("journey state is unreadable")
        journey = self._journeys.get(journey_id)
        return dict(journey) if journey else None

    def switches(self, journey_id=None):
        return list(self._switches)

    def system_health(self):
        return self._health

    def source_available(self, path):
        published = {
            self.incidents_path: self._incidents,
            self.features_path: self._features,
            self.state_path: self._journeys,
            self.switch_path: self._switches,
        }
        return path in published and published[path] is not None


def journey(journey_id=JOURNEY_ID, label="Home to Office", status="active"):
    return {"journey_id": journey_id, "label": label, "status": status, "total_length_km": 14.2,
            "updated_at": fresh(),
            "spine": [{"junction_id": "J001", "cum_m": 0.0},
                      {"junction_id": "J002", "cum_m": 1200.0}],
            "route_nodes": ["J001", "J002"],
            "decision": {"journey_id": journey_id, "action": "hold", "progress_pct": 40.0,
                         "remaining_km": 8.1, "eta_min": 27.0, "baseline_eta_min": 26.0,
                         "eta_delta_min": 1.0, "incident_ids": [], "confidence": "medium",
                         "updated_at": fresh()}}


def default_sources(**overrides):
    kwargs = {"incidents": {"made_at": fresh(), "active_events": [dict(INCIDENT_EVENT)],
                            "recent_resolved": []},
              "features": {"made_at": fresh(), "areas": []},
              "journeys": {JOURNEY_ID: journey(), OTHER_JOURNEY_ID: journey(OTHER_JOURNEY_ID)},
              "switches": []}
    kwargs.update(overrides)
    return FakeSources(**kwargs)


# --------------------------------------------------------------------------- fixtures

class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = os.path.join(self.tmp.name, "interaction.sqlite")
        self.services = None
        self.clients = []

    def make_services(self, sources=None, **config_kwargs):
        config_kwargs.setdefault("memory_db_path", self.db_path)
        config_kwargs.setdefault("llm_provider", "mock")
        # The store must be enabled for the notification/subscription endpoints to do anything.
        # The background worker never starts here because the app lifespan only runs when a
        # TestClient is used as a context manager, which these tests deliberately avoid.
        config_kwargs.setdefault("notifications_enabled", True)
        config_kwargs.setdefault("notification_cooldown_min", 0.0)
        config = InteractionConfig(**config_kwargs)
        services = InteractionServices(config=config, sources=sources or default_sources())
        self.addCleanup(services.memory.close)
        self.addCleanup(services.store.close)
        self.addCleanup(services.guard.close)
        self.services = services
        return services

    def make_client(self, services=None):
        services = services or self.services or self.make_services()
        app = srv.create_app(model_service_url="http://fake", graph=None, spots=None,
                             interaction=services)
        client = TestClient(app)
        self.clients.append(client)
        return client

    def new_session(self, client, message="status of my journey"):
        response = client.post("/api/chat", json={"message": message})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        return body["session_id"], body["session_token"]

    @staticmethod
    def auth(token):
        return {"Authorization": "Bearer %s" % token}


# --------------------------------------------------------------------------- 1. chat

class TestChat(ApiTestCase):
    def test_successful_mock_chat(self):
        client = self.make_client()
        body = client.post("/api/chat", json={"message": "are there any traffic incidents?"}).json()
        self.assertTrue(body["ok"], body)
        self.assertEqual(body["provider"], "mock")
        self.assertTrue(body["response"])
        self.assertTrue(body["session_id"].startswith("cs-"))
        self.assertTrue(body["session_token"])
        names = [call["name"] for call in body["tools_used"]]
        self.assertIn("get_incidents", names)
        self.assertIn("sources", body["data"])
        self.assertEqual(body["data"]["sources"].get("get_incidents"), "live")

    def test_new_session_is_created_when_none_given(self):
        client = self.make_client()
        first = client.post("/api/chat", json={"message": "hi"}).json()
        second = client.post("/api/chat", json={"message": "hi"}).json()
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertNotEqual(first["session_token"], second["session_token"])

    def test_follow_up_uses_memory(self):
        client = self.make_client()
        session_id, token = self.new_session(client, "status of my journey")
        reply = client.post("/api/chat", json={"message": "what about the delay?",
                                               "session_id": session_id},
                            headers=self.auth(token)).json()
        self.assertEqual(reply["session_id"], session_id)
        self.assertIsNone(reply["session_token"])             # token is only issued once
        names = [call["name"] for call in reply["tools_used"]]
        self.assertTrue(names, reply)
        self.assertIn("get_journeys", names)
        history = client.get("/api/chat/history", params={"session_id": session_id},
                             headers=self.auth(token)).json()
        self.assertEqual(history["count"], 4)                 # 2 questions + 2 answers

    def test_chat_does_not_auto_subscribe(self):
        client = self.make_client()
        session_id, token = self.new_session(client, "status of my journey")
        subs = client.get("/api/subscriptions", params={"session_id": session_id},
                          headers=self.auth(token)).json()
        self.assertEqual(subs["count"], 0)
        self.assertEqual(self.services.store.subscriptions(), [])

    def test_invalid_requests(self):
        client = self.make_client()
        self.assertEqual(client.post("/api/chat", json={"message": ""}).status_code, 400)
        self.assertEqual(client.post("/api/chat", json={"message": "   "}).status_code, 400)
        self.assertEqual(client.post("/api/chat", json={"message": "x" * 3000}).status_code, 413)
        self.assertEqual(client.post("/api/chat", json={"session_id": "cs-abc"}).status_code, 422)
        bad = client.post("/api/chat", content=b"{not json",
                          headers={"Content-Type": "application/json"})
        self.assertEqual(bad.status_code, 422)

    def test_malformed_session_id_is_rejected(self):
        client = self.make_client()
        response = client.post("/api/chat", json={"message": "hi", "session_id": "not-a-session"},
                               headers=self.auth("whatever"))
        self.assertEqual(response.status_code, 400)

    def test_data_source_failure_still_answers(self):
        services = self.make_services(default_sources(fail=("incidents",)))
        client = self.make_client(services)
        body = client.post("/api/chat", json={"message": "any incidents?"}).json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["data"]["sources"].get("get_incidents"), "unavailable")


# --------------------------------------------------------------------------- 2. session security

class TestSessionSecurity(ApiTestCase):
    def test_session_without_token_is_rejected(self):
        client = self.make_client()
        session_id, _ = self.new_session(client)
        response = client.post("/api/chat", json={"message": "again", "session_id": session_id})
        self.assertEqual(response.status_code, 401)

    def test_invalid_token_is_rejected(self):
        client = self.make_client()
        session_id, _ = self.new_session(client)
        response = client.get("/api/chat/history", params={"session_id": session_id},
                              headers=self.auth("totally-made-up"))
        self.assertEqual(response.status_code, 401)

    def test_token_from_another_session_is_forbidden(self):
        client = self.make_client()
        a, _ = self.new_session(client, "hi")
        _, token_b = self.new_session(client, "hi")
        response = client.get("/api/chat/history", params={"session_id": a},
                              headers=self.auth(token_b))
        self.assertEqual(response.status_code, 403)

    def test_owner_can_read_own_history(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        response = client.get("/api/chat/history", params={"session_id": session_id},
                              headers=self.auth(token))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], session_id)

    def test_notifications_require_ownership(self):
        client = self.make_client()
        session_id, _ = self.new_session(client)
        self.assertEqual(client.get("/api/notifications",
                                    params={"session_id": session_id}).status_code, 401)


# --------------------------------------------------------------------------- 3. history

class TestHistory(ApiTestCase):
    def test_retrieve_and_clear_only_this_session(self):
        client = self.make_client()
        a, token_a = self.new_session(client, "hi")
        b, token_b = self.new_session(client, "hi")
        client.post("/api/chat", json={"message": "any incidents?", "session_id": a},
                    headers=self.auth(token_a))
        cleared = client.delete("/api/chat/history", params={"session_id": a},
                                headers=self.auth(token_a)).json()
        self.assertGreater(cleared["cleared"], 0)
        self.assertEqual(client.get("/api/chat/history", params={"session_id": a},
                                    headers=self.auth(token_a)).json()["count"], 0)
        # The other conversation is untouched.
        self.assertEqual(client.get("/api/chat/history", params={"session_id": b},
                                    headers=self.auth(token_b)).json()["count"], 2)

    def test_clearing_does_not_touch_notifications_or_subscriptions(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        self.services.store.add(Notification(session_id=session_id, notification_type="incident",
                                             title="t", message="m", severity="high"))
        self.services.store.subscribe(session_id, JOURNEY_ID)
        client.delete("/api/chat/history", params={"session_id": session_id},
                      headers=self.auth(token))
        self.assertEqual(self.services.store.notification_count(session_id), 1)
        self.assertEqual(self.services.store.subscribed_journeys(session_id), [JOURNEY_ID])

    def test_limit_is_bounded(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        out = client.get("/api/chat/history", params={"session_id": session_id, "limit": 99999},
                         headers=self.auth(token)).json()
        self.assertLessEqual(out["limit"], 200)
        self.assertEqual(client.get("/api/chat/history",
                                    params={"session_id": session_id, "limit": -1},
                                    headers=self.auth(token)).status_code, 400)


# --------------------------------------------------------------------------- 4. notifications

class TestNotifications(ApiTestCase):
    def seed(self, session_id, count=3, read_first=False):
        ids = []
        for index in range(count):
            notification = Notification(session_id=session_id, notification_type="incident",
                                        title="Alert %d" % index, message="m", severity="moderate",
                                        journey_id=JOURNEY_ID)
            ids.append(self.services.store.add(notification, dedup_key="k%d" % index,
                                               signature="s%d" % index))
        if read_first and ids:
            self.services.store.mark_read(session_id, ids[0])
        return ids

    def test_list_and_unread_count(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        self.seed(session_id, count=3, read_first=True)
        listing = client.get("/api/notifications", params={"session_id": session_id},
                             headers=self.auth(token)).json()
        self.assertEqual(listing["count"], 3)
        self.assertEqual(listing["unread"], 2)
        unread_only = client.get("/api/notifications",
                                 params={"session_id": session_id, "unread_only": True},
                                 headers=self.auth(token)).json()
        self.assertEqual(unread_only["count"], 2)
        count = client.get("/api/notifications/unread-count",
                           params={"session_id": session_id}, headers=self.auth(token)).json()
        self.assertEqual(count["unread"], 2)

    def test_notifications_are_session_specific(self):
        client = self.make_client()
        a, token_a = self.new_session(client)
        b, token_b = self.new_session(client)
        self.seed(a, count=2)
        self.assertEqual(client.get("/api/notifications", params={"session_id": a},
                                    headers=self.auth(token_a)).json()["count"], 2)
        self.assertEqual(client.get("/api/notifications", params={"session_id": b},
                                    headers=self.auth(token_b)).json()["count"], 0)

    def test_mark_one_and_all_read(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        ids = self.seed(session_id, count=3)
        one = client.post("/api/notifications/read",
                          json={"session_id": session_id, "notification_id": ids[0]},
                          headers=self.auth(token)).json()
        self.assertEqual(one["marked"], 1)
        self.assertEqual(client.get("/api/notifications/unread-count",
                                    params={"session_id": session_id},
                                    headers=self.auth(token)).json()["unread"], 2)
        allof = client.post("/api/notifications/read", json={"session_id": session_id},
                            headers=self.auth(token)).json()
        self.assertEqual(allof["marked"], 2)
        self.assertEqual(client.get("/api/notifications/unread-count",
                                    params={"session_id": session_id},
                                    headers=self.auth(token)).json()["unread"], 0)

    def test_unknown_notification_id(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        response = client.post("/api/notifications/read",
                               json={"session_id": session_id, "notification_id": "nt-nope"},
                               headers=self.auth(token))
        self.assertEqual(response.status_code, 404)

    def test_disabled_notifications(self):
        services = self.make_services(notifications_enabled=False)
        client = self.make_client(services)
        session_id, token = self.new_session(client)
        listing = client.get("/api/notifications", params={"session_id": session_id},
                             headers=self.auth(token)).json()
        self.assertFalse(listing["enabled"])
        self.assertEqual(listing["count"], 0)
        self.assertEqual(client.get("/api/notifications/unread-count",
                                    params={"session_id": session_id},
                                    headers=self.auth(token)).json()["unread"], 0)
        marked = client.post("/api/notifications/read", json={"session_id": session_id},
                             headers=self.auth(token)).json()
        self.assertFalse(marked["enabled"])
        self.assertEqual(marked["marked"], 0)


# --------------------------------------------------------------------------- 5. subscriptions

class TestSubscriptions(ApiTestCase):
    def test_subscribe_list_unsubscribe(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        created = client.post("/api/subscriptions",
                              json={"session_id": session_id, "journey_id": JOURNEY_ID},
                              headers=self.auth(token)).json()
        self.assertTrue(created["subscribed"])
        listed = client.get("/api/subscriptions", params={"session_id": session_id},
                            headers=self.auth(token)).json()
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["subscriptions"][0]["journey_id"], JOURNEY_ID)
        removed = client.delete("/api/subscriptions/%s" % JOURNEY_ID,
                                params={"session_id": session_id},
                                headers=self.auth(token)).json()
        self.assertTrue(removed["removed"])
        self.assertEqual(client.get("/api/subscriptions", params={"session_id": session_id},
                                    headers=self.auth(token)).json()["count"], 0)

    def test_invalid_journey_id_is_rejected(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        response = client.post("/api/subscriptions",
                               json={"session_id": session_id, "journey_id": "jr-nonexistent"},
                               headers=self.auth(token))
        self.assertEqual(response.status_code, 404)

    def test_subscriptions_are_session_specific(self):
        client = self.make_client()
        a, token_a = self.new_session(client)
        b, token_b = self.new_session(client)
        client.post("/api/subscriptions", json={"session_id": a, "journey_id": JOURNEY_ID},
                    headers=self.auth(token_a))
        self.assertEqual(client.get("/api/subscriptions", params={"session_id": b},
                                    headers=self.auth(token_b)).json()["count"], 0)

    def test_unsubscribe_unknown_is_graceful(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        removed = client.delete("/api/subscriptions/%s" % JOURNEY_ID,
                                params={"session_id": session_id},
                                headers=self.auth(token)).json()
        self.assertFalse(removed["removed"])


# --------------------------------------------------------------------------- 6. status + resilience

class TestStatus(ApiTestCase):
    def test_health_endpoint(self):
        client = self.make_client()
        body = client.get("/api/interaction/status").json()
        self.assertTrue(body["available"])
        self.assertTrue(body["agent"]["available"])
        self.assertEqual(body["llm"]["provider"], "mock")
        self.assertFalse(body["llm"]["uses_bedrock"])
        self.assertIn("tools", body["agent"])
        self.assertIn("enabled", body["memory"])
        self.assertIn("enabled", body["notifications"])
        self.assertIn("notification_runtime", body)
        self.assertIn("last_processing_cycle_at", body)
        self.assertIn("data_sources", body)
        self.assertIn("incidents_published", body["data_sources"])

    def test_status_leaks_no_secrets_or_paths(self):
        client = self.make_client()
        raw = client.get("/api/interaction/status").text
        self.assertNotIn(self.db_path, raw)
        self.assertNotIn("session_token", raw)
        self.assertNotIn("system_prompt", raw)

    def test_status_reports_missing_bedrock_seam_without_calling_it(self):
        services = self.make_services(llm_provider="bedrock")
        client = self.make_client(services)
        body = client.get("/api/interaction/status").json()
        self.assertTrue(body["llm"]["uses_bedrock"])
        self.assertEqual(body["llm"]["model"], services.config.bedrock_model_id)

    def test_unavailable_services_report_unavailable(self):
        # A container that failed to build is represented by include_router(None).
        from webapp.interaction_api import create_interaction_router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(create_interaction_router(None))
        client = TestClient(app)
        self.assertEqual(client.get("/api/interaction/status").json()["available"], False)
        self.assertEqual(client.post("/api/chat", json={"message": "hi"}).status_code, 503)

    def test_existing_endpoints_still_work(self):
        client = self.make_client()
        self.assertEqual(client.get("/api/journeys").status_code, 200)
        self.assertEqual(client.get("/api/journey-runtime").status_code, 200)
        self.assertEqual(client.get("/api/journey/jr-missing").status_code, 404)
        self.assertEqual(client.get("/api/does-not-exist").status_code, 404)


class TestResilience(ApiTestCase):
    def test_database_failure_degrades_gracefully(self):
        # Point the shared database at a directory, so opening it fails; nothing should 500.
        services = InteractionServices(config=InteractionConfig(memory_db_path=self.tmp.name,
                                                                llm_provider="mock",
                                                                notifications_enabled=False))
        self.addCleanup(services.memory.close)
        self.addCleanup(services.store.close)
        self.addCleanup(services.guard.close)
        client = self.make_client(services)
        chat = client.post("/api/chat", json={"message": "any incidents?"})
        self.assertEqual(chat.status_code, 200)
        self.assertTrue(chat.json()["ok"])
        session_id, token = chat.json()["session_id"], chat.json()["session_token"]
        self.assertEqual(client.get("/api/chat/history", params={"session_id": session_id},
                                    headers=self.auth(token)).json()["count"], 0)
        self.assertEqual(client.get("/api/notifications", params={"session_id": session_id},
                                    headers=self.auth(token)).json()["count"], 0)

    def test_concurrent_requests(self):
        services = self.make_services(memory_max_messages=100)
        client = self.make_client(services)
        session_id, token = self.new_session(client)
        results = []

        def worker(index):
            response = client.post("/api/chat",
                                   json={"message": "any incidents? %d" % index,
                                         "session_id": session_id},
                                   headers=self.auth(token))
            results.append(response.status_code)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)
        self.assertEqual(results.count(200), 10, results)
        self.assertEqual(len(self.services.memory.history(session_id)), 22)   # 1 + 10 questions + 10 answers


if __name__ == "__main__":
    unittest.main(verbosity=2)
