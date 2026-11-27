"""Phase 5 tests: the "Traffic Assistant" frontend and the API contract it depends on.

    python -m unittest test_interaction_agent_frontend -v

The chatbot is plain JavaScript inside ``webapp/templates/index.html``; there is no browser test
runner in this project, so the checks come in two honest kinds, both real:

  * API contract tests -- exactly the routes and payloads the page calls (``POST /api/chat``,
    ``/api/chat/history``, ``/api/notifications*``, ``/api/subscriptions*``,
    ``/api/interaction/session``, ``/api/interaction/status``) exercised through the real app with a
    TestClient. The harness (``ApiTestCase``) is shared with ``test_interaction_agent_api`` so the
    fixtures and conventions stay identical.
  * Static template tests -- the served page really contains the chatbot markup, the suggested
    questions, and the safety properties the brief demands (untrusted text rendered with
    ``textContent``, a single poll timer, credentials only in the ``Authorization`` header, no token
    in a URL). These are assertions about the shipped file, not a claim that a browser ran.

Nothing here touches AWS, Kafka, Docker, the network or the runtime database.
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse the exact fixtures and base class the Phase 4 API tests use, so both suites stay in step.
from test_interaction_agent_api import (ApiTestCase, FakeSources, JOURNEY_ID, OTHER_JOURNEY_ID,
                                        default_sources, journey)
from webapp.interaction_agent import Notification

TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "webapp", "templates", "index.html")

# The exact wording the brief lists for the suggested questions.
SUGGESTED_QUESTIONS = [
    "Are there any traffic incidents?",
    "What are the current congestion conditions?",
    "What is the status of my journey?",
    "Is there a faster route?",
    "Is the traffic monitoring system running?",
]


def read_template():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


# --------------------------------------------------------------------------- 1. chatbot contract

class TestChatContract(ApiTestCase):
    """The chat flow the browser performs: bootstrap a session, ask, follow up, reload, clear."""

    def bootstrap(self, client):
        """The page's very first call, before any question: POST /api/interaction/session."""
        response = client.post("/api/interaction/session")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["session_id"].startswith("cs-"))
        self.assertTrue(body["session_token"])
        return body["session_id"], body["session_token"]

    def test_session_bootstrap_is_empty_and_usable_before_first_question(self):
        client = self.make_client()
        session_id, token = self.bootstrap(client)
        # A session minted up front stores no message, so the history is empty ...
        history = client.get("/api/chat/history", params={"session_id": session_id},
                             headers=self.auth(token)).json()
        self.assertEqual(history["count"], 0)
        # ... but its credential already authorises the alert and monitoring panels.
        self.assertEqual(client.get("/api/notifications/unread-count",
                                    params={"session_id": session_id},
                                    headers=self.auth(token)).status_code, 200)
        self.assertEqual(client.get("/api/subscriptions", params={"session_id": session_id},
                                    headers=self.auth(token)).status_code, 200)

    def test_first_message_with_bootstrapped_session_issues_no_second_token(self):
        client = self.make_client()
        session_id, token = self.bootstrap(client)
        body = client.post("/api/chat", json={"message": "hi", "session_id": session_id},
                           headers=self.auth(token)).json()
        self.assertTrue(body["ok"], body)
        self.assertEqual(body["provider"], "mock")
        self.assertEqual(body["session_id"], session_id)
        self.assertIsNone(body["session_token"])       # the credential is only issued once
        self.assertTrue(body["response"])

    def test_new_conversation_mints_fresh_credentials(self):
        client = self.make_client()
        first = client.post("/api/chat", json={"message": "hi"}).json()
        second = client.post("/api/chat", json={"message": "hi"}).json()
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertNotEqual(first["session_token"], second["session_token"])

    def test_follow_up_and_history_survive_a_refresh(self):
        client = self.make_client()
        session_id, token = self.new_session(client, "status of my journey")
        client.post("/api/chat", json={"message": "what about the delay?", "session_id": session_id},
                    headers=self.auth(token))
        # A page refresh keeps the credentials in browser storage and simply re-reads the history.
        history = client.get("/api/chat/history", params={"session_id": session_id},
                             headers=self.auth(token)).json()
        self.assertEqual(history["count"], 4)                       # 2 questions + 2 answers
        roles = [message["role"] for message in history["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
        stamps = [message["created_at"] for message in history["messages"]]
        self.assertEqual(stamps, sorted(stamps))                    # chronological order
        self.assertTrue(all(message["content"] for message in history["messages"]))

    def test_clear_conversation_keeps_notifications_and_subscriptions(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        self.services.store.subscribe(session_id, JOURNEY_ID)
        self.services.store.add(Notification(session_id=session_id, notification_type="incident",
                                             title="t", message="m", severity="high",
                                             journey_id=JOURNEY_ID))
        cleared = client.delete("/api/chat/history", params={"session_id": session_id},
                                headers=self.auth(token)).json()
        self.assertGreater(cleared["cleared"], 0)
        self.assertEqual(client.get("/api/chat/history", params={"session_id": session_id},
                                    headers=self.auth(token)).json()["count"], 0)
        self.assertEqual(self.services.store.notification_count(session_id), 1)
        self.assertEqual(self.services.store.subscribed_journeys(session_id), [JOURNEY_ID])

    def test_unsuccessful_and_oversized_messages(self):
        client = self.make_client()
        self.assertEqual(client.post("/api/chat", json={"message": ""}).status_code, 400)
        self.assertEqual(client.post("/api/chat", json={"message": "   "}).status_code, 400)
        self.assertEqual(client.post("/api/chat", json={"message": "x" * 2001}).status_code, 413)

    def test_invalid_credentials_are_rejected(self):
        client = self.make_client()
        session_id, _ = self.new_session(client)
        # Refreshed page with no stored token, then with a tampered one.
        self.assertEqual(client.post("/api/chat", json={"message": "hi", "session_id": session_id})
                         .status_code, 401)
        self.assertEqual(client.get("/api/chat/history", params={"session_id": session_id},
                                    headers=self.auth("made-up")).status_code, 401)
        # A token that belongs to a different conversation.
        _, other_token = self.new_session(client)
        self.assertEqual(client.get("/api/chat/history", params={"session_id": session_id},
                                    headers=self.auth(other_token)).status_code, 403)

    def test_data_source_failure_is_handled_not_fatal(self):
        services = self.make_services(default_sources(fail=("incidents",)))
        client = self.make_client(services)
        body = client.post("/api/chat", json={"message": "any incidents?"}).json()
        self.assertTrue(body["ok"])                                 # offline-safe answer, no 500
        self.assertEqual(body["data"]["sources"].get("get_incidents"), "unavailable")


# --------------------------------------------------------------------------- 2. notifications

class TestNotificationContract(ApiTestCase):
    def seed(self, session_id, count=3, read_first=False):
        ids = []
        for index in range(count):
            notification = Notification(session_id=session_id, notification_type="incident",
                                        title="Alert %d" % index, message="m",
                                        severity="moderate", journey_id=JOURNEY_ID)
            ids.append(self.services.store.add(notification, dedup_key="k%d" % index,
                                               signature="s%d" % index))
        if read_first and ids:
            self.services.store.mark_read(session_id, ids[0])
        return ids

    def test_empty_state_when_no_notifications_exist(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        listing = client.get("/api/notifications", params={"session_id": session_id},
                             headers=self.auth(token)).json()
        self.assertTrue(listing["enabled"])
        self.assertEqual(listing["count"], 0)                      # nothing fabricated
        self.assertEqual(listing["unread"], 0)
        self.assertEqual(client.get("/api/notifications/unread-count",
                                    params={"session_id": session_id},
                                    headers=self.auth(token)).json()["unread"], 0)

    def test_listing_carries_severity_timestamp_and_read_flag(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        self.seed(session_id, count=1)
        row = client.get("/api/notifications", params={"session_id": session_id},
                         headers=self.auth(token)).json()["notifications"][0]
        for field in ("notification_id", "title", "message", "severity", "created_at", "read"):
            self.assertIn(field, row)
        self.assertEqual(row["severity"], "moderate")
        self.assertFalse(row["read"])

    def test_unread_badge_updates_and_marks_read(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        ids = self.seed(session_id, count=3)
        self.assertEqual(client.get("/api/notifications/unread-count",
                                    params={"session_id": session_id},
                                    headers=self.auth(token)).json()["unread"], 3)
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

    def test_unknown_notification_id_is_a_clear_404(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        response = client.post("/api/notifications/read",
                               json={"session_id": session_id, "notification_id": "nt-nope"},
                               headers=self.auth(token))
        self.assertEqual(response.status_code, 404)

    def test_seeded_notification_appears_after_a_poll(self):
        # The runtime writes with the same store the panel reads: a screenshot-style end-to-end.
        client = self.make_client()
        session_id, token = self.new_session(client)
        self.services.store.add(Notification(session_id=session_id, notification_type="journey_delay",
                                             title="Delay ahead", message="+6 min on your route",
                                             severity="high", journey_id=JOURNEY_ID))
        unread = client.get("/api/notifications/unread-count", params={"session_id": session_id},
                            headers=self.auth(token)).json()["unread"]
        self.assertEqual(unread, 1)


# --------------------------------------------------------------------------- 3. subscriptions

class TestSubscriptionContract(ApiTestCase):
    def test_monitor_then_stop_monitoring(self):
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

    def test_nothing_is_auto_subscribed(self):
        client = self.make_client()
        session_id, token = self.new_session(client, "status of my journey")
        self.assertEqual(client.get("/api/subscriptions", params={"session_id": session_id},
                                    headers=self.auth(token)).json()["count"], 0)

    def test_invalid_journey_id_is_rejected(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        response = client.post("/api/subscriptions",
                               json={"session_id": session_id, "journey_id": "jr-missing"},
                               headers=self.auth(token))
        self.assertEqual(response.status_code, 404)

    def test_stopping_an_unmonitored_journey_is_graceful(self):
        client = self.make_client()
        session_id, token = self.new_session(client)
        removed = client.delete("/api/subscriptions/%s" % JOURNEY_ID,
                                params={"session_id": session_id},
                                headers=self.auth(token)).json()
        self.assertFalse(removed["removed"])


# --------------------------------------------------------------------------- 4. agent status

class TestStatusContract(ApiTestCase):
    def test_status_drives_the_connection_chip(self):
        client = self.make_client()
        body = client.get("/api/interaction/status").json()
        self.assertTrue(body["available"])
        self.assertTrue(body["agent"]["available"])
        self.assertEqual(body["llm"]["provider"], "mock")
        self.assertFalse(body["llm"]["uses_bedrock"])              # the page shows "(mock)"
        self.assertIn("notification_runtime", body)
        self.assertIn("enabled", body["notifications"])

    def test_status_is_provider_agnostic(self):
        # Switching the backend to Bedrock must not need a frontend redesign: the same field flips.
        services = self.make_services(llm_provider="bedrock")
        client = self.make_client(services)
        body = client.get("/api/interaction/status").json()
        self.assertTrue(body["llm"]["uses_bedrock"])
        self.assertEqual(body["llm"]["model"], services.config.bedrock_model_id)


# --------------------------------------------------------------------------- 5. dashboard intact

class TestDashboardIntact(ApiTestCase):
    def test_existing_endpoints_still_work(self):
        client = self.make_client()
        self.assertEqual(client.get("/api/journeys").status_code, 200)
        self.assertEqual(client.get("/api/journey-runtime").status_code, 200)
        self.assertEqual(client.get("/api/journey/jr-missing").status_code, 404)

    def test_served_page_contains_the_chatbot_and_the_map(self):
        client = self.make_client()
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn('id="ta-launcher"', page.text)
        self.assertIn("Traffic Assistant", page.text)
        self.assertIn("Compute Routes", page.text)                  # the original route panel
        self.assertIn("circle_marker", page.text)                   # the folium/Leaflet map


# --------------------------------------------------------------------------- 6. template safety

class TestTemplateStatic(unittest.TestCase):
    """Assertions about the shipped template file. No browser runs here."""

    @classmethod
    def setUpClass(cls):
        cls.page = read_template()
        # The whole Traffic Assistant module, for the "must not" checks below.
        cls.module = cls.page[cls.page.index("const TA_SESSION_KEY"):]

    def test_required_markup_is_present(self):
        for element_id in ("ta-launcher", "ta-chat", "ta-chat-title", "ta-msgs", "ta-suggests",
                           "ta-form", "ta-input", "ta-send", "ta-new", "ta-clear", "ta-chat-close",
                           "ta-bell", "ta-badge", "ta-alerts", "ta-alerts-close", "ta-mark-all",
                           "ta-alerts-list", "ta-subs-list", "ta-status-chip", "ta-status-dot",
                           "ta-status-text", "ta-chat-dot"):
            self.assertIn('id="%s"' % element_id, self.page, element_id)

    def test_header_is_titled_traffic_assistant(self):
        self.assertIn('<h2 id="ta-chat-title">Traffic Assistant</h2>', self.page)

    def test_every_suggested_question_matches_a_supported_capability(self):
        for question in SUGGESTED_QUESTIONS:
            self.assertIn(question, self.page, question)

    def test_session_credentials_go_only_in_the_authorization_header(self):
        self.assertIn('"bangalore.interaction.session"', self.module)
        self.assertIn("Authorization", self.module)
        self.assertIn("Bearer ", self.module)
        # Never in a URL or query string.
        self.assertNotIn("session_token=", self.page)
        self.assertNotIn("?token", self.page)

    def test_untrusted_text_is_rendered_with_textContent_only(self):
        # No innerHTML writes anywhere in the module: message and notification text is data, not
        # markup. (The prose "never innerHTML" appears in a comment, so match the property write.)
        self.assertNotIn(".innerHTML", self.module)
        self.assertIn("textContent = content", self.module)         # the message body

    def test_a_single_poll_timer_with_an_overlap_guard(self):
        self.assertIn("let taPollTimer = null", self.module)
        self.assertIn("if (taPollTimer) return", self.module)       # never a second timer
        self.assertIn("setInterval(taPollTick", self.module)
        self.assertIn("taPollBusy", self.module)                    # never overlapping requests
        self.assertIn("document.hidden", self.module)               # pause when the tab is hidden
        self.assertIn("visibilitychange", self.module)

    def test_enter_sends_and_shift_enter_is_multiline(self):
        self.assertIn("shiftKey", self.module)
        self.assertIn("isComposing", self.module)                   # respect IME composition

    def test_clear_conversation_confirms_first(self):
        self.assertIn("confirm(", self.module)

    def test_responsive_layout_rules_exist(self):
        self.assertIn("@media (max-width: 640px)", self.page)
        self.assertIn(".ta-chat", self.page)
        self.assertIn(".ta-alerts", self.page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
