"""Isolated unit tests for Phase 2: persistent conversation memory and context management.

    python -m unittest test_interaction_agent_memory -v

Every test uses a temporary SQLite database (or none at all). Nothing here touches the runtime
database, AWS, Kafka, Docker or the network.
"""
import os
import sqlite3
import sys
import tempfile
import threading
import unittest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.journey_memory import iso, utc_now
from webapp.interaction_agent import (ConversationMemory, ConfigError, InteractionConfig,
                                      MockLLMProvider, build_agent, build_default_registry,
                                      detect_followup)
from webapp.interaction_agent.agent import (EMPTY_QUESTION_REPLY, MAX_STEPS, PROVIDER_ERROR_REPLY,
                                            InteractionAgent, context_update)
from webapp.interaction_agent.llm import CLARIFY, LLMResponse, ToolCall
from webapp.interaction_agent.memory import (DEFAULT_DB_PATH, LIST_CONTEXT_KEYS, MAX_CONTEXT_ITEMS,
                                             MESSAGE_TABLE, META_TABLE, SESSION_TABLE)


# --------------------------------------------------------------------------- fixtures

def fresh():
    return iso(utc_now())


INCIDENT_EVENT = {
    "event_id": "evt-1", "event_type": "congestion", "status": "active", "severity": "heavy",
    "area": "Silk Board", "anchor_junction_id": "J001", "anchor_name": "Silk Board Junction",
    "affected_junction_count": 4, "avg_current_speed_kmh": 8.5, "avg_free_flow_speed_kmh": 45.0,
}

FEATURE_AREA = {
    "area": "Koramangala", "overall_congestion_level": "heavy", "avg_current_speed_kmh": 12.0,
    "avg_free_flow_speed_kmh": 40.0, "avg_predicted_speed_kmh": 10.0, "sensor_coverage_pct": 90.0,
}

JOURNEY_ID = "jr-ff466f71c8"

JOURNEY = {
    "journey_id": JOURNEY_ID, "label": "Home to Office", "status": "active",
    "total_length_km": 14.2, "updated_at": fresh(),
    "decision": {"journey_id": JOURNEY_ID, "action": "recommend_switch", "progress_pct": 40.0,
                 "remaining_km": 8.1, "eta_min": 22.0, "baseline_eta_min": 26.0, "eta_delta_min": -4.0,
                 "confidence": "high", "updated_at": fresh(), "rationale": "bypass is faster",
                 "alternative": {"rank": 1, "eta_min": 18.0, "gain_min": 4.0, "gain_pct": 18.0,
                                 "total_length_km": 12.9, "turn_off_after_km": 1.2}},
}

HEALTH_READY = {"status": "ok", "ready": True, "mode": "pipeline_test", "step_minutes": 2,
                "horizons_min": [5, 10, 15], "newest_slot": fresh(), "latest_poll_id": "poll-9",
                "sensor_coverage_pct": 88.0, "warmup_steps": 0, "not_ready_reason": None}


class FakeSources(object):
    """In-memory stand-in for LiveDataSources, with a read counter to prove fresh tool runs."""

    def __init__(self, incidents=None, features=None, journeys=None, switches=None, health=None,
                 stale_after_min=10.0):
        self._incidents = incidents
        self._features = features
        self._journeys = journeys or []
        self._switches = switches or []
        self._health = health
        self.stale_after_min = stale_after_min
        self.model_service_url = "http://fake-model:8001"
        self.incidents_path = "incidents.json"
        self.features_path = "features.json"
        self.incident_reads = 0

    def incidents(self):
        self.incident_reads += 1
        return self._incidents

    def features(self):
        return self._features

    def journeys(self):
        return [dict(journey) for journey in self._journeys]

    def journey(self, journey_id):
        for journey in self._journeys:
            if journey.get("journey_id") == journey_id:
                return dict(journey)
        return None

    def switches(self, journey_id=None):
        return [dict(row) for row in self._switches]

    def system_health(self):
        return self._health

    def source_available(self, path):
        return False


class RecordingProvider(object):
    """The real mock provider, with every call it received kept for inspection."""

    name = "mock"

    def __init__(self):
        self.inner = MockLLMProvider()
        self.seen = []

    def converse(self, messages, tools, system="", context=None):
        self.seen.append({"messages": [dict(m) for m in messages], "system": system,
                          "context": dict(context or {})})
        return self.inner.converse(messages, tools, system=system, context=context)

    def first_calls(self):
        """Only each turn's first provider call, before model-requested tools run."""
        first_calls = []
        for call in self.seen:
            messages = call["messages"]
            if not messages:
                continue
            last = messages[-1]
            if last.get("role") == "user":
                first_calls.append(call)
            elif (last.get("role") == "tool"
                  and str(last.get("tool_call_id", "")).startswith("required-")):
                first_calls.append(call)
        return first_calls


class MemoryTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = os.path.join(self.tmp.name, "memory.sqlite")

    def make_memory(self, **kwargs):
        kwargs.setdefault("db_path", self.db_path)
        memory = ConversationMemory(**kwargs)
        self.addCleanup(memory.close)
        return memory

    def agent_with(self, sources, memory, provider=None):
        return InteractionAgent(config=InteractionConfig.from_env({}),
                                provider=provider or MockLLMProvider(),
                                registry=build_default_registry(sources), memory=memory)


def table_names(path):
    con = sqlite3.connect(path)
    try:
        return {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


# --------------------------------------------------------------------------- 1. configuration

class TestPhase2Config(unittest.TestCase):
    def test_memory_defaults(self):
        config = InteractionConfig.from_env({})
        self.assertTrue(config.memory_enabled)
        self.assertEqual(config.memory_max_messages, 20)
        self.assertTrue(config.memory_db_path.endswith("interaction_memory.sqlite"))
        self.assertEqual(config.memory_db_path, DEFAULT_DB_PATH)

    def test_bedrock_default_is_nova_lite(self):
        self.assertEqual(InteractionConfig.from_env({}).bedrock_model_id, "amazon.nova-lite-v1:0")

    def test_memory_env_overrides(self):
        config = InteractionConfig.from_env({
            "INTERACTION_MEMORY_DB_PATH": "/tmp/custom.sqlite",
            "INTERACTION_MEMORY_MAX_MESSAGES": "5",
            "INTERACTION_MEMORY_ENABLED": "false"})
        self.assertEqual(config.memory_db_path, "/tmp/custom.sqlite")
        self.assertEqual(config.memory_max_messages, 5)
        self.assertFalse(config.memory_enabled)

    def test_memory_enabled_accepts_truthy_words(self):
        for value in ("1", "true", "YES", "on"):
            self.assertTrue(InteractionConfig.from_env(
                {"INTERACTION_MEMORY_ENABLED": value}).memory_enabled, value)

    def test_bad_memory_values_rejected(self):
        with self.assertRaises(ConfigError):
            InteractionConfig.from_env({"INTERACTION_MEMORY_ENABLED": "maybe"})
        with self.assertRaises(ConfigError):
            InteractionConfig.from_env({"INTERACTION_MEMORY_MAX_MESSAGES": "0"})
        with self.assertRaises(ConfigError):
            InteractionConfig.from_env({"INTERACTION_MEMORY_DB_PATH": "   "})


# --------------------------------------------------------------------------- 2. schema

class TestMemorySchema(MemoryTestCase):
    def test_initialization_creates_namespaced_tables(self):
        memory = self.make_memory()
        self.assertTrue(memory.connect())
        names = table_names(self.db_path)
        self.assertLessEqual({SESSION_TABLE, MESSAGE_TABLE, META_TABLE}, names)
        con = sqlite3.connect(self.db_path)
        try:
            version = con.execute("SELECT value FROM %s WHERE key='schema_version'" % META_TABLE)
            self.assertEqual(version.fetchone()[0], "1")
        finally:
            con.close()

    def test_opening_is_lazy(self):
        memory = self.make_memory()
        self.assertFalse(os.path.exists(self.db_path))
        self.assertFalse(memory.available)
        memory.create_session()
        self.assertTrue(os.path.exists(self.db_path))

    def test_existing_database_is_preserved(self):
        # Stand in for the LangGraph checkpoint store that already lives in the runtime file.
        con = sqlite3.connect(self.db_path)
        con.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT, data BLOB)")
        con.execute("INSERT INTO checkpoints VALUES ('t1', 'c1', X'00')")
        con.commit()
        con.close()

        memory = self.make_memory()
        self.assertTrue(memory.connect())
        memory.create_session("cs-keep")
        memory.add_message("cs-keep", "user", "hello")

        names = table_names(self.db_path)
        self.assertIn("checkpoints", names)                      # untouched, still there
        con = sqlite3.connect(self.db_path)
        try:
            rows = con.execute("SELECT thread_id, checkpoint_id FROM checkpoints").fetchall()
        finally:
            con.close()
        self.assertEqual(rows, [("t1", "c1")])                    # and its data is intact

    def test_reopening_an_existing_schema_is_idempotent(self):
        first = self.make_memory()
        first.create_session("cs-1")
        first.close()
        second = self.make_memory()
        self.assertTrue(second.connect())
        self.assertEqual(second.message_count("cs-1"), 0)


# --------------------------------------------------------------------------- 3. sessions

class TestSessions(MemoryTestCase):
    def test_create_session_returns_unique_ids(self):
        memory = self.make_memory()
        first, second = memory.create_session(), memory.create_session()
        self.assertTrue(first.startswith("cs-") and second.startswith("cs-"))
        self.assertNotEqual(first, second)

    def test_create_session_with_explicit_id_is_idempotent(self):
        memory = self.make_memory()
        self.assertEqual(memory.create_session("cs-mine"), "cs-mine")
        memory.add_message("cs-mine", "user", "hi")
        self.assertEqual(memory.create_session("cs-mine"), "cs-mine")
        self.assertEqual(memory.message_count("cs-mine"), 1)

    def test_get_session_returns_metadata(self):
        memory = self.make_memory()
        memory.create_session("cs-1")
        session = memory.get_session("cs-1")
        self.assertEqual(session["session_id"], "cs-1")
        self.assertEqual(session["context"], {})
        self.assertIsNotNone(session["created_at"])
        self.assertIsNone(memory.get_session("cs-nope"))

    def test_ensure_session_creates_only_when_missing(self):
        memory = self.make_memory()
        created = memory.ensure_session("cs-1")
        self.assertEqual(created, "cs-1")
        self.assertEqual(memory.ensure_session("cs-1"), "cs-1")
        self.assertEqual(len(memory.session_ids()), 1)

    def test_sessions_are_isolated(self):
        memory = self.make_memory()
        memory.add_message("cs-a", "user", "for a")
        memory.add_message("cs-b", "user", "for b")
        self.assertEqual([m["content"] for m in memory.recent_messages("cs-a")], ["for a"])
        self.assertEqual([m["content"] for m in memory.recent_messages("cs-b")], ["for b"])
        self.assertEqual(memory.message_count("cs-a"), 1)
        self.assertEqual(memory.message_count("cs-b"), 1)


# --------------------------------------------------------------------------- 4. messages

class TestMessages(MemoryTestCase):
    def test_store_and_retrieve(self):
        memory = self.make_memory()
        memory.add_message("cs-1", "user", "Are there any incidents?")
        memory.add_message("cs-1", "assistant", "No congestion incidents are being reported.")
        rows = memory.recent_messages("cs-1")
        self.assertEqual([row["role"] for row in rows], ["user", "assistant"])
        self.assertEqual(rows[0]["content"], "Are there any incidents?")
        self.assertTrue(rows[0]["created_at"])

    def test_chronological_ordering(self):
        memory = self.make_memory()
        for index in range(4):
            memory.add_message("cs-1", "user" if index % 2 == 0 else "assistant", "m%d" % index)
        rows = memory.recent_messages("cs-1")
        self.assertEqual([row["content"] for row in rows], ["m0", "m1", "m2", "m3"])
        self.assertEqual([row["id"] for row in rows], sorted(row["id"] for row in rows))

    def test_history_is_role_and_content_only(self):
        memory = self.make_memory()
        memory.add_message("cs-1", "user", "incidents?")
        memory.add_message("cs-1", "assistant", "none", tool_calls=[{"name": "get_incidents"}])
        self.assertEqual(memory.history("cs-1"),
                         [{"role": "user", "content": "incidents?"},
                          {"role": "assistant", "content": "none"}])

    def test_tool_calls_are_recorded_but_kept_out_of_history(self):
        memory = self.make_memory()
        memory.add_message("cs-1", "assistant", "answer", tool_calls=[{"name": "get_incidents",
                                                                      "availability": "live"}])
        stored = memory.recent_messages("cs-1")[0]
        self.assertEqual(stored["tool_calls"][0]["name"], "get_incidents")
        self.assertNotIn("tool_calls", memory.history("cs-1")[0])

    def test_invalid_role_is_rejected(self):
        memory = self.make_memory()
        with self.assertRaises(ValueError):
            memory.add_message("cs-1", "system", "not a real turn")

    def test_message_count(self):
        memory = self.make_memory()
        self.assertEqual(memory.message_count("cs-1"), 0)
        memory.add_message("cs-1", "user", "a")
        self.assertEqual(memory.message_count("cs-1"), 1)


# --------------------------------------------------------------------------- 5. history limits

class TestHistoryLimits(MemoryTestCase):
    def test_recent_messages_returns_the_newest_in_order(self):
        memory = self.make_memory(max_messages=4)
        for index in range(6):
            memory.add_message("cs-1", "user", "m%d" % index)
        rows = memory.recent_messages("cs-1")
        self.assertEqual([row["content"] for row in rows], ["m2", "m3", "m4", "m5"])

    def test_explicit_limit(self):
        memory = self.make_memory()
        for index in range(6):
            memory.add_message("cs-1", "user", "m%d" % index)
        self.assertEqual([row["content"] for row in memory.recent_messages("cs-1", 2)],
                         ["m4", "m5"])
        self.assertEqual(memory.recent_messages("cs-1", 0), [])

    def test_history_respects_the_limit(self):
        memory = self.make_memory(max_messages=2)
        memory.add_message("cs-1", "user", "a")
        memory.add_message("cs-1", "assistant", "b")
        memory.add_message("cs-1", "user", "c")
        self.assertEqual([row["content"] for row in memory.history("cs-1")], ["b", "c"])


# --------------------------------------------------------------------------- 6. context

class TestContext(MemoryTestCase):
    def test_set_and_get(self):
        memory = self.make_memory()
        memory.update_context("cs-1", last_intent="journey")
        self.assertEqual(memory.get_context("cs-1")["last_intent"], "journey")
        self.assertEqual(memory.get_context("cs-unknown"), {})

    def test_lists_merge_newest_first_and_de_duplicate(self):
        memory = self.make_memory()
        memory.update_context("cs-1", journey_ids=["jr-a", "jr-b"])
        context = memory.update_context("cs-1", journey_ids=["jr-b", "jr-c"])
        self.assertEqual(context["journey_ids"], ["jr-b", "jr-c", "jr-a"])

    def test_lists_are_capped(self):
        memory = self.make_memory()
        memory.update_context("cs-1", places=["p%d" % index for index in range(9)])
        self.assertEqual(len(memory.get_context("cs-1")["places"]), MAX_CONTEXT_ITEMS)

    def test_none_values_do_not_clobber(self):
        memory = self.make_memory()
        memory.update_context("cs-1", last_intent="incidents")
        memory.update_context("cs-1", last_intent=None, places=["Silk Board"])
        context = memory.get_context("cs-1")
        self.assertEqual(context["last_intent"], "incidents")
        self.assertEqual(context["places"], ["Silk Board"])

    def test_list_context_keys_are_the_documented_ones(self):
        self.assertEqual(set(LIST_CONTEXT_KEYS), {"journey_ids", "places"})

    def test_context_survives_reopen(self):
        memory = self.make_memory()
        memory.update_context("cs-1", last_intent="alternatives", journey_ids=[JOURNEY_ID])
        memory.close()
        reopened = self.make_memory()
        self.assertEqual(reopened.get_context("cs-1")["journey_ids"], [JOURNEY_ID])


# --------------------------------------------------------------------------- 7. clearing

class TestClearSession(MemoryTestCase):
    def test_clear_removes_messages_and_context_but_keeps_the_session(self):
        memory = self.make_memory()
        memory.add_message("cs-1", "user", "a")
        memory.add_message("cs-1", "assistant", "b")
        memory.update_context("cs-1", last_intent="journey")
        self.assertEqual(memory.clear_session("cs-1"), 2)
        self.assertEqual(memory.message_count("cs-1"), 0)
        self.assertEqual(memory.get_context("cs-1"), {})
        self.assertIsNotNone(memory.get_session("cs-1"))

    def test_clear_touches_only_the_named_session(self):
        memory = self.make_memory()
        memory.add_message("cs-a", "user", "a")
        memory.add_message("cs-b", "user", "b")
        memory.clear_session("cs-a")
        self.assertEqual(memory.message_count("cs-a"), 0)
        self.assertEqual(memory.message_count("cs-b"), 1)


# --------------------------------------------------------------------------- 8. persistence

class TestPersistence(MemoryTestCase):
    def test_messages_survive_a_reopen(self):
        memory = self.make_memory()
        session = memory.create_session()
        memory.add_message(session, "user", "Are there any traffic incidents?")
        memory.add_message(session, "assistant", "One congestion event is open.")
        memory.close()

        reopened = self.make_memory()
        rows = reopened.recent_messages(session)
        self.assertEqual([row["content"] for row in rows],
                         ["Are there any traffic incidents?", "One congestion event is open."])

    def test_a_second_handle_sees_the_same_conversation(self):
        first = self.make_memory()
        session = first.create_session("cs-shared")
        first.add_message("cs-shared", "user", "hello")
        second = self.make_memory()
        self.assertEqual(second.message_count(session), 1)
        self.assertEqual(second.history(session)[0]["content"], "hello")

    def test_concurrent_writers_do_not_lose_messages(self):
        memory = self.make_memory()
        memory.create_session("cs-threads")
        errors = []

        def writer(worker):
            try:
                for index in range(10):
                    memory.add_message("cs-threads", "user", "w%d-%d" % (worker, index))
            except Exception as exc:                                    # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(worker,)) for worker in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(memory.message_count("cs-threads"), 40)


# --------------------------------------------------------------------------- 9. disabled / broken

class TestMemoryDisabledAndBroken(MemoryTestCase):
    def test_disabled_memory_is_a_no_op(self):
        memory = self.make_memory(enabled=False)
        session = memory.create_session()
        self.assertTrue(session)
        self.assertFalse(memory.enabled)
        self.assertFalse(memory.available)
        self.assertFalse(memory.add_message(session, "user", "hi"))
        self.assertEqual(memory.recent_messages(session), [])
        self.assertEqual(memory.get_context(session), {})
        self.assertEqual(memory.clear_session(session), 0)
        self.assertEqual(memory.message_count(session), 0)
        self.assertFalse(os.path.exists(self.db_path))

    def test_unusable_path_degrades_instead_of_raising(self):
        memory = ConversationMemory(db_path=self.tmp.name)          # a directory, not a file
        self.addCleanup(memory.close)
        self.assertFalse(memory.connect())
        self.assertFalse(memory.available)
        session = memory.create_session()
        self.assertTrue(session)                                    # the caller still gets an id
        self.assertFalse(memory.add_message(session, "user", "hi"))
        self.assertEqual(memory.recent_messages(session), [])

    def test_agent_still_answers_when_memory_is_broken(self):
        sources = FakeSources(incidents={"made_at": fresh(), "active_events": [INCIDENT_EVENT]})
        memory = ConversationMemory(db_path=self.tmp.name)
        self.addCleanup(memory.close)
        agent = self.agent_with(sources, memory)
        reply = agent.answer("Are there any traffic incidents?", session_id="cs-1")
        self.assertTrue(reply.ok)
        self.assertIn("Silk Board", reply.text)
        self.assertEqual(reply.session_id, "cs-1")

    def test_agent_still_answers_when_memory_is_disabled(self):
        sources = FakeSources(incidents={"made_at": fresh(), "active_events": [INCIDENT_EVENT]})
        agent = self.agent_with(sources, self.make_memory(enabled=False))
        reply = agent.answer("Are there any traffic incidents?", session_id="cs-1")
        self.assertIn("Silk Board", reply.text)
        self.assertEqual(reply.session_id, "cs-1")
        self.assertIsNone(agent.new_session())


# --------------------------------------------------------------------------- 10. follow-ups

class TestFollowupResolution(unittest.TestCase):
    def test_followups_resolve_from_context(self):
        cases = (
            ("What about the delay?", {"last_intent": "journey"}, "journey"),
            ("How much time can I save?", {"last_intent": "alternatives"}, "alternatives"),
            ("Where are they?", {"last_intent": "incidents"}, "incidents"),
            ("How long is that?", {"last_intent": "journey"}, "journey"),
            ("Is it up?", {"last_intent": "system"}, "system"),
        )
        for question, context, intent in cases:
            self.assertEqual(detect_followup(question, context), intent, question)

    def test_journey_context_wins_when_no_topic_is_named(self):
        self.assertEqual(detect_followup("Anything else?", {"journey_ids": [JOURNEY_ID]}), "journey")

    def test_unresolvable_followup_returns_none(self):
        self.assertIsNone(detect_followup("What about it?", {}))
        self.assertIsNone(detect_followup("Tell me a joke about cats", {"last_intent": "journey"}))

    def test_mock_asks_for_clarification_without_context(self):
        provider = MockLLMProvider()
        response = provider.converse([{"role": "user", "content": "What about the delay?"}], [])
        self.assertFalse(response.wants_tools)
        self.assertEqual(response.text, CLARIFY)

    def test_mock_resolves_the_delay_to_the_known_journey(self):
        provider = MockLLMProvider()
        response = provider.converse([{"role": "user", "content": "What about the delay?"}], [],
                                     context={"last_intent": "journey", "journey_ids": [JOURNEY_ID]})
        self.assertEqual(response.tool_calls[0].name, "get_journey")
        self.assertEqual(response.tool_calls[0].arguments, {"journey_id": JOURNEY_ID})

    def test_mock_uses_the_last_topic_for_an_unresolvable_phrasing(self):
        provider = MockLLMProvider()
        response = provider.converse([{"role": "user", "content": "Anything else?"}], [],
                                     context={"last_intent": "alternatives"})
        self.assertEqual(response.tool_calls[0].name, "get_route_recommendations")


class TestTurnScoping(unittest.TestCase):
    """An old tool result must never be mistaken for this turn's traffic."""

    def test_previous_turn_tool_result_is_not_reused(self):
        provider = MockLLMProvider()
        messages = [
            {"role": "user", "content": "Are there any traffic incidents?"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "get_incidents"}]},
            {"role": "tool", "tool_call_id": "call-1", "name": "get_incidents",
             "result": {"ok": True, "availability": "live", "headline": "OLD RESULT",
                        "details": ["15 events"]}},
            {"role": "user", "content": "Anything else?"},
        ]
        response = provider.converse(messages, [], context={"last_intent": "incidents"})
        self.assertTrue(response.wants_tools)                    # a fresh call, not the old answer
        self.assertEqual(response.tool_calls[0].name, "get_incidents")
        self.assertNotIn("OLD RESULT", response.text or "")

    def test_a_tool_result_in_the_current_turn_is_still_used(self):
        provider = MockLLMProvider()
        messages = [
            {"role": "user", "content": "Are there any traffic incidents?"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "get_incidents"}]},
            {"role": "tool", "tool_call_id": "call-1", "name": "get_incidents",
             "result": {"ok": True, "availability": "live", "headline": "FRESH RESULT",
                        "details": []}},
        ]
        response = provider.converse(messages, [])
        self.assertFalse(response.wants_tools)
        self.assertIn("FRESH RESULT", response.text)


# --------------------------------------------------------------------------- 11. multi-turn agent

class TestContextAwareAgent(MemoryTestCase):
    def test_ambiguous_delay_followup_without_context_is_not_prefetched(self):
        provider = RecordingProvider()
        agent = self.agent_with(FakeSources(journeys=[JOURNEY]), self.make_memory(),
                                provider=provider)

        reply = agent.answer("What about the delay?")

        self.assertEqual(reply.tool_calls, [])
        self.assertEqual(reply.text, CLARIFY)

    def test_followup_uses_the_journey_from_the_previous_turn(self):
        sources = FakeSources(journeys=[JOURNEY])
        memory = self.make_memory()
        provider = RecordingProvider()
        agent = self.agent_with(sources, memory, provider=provider)
        session = agent.new_session()

        first = agent.answer("What is the status of my journey?", session_id=session)
        self.assertEqual(first.tool_calls[0].name, "get_journeys")

        second = agent.answer("What about the delay?", session_id=session)
        self.assertEqual(second.tool_calls[0].name, "get_journey")
        self.assertEqual(second.tool_calls[0].arguments, {"journey_id": JOURNEY_ID})
        self.assertEqual(second.session_id, session)

    def test_history_is_replayed_to_the_provider(self):
        sources = FakeSources(journeys=[JOURNEY])
        provider = RecordingProvider()
        agent = self.agent_with(sources, self.make_memory(), provider=provider)
        session = agent.new_session()
        agent.answer("What is the status of my journey?", session_id=session)
        agent.answer("What about the delay?", session_id=session)

        turn_two = provider.first_calls()[1]
        self.assertEqual([message["role"] for message in turn_two["messages"][:3]],
                         ["user", "assistant", "user"])
        self.assertEqual(turn_two["messages"][2]["content"], "What about the delay?")
        self.assertEqual(turn_two["messages"][-2]["role"], "assistant")
        self.assertEqual(turn_two["messages"][-2]["tool_calls"][0]["id"],
                         "required-get_journey")
        self.assertEqual(turn_two["messages"][-1]["role"], "tool")
        self.assertEqual(turn_two["context"]["journey_ids"], [JOURNEY_ID])

    def test_conversation_is_stored_across_turns(self):
        sources = FakeSources(journeys=[JOURNEY])
        memory = self.make_memory()
        agent = self.agent_with(sources, memory)
        session = agent.new_session()
        agent.answer("What is the status of my journey?", session_id=session)
        agent.answer("What about the delay?", session_id=session)
        self.assertEqual([row["role"] for row in memory.recent_messages(session)],
                         ["user", "assistant", "user", "assistant"])
        self.assertEqual(memory.get_context(session)["last_intent"], "journey")

    def test_history_limit_bounds_what_is_replayed(self):
        sources = FakeSources(journeys=[JOURNEY])
        provider = RecordingProvider()
        agent = self.agent_with(sources, self.make_memory(max_messages=2), provider=provider)
        session = agent.new_session()
        agent.answer("What is the status of my journey?", session_id=session)
        agent.answer("What about the delay?", session_id=session)
        agent.answer("And the alternative?", session_id=session)

        last_turn = provider.first_calls()[-1]
        self.assertEqual([message["role"] for message in last_turn["messages"][:3]],
                         ["user", "assistant", "user"])
        self.assertEqual(last_turn["messages"][2]["content"], "And the alternative?")
        self.assertEqual(last_turn["messages"][-1]["role"], "tool")

    def test_incident_place_is_remembered_for_the_next_turn(self):
        sources = FakeSources(incidents={"made_at": fresh(), "active_events": [INCIDENT_EVENT]})
        memory = self.make_memory()
        agent = self.agent_with(sources, memory)
        session = agent.new_session()
        agent.answer("Are there any traffic incidents?", session_id=session)
        context = memory.get_context(session)
        self.assertEqual(context["last_intent"], "incidents")
        self.assertEqual(context["places"], ["Silk Board"])

    def test_source_reads_are_fresh_on_every_turn(self):
        sources = FakeSources(incidents={"made_at": fresh(), "active_events": [INCIDENT_EVENT]})
        agent = self.agent_with(sources, self.make_memory())
        session = agent.new_session()
        first = agent.answer("Are there any traffic incidents?", session_id=session)
        second = agent.answer("Where are they?", session_id=session)
        self.assertEqual(first.tool_calls[0].name, "get_incidents")
        self.assertEqual(second.tool_calls[0].name, "get_incidents")
        self.assertEqual(sources.incident_reads, 2)              # re-read, not replayed

    def test_second_session_is_isolated(self):
        sources = FakeSources(journeys=[JOURNEY])
        memory = self.make_memory()
        agent = self.agent_with(sources, memory)
        one, two = agent.new_session(), agent.new_session()
        agent.answer("What is the status of my journey?", session_id=one)
        self.assertEqual(memory.message_count(two), 0)
        self.assertEqual(memory.get_context(two), {})

    def test_clearing_a_session_ends_its_context(self):
        sources = FakeSources(journeys=[JOURNEY])
        memory = self.make_memory()
        agent = self.agent_with(sources, memory)
        session = agent.new_session()
        agent.answer("What is the status of my journey?", session_id=session)
        memory.clear_session(session)
        self.assertEqual(memory.get_context(session), {})

    def test_multi_turn_example_end_to_end(self):
        """The worked example from the deliverable notes, asserted as one conversation."""
        sources = FakeSources(
            incidents={"made_at": fresh(), "active_events": [INCIDENT_EVENT]},
            journeys=[JOURNEY])
        agent = self.agent_with(sources, self.make_memory())
        session = agent.new_session()

        self.assertIn("Silk Board", agent.answer("Are there any traffic incidents?",
                                                 session_id=session).text)
        self.assertIn("Home to Office", agent.answer("What is the status of my journey?",
                                                     session_id=session).text)
        followup = agent.answer("What about the delay?", session_id=session)
        self.assertIn("Home to Office", followup.text)
        self.assertIn("ETA 22.0 min", followup.text)


# --------------------------------------------------------------------------- 12. backwards compatibility

class TestPhase1Compatibility(MemoryTestCase):
    def test_stateless_turn_returns_no_session_and_touches_no_database(self):
        sources = FakeSources(incidents={"made_at": fresh(), "active_events": [INCIDENT_EVENT]})
        agent = self.agent_with(sources, self.make_memory())
        reply = agent.answer("Are there any traffic incidents?")
        self.assertIsNone(reply.session_id)
        self.assertFalse(agent.memory.available)
        self.assertFalse(os.path.exists(self.db_path))
        self.assertIn("Silk Board", reply.text)

    def test_default_agent_builds_without_opening_the_runtime_database(self):
        agent = build_agent(InteractionConfig.from_env({}))
        reply = agent.answer("Is the traffic prediction system running?")
        self.assertTrue(reply.ok)
        self.assertIsNone(reply.session_id)
        self.assertFalse(agent.memory.available)                  # still never opened

    def test_reply_dict_gains_session_id_without_losing_fields(self):
        sources = FakeSources(journeys=[JOURNEY])
        agent = self.agent_with(sources, self.make_memory())
        stateless = agent.answer("What is the status of my journey?").as_dict()
        self.assertEqual(stateless["session_id"], None)
        self.assertLessEqual({"text", "provider", "tool_calls", "steps", "error", "ok"}, set(stateless))

    def test_phase1_style_provider_is_still_called_without_context(self):
        """A provider written against the Phase 1 signature must keep working statelessly."""
        calls = []

        class LegacyProvider(object):
            name = "legacy"

            def converse(self, messages, tools, system=""):
                calls.append(messages)
                return LLMResponse(text="ok")

        agent = InteractionAgent(config=InteractionConfig.from_env({}),
                                 provider=LegacyProvider(),
                                 registry=build_default_registry(FakeSources()),
                                 memory=self.make_memory())
        reply = agent.answer("hello")
        self.assertEqual(reply.text, "ok")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(reply.session_id)

    def test_empty_question_still_short_circuits_in_a_session(self):
        agent = self.agent_with(FakeSources(), self.make_memory())
        reply = agent.answer("   ", session_id="cs-1")
        self.assertEqual(reply.text, EMPTY_QUESTION_REPLY)
        self.assertEqual(reply.session_id, "cs-1")

    def test_provider_failure_is_still_graceful_in_a_session(self):
        class ExplodingProvider(object):
            name = "exploding"

            def converse(self, messages, tools, system="", context=None):
                raise RuntimeError("bedrock unreachable")

        agent = InteractionAgent(config=InteractionConfig.from_env({}),
                                 provider=ExplodingProvider(),
                                 registry=build_default_registry(FakeSources()),
                                 memory=self.make_memory())
        reply = agent.answer("incidents?", session_id="cs-1")
        self.assertEqual(reply.text, PROVIDER_ERROR_REPLY)
        self.assertIsNotNone(reply.error)

    def test_step_limit_still_applies_in_a_session(self):
        class LoopingProvider(object):
            name = "looping"

            def converse(self, messages, tools, system="", context=None):
                return LLMResponse(tool_calls=[ToolCall("get_incidents")])

        agent = InteractionAgent(config=InteractionConfig.from_env({}),
                                 provider=LoopingProvider(),
                                 registry=build_default_registry(FakeSources()),
                                 memory=self.make_memory())
        reply = agent.answer("incidents?", session_id="cs-1")
        self.assertEqual(reply.steps, MAX_STEPS)


# --------------------------------------------------------------------------- 13. context extraction

class TestContextUpdate(unittest.TestCase):
    def test_journey_ids_come_from_tool_data_only(self):
        results = [{"data": {"journey": {"journey_id": JOURNEY_ID}}}]
        self.assertEqual(context_update("hello", [], results)["journey_ids"], [JOURNEY_ID])

    def test_journey_id_in_the_question_is_not_treated_as_known(self):
        changes = context_update("What about journey %s?" % JOURNEY_ID, [], [])
        self.assertNotIn("journey_ids", changes)

    def test_places_come_from_incidents_and_areas(self):
        results = [{"data": {"events": [{"area": "Silk Board"}]}},
                   {"data": {"areas": [{"area": "Koramangala"}]}}]
        self.assertEqual(context_update("q", [], results)["places"], ["Silk Board", "Koramangala"])

    def test_last_intent_follows_the_tool_used(self):
        class Call(object):
            name = "get_route_recommendations"
        self.assertEqual(context_update("q", [Call()], [])["last_intent"], "alternatives")

    def test_nothing_is_invented_when_no_tool_ran(self):
        self.assertEqual(context_update("hello", [], []), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
