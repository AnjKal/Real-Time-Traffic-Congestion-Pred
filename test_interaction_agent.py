"""Isolated unit tests for the User Interaction Agent (Phase 1).

    python -m unittest test_interaction_agent -v

Nothing here needs AWS, Kafka, Docker, TensorFlow or the network. The tools are exercised against
in-memory fake data sources; the one HTTP call (the model service ``/health``) is stubbed.
"""
import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.journey_memory import iso, utc_now
from webapp.interaction_agent import (ConfigError, InteractionConfig, MockLLMProvider,
                                      ProviderUnavailable, ToolCall, ToolRegistry, build_agent,
                                      build_default_registry)
from webapp.interaction_agent.agent import (EMPTY_QUESTION_REPLY, MAX_STEPS, NO_ANSWER_REPLY,
                                            PROVIDER_ERROR_REPLY, STEP_LIMIT_REPLY,
                                            InteractionAgent)
from webapp.interaction_agent.llm import CLARIFY, LLMResponse, as_tool_config, detect_intent
from webapp.interaction_agent.tools import (LiveDataSources, get_congestion_summary, get_incidents,
                                            get_journey, get_journeys, get_route_recommendations,
                                            get_system_status)


# --------------------------------------------------------------------------- fixtures

def fresh():
    return iso(utc_now())


def old(minutes=60):
    from datetime import timedelta
    return iso(utc_now() - timedelta(minutes=minutes))


INCIDENT_EVENT = {
    "event_id": "evt-1", "event_type": "congestion", "status": "active", "severity": "heavy",
    "area": "Silk Board", "anchor_junction_id": "J001", "anchor_name": "Silk Board Junction",
    "affected_junction_count": 4, "avg_current_speed_kmh": 8.5, "avg_free_flow_speed_kmh": 45.0,
    "avg_predicted_speed_kmh": 7.0, "forecast_horizon_min": 15,
}

FEATURE_AREA = {
    "area": "Koramangala", "overall_congestion_level": "heavy", "avg_current_speed_kmh": 12.0,
    "avg_free_flow_speed_kmh": 40.0, "avg_predicted_speed_kmh": 10.0, "avg_delay_min_per_km": 2.4,
    "reporting_junction_count": 18, "total_junction_count": 20, "sensor_coverage_pct": 90.0,
}

JOURNEY = {
    "journey_id": "jr-ff466f71c8", "label": "Home to Office", "status": "active",
    "total_length_km": 14.2, "updated_at": fresh(),
    "decision": {"journey_id": "jr-ff466f71c8", "action": "recommend_switch", "progress_pct": 40.0,
                 "remaining_km": 8.1, "eta_min": 22.0, "baseline_eta_min": 26.0, "eta_delta_min": -4.0,
                 "confidence": "high", "updated_at": fresh(), "rationale": "bypass is faster",
                 "alternative": {"rank": 1, "eta_min": 18.0, "gain_min": 4.0, "gain_pct": 18.0,
                                 "total_length_km": 12.9, "turn_off_after_km": 1.2}},
}

SWITCH_ROW = {"journey_id": "jr-ff466f71c8", "switched_at": old(5), "gain_min": 4.0,
              "gain_realised_min": None, "reason": "bypass"}

HEALTH_READY = {"status": "ok", "ready": True, "mode": "pipeline_test", "step_minutes": 2,
                "horizons_min": [5, 10, 15], "newest_slot": fresh(), "latest_poll_id": "poll-9",
                "sensor_coverage_pct": 88.0, "warmup_steps": 0, "not_ready_reason": None}

HEALTH_WARMING = dict(HEALTH_READY, ready=False, not_ready_reason="collecting history")


class FakeSources(object):
    """In-memory stand-in for :class:`LiveDataSources`; no files, no HTTP."""

    def __init__(self, incidents=None, features=None, journeys=None, switches=None,
                 health=None, health_raises=False, published=(), stale_after_min=10.0):
        self._incidents = incidents
        self._features = features
        self._journeys = journeys or []
        self._switches = switches or []
        self._health = health
        self._health_raises = health_raises
        self.stale_after_min = stale_after_min
        self.model_service_url = "http://fake-model:8001"
        self.incidents_path = "incidents.json"
        self.features_path = "features.json"
        self._published = set(published)

    def incidents(self):
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
        if self._health_raises:
            return None
        return self._health

    def source_available(self, path):
        return path in self._published


class ScriptedProvider(object):
    """Returns a fixed list of :class:`LLMResponse` objects, one per call."""

    name = "scripted"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def converse(self, messages, tools, system=""):
        self.calls.append({"messages": list(messages), "tools": list(tools), "system": system})
        return self._responses.pop(0)


class ExplodingProvider(object):
    name = "exploding"

    def converse(self, messages, tools, system=""):
        raise RuntimeError("bedrock unreachable")


# --------------------------------------------------------------------------- 1. configuration

class TestConfig(unittest.TestCase):
    def test_defaults_are_local_mock(self):
        config = InteractionConfig.from_env({})
        self.assertEqual(config.llm_provider, "mock")
        self.assertFalse(config.uses_bedrock)
        self.assertEqual(config.aws_region, "ap-south-1")
        self.assertTrue(config.bedrock_model_id)
        self.assertGreater(config.bedrock_max_tokens, 0)

    def test_bedrock_fields_are_read_from_env(self):
        config = InteractionConfig.from_env({
            "INTERACTION_LLM_PROVIDER": "bedrock", "AWS_REGION": "us-east-1",
            "BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "BEDROCK_MAX_TOKENS": "2048"})
        self.assertTrue(config.uses_bedrock)
        self.assertEqual(config.aws_region, "us-east-1")
        self.assertEqual(config.bedrock_max_tokens, 2048)

    def test_default_region_used_when_aws_region_absent(self):
        config = InteractionConfig.from_env({"AWS_DEFAULT_REGION": "eu-west-1",
                                             "INTERACTION_LLM_PROVIDER": "bedrock"})
        self.assertEqual(config.aws_region, "eu-west-1")

    def test_unknown_provider_rejected(self):
        with self.assertRaises(ConfigError):
            InteractionConfig.from_env({"INTERACTION_LLM_PROVIDER": "openai"})

    def test_bad_numeric_values_rejected(self):
        with self.assertRaises(ConfigError):
            InteractionConfig.from_env({"BEDROCK_MAX_TOKENS": "lots"})
        with self.assertRaises(ConfigError):
            InteractionConfig.from_env({"INTERACTION_STALE_AFTER_MIN": "0"})

    def test_no_credentials_are_stored(self):
        self.assertFalse(any("KEY" in name or "SECRET" in name
                             for name in InteractionConfig.__dataclass_fields__))


# --------------------------------------------------------------------------- 2. mock provider

class TestMockProvider(unittest.TestCase):
    def setUp(self):
        self.provider = MockLLMProvider()

    def answer(self, text):
        return self.provider.converse([{"role": "user", "content": text}], tools=[])

    def test_intent_detection(self):
        cases = {
            "Are there any traffic incidents?": "incidents",
            "What is the current congestion status?": "congestion",
            "What is the status of my journey?": "journey",
            "Are there any alternative route recommendations?": "alternatives",
            "Is the traffic prediction system running?": "system",
        }
        for question, intent in cases.items():
            self.assertEqual(detect_intent(question), intent, question)

    def test_supported_questions_map_to_one_tool_call(self):
        cases = {
            "Are there any traffic incidents?": "get_incidents",
            "What is the current congestion status?": "get_congestion_summary",
            "What is the status of my journey?": "get_journeys",
            "Are there any alternative route recommendations?": "get_route_recommendations",
            "Is the traffic prediction system running?": "get_system_status",
        }
        for question, tool in cases.items():
            response = self.answer(question)
            self.assertTrue(response.wants_tools, question)
            self.assertEqual(len(response.tool_calls), 1)
            self.assertEqual(response.tool_calls[0].name, tool, question)

    def test_journey_id_selects_single_journey_tool(self):
        response = self.answer("What is the status of my journey jr-ff466f71c8?")
        self.assertEqual(response.tool_calls[0].name, "get_journey")
        self.assertEqual(response.tool_calls[0].arguments, {"journey_id": "jr-ff466f71c8"})

    def test_unknown_question_asks_for_clarification(self):
        response = self.answer("Tell me a joke about cats")
        self.assertFalse(response.wants_tools)
        self.assertEqual(response.text, CLARIFY)

    def test_composes_answer_from_tool_result(self):
        message = {"role": "tool", "name": "get_incidents", "result": {
            "ok": True, "availability": "live", "headline": "1 congestion event(s) are open.",
            "details": ["Silk Board: active, heavy"]}}
        response = self.provider.converse([{"role": "user", "content": "incidents"}, message], tools=[])
        self.assertIn("1 congestion event(s) are open.", response.text)
        self.assertIn("Silk Board", response.text)

    def test_adds_availability_note_and_is_not_a_language_model(self):
        for availability, phrase in (("missing", "not been published"),
                                     ("stale", "out of date"),
                                     ("unavailable", "not reachable")):
            message = {"role": "tool", "result": {"ok": True, "availability": availability,
                                                  "headline": "h", "details": []}}
            response = self.provider.converse([message], tools=[])
            self.assertIn(phrase, response.text)

    def test_failed_tool_result_becomes_apology(self):
        message = {"role": "tool", "result": {"ok": False, "error": "tool failed"}}
        response = self.provider.converse([message], tools=[])
        self.assertIn("couldn't retrieve", response.text)

    def test_as_tool_config_shape(self):
        specs = [{"toolSpec": {"name": "get_incidents"}}]
        self.assertEqual(as_tool_config(specs), {"tools": specs})


# --------------------------------------------------------------------------- 3. tool execution

class TestTools(unittest.TestCase):
    def test_incidents_live(self):
        sources = FakeSources(incidents={"made_at": fresh(), "active_events": [INCIDENT_EVENT],
                                         "recent_resolved": []})
        result = get_incidents(sources)
        self.assertTrue(result["ok"])
        self.assertEqual(result["availability"], "live")
        self.assertEqual(result["data"]["event_count"], 1)
        self.assertIn("Silk Board", result["details"][0])

    def test_incidents_empty(self):
        result = get_incidents(FakeSources(incidents={"made_at": fresh(), "active_events": []}))
        self.assertEqual(result["availability"], "empty")
        self.assertIn("No congestion incidents", result["headline"])

    def test_incidents_missing(self):
        result = get_incidents(FakeSources(incidents=None))
        self.assertEqual(result["availability"], "missing")
        self.assertEqual(result["data"]["events"], [])

    def test_incidents_stale(self):
        sources = FakeSources(incidents={"made_at": old(30), "active_events": [INCIDENT_EVENT]})
        result = get_incidents(sources)
        self.assertEqual(result["availability"], "stale")
        self.assertIsNotNone(result["data"]["age_min"])

    def test_congestion_summary(self):
        sources = FakeSources(features={"made_at": fresh(), "areas": [FEATURE_AREA]})
        result = get_congestion_summary(sources)
        self.assertEqual(result["availability"], "live")
        self.assertEqual(result["data"]["overall_level"], "heavy")
        self.assertIn("Koramangala", result["details"][0])

    def test_congestion_missing_does_not_invent_areas(self):
        result = get_congestion_summary(FakeSources(features=None))
        self.assertEqual(result["availability"], "missing")
        self.assertEqual(result["data"]["areas"], [])

    def test_journeys_empty(self):
        result = get_journeys(FakeSources(journeys=[]))
        self.assertEqual(result["availability"], "empty")
        self.assertEqual(result["data"]["journey_count"], 0)

    def test_journey_found_and_not_found(self):
        sources = FakeSources(journeys=[JOURNEY])
        found = get_journey(sources, "jr-ff466f71c8")
        self.assertEqual(found["availability"], "live")
        self.assertEqual(found["data"]["journey"]["eta_min"], 22.0)
        missing = get_journey(sources, "jr-doesnotexist")
        self.assertEqual(missing["availability"], "missing")

    def test_route_recommendations_only_from_journey_agent(self):
        result = get_route_recommendations(FakeSources(journeys=[JOURNEY]))
        self.assertEqual(result["availability"], "live")
        self.assertEqual(len(result["data"]["recommendations"]), 1)
        self.assertEqual(result["data"]["recommendations"][0]["alternative"]["rank"], 1)

    def test_no_alternative_reported_when_none_exists(self):
        journey = dict(JOURNEY)
        journey["decision"] = dict(JOURNEY["decision"], action="hold", alternative=None)
        result = get_route_recommendations(FakeSources(journeys=[journey]))
        self.assertEqual(result["availability"], "live")
        self.assertEqual(result["data"]["recommendations"], [])
        self.assertEqual(result["data"]["decisions"][0]["action"], "hold")
        self.assertIn("No route switch", result["headline"])

    def test_route_recommendations_without_journeys(self):
        result = get_route_recommendations(FakeSources(journeys=[]))
        self.assertEqual(result["availability"], "empty")
        self.assertIn("not watching a journey", result["headline"])

    def test_system_status_ready(self):
        result = get_system_status(FakeSources(health=HEALTH_READY, published=("incidents.json",)))
        self.assertEqual(result["availability"], "live")
        self.assertTrue(result["data"]["ready"])
        self.assertIn("running", result["headline"])

    def test_system_status_warming_up(self):
        result = get_system_status(FakeSources(health=HEALTH_WARMING))
        self.assertEqual(result["availability"], "unavailable")
        self.assertFalse(result["data"]["ready"])
        self.assertIn("no forecast yet", result["headline"])

    def test_system_status_unreachable(self):
        result = get_system_status(FakeSources(health_raises=True))
        self.assertEqual(result["availability"], "unavailable")
        self.assertFalse(result["data"]["reachable"])

    def test_registry_specs_are_bedrock_shaped(self):
        registry = build_default_registry(FakeSources())
        self.assertEqual(registry.names(), ["get_system_status", "get_incidents",
                                            "get_congestion_summary", "get_journeys",
                                            "get_journey", "get_route_recommendations"])
        for spec in registry.specs():
            self.assertIn("toolSpec", spec)
            self.assertIn("inputSchema", spec["toolSpec"])
            self.assertIn("json", spec["toolSpec"]["inputSchema"])

    def test_registry_call_never_raises(self):
        registry = build_default_registry(FakeSources(journeys=[JOURNEY]))
        self.assertTrue(registry.call("get_incidents")["ok"])
        unknown = registry.call("no_such_tool")
        self.assertFalse(unknown["ok"])
        self.assertEqual(unknown["availability"], "unavailable")

    def test_registry_reports_bad_arguments_instead_of_crashing(self):
        registry = ToolRegistry().register("needs_id", "d", {"type": "object"},
                                           lambda journey_id: {"ok": True})
        result = registry.call("needs_id", {})
        self.assertFalse(result["ok"])
        self.assertIn("could not be called", result["headline"])

    def test_registry_reports_handler_failure_instead_of_crashing(self):
        def boom(**kwargs):
            raise ValueError("kaboom")
        registry = ToolRegistry().register("boom", "d", {"type": "object"}, boom)
        result = registry.call("boom")
        self.assertFalse(result["ok"])
        self.assertIn("failed", result["headline"])


class TestRealDataSourcesOffline(unittest.TestCase):
    """The real read layer, with a temp dir and a stubbed health fetcher."""

    def test_reads_published_files_and_uses_injected_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "incidents.json"), "w") as handle:
                json.dump({"made_at": fresh(), "active_events": [INCIDENT_EVENT]}, handle)
            sources = LiveDataSources(live_dir=tmp, health_fetcher=lambda: HEALTH_READY)
            self.assertEqual(get_incidents(sources)["availability"], "live")
            self.assertTrue(get_system_status(sources)["data"]["reachable"])

    def test_missing_file_is_missing_not_an_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = LiveDataSources(live_dir=tmp, health_fetcher=lambda: None)
            self.assertEqual(get_incidents(sources)["availability"], "missing")

    def test_unreadable_file_is_treated_as_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "features.json"), "w") as handle:
                handle.write("{not json")
            sources = LiveDataSources(live_dir=tmp)
            self.assertEqual(get_congestion_summary(sources)["availability"], "unavailable")

    def test_health_fetcher_failure_becomes_unavailable(self):
        def boom():
            raise ConnectionError("connection refused")
        sources = LiveDataSources(live_dir=tempfile.gettempdir(), health_fetcher=boom)
        self.assertIsNone(sources.system_health())
        self.assertEqual(get_system_status(sources)["availability"], "unavailable")

    def test_from_config_uses_config_values(self):
        config = InteractionConfig.from_env({"MODEL_SERVICE_URL": "http://127.0.0.1:9999/"})
        sources = LiveDataSources.from_config(config)
        self.assertEqual(sources.model_service_url, "http://127.0.0.1:9999")


# --------------------------------------------------------------------------- 4. orchestration

class TestAgentOrchestration(unittest.TestCase):
    def test_tool_call_then_answer(self):
        provider = ScriptedProvider([
            LLMResponse(tool_calls=[ToolCall("get_incidents", {}, "call-1")]),
            LLMResponse(text="One congestion event is open near Silk Board."),
        ])
        registry = build_default_registry(FakeSources(incidents={"made_at": fresh(),
                                                                "active_events": [INCIDENT_EVENT]}))
        agent = InteractionAgent(provider=provider, registry=registry)
        reply = agent.answer("Are there any traffic incidents?")
        self.assertTrue(reply.ok)
        self.assertEqual(reply.steps, 2)
        self.assertEqual([call.name for call in reply.tool_calls],
                         ["get_incidents", "get_journeys"])
        self.assertEqual(reply.tool_calls[0].name, "get_incidents")
        self.assertEqual(reply.tool_calls[0].availability, "live")
        self.assertIn("Silk Board", reply.text)

    def test_tool_result_is_fed_back_to_the_provider(self):
        provider = ScriptedProvider([LLMResponse(tool_calls=[ToolCall("get_incidents")]),
                                     LLMResponse(text="done")])
        agent = InteractionAgent(provider=provider,
                                 registry=build_default_registry(FakeSources(incidents=None)))
        agent.answer("incidents?")
        second_call = provider.calls[1]["messages"]
        self.assertEqual(second_call[-1]["role"], "tool")
        self.assertEqual(second_call[-1]["result"]["availability"], "missing")

    def test_repeated_tool_calls_are_bounded(self):
        responses = [LLMResponse(tool_calls=[ToolCall("get_incidents")]) for _ in range(10)]
        agent = InteractionAgent(provider=ScriptedProvider(responses),
                                 registry=build_default_registry(FakeSources(incidents=None)))
        reply = agent.answer("incidents?")
        self.assertEqual(reply.steps, MAX_STEPS)
        self.assertEqual(reply.text, STEP_LIMIT_REPLY)

    def test_provider_failure_is_graceful(self):
        agent = InteractionAgent(provider=ExplodingProvider(),
                                 registry=build_default_registry(FakeSources()))
        reply = agent.answer("Are there any traffic incidents?")
        self.assertFalse(reply.ok)
        self.assertEqual(reply.text, PROVIDER_ERROR_REPLY)
        self.assertIn("bedrock unreachable", reply.error)

    def test_empty_question_short_circuits(self):
        provider = ScriptedProvider([])
        agent = InteractionAgent(provider=provider, registry=build_default_registry(FakeSources()))
        reply = agent.answer("   ")
        self.assertEqual(reply.text, EMPTY_QUESTION_REPLY)
        self.assertEqual(provider.calls, [])

    def test_provider_with_neither_text_nor_tools_falls_back(self):
        agent = InteractionAgent(provider=ScriptedProvider([LLMResponse()]),
                                 registry=build_default_registry(FakeSources()))
        self.assertEqual(agent.answer("incidents?").text, NO_ANSWER_REPLY)

    def test_system_prompt_is_passed_to_provider(self):
        provider = ScriptedProvider([LLMResponse(text="ok")])
        agent = InteractionAgent(provider=provider, registry=build_default_registry(FakeSources()))
        agent.answer("congestion?")
        self.assertIn("read-only", provider.calls[0]["system"].lower())

    def test_unknown_tool_call_is_reported_not_raised(self):
        provider = ScriptedProvider([LLMResponse(tool_calls=[ToolCall("delete_everything")]),
                                     LLMResponse(text="I can't do that.")])
        agent = InteractionAgent(provider=provider, registry=build_default_registry(FakeSources()))
        reply = agent.answer("do something")
        self.assertFalse(reply.tool_calls[0].ok)
        self.assertEqual(reply.tool_calls[0].availability, "unavailable")


class TestAgentEndToEndMock(unittest.TestCase):
    """The real agent + real mock provider + real tools over fake data. No AWS."""

    def agent_for(self, sources):
        return build_agent_with_sources(sources)

    def test_incidents_question_reports_real_data(self):
        reply = self.agent_for(FakeSources(incidents={"made_at": fresh(),
                                                     "active_events": [INCIDENT_EVENT]})) \
            .answer("Are there any traffic incidents?")
        self.assertEqual(reply.provider, "mock")
        self.assertEqual(reply.tool_calls[0].name, "get_incidents")
        self.assertIn("Silk Board", reply.text)

    def test_incidents_question_never_fabricates(self):
        reply = self.agent_for(FakeSources(incidents={"made_at": fresh(), "active_events": []})) \
            .answer("Are there any traffic incidents?")
        self.assertIn("No congestion incidents", reply.text)

    def test_congestion_with_no_data_says_so(self):
        reply = self.agent_for(FakeSources(features=None)).answer("What is the current congestion status?")
        self.assertIn("not", reply.text.lower())

    def test_journey_question_reports_published_decision(self):
        reply = self.agent_for(FakeSources(journeys=[JOURNEY])).answer("What is the status of my journey?")
        self.assertIn("Home to Office", reply.text)
        self.assertIn("ETA 22.0 min", reply.text)

    def test_alternative_question_reports_journey_agents_recommendation(self):
        reply = self.agent_for(FakeSources(journeys=[JOURNEY])) \
            .answer("Are there any alternative route recommendations?")
        self.assertIn("save", reply.text.lower())

    def test_alternative_question_without_journeys_says_so(self):
        reply = self.agent_for(FakeSources(journeys=[])) \
            .answer("Are there any alternative route recommendations?")
        self.assertIn("not watching a journey", reply.text)

    def test_system_question_reports_readiness(self):
        reply = self.agent_for(FakeSources(health=HEALTH_READY)) \
            .answer("Is the traffic prediction system running?")
        self.assertIn("running", reply.text)

    def test_system_question_when_unreachable(self):
        reply = self.agent_for(FakeSources(health_raises=True)) \
            .answer("Is the traffic prediction system running?")
        self.assertIn("can't reach", reply.text)

    def test_default_agent_builds_with_mock_and_no_aws(self):
        agent = build_agent(InteractionConfig.from_env({}))
        self.assertEqual(agent.provider.name, "mock")
        reply = agent.answer("hello")
        self.assertTrue(reply.ok)
        self.assertFalse(reply.tool_calls)

    def test_reply_is_json_serializable(self):
        reply = self.agent_for(FakeSources(incidents={"made_at": fresh(),
                                                      "active_events": [INCIDENT_EVENT]})) \
            .answer("Are there any traffic incidents?")
        self.assertIn("text", json.loads(json.dumps(reply.as_dict())))


def build_agent_with_sources(sources):
    """An agent wired to the real mock provider and the real tools, over injected fake data."""
    return InteractionAgent(provider=MockLLMProvider(), registry=build_default_registry(sources))


if __name__ == "__main__":
    unittest.main(verbosity=2)
