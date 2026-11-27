"""Grounding regressions using the producer snapshot contracts."""
import os
import tempfile
import unittest
from datetime import timedelta

from webapp.interaction_agent import (ConversationMemory, InteractionAgent, InteractionConfig,
                                      LLMResponse, LiveDataSources, MockLLMProvider,
                                      build_default_registry)
from webapp.interaction_agent.agent import (_has_unsupported_claim, _tool_fallback,
                                            strip_thinking_tags)
from webapp.interaction_agent.tools import (get_incidents, get_journey, get_journeys,
                                            get_route_recommendations)
from webapp.journey_memory import iso, utc_now, write_json_atomic


JOURNEY_ID = "jr-a1b2c3d4e5"


def _journey_state(status="active", journey_id=JOURNEY_ID, incidents_ahead=0,
                   incident_ids=(), updated_at=None):
    now = updated_at or iso(utc_now())
    journey = {
        "journey_id": journey_id,
        "label": "Home to Office",
        "status": status,
        "updated_at": now,
        "total_length_km": 14.2,
        "baseline_eta_min": 25.7,
        "suggested": None,
        "switch_count": 0,
    }
    action = "arrived" if status == "arrived" else "hold"
    route_check = ([{"tool": "check_incidents_on_route", "ok": True,
                     "summary": "%s reported and 0 watched on the road ahead"
                     % incidents_ahead}] if status == "active" else [])
    decision = {
        "record_type": "journey_event",
        "schema_version": 1,
        "journey_id": journey_id,
        "status": status,
        "action": action,
        "updated_at": now,
        "eta_min": 25.7 if status == "active" else None,
        "remaining_eta_min": 17.2 if status == "active" else None,
        "baseline_eta_min": 25.7,
        "eta_delta_min": None,
        "achieved_total_min": 26.0 if status == "arrived" else None,
        "switch_count": 0,
        "incidents_ahead": incidents_ahead if status == "active" else None,
        "incident_ids": list(incident_ids),
        "tool_calls": route_check,
        "alternative": None,
    }
    return {"updated_at": now, "journeys": {journey_id: journey},
            "decisions": {journey_id: [decision]}}


class FixedProvider:
    name = "bedrock"

    def __init__(self, answer):
        self.answer_text = answer
        self.requests = []

    def converse(self, messages, tools, system="", context=None):
        self.requests.append({"messages": list(messages), "system": system, "context": context})
        return LLMResponse(text=self.answer_text)


class GroundingTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.live_dir = self.temp.name
        self.sources = LiveDataSources(
            live_dir=self.live_dir, stale_after_min=10.0,
            health_fetcher=lambda: {"ready": True, "mode": "pipeline_test",
                                    "horizons_min": [2, 14, 24], "warmup_steps": 1})

    def write(self, name, data):
        write_json_atomic(os.path.join(self.live_dir, name), data)

    def write_journeys(self, status="active"):
        self.write("journey_state.json", _journey_state(status))
        self.write("journey_switches.json", {"switch_count": 0, "switches": []})

    def write_incidents(self, events):
        now = iso(utc_now())
        self.write("incidents.json", {
            "record_type": "incident_snapshot", "schema_version": 1,
            "made_at": now, "timestamp": now, "active_events": events,
            "recent_resolved": [],
        })

    def agent(self, provider=None, memory=None):
        config = InteractionConfig.from_env({"INTERACTION_MEMORY_DB_PATH": ":memory:"})
        return InteractionAgent(config=config, provider=provider or MockLLMProvider(),
                                registry=build_default_registry(self.sources), memory=memory)


class TestProducerSnapshotContracts(GroundingTestCase):
    def test_journey_tools_preserve_recorded_rationale_and_route_evaluation(self):
        state = _journey_state()
        decision = state["decisions"][JOURNEY_ID][0]
        decision.update({
            "eta_min": 29.7,
            "eta_delta_min": 4.0,
            "rationale": "Your route still holds: 17.2 min left, about 4 min later than "
                         "when you chose this route.",
            "evidence": ["no_alternative_saves_enough_to_be_worth_it"],
            "tool_calls": [
                {"tool": "score_current_route", "ok": True,
                 "summary": "17.2 min left; due +4.0 min against the promise"},
                {"tool": "check_incidents_on_route", "ok": True,
                 "summary": "0 reported and 0 watched on the road ahead"},
                {"tool": "find_alternatives", "ok": True,
                 "summary": "3 searched, 0 usable"},
            ],
        })
        self.write("journey_state.json", state)
        self.write("journey_switches.json", {"switch_count": 0, "switches": []})

        journey = get_journeys(self.sources)["data"]["journeys"][0]
        self.assertEqual(journey["rationale"], decision["rationale"])
        self.assertEqual(journey["evidence"], decision["evidence"])
        self.assertEqual([item["tool"] for item in journey["evaluation"]],
                         ["score_current_route", "check_incidents_on_route",
                          "find_alternatives"])

        routes = get_route_recommendations(self.sources)
        self.assertEqual(routes["availability"], "live")
        route_decision = routes["data"]["decisions"][0]
        self.assertEqual(route_decision["rationale"], decision["rationale"])
        self.assertEqual(route_decision["eta_delta_min"], 4.0)
        self.assertEqual(route_decision["evaluation"][-1]["summary"], "3 searched, 0 usable")

    def test_grounding_accepts_delay_and_remaining_time_with_correct_metrics(self):
        self.write_journeys()
        state = _journey_state()
        decision = state["decisions"][JOURNEY_ID][0]
        decision.update({
            "eta_min": 29.7,
            "eta_delta_min": 4.0,
            "rationale": "Your route still holds: 17.2 min left, about 4 min later than "
                         "when you chose this route.",
        })
        self.write("journey_state.json", state)
        source = get_journeys(self.sources)
        answer = ("You're projected to arrive about 4 minutes later than promised, "
                  "with 17.2 minutes to go.")

        self.assertFalse(_has_unsupported_claim(answer, [source]))

    def test_grounding_rejection_logs_only_reason_and_tool_names(self):
        self.write_journeys()
        source = get_journeys(self.sources)
        with self.assertLogs("webapp.interaction_agent.agent", level="INFO") as logs:
            rejected = _has_unsupported_claim("The current ETA is 999 minutes.", [source])

        self.assertTrue(rejected)
        log_text = "\n".join(logs.output)
        self.assertIn("reason=unsupported_numeric_value", log_text)
        self.assertIn("get_journeys", log_text)
        self.assertNotIn("999", log_text)

    def test_grounded_fallback_uses_selected_source_not_all_tool_summaries(self):
        result = _tool_fallback([
            {"tool": "get_incidents", "ok": True, "availability": "live",
             "headline": "Three congestion events are open.",
             "details": ["First city-wide event.", "Second city-wide event."]},
            {"tool": "get_journeys", "ok": True, "availability": "stale",
             "headline": "Journey data is stale.",
             "details": ["A stale journey detail."]},
        ], ["get_incidents"])

        self.assertIn("Three congestion events are open.", result)
        self.assertIn("First city-wide event.", result)
        self.assertNotIn("Second city-wide event.", result)
        self.assertNotIn("stale", result.lower())

    def test_journey_fallback_uses_producer_rationale_instead_of_raw_summary(self):
        rationale = "Your route still holds: 17.2 min left, about 4 min later than promised."
        result = _tool_fallback([
            {"tool": "get_incidents", "ok": True, "availability": "live",
             "headline": "Six city-wide incidents are open."},
            {"tool": "get_journeys", "ok": True, "availability": "live",
             "data": {"journeys": [{
                 "journey_state": "active", "stale": False, "rationale": rationale,
             }]},
             "details": ["Raw journey summary with unrelated incident count."]},
        ], ["get_journeys"])

        self.assertEqual(result, rationale)

    def test_journey_fallback_does_not_repeat_unsupported_best_route_claim(self):
        result = _tool_fallback([{
            "tool": "get_journeys", "ok": True, "availability": "live",
            "data": {"journeys": [{
                "journey_state": "active", "stale": False,
                "rationale": "This route is the best way to the destination.",
            }]},
            "details": ["Home to Office: active, action hold"],
        }], ["get_journeys"])

        self.assertEqual(result, "Home to Office: active, action hold")
        self.assertNotIn("best", result.lower())

    def test_incident_snapshot_separates_observation_and_pipeline_test_forecast(self):
        now = iso(utc_now())
        event = {
            "event_id": "ce-J001-test",
            "event_type": "observed_slowdown",
            "status": "active",
            "severity": "moderate",
            "area": "Silk Board",
            "anchor_junction_id": "J001",
            "anchor_name": "Silk Board Junction",
            "affected_junction_ids": ["J001"],
            "affected_junction_count": 1,
            "avg_current_speed_kmh": 8.5,
            "avg_free_flow_speed_kmh": 45.0,
            "avg_predicted_speed_kmh": 7.0,
            "forecast_horizon_min": 15,
            "model_mode": "pipeline_test",
            "warmup_steps": 1,
            "forecast_used": True,
            "updated_at": now,
        }
        self.write("incidents.json", {
            "record_type": "incident_snapshot", "schema_version": 1, "poll_id": "poll-1",
            "timestamp": now, "made_at": now, "step_minutes": 2,
            "model_mode": "pipeline_test", "active_events": [event], "recent_resolved": [],
        })

        result = get_incidents(self.sources)
        self.assertEqual(result["availability"], "live")
        self.assertIn("observed TomTom speed", result["details"][0])
        self.assertIn("DCRNN forecast", result["details"][0])
        self.assertIn("pipeline-test", result["details"][0])
        self.assertEqual(result["data"]["events"][0]["warmup_steps"], 1)

    def test_completed_producer_journey_is_not_an_active_route_recommendation(self):
        self.write_journeys("arrived")

        result = get_journeys(self.sources)
        journey = result["data"]["journeys"][0]
        self.assertEqual(journey["journey_state"], "completed")
        self.assertEqual(journey["achieved_total_min"], 26.0)
        self.assertEqual(journey["baseline_eta_min"], 25.7)
        self.assertEqual(journey["switch_count"], 0)

        alternatives = get_route_recommendations(self.sources)
        self.assertEqual(alternatives["availability"], "empty")
        self.assertEqual(alternatives["data"]["recommendations"], [])
        self.assertEqual(alternatives["data"]["switch_count"], 0)

    def test_route_incident_and_eta_fields_are_preserved_with_distinct_scope(self):
        self.write_journeys()

        result = get_journeys(self.sources)
        journey = result["data"]["journeys"][0]

        self.assertEqual(journey["route_incident_state"], "current")
        self.assertEqual(journey["incidents_ahead"], 0)
        self.assertEqual(journey["incident_ids"], [])
        self.assertEqual(journey["total_eta_min"], 25.7)
        self.assertEqual(journey["remaining_eta_min"], 17.2)
        self.assertIn("route check: 0 reported", result["details"][0])

    def test_observed_speed_cannot_be_substituted_with_a_forecast_value(self):
        self.write("features.json", {
            "made_at": iso(utc_now()),
            "areas": [{
                "area": "Koramangala",
                "overall_congestion_level": "moderate",
                "avg_current_speed_kmh": 12.0,
                "avg_free_flow_speed_kmh": 40.0,
                "avg_predicted_speed_kmh": 10.0,
            }],
        })
        reply = self.agent(FixedProvider(
            "Observed TomTom speed in Koramangala is 10 km/h.")).answer(
                "What is the current speed in Koramangala?")

        self.assertNotIn("Observed TomTom speed in Koramangala is 10", reply.text)
        self.assertIn("observed TomTom speed 12.0 km/h", reply.text)
        self.assertIn("DCRNN forecast 10.0 km/h", reply.text)

    def test_system_readiness_claim_must_match_health_data(self):
        reply = self.agent(FixedProvider(
            "The prediction system is not ready.")).answer(
                "Is the prediction system running?")

        self.assertNotIn("not ready", reply.text.lower())
        self.assertIn("prediction system is running", reply.text.lower())

    def test_nonexistent_journey_id_is_not_presented_as_a_current_journey(self):
        self.write_journeys("arrived")

        result = get_journey(self.sources, "jr-deadbeef00")
        self.assertEqual(result["availability"], "missing")
        self.assertNotIn("journey", result["data"])
        self.assertEqual(result["data"]["requested_journey_id"], "jr-deadbeef00")
        self.assertEqual(result["data"]["snapshot_availability"], "live")

    def test_snapshot_without_a_valid_timestamp_is_unavailable_not_live(self):
        self.write("incidents.json", {"active_events": [], "recent_resolved": []})
        self.assertEqual(get_incidents(self.sources)["availability"], "unavailable")

    def test_missing_and_invalid_journey_snapshots_are_distinguished(self):
        self.assertEqual(get_journeys(self.sources)["availability"], "missing")
        self.write("journey_state.json", {"journeys": []})
        self.assertEqual(get_journeys(self.sources)["availability"], "unavailable")


class TestLiveGrounding(GroundingTestCase):
    def test_current_speed_question_fetches_a_fresh_traffic_snapshot(self):
        self.write("features.json", {"made_at": iso(utc_now()), "areas": []})
        reply = self.agent(FixedProvider("I've checked the latest traffic snapshot.")).answer(
            "What's the current speed?")

        self.assertEqual([call.name for call in reply.tool_calls], ["get_congestion_summary"])

    def test_each_request_reads_the_latest_producer_snapshot(self):
        self.write_journeys("active")
        agent = self.agent()

        active = agent.answer("What is the status of my journey?")
        self.assertIn("active", active.text.lower())

        self.write_journeys("arrived")
        arrived = agent.answer("What is the status of my journey?")
        self.assertIn("completed", arrived.text.lower())
        self.assertIn("26.0 min", arrived.text)
        self.assertNotIn("active", arrived.text.lower())

    def test_model_cannot_invent_journeys_delays_or_alternatives(self):
        self.write_journeys("arrived")
        provider = FixedProvider(
            "Two journeys are active. Journey jr-ffffffff is 10 minutes late; "
            "take the Outer Ring Road to save 8 minutes.")
        reply = self.agent(provider).answer("What is the status of my journey?")

        self.assertEqual([call.name for call in reply.tool_calls], ["get_journeys"])
        self.assertNotIn("jr-ffffffff", reply.text)
        self.assertNotIn("10 minutes", reply.text)
        self.assertNotIn("Outer Ring Road", reply.text)
        self.assertIn("completed", reply.text.lower())

    def test_stale_history_cannot_override_the_fresh_journey_snapshot(self):
        self.write_journeys("active")
        memory = ConversationMemory(db_path=":memory:", enabled=True)
        self.addCleanup(memory.close)
        session_id = memory.create_session()
        memory.add_message(session_id, "user", "What was my journey status?")
        memory.add_message(session_id, "assistant",
                           "There are two journeys; jr-deadbeef00 is 10 minutes late.")
        memory.update_context(session_id, last_intent="journey",
                              journey_ids=["jr-deadbeef00"])
        provider = FixedProvider("There are two journeys; jr-deadbeef00 is 10 minutes late.")
        reply = self.agent(provider, memory).answer(
            "What is the status of my journey?", session_id=session_id)

        self.assertEqual([call.name for call in reply.tool_calls], ["get_journeys"])
        self.assertNotIn("jr-deadbeef00", reply.text)
        self.assertNotIn("10 minutes", reply.text)
        self.assertIn("active", reply.text.lower())

    def test_city_incidents_cannot_be_misreported_as_incidents_ahead(self):
        self.write_incidents([
            {"event_id": "evt-city-1", "event_type": "congestion", "status": "active",
             "severity": "heavy", "area": "Indiranagar"},
            {"event_id": "evt-city-2", "event_type": "congestion", "status": "active",
             "severity": "moderate", "area": "Jayanagar"},
        ])
        self.write_journeys()
        provider = FixedProvider(
            "Yes, there are two high-severity traffic incidents ahead. "
            "One is in Indiranagar and one is in Jayanagar.")

        reply = self.agent(provider).answer("Any traffic incidents aahead of me?")

        self.assertEqual([call.name for call in reply.tool_calls],
                         ["get_incidents", "get_journeys"])
        self.assertIn("reports no active incidents ahead", reply.text)
        self.assertNotIn("Indiranagar", reply.text)
        self.assertNotIn("Jayanagar", reply.text)

    def test_route_incident_locations_require_a_matching_route_event_id(self):
        event = {"event_id": "evt-route", "event_type": "congestion", "status": "active",
                 "severity": "heavy", "area": "Silk Board"}
        self.write_incidents([event])
        self.write("journey_state.json", _journey_state(
            incidents_ahead=1, incident_ids=["evt-route"]))
        self.write("journey_switches.json", {"switch_count": 0, "switches": []})
        answer = "One heavy incident is reported ahead on your route near Silk Board."

        reply = self.agent(FixedProvider(answer)).answer(
            "Could you check whether my route has a reported incident?")

        self.assertEqual(reply.text, answer)

    def test_contextual_followup_uses_fresh_route_evidence_not_old_history(self):
        self.write_incidents([
            {"event_id": "evt-city", "event_type": "congestion", "status": "active",
             "severity": "heavy", "area": "Indiranagar"},
        ])
        self.write_journeys()
        memory = ConversationMemory(db_path=":memory:", enabled=True)
        self.addCleanup(memory.close)
        session_id = memory.create_session()
        memory.add_message(session_id, "user", "Are there incidents ahead?")
        memory.add_message(session_id, "assistant",
                           "Two high-severity incidents are ahead in Indiranagar.")
        memory.update_context(session_id, last_intent="incidents", journey_ids=[JOURNEY_ID])

        reply = self.agent(FixedProvider(
            "Two high-severity incidents are ahead in Indiranagar."), memory).answer(
                "What about up ahead now?", session_id=session_id)

        self.assertEqual([call.name for call in reply.tool_calls],
                         ["get_incidents", "get_journeys"])
        self.assertIn("no active incidents ahead", reply.text)
        self.assertNotIn("Indiranagar", reply.text)

    def test_multiple_active_journeys_trigger_route_clarification(self):
        first = _journey_state()
        second = _journey_state(journey_id="jr-abcdef0123")
        first["journeys"].update(second["journeys"])
        first["decisions"].update(second["decisions"])
        self.write("journey_state.json", first)
        self.write("journey_switches.json", {"switch_count": 0, "switches": []})
        self.write_incidents([])

        reply = self.agent(FixedProvider("There are no incidents ahead on your route.")).answer(
            "Are there any incidents ahead?")

        self.assertIn("multiple active journeys", reply.text.lower())
        self.assertIn("which journey", reply.text.lower())

    def test_stale_route_incident_claim_uses_one_stale_warning(self):
        old_stamp = iso(utc_now() - timedelta(minutes=45))
        self.write("journey_state.json", _journey_state(updated_at=old_stamp))
        self.write("journey_switches.json", {"switch_count": 0, "switches": []})
        self.write_incidents([])
        provider = FixedProvider(
            "The journey data is stale. There is one incident ahead. The data is stale.")

        reply = self.agent(provider).answer("Any incidents ahead?")

        self.assertEqual(reply.text.count("stale"), 1)
        self.assertIn("can't confirm current incidents ahead", reply.text)

    def test_hold_decision_does_not_support_a_fastest_route_claim(self):
        self.write_journeys()

        for claim in ("Your current route is definitely the fastest.",
                      "Your current route is the best way to go."):
            with self.subTest(claim=claim):
                reply = self.agent(FixedProvider(claim)).answer("How is my route doing?")

                self.assertNotIn("fastest", reply.text.lower())
                self.assertNotIn("best way", reply.text.lower())
                self.assertIn("action hold", reply.text.lower())

    def test_total_and_remaining_eta_cannot_be_swapped(self):
        self.write_journeys()

        reply = self.agent(FixedProvider("The remaining ETA is 25.7 minutes.")).answer(
            "How long is left on my journey?")

        self.assertNotIn("remaining ETA is 25.7", reply.text)
        self.assertIn("total ETA 25.7 min", reply.text)
        self.assertIn("remaining ETA 17.2 min", reply.text)

    def test_route_incident_snapshot_is_refreshed_between_requests(self):
        self.write_incidents([])
        self.write_journeys()
        agent = self.agent(FixedProvider("There are no active incidents ahead."))

        first = agent.answer("Are there any traffic incidents ahead?")
        self.write("journey_state.json", _journey_state(
            incidents_ahead=1, incident_ids=["evt-new"]))
        second = agent.answer("Are there any traffic incidents ahead?")

        self.assertIn("no active incidents ahead", first.text)
        self.assertIn("reports 1 active incident ahead", second.text)

    def test_malformed_thinking_tags_are_removed_from_reply_and_memory(self):
        self.write_journeys("arrived")
        memory = ConversationMemory(db_path=":memory:", enabled=True)
        self.addCleanup(memory.close)
        provider = FixedProvider("Your journey is completed."
                                 "<ThInKiNg private>hidden reasoning about 999 minutes")
        agent = self.agent(provider, memory)
        session_id = agent.new_session()
        reply = agent.answer("What is the status of my journey?", session_id=session_id)

        self.assertEqual(reply.text, "Your journey is completed.")
        self.assertNotIn("thinking", memory.history(session_id)[-1]["content"].lower())
        self.assertNotIn("hidden reasoning", memory.history(session_id)[-1]["content"])
        self.assertEqual(strip_thinking_tags("Answer</thinking>"), "Answer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
