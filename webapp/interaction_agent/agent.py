"""The User Interaction Agent: turn a traveller's question into a grounded answer.

One turn is a small tool-calling loop, the same shape an Amazon Bedrock agent would use:

    user question
      -> provider.converse(messages, tool specs, system prompt)
      -> either a final answer, or one or more tool calls
      -> run each tool through the registry (they are read-only and never raise)
      -> feed the results back to the provider
      -> provider returns the answer, grounded in those results

The loop is bounded by :data:`MAX_STEPS` so a misbehaving provider cannot spin forever. Nothing is
invented at any point: a tool reports what the other two agents published, or reports that the data
is missing, empty, stale or unavailable, and the agent repeats exactly that. When the provider
itself cannot be reached, the agent says so instead of answering from nothing.

Conversation memory (optional, Phase 2)
---------------------------------------
``answer(message, session_id=...)`` makes the turn part of a conversation: earlier user/assistant
text is loaded and sent to the provider as context, and the new question and answer are stored.
Only text is stored -- never tool results -- so every turn runs its tools fresh and a turn can never
be answered from stale evidence. Follow-up questions like "what about the delay?" are resolved from
session context (the last topic, the journeys and places already discussed).

With no ``session_id`` the agent behaves exactly as it did before memory existed: a stateless,
single turn that touches no database. Memory failures degrade to that same stateless behaviour.
There is still no proactive notification -- that is a later phase.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from webapp.interaction_agent.config import InteractionConfig
from webapp.interaction_agent.llm import (BedrockConverseProvider, INTENT_TOOL, JOURNEY_ID_RE,
                                          TOOL_INTENT, ToolCall, build_llm_provider,
                                          detect_followup, detect_intent,
                                          is_contextual_followup)
from webapp.interaction_agent.memory import ConversationMemory
from webapp.interaction_agent.prompts import build_system_prompt
from webapp.interaction_agent.tools import build_default_registry

LOG = logging.getLogger(__name__)

# A grounded turn needs at most: understand -> call a tool -> answer. A small ceiling absorbs a
# provider that chains a second call without letting a broken one loop forever.
MAX_STEPS = 4

EMPTY_QUESTION_REPLY = "Please ask me something about traffic, your journey or the prediction system."

PROVIDER_ERROR_REPLY = ("I can't reach the language model right now, so I can't answer that. "
                        "The underlying traffic data is unaffected.")

NO_ANSWER_REPLY = "I don't have an answer for that right now."

STEP_LIMIT_REPLY = ("I couldn't finish looking that up. Please try rephrasing the question.")
GROUNDING_RETRY_INSTRUCTION = (
    "Your draft included current traffic or journey claims that are not supported by the tool "
    "results in this turn. Use the relevant available read-only tool or tools to verify them. "
    "If the data is missing, stale, unavailable, or insufficient, explain that limitation "
    "without making the unsupported claim."
)

_THINKING_BLOCK_RE = re.compile(r"<thinking\b[^>]*>.*?(?:</thinking\s*>|$)", re.IGNORECASE | re.DOTALL)
_THINKING_CLOSE_RE = re.compile(r"</thinking\s*>", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?(?![\w.])")
_NUMBER_WORDS = {
    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0, "fifteen": 15.0,
    "sixteen": 16.0, "seventeen": 17.0, "eighteen": 18.0, "nineteen": 19.0, "twenty": 20.0,
}
_ALTERNATIVE_CLAIM_RE = re.compile(
    r"\b(?:alternative route|alternate route|another route|different route|faster route|"
    r"shorter route|route recommendation|route change|recommend(?:ed|ation)?\s+(?:an?\s+)?"
    r"route|rerout(?:e|ing)|detour|switch(?:ed)?\s+to|save(?:s|d)?\s+\d+)\b",
    re.IGNORECASE)
_FASTEST_CLAIM_RE = re.compile(
    r"\b(?:route|way|option)\b.{0,50}\b(?:fastest|quickest|shortest|best|optimal)\b|"
    r"\b(?:fastest|quickest|shortest|best|optimal)\s+(?:route|way|option)\b",
    re.IGNORECASE)
_ROUTE_INCIDENT_CLAIM_RE = re.compile(
    r"\b(?:incidents?|accidents?|jams?|closures?|hazards?|congestion(?:\s+events?)?|"
    r"traffic\s+conditions?)\b.{0,90}\b(?:ahead|in\s+front|on\s+(?:the\s+)?"
    r"(?:road|route|way)|affecting\s+(?:your\s+)?(?:route|journey)|"
    r"along\s+(?:your\s+)?(?:route|way))\b|"
    r"\b(?:ahead|in\s+front|on\s+(?:the\s+)?(?:road|route|way)|"
    r"affecting\s+(?:your\s+)?(?:route|journey)|along\s+(?:your\s+)?"
    r"(?:route|way))\b.{0,90}\b(?:incidents?|accidents?|jams?|closures?|"
    r"hazards?|congestion(?:\s+events?)?|traffic\s+conditions?)\b",
    re.IGNORECASE)
_CITY_INCIDENT_CLAIM_RE = re.compile(
    r"\b(?:there\s+(?:are|is)|i\s+(?:see|found|can\s+confirm)|"
    r"(?:no|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b|"
    r"(?:reported|open|active|current|resolved))\b.{0,70}\b"
    r"(?:incidents?|accidents?|jams?|closures?|hazards?|congestion\s+events?)\b|"
    r"\b(?:incidents?|accidents?|jams?|closures?|hazards?|congestion\s+events?)\b.{0,55}"
    r"\b(?:are|is|remain|reported|located|open|active|resolved)\b",
    re.IGNORECASE)
_SYSTEM_STATUS_CLAIM_RE = re.compile(
    r"\b(?:prediction system|system|model service|forecast service|the model|the service)\b"
    r".{0,70}\b(?:running|up|ready|healthy|reachable|available|down|unavailable)\b|"
    r"\b(?:running|up|ready|healthy|reachable|available|down|unavailable)\b.{0,70}"
    r"\b(?:prediction system|system|model service|forecast service|the model|the service)\b",
    re.IGNORECASE)
_ROUTE_NAME_RE = re.compile(
    r"\b(?:via|through|along|take|use)\s+(?:the\s+)?"
    r"([a-z][a-z0-9'-]*(?:\s+[a-z0-9'-]+){0,4}\s+"
    r"(?:road|street|highway|flyover|bypass|expressway|junction))\b",
    re.IGNORECASE)
_SOURCE_TOOLS = set(INTENT_TOOL.values()) | {"get_journey"}


def strip_thinking_tags(text: str) -> str:
    """Remove Nova reasoning blocks, including an opening tag with no closing tag."""
    cleaned = _THINKING_BLOCK_RE.sub("", str(text or ""))
    return _THINKING_CLOSE_RE.sub("", cleaned).strip()


def _required_tool_call(question, context):
    """Resolve the live source to read before the model can answer a live-data question."""
    match = JOURNEY_ID_RE.search(question or "")
    if match:
        return "get_journey", {"journey_id": match.group(1)}
    followup_intent = detect_followup(question, context)
    direct_intent = detect_intent(question)
    if direct_intent and direct_intent != "journey":
        return INTENT_TOOL[direct_intent], {}
    if direct_intent == "journey" and not is_contextual_followup(question):
        return "get_journeys", {}

    intent = followup_intent or direct_intent
    journey_ids = (context or {}).get("journey_ids") or []
    mentions_journey = bool(re.search(r"\b(?:journey|trip|route)\b", question,
                                      re.IGNORECASE))
    if intent == "journey":
        if len(journey_ids) == 1:
            return "get_journey", {"journey_id": journey_ids[0]}
        if journey_ids or mentions_journey:
            return "get_journeys", {}
        return None
    tool = INTENT_TOOL.get(intent)
    return (tool, {}) if tool else None


def _required_tool_calls(question, context):
    """Fetch complementary evidence for incident questions before the model answers."""
    required = _required_tool_call(question, context)
    if not required:
        return []
    calls = [required]
    if required[0] == "get_incidents":
        calls.append(("get_journeys", {}))
    return calls


def _tool_fallback(results, preferred_tools=None):
    """Return a concise fallback from the tool(s) selected for this question."""
    preferred_tools = list(dict.fromkeys(preferred_tools or []))
    selected = []
    for name in preferred_tools:
        result = next((item for item in reversed(results) if item.get("tool") == name), None)
        if result is not None:
            selected.append(result)
    if not selected and results:
        selected = [results[-1]]
    lines = [_tool_result_fallback(result) for result in selected]
    return "\n".join(line for line in lines if line).strip() or NO_ANSWER_REPLY


def _tool_result_fallback(result):
    if not result.get("ok", True):
        return "I couldn't retrieve that information right now."

    tool = result.get("tool")
    availability = result.get("availability")
    data = result.get("data") or {}
    if availability == "stale":
        if tool in ("get_journey", "get_journeys", "get_route_recommendations"):
            return "The latest journey decision is stale, so I can't confirm current journey details."
        return "That source is stale, so I can't confirm current conditions."
    if availability == "missing":
        if (data.get("requested_journey_id")
                and data.get("snapshot_availability") not in ("missing", "unavailable")):
            return "That journey isn't present in the current snapshot."
        return "That data has not been published yet."
    if availability == "unavailable":
        return str(result.get("headline") or "That source is unavailable right now.")

    if tool in ("get_journey", "get_journeys", "get_route_recommendations"):
        if tool == "get_route_recommendations":
            journeys = data.get("decisions") or []
        elif isinstance(data.get("journey"), dict):
            journeys = [data["journey"]]
        else:
            journeys = data.get("journeys") or []

        current = [item for item in journeys
                   if isinstance(item, dict) and item.get("journey_state") == "active"
                   and not item.get("stale")]
        if len(current) > 1:
            return ("I found multiple active journeys, so I can't tell which one you mean. "
                    "Which journey are you asking about?")
        if len(current) == 1:
            rationale = current[0].get("rationale")
            if rationale and not _FASTEST_CLAIM_RE.search(str(rationale)):
                return str(rationale)
        active = [item for item in journeys
                  if isinstance(item, dict) and item.get("journey_state") == "active"]
        if active:
            if any(not item.get("stale") for item in active):
                details = result.get("details") or []
                if details:
                    return str(details[0])
                return str(result.get("headline") or "Your journey is being watched.")
            return "The latest journey decision is stale, so I can't confirm current journey details."
        completed = [item for item in journeys
                     if isinstance(item, dict) and item.get("journey_state") == "completed"]
        if completed:
            rationale = completed[0].get("rationale")
            if rationale and not _FASTEST_CLAIM_RE.search(str(rationale)):
                return str(rationale)
            details = result.get("details") or []
            if details:
                return str(details[0])
            return "That journey is completed; there is no current route recommendation."
        if availability == "empty":
            return str(result.get("headline") or "No journey is being watched right now.")

    headline = str(result.get("headline") or "").strip()
    if availability == "empty" or tool == "get_system_status" or not headline:
        return headline or NO_ANSWER_REPLY
    details = result.get("details") or []
    if tool in ("get_incidents", "get_congestion_summary") and details:
        return headline + " " + str(details[0])
    return headline


def _result_evidence(results):
    """Return evidence strings, numbers, journey ids, statuses and recommendations for a turn."""
    evidence_parts = []
    numbers = set()
    typed_numbers = {"delay": set(), "eta": set(), "remaining_eta": set(),
                     "saving": set(), "distance": set(),
                     "speed": set(), "progress": set(), "horizon": set(),
                     "observed_speed": set(), "forecast_speed": set(),
                     "free_flow_speed": set(),
                     "journey_count": set(), "switch_count": set(),
                     "city_incident_count": set(), "active_incident_count": set(),
                     "route_incident_count": set()}
    journey_ids = set()
    statuses = set()
    has_recommendation = False
    has_journey_result = False

    def collect_numbers(value, key=""):
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            numbers.add(float(value))
            metric = {
                "eta_delta_min": "delay", "delay_min": "delay",
                "eta_min": "eta", "total_eta_min": "eta", "baseline_eta_min": "eta",
                "remaining_eta_min": "remaining_eta",
                "achieved_total_min": "eta", "gain_min": "saving", "gain_pct": "saving",
                "progress_pct": "progress", "forecast_horizon_min": "horizon",
                "journey_count": "journey_count", "switch_count": "switch_count",
                "event_count": "city_incident_count",
                "active_count": "active_incident_count",
                "incidents_ahead": "route_incident_count",
                "total_length_km": "distance", "remaining_km": "distance",
                "turn_off_after_km": "distance",
                "avg_current_speed_kmh": "observed_speed",
                "avg_free_flow_speed_kmh": "free_flow_speed",
                "avg_predicted_speed_kmh": "forecast_speed",
            }.get(key)
            if metric:
                typed_numbers[metric].add(float(value))
                if metric in ("observed_speed", "forecast_speed", "free_flow_speed"):
                    typed_numbers["speed"].add(float(value))
        elif isinstance(value, dict):
            for item_key, item in value.items():
                collect_numbers(item, str(item_key))
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect_numbers(item, key)
        elif isinstance(value, str):
            for match in _NUMBER_RE.finditer(value):
                try:
                    number = float(match.group(0).rstrip("%"))
                    numbers.add(number)
                    if key == "rationale":
                        metric = _numeric_metric(value, match.start(), match.end())
                        if metric:
                            typed_numbers[metric].add(number)
                except ValueError:
                    continue

    for result in results:
        has_journey_result = has_journey_result or result.get("tool") in (
            "get_journey", "get_journeys", "get_route_recommendations")
        data = result.get("data") or {}
        evidence_parts.extend([str(result.get("headline") or "")])
        evidence_parts.extend(str(item) for item in (result.get("details") or []))
        evidence_data = data
        if result.get("tool") in ("get_journey", "get_journeys"):
            journey_items = ([data["journey"]] if isinstance(data.get("journey"), dict) else
                             data.get("journeys") or [])
            fresh_items = [item for item in journey_items
                           if isinstance(item, dict) and not item.get("stale")]
            if "journey" in data:
                evidence_data = {"journey": fresh_items[0]} if fresh_items else {}
            elif "journeys" in data:
                evidence_data = {"journey_count": data.get("journey_count", 0),
                                 "journeys": fresh_items}
            else:
                evidence_data = data
        elif result.get("tool") == "get_route_recommendations":
            evidence_data = dict(data)
            evidence_data["decisions"] = [
                item for item in data.get("decisions") or []
                if isinstance(item, dict) and not item.get("stale")
            ]
        evidence_parts.append(str(evidence_data))
        collect_numbers(evidence_data)
        for key in ("journey",):
            item = evidence_data.get(key)
            if isinstance(item, dict):
                if item.get("journey_id"):
                    journey_ids.add(str(item["journey_id"]))
                if item.get("journey_state"):
                    statuses.add(str(item["journey_state"]).lower())
        for key in ("journeys", "recommendations", "decisions"):
            for item in evidence_data.get(key) or []:
                if not isinstance(item, dict):
                    continue
                if item.get("journey_id"):
                    journey_ids.add(str(item["journey_id"]))
                if item.get("journey_state"):
                    statuses.add(str(item["journey_state"]).lower())
                if isinstance(item.get("alternative"), dict):
                    has_recommendation = True
        if isinstance(evidence_data.get("recommendations"), list) \
                and evidence_data["recommendations"]:
            has_recommendation = any(
                isinstance(item, dict) and isinstance(item.get("alternative"), dict)
                for item in evidence_data["recommendations"])
    evidence = "\n".join(evidence_parts).lower()
    for match in _NUMBER_RE.finditer(evidence):
        try:
            numbers.add(float(match.group(0).rstrip("%")))
        except ValueError:
            continue
    return (evidence, numbers, typed_numbers, journey_ids, statuses, has_recommendation,
            has_journey_result)


def _has_positive_claim(text, terms):
    for match in re.finditer(r"\b(?:%s)\b" % "|".join(terms), text, re.IGNORECASE):
        prefix = text[max(0, match.start() - 24):match.start()]
        if not re.search(r"\b(?:no|not|isn't|aren't|never|without)\s+(?:any\s+)?$", prefix,
                         re.IGNORECASE):
            return True
    return False


def _numeric_metric(text, number_start, number_end):
    before = text[:number_start].lower()
    after = text[number_end:].lower()
    nearby = before[-24:] + " " + after[:24]
    number_text = text[number_start:number_end]
    if re.match(r"\s*(?:km/h|mph)\b", after):
        speed_labels = (
            ("observed_speed", r"\b(?:observed|current|tomtom)\b"),
            ("forecast_speed", r"\b(?:forecast|predicted|dcrnn)\b"),
            ("free_flow_speed", r"\bfree[- ]flow\b"),
        )
        preceding_labels = [
            (match.start(), metric)
            for metric, pattern in speed_labels
            for match in re.finditer(pattern, before[-80:])
        ]
        if preceding_labels:
            return max(preceding_labels)[1]
        following_labels = [
            (match.start(), metric)
            for metric, pattern in speed_labels
            for match in re.finditer(pattern, after[:32])
        ]
        if following_labels:
            return min(following_labels)[1]
        return "speed"
    if re.match(r"\s*(?:km|kilomet(?:er|re)s?|miles?)\b", after):
        return "distance"
    if (re.search(r"\b(?:delay|delayed|late|behind|later|earlier)\b", nearby)
            or re.search(r"\b(?:after|before)\b.{0,24}"
                         r"\b(?:baseline|promise|expected|schedule)\b", nearby)):
        return "delay"
    if re.search(r"\b(?:saving|save|saved|gain)\b", nearby):
        return "saving"
    if number_text.endswith("%") or re.match(r"\s*%", after):
        return "progress"
    if re.search(r"\b(?:incident|accident|jam|closure|hazard)s?\b", nearby):
        if re.search(r"\b(?:ahead|route|journey|path|in\s+front|on\s+the\s+road)\b",
                     nearby):
            return "route_incident_count"
        if re.search(r"\bactive\b", nearby):
            return "active_incident_count"
        return "city_incident_count"
    if re.search(r"\bcongestion\s+events?\b", nearby):
        if re.search(r"\b(?:ahead|route|journey|path|in\s+front|on\s+the\s+road)\b",
                     nearby):
            return "route_incident_count"
        return "city_incident_count"
    if (re.search(r"\b(?:remaining|left)\s+(?:travel\s+)?"
                  r"(?:eta|arrival(?:\s+time)?|time)\s*(?:(?:is|of|:)\s*)?$",
                  before[-40:])
            or re.search(r"\b(?:eta|arrival(?:\s+time)?|time)\s+"
                         r"(?:remaining|left)\s*$", before[-32:])
            or re.search(r"\b(?:remaining|left)\s*(?:(?:is|of|:)\s*)?$", before[-24:])
            or re.match(r"\s*min(?:ute)?s?\s+(?:left|remaining|to\s+go)\b", after)):
        return "remaining_eta"
    if re.search(r"\b(?:speed|km/h)\b", nearby):
        if re.search(r"\b(?:observed|current|tomtom)\b", nearby):
            return "observed_speed"
        if re.search(r"\b(?:forecast|predicted|dcrnn)\b", nearby):
            return "forecast_speed"
        if re.search(r"\bfree[- ]flow\b", nearby):
            return "free_flow_speed"
        return "speed"
    if re.search(r"\bprogress\b", nearby):
        return "progress"
    if re.search(r"\b(?:forecast|horizon)\b", nearby):
        return "horizon"
    if (re.search(r"\b(?:eta|arrival|arrive|travel\s+time)\b", before[-22:])
            or re.match(r"\s*min(?:ute)?s?\b", after)):
        return "eta"
    if re.search(r"\bjourneys?\b", nearby):
        return "journey_count"
    if re.search(r"\bswitch(?:es)?\b", nearby):
        return "switch_count"
    return None


def _contains_specific_live_claim(text):
    if JOURNEY_ID_RE.search(text) or _NUMBER_RE.search(text):
        return True
    for word in _NUMBER_WORDS:
        match = re.search(r"\b%s\b" % word, text, re.IGNORECASE)
        if match and _numeric_metric(text, match.start(), match.end()):
            return True
    if re.search(r"\b(?:no|zero|none)\s+(?:(?:active|current|traffic|alternative)\s+)*"
                 r"(?:incidents?|journeys?|routes?|switch(?:es)?)\b", text, re.IGNORECASE):
        return True
    if (_ALTERNATIVE_CLAIM_RE.search(text) or _FASTEST_CLAIM_RE.search(text)
            or _ROUTE_INCIDENT_CLAIM_RE.search(text) or _ROUTE_NAME_RE.search(text)):
        return True
    if _CITY_INCIDENT_CLAIM_RE.search(text):
        return True
    if _SYSTEM_STATUS_CLAIM_RE.search(text):
        return True
    if any(_has_positive_claim(text, (term,)) for term in
           ("active", "ongoing", "completed", "complete", "arrived", "finished", "expired")):
        return True
    return any(re.search(r"\b%s\b" % re.escape(term), text, re.IGNORECASE)
               for term in ("heavy", "moderate", "free-flow", "congested", "blocked",
                            "accident", "slowdown", "slow", "smooth"))


def _has_unsupported_claim(text, results):
    def reject(reason):
        source_names = sorted({
            str(result.get("tool") or "unknown") for result in results
            if result.get("tool") in _SOURCE_TOOLS
        })
        LOG.info("Grounding rejected provider response: reason=%s tools=%s",
                 reason, source_names)
        return True

    source_results = [result for result in results if result.get("tool") in _SOURCE_TOOLS]
    if not source_results:
        return (reject("live_claim_without_source")
                if _contains_specific_live_claim(text) else False)
    unavailable_results = [
        result for result in source_results
        if not result.get("ok", True)
        or result.get("availability") in ("missing", "stale", "unavailable")
    ]
    if any(_claims_current_source_data(text, result.get("tool"))
           for result in unavailable_results):
        return reject("current_claim_from_unavailable_source")
    available_results = [
        result for result in source_results
        if result.get("ok", True)
        and result.get("availability") not in ("missing", "stale", "unavailable")
    ]
    if not available_results:
        return (reject("live_claim_without_available_source")
                if _contains_specific_live_claim(text) else False)
    (evidence, numbers, typed_numbers, journey_ids, statuses, has_recommendation,
     has_journey_result) = \
        _result_evidence(available_results)
    has_journey_result = has_journey_result or any(
        result.get("tool") in ("get_journey", "get_journeys") for result in source_results)
    if (_CITY_INCIDENT_CLAIM_RE.search(text)
            and not _ROUTE_INCIDENT_CLAIM_RE.search(text)
            and _unsupported_city_incident_claim(text, available_results)):
        return reject("unsupported_city_incident_claim")
    if (_SYSTEM_STATUS_CLAIM_RE.search(text)
            and _unsupported_system_status_claim(text, available_results)):
        return reject("unsupported_system_status_claim")
    if _FASTEST_CLAIM_RE.search(text):
        return reject("unsupported_fastest_route_claim")
    journey_id_matches = list(JOURNEY_ID_RE.finditer(text))
    if any(match.group(1) not in journey_ids for match in journey_id_matches):
        return reject("unknown_journey_id")
    journey_id_spans = [match.span() for match in journey_id_matches]
    for match in _NUMBER_RE.finditer(text):
        if any(start <= match.start() and match.end() <= end
               for start, end in journey_id_spans):
            continue
        try:
            value = float(match.group(0).rstrip("%"))
        except ValueError:
            continue
        if not any(abs(value - supported) < 0.000001 for supported in numbers):
            return reject("unsupported_numeric_value")
        metric = _numeric_metric(text, match.start(), match.end())
        if metric and not any(abs(value - supported) < 0.000001
                              for supported in typed_numbers[metric]):
            return reject("numeric_value_mismatched_metric_%s" % metric)
    for word, value in _NUMBER_WORDS.items():
        match = re.search(r"\b%s\b" % word, text, re.IGNORECASE)
        if match:
            if not any(abs(value - supported) < 0.000001 for supported in numbers):
                return reject("unsupported_number_word")
            metric = _numeric_metric(text, match.start(), match.end())
            if metric and not any(abs(value - supported) < 0.000001
                                  for supported in typed_numbers[metric]):
                return reject("number_word_mismatched_metric_%s" % metric)
    if _ALTERNATIVE_CLAIM_RE.search(text) and not has_recommendation:
        return reject("alternative_claim_without_recommendation")
    for match in _ROUTE_NAME_RE.finditer(text):
        if match.group(1).lower() not in evidence:
            return reject("unsupported_route_name")
    if has_journey_result:
        if (_has_positive_claim(text, ("active", "ongoing"))
                and "active" not in statuses):
            return reject("unsupported_active_journey_status")
        if (_has_positive_claim(text, ("completed", "complete", "arrived", "finished"))
                and "completed" not in statuses):
            return reject("unsupported_completed_journey_status")
        if _has_positive_claim(text, ("expired",)) and "expired" not in statuses:
            return reject("unsupported_expired_journey_status")
    for term in ("heavy", "moderate", "free-flow", "congested", "blocked", "accident",
                 "slowdown", "slow", "smooth"):
        if re.search(r"\b%s\b" % re.escape(term), text, re.IGNORECASE) and term not in evidence:
            return reject("unsupported_traffic_condition")
    if _ROUTE_INCIDENT_CLAIM_RE.search(text) and _unsupported_route_incident_claim(
            text, available_results):
        return reject("unsupported_route_incident_claim")
    return False


def _claims_current_source_data(text, tool):
    if tool == "get_incidents":
        return bool(_CITY_INCIDENT_CLAIM_RE.search(text)
                    and not _ROUTE_INCIDENT_CLAIM_RE.search(text))
    if tool in ("get_journey", "get_journeys"):
        return bool(
            _ROUTE_INCIDENT_CLAIM_RE.search(text)
            or _FASTEST_CLAIM_RE.search(text)
            or JOURNEY_ID_RE.search(text)
            or _NUMBER_RE.search(text)
            or any(_has_positive_claim(text, (term,)) for term in
                   ("active", "ongoing", "completed", "complete", "arrived", "finished",
                    "expired"))
        )
    if tool == "get_route_recommendations":
        return bool(_ALTERNATIVE_CLAIM_RE.search(text) or _FASTEST_CLAIM_RE.search(text))
    if tool == "get_congestion_summary":
        return bool(
            _NUMBER_RE.search(text)
            or any(re.search(r"\b%s\b" % re.escape(term), text, re.IGNORECASE)
                   for term in ("heavy", "moderate", "free-flow", "congested", "blocked",
                                "slowdown", "slow", "smooth"))
        )
    if tool == "get_system_status":
        return bool(_SYSTEM_STATUS_CLAIM_RE.search(text))
    return False


def _unsupported_city_incident_claim(text, results):
    incident_results = [
        result for result in results if result.get("tool") == "get_incidents"
        and result.get("availability") in ("live", "empty")
    ]
    if not incident_results:
        return True
    data = incident_results[-1].get("data") or {}
    total = data.get("event_count")
    active = data.get("active_count")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        return True
    if not isinstance(active, int) or isinstance(active, bool) or active < 0:
        active = None
    negative = bool(re.search(
        r"\b(?:no|none|zero|not\s+any|without)\b.{0,55}\b"
        r"(?:incidents?|accidents?|jams?|closures?|hazards?)\b",
        text, re.IGNORECASE))
    count = active if active is not None and re.search(r"\bactive\b", text, re.IGNORECASE) \
        else total
    positive = not negative
    return bool((negative and count > 0) or (positive and count == 0))


def _unsupported_system_status_claim(text, results):
    matches = [result for result in results if result.get("tool") == "get_system_status"]
    if not matches:
        return True
    data = matches[-1].get("data") or {}
    reachable = data.get("reachable")
    ready = data.get("ready")
    lowered = text.lower()
    if re.search(r"\b(?:healthy)\b", lowered):
        return True
    negative_reachability = bool(re.search(
        r"\b(?:not\s+running|not\s+reachable|down|unavailable)\b", lowered))
    positive_reachability = bool(re.search(r"\b(?:running|reachable|up)\b", lowered))
    if negative_reachability and reachable is True:
        return True
    if positive_reachability and reachable is False:
        return True
    negative_readiness = bool(re.search(r"\b(?:not\s+ready|no\s+forecast)\b", lowered))
    positive_readiness = bool(re.search(r"\bready\b", lowered) and not negative_readiness)
    if negative_readiness and ready is True:
        return True
    if positive_readiness and ready is False:
        return True
    if re.search(r"\bforecasts?\s+(?:are\s+)?available\b", lowered):
        horizons = data.get("horizons_min")
        if ready is not True or not horizons:
            return True
    return False


def _route_journeys(results):
    journeys = []
    snapshot_availability = None
    for result in results:
        if result.get("tool") not in (
                "get_journey", "get_journeys", "get_route_recommendations"):
            continue
        snapshot_availability = result.get("availability")
        data = result.get("data") or {}
        if isinstance(data.get("journey"), dict):
            journeys.append(data["journey"])
        journeys.extend(item for item in data.get("journeys") or [] if isinstance(item, dict))
        journeys.extend(item for item in data.get("decisions") or [] if isinstance(item, dict))
    return journeys, snapshot_availability


def _city_incident_events(results):
    events = []
    for result in results:
        if (result.get("tool") != "get_incidents"
                or result.get("availability") not in ("live", "empty")):
            continue
        events.extend(item for item in (result.get("data") or {}).get("events") or []
                      if isinstance(item, dict))
    return events


def _unsupported_route_incident_claim(text, results):
    journeys, _ = _route_journeys(results)
    possible_routes = [item for item in journeys if item.get("journey_state") == "active"]
    current = [item for item in journeys
               if item.get("journey_state") == "active" and not item.get("stale")
               and item.get("route_incident_state") == "current"
               and item.get("incidents_ahead") is not None]
    if not current:
        return True
    if len(possible_routes) > 1:
        identifiers = set(JOURNEY_ID_RE.findall(text))
        labels = {str(item.get("label") or "").lower() for item in possible_routes}
        if not identifiers and not any(label and label in text.lower() for label in labels):
            return True

    counts = [item["incidents_ahead"] for item in current]
    is_negative = bool(re.search(
        r"\b(?:no|not|none|zero|without)\b.{0,55}\b(?:incidents?|accidents?|"
        r"jams?|closures?|hazards?)\b", text, re.IGNORECASE))
    is_positive = bool(re.search(
        r"\b(?:incidents?|accidents?|jams?|closures?|hazards?)\b", text, re.IGNORECASE)
        and not is_negative)
    if is_negative and any(count > 0 for count in counts):
        return True
    if is_positive and all(count == 0 for count in counts):
        return True

    route_ids = {
        str(event_id)
        for item in current
        for event_id in item.get("incident_ids") or []
    }
    route_events = [event for event in _city_incident_events(results)
                    if str(event.get("event_id")) in route_ids]
    all_events = _city_incident_events(results)
    lowered = text.lower()
    for event in all_events:
        names = (event.get("area"), event.get("anchor_name"))
        if any(name and str(name).lower() in lowered for name in names):
            if str(event.get("event_id")) not in route_ids:
                return True
    mentioned_severities = {
        str(event.get("severity") or "").lower()
        for event in all_events
        if event.get("severity") and
        re.search(r"\b%s\b" % re.escape(str(event["severity"]).lower()),
                  text, re.IGNORECASE)
    }
    supported_severities = {
        str(event.get("severity") or "").lower() for event in route_events
        if event.get("severity")
    }
    return bool(mentioned_severities - supported_severities)


def _route_incident_fallback(results):
    journeys, availability = _route_journeys(results)
    current = [item for item in journeys
               if item.get("journey_state") == "active" and not item.get("stale")]
    possible_routes = [item for item in journeys if item.get("journey_state") == "active"]
    checked = [item for item in current
               if item.get("route_incident_state") == "current"
               and item.get("incidents_ahead") is not None]
    if len(possible_routes) > 1:
        return ("I found multiple active journeys, so I can't tell which route you mean. "
                "Which journey are you asking about?")
    if len(checked) > 1:
        return ("I found multiple active journeys, so I can't tell which route you mean. "
                "Which journey are you asking about?")
    if len(checked) == 1:
        journey = checked[0]
        count = int(journey["incidents_ahead"])
        label = journey.get("label") or journey.get("journey_id") or "your journey"
        if count == 0:
            return ("The latest Journey Intelligence check for %s reports no active incidents "
                    "ahead. That does not guarantee the road is completely safe."
                    % label)
        route_ids = {str(value) for value in journey.get("incident_ids") or []}
        matching = [event for event in _city_incident_events(results)
                    if str(event.get("event_id")) in route_ids]
        locations = [
            "%s%s" % (event.get("area") or event.get("anchor_name"),
                      " (%s)" % event["severity"] if event.get("severity") else "")
            for event in matching if event.get("area") or event.get("anchor_name")
        ]
        detail = (" Matching city-wide records: %s." % ", ".join(locations)
                  if locations else "")
        incident_word = "incident" if count == 1 else "incidents"
        return ("The latest Journey Intelligence check reports %s active %s ahead for %s.%s"
                % (count, incident_word, label, detail))
    if any(item.get("route_incident_state") == "stale" for item in journeys):
        return "The route-incident decision is stale, so I can't confirm current incidents ahead."
    if journeys and all(item.get("journey_state") == "completed" for item in journeys):
        return "That journey is completed; there is no current route-incident check for it."
    if availability == "missing":
        return "Journey data has not been published yet, so I can't confirm incidents ahead."
    if availability == "unavailable":
        return "Journey data is unavailable, so I can't confirm incidents ahead."
    if availability == "empty":
        return "No journey is being watched, so there is no current route to check for incidents."
    if not journeys:
        return "City-wide incident data does not establish whether an incident is on your route."
    if any(item.get("route_incident_state") == "not_checked" for item in journeys):
        return "The current journey data has no route-incident check, so I can't confirm incidents ahead."
    if any(item.get("route_incident_state") == "unavailable" for item in journeys):
        return "The route-incident data is incomplete, so I can't confirm incidents ahead."
    return None

@dataclass(frozen=True)
class ToolInvocation:
    """A record of one tool call this turn made, kept for the caller and for tests."""

    name: str
    arguments: Dict[str, Any]
    ok: bool
    availability: str
    headline: str

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments), "ok": self.ok,
                "availability": self.availability, "headline": self.headline}


@dataclass(frozen=True)
class AgentReply:
    """The result of one turn: the text to show, plus how it was produced.

    ``session_id`` is None for a stateless turn and the conversation id otherwise, so a caller that
    knows nothing about memory keeps working unchanged.
    """

    text: str
    provider: str
    tool_calls: List[ToolInvocation] = field(default_factory=list)
    steps: int = 0
    error: Optional[str] = None
    session_id: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "provider": self.provider, "steps": self.steps,
                "error": self.error, "ok": self.ok, "session_id": self.session_id,
                "tool_calls": [call.as_dict() for call in self.tool_calls]}


class InteractionAgent:
    """Orchestrates a provider, its tools, the published traffic data and (optionally) memory.

    Everything is injectable so tests run with a stub provider, a stub registry and a temporary
    database, and need no AWS, no Kafka and no files.
    """

    name = "interaction"

    def __init__(self, config: Optional[InteractionConfig] = None, provider=None, registry=None,
                 system_prompt: Optional[str] = None, memory: Optional[ConversationMemory] = None):
        self.config = config or InteractionConfig.from_env()
        self.provider = provider or build_llm_provider(self.config)
        self.registry = registry or build_default_registry()
        self.system_prompt = system_prompt or build_system_prompt()
        # Constructing memory opens nothing; the database is only touched once a session is used.
        self.memory = memory if memory is not None else ConversationMemory.from_config(self.config)

    # -- sessions --------------------------------------------------------------
    def new_session(self, context=None) -> Optional[str]:
        """Start a new conversation. Returns None when memory is switched off."""
        if not self.memory.enabled:
            return None
        return self.memory.create_session(context=context)

    # -- one turn --------------------------------------------------------------
    def answer(self, message: str, session_id: Optional[str] = None) -> AgentReply:
        """Answer one user message. Never raises; failures come back as an :class:`AgentReply`.

        Without ``session_id`` this is a stateless single turn. With one, the turn is part of a
        conversation: prior messages are replayed as context, this turn is stored, and the same
        ``session_id`` comes back on the reply.
        """
        text = (message or "").strip()
        if not text:
            return AgentReply(text=EMPTY_QUESTION_REPLY, provider=self.provider.name,
                              session_id=session_id)

        session = session_id or None
        use_memory = bool(session) and self.memory.enabled
        history: List[Dict[str, str]] = []
        context: Optional[Dict[str, Any]] = None
        if use_memory:
            session = self.memory.ensure_session(session)
            history = [{"role": item["role"],
                        "content": strip_thinking_tags(item.get("content", ""))}
                       for item in self.memory.history(session)]
            context = self.memory.get_context(session)
            self.memory.add_message(session, "user", text)

        messages: List[Dict[str, Any]] = list(history) + [{"role": "user", "content": text}]
        tool_specs = self.registry.specs()
        invocations: List[ToolInvocation] = []
        results: List[Dict[str, Any]] = []
        preferred_tools: List[str] = []

        required_calls = ([] if isinstance(self.provider, BedrockConverseProvider)
                          else _required_tool_calls(text, context))
        if required_calls:
            preferred_tools.append(required_calls[0][0])
        for name, arguments in required_calls:
            call_id = "required-" + name
            result = self.registry.call(name, arguments)
            results.append(result)
            invocations.append(ToolInvocation(
                name=name, arguments=dict(arguments), ok=bool(result.get("ok", True)),
                availability=str(result.get("availability") or ""),
                headline=str(result.get("headline") or "")))
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"id": call_id, "name": name,
                                             "arguments": dict(arguments)}]})
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "name": name, "result": result})

        final_text: Optional[str] = None
        error: Optional[str] = None
        step = 0
        grounding_retry_used = False
        while step < MAX_STEPS:
            step += 1
            response = self._converse(messages, tool_specs, context)
            if isinstance(response, AgentReply):                 # provider failed, already shaped
                final_text, error = response.text, response.error
                break
            if not response.wants_tools:
                candidate = (response.text or "").strip() or NO_ANSWER_REPLY
                if (isinstance(self.provider, BedrockConverseProvider)
                        and not grounding_retry_used
                        and candidate not in (NO_ANSWER_REPLY, STEP_LIMIT_REPLY)
                        and _has_unsupported_claim(candidate, results)):
                    grounding_retry_used = True
                    messages.append({"role": "assistant", "content": strip_thinking_tags(candidate)})
                    messages.append({"role": "user", "content": GROUNDING_RETRY_INSTRUCTION})
                    continue
                final_text = candidate
                break
            response_calls = list(response.tool_calls)
            for call in response_calls:
                if call.name not in preferred_tools:
                    preferred_tools.append(call.name)
            if (isinstance(self.provider, BedrockConverseProvider)
                    and any(call.name == "get_incidents" for call in response_calls)
                    and not any(call.name == "get_journeys" for call in response_calls)
                    and not any(result.get("tool") in ("get_journey", "get_journeys")
                                for result in results)):
                response_calls.append(ToolCall(
                    name="get_journeys", arguments={},
                    id="related-get_journeys-%d" % step))
            messages.append({"role": "assistant", "content": response.text or "",
                             "tool_calls": [{"id": call.id, "name": call.name,
                                             "arguments": dict(call.arguments)}
                                            for call in response_calls]})
            for call in response_calls:
                cached = next((previous for previous, invocation in zip(
                    reversed(results), reversed(invocations))
                    if invocation.name == call.name
                    and invocation.arguments == dict(call.arguments)), None)
                if cached is None:
                    result = self.registry.call(call.name, call.arguments)
                    results.append(result)
                    invocations.append(ToolInvocation(
                        name=call.name, arguments=dict(call.arguments),
                        ok=bool(result.get("ok", True)),
                        availability=str(result.get("availability") or ""),
                        headline=str(result.get("headline") or "")))
                else:
                    result = cached
                messages.append({"role": "tool", "tool_call_id": call.id or call.name,
                                 "name": call.name, "result": result})
        if final_text is None:
            final_text = STEP_LIMIT_REPLY
        final_text = strip_thinking_tags(final_text) or NO_ANSWER_REPLY
        if (error is None and final_text not in (NO_ANSWER_REPLY, STEP_LIMIT_REPLY)
                and _has_unsupported_claim(final_text, results)):
            LOG.info("Using grounded fallback: selected_tools=%s result_tools=%s",
                     preferred_tools,
                     sorted({str(result.get("tool") or "unknown") for result in results}))
            if _ROUTE_INCIDENT_CLAIM_RE.search(final_text):
                final_text = (_route_incident_fallback(results)
                              or _tool_fallback(results, preferred_tools)
                              if results else
                              "I can't confirm route incidents without current journey data.")
            else:
                final_text = (_tool_fallback(results, preferred_tools)
                              if results else NO_ANSWER_REPLY)

        if use_memory:
            self._persist(session, text, final_text, invocations, results)

        return AgentReply(text=final_text, provider=self.provider.name, tool_calls=invocations,
                          steps=step, error=error, session_id=session)

    # -- internals -------------------------------------------------------------
    def _converse(self, messages, tool_specs, context):
        """Call the provider. On failure return a ready-made reply instead of raising."""
        try:
            if context is None:                                  # stateless: Phase 1 call shape
                return self.provider.converse(messages, tool_specs, system=self.system_prompt)
            return self.provider.converse(messages, tool_specs, system=self.system_prompt,
                                          context=context)
        except Exception as exc:                                 # a bad provider must not 500 the turn
            LOG.warning("interaction agent provider %r failed: %s", self.provider.name, exc)
            return AgentReply(text=PROVIDER_ERROR_REPLY, provider=self.provider.name,
                              error=str(exc))

    def _persist(self, session, question, answer_text, invocations, results):
        """Store the turn. A memory failure must never lose the answer the traveller already has."""
        try:
            self.memory.add_message(session, "assistant", answer_text,
                                    tool_calls=[call.as_dict() for call in invocations] or None)
            self.memory.update_context(session, **context_update(question, invocations, results))
        except Exception as exc:                                 # noqa: BLE001 - memory is optional
            LOG.warning("could not persist interaction turn for %s: %s", session, exc)

    def tool_names(self) -> List[str]:
        """The read-only tools this agent may use, in registration order."""
        return self.registry.names()


def context_update(question, invocations, results) -> Dict[str, Any]:
    """The context to carry forward from a turn: topic, journey ids and places.

    Pulled only from tool results and the question itself, so nothing is remembered that a tool did
    not actually report.
    """
    changes: Dict[str, Any] = {}
    intents = []
    for invocation in invocations:
        intent = TOOL_INTENT.get(invocation.name)
        if intent and intent not in intents:
            intents.append(intent)
    primary_intent = detect_intent(question)
    if primary_intent:
        changes["last_intent"] = primary_intent
    elif intents:
        changes["last_intent"] = intents[0]

    journey_ids = []
    places: List[str] = []
    for result in results:
        data = result.get("data") or {}
        journey_ids.extend(_journey_ids_from(data))
        places.extend(_places_from(data))
    if journey_ids:
        changes["journey_ids"] = journey_ids
    if places:
        changes["places"] = places
    return changes


def _journey_ids_from(data) -> List[str]:
    ids = []
    journey = data.get("journey")
    if isinstance(journey, dict) and journey.get("journey_id"):
        ids.append(journey["journey_id"])
    for key in ("journeys", "recommendations"):
        for item in data.get(key) or []:
            if isinstance(item, dict) and item.get("journey_id"):
                ids.append(item["journey_id"])
    return ids


def _places_from(data) -> List[str]:
    places = []
    for event in data.get("events") or []:
        if isinstance(event, dict):
            area = event.get("area") or event.get("anchor_name")
            if area:
                places.append(area)
    for area in data.get("areas") or []:
        if isinstance(area, dict) and area.get("area"):
            places.append(area["area"])
    return places


def build_agent(config: Optional[InteractionConfig] = None) -> InteractionAgent:
    """Build the agent with the configured provider and the real read-only tools."""
    return InteractionAgent(config=config)


def answer(message: str, session_id: Optional[str] = None,
           config: Optional[InteractionConfig] = None) -> AgentReply:
    """Convenience one-shot: build a default agent and answer a single message."""
    return build_agent(config).answer(message, session_id=session_id)
