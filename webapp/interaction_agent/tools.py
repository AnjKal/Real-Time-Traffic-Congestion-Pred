"""Read-only tools for the User Interaction Agent.

Every tool answers a question from data the other two agents have ALREADY published. Nothing here
re-derives congestion, re-runs routing or runs the model: the interaction agent is a consumer, so it
reads the same files the web app serves and nothing else --

  * ``data/live/incidents.json``       written by the incident monitoring agent
  * ``data/live/features.json``        written by the feature aggregator
  * ``data/live/journey_state.json``   written by the journey intelligence agent
  * ``data/live/journey_switches.json`` the journey agent's switch ledger
  * ``<model service>/health``          the running prediction system

Journeys are read through the existing ``webapp.journey_memory.JourneyMemory`` so the on-disk
schema is not duplicated here. The model service is the one thing that is a separate process, so
its health is the only HTTP call (the same call the web app already makes).

Each tool returns a structured result and NEVER raises. A missing file, an empty list, stale data
or an unreachable service all come back as an explicit ``availability`` value, so the agent can say
"I don't know" instead of inventing an incident, a prediction, a journey or an alternative route.

Availability values:
  live         source present, fresh, and has content
  empty        source present and fresh, but there is nothing to report
  stale        source present but older than ``stale_after_min``
  missing      source has never been published
  unavailable  source could not be reached (e.g. the model service is down)
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from webapp.journey_memory import JourneyMemory, parse_iso, utc_now

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LIVE_DIR = os.path.join(PROJECT_ROOT, "data", "live")

NO_ARGS_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}}

_SEVERITY_ORDER = {"heavy": 2, "moderate": 1, "free": 0}


def _num(value, digits=1):
    """Format a number for a human line; None becomes N/A rather than a guess."""
    if value is None:
        return "N/A"
    try:
        return ("%%.%df" % digits) % float(value)
    except (TypeError, ValueError):
        return "N/A"


def _age_minutes(stamp):
    moment = parse_iso(stamp)
    if moment is None:
        return None
    return round(max(0.0, (utc_now() - moment).total_seconds() / 60.0), 1)


def _read_json(path):
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (IOError, ValueError):
        return None


def _result(tool, availability, headline, details=None, data=None, ok=True, error=None):
    return {"tool": tool, "ok": ok, "availability": availability, "headline": headline,
            "details": list(details or []), "data": data or {}, "error": error}


def _snapshot_failure(sources, path, snapshot, source_name):
    """Return an explicit availability result when a snapshot cannot be trusted."""
    if snapshot is None:
        if not sources.source_available(path):
            return "missing", "The %s has not published a snapshot yet." % source_name
        return "unavailable", "The %s snapshot is present but could not be read." % source_name
    if not isinstance(snapshot, dict):
        return "unavailable", "The %s snapshot has an invalid format." % source_name
    stamp = snapshot.get("made_at") or snapshot.get("timestamp")
    if _age_minutes(stamp) is None:
        return "unavailable", "The %s snapshot has no valid freshness timestamp." % source_name
    return None


# --------------------------------------------------------------------------- data sources

class LiveDataSources:
    """The real read layer. Everything is injectable so tests need no files, HTTP or AWS.

    ``health_fetcher`` returns the model service ``/health`` body or raises; tests pass a stub.
    """

    def __init__(self, live_dir=DEFAULT_LIVE_DIR, model_service_url="http://127.0.0.1:8001",
                 timeout=10.0, stale_after_min=10.0, health_fetcher=None):
        self.live_dir = live_dir
        self.incidents_path = os.path.join(live_dir, "incidents.json")
        self.features_path = os.path.join(live_dir, "features.json")
        self.state_path = os.path.join(live_dir, "journey_state.json")
        self.switch_path = os.path.join(live_dir, "journey_switches.json")
        self.model_service_url = (model_service_url or "").rstrip("/")
        self.timeout = timeout
        self.stale_after_min = stale_after_min
        self._health_fetcher = health_fetcher or self._fetch_health

    @classmethod
    def from_config(cls, config=None):
        if config is None:
            from webapp.interaction_agent.config import InteractionConfig
            config = InteractionConfig.from_env()
        return cls(model_service_url=config.model_service_url, timeout=config.request_timeout_s,
                   stale_after_min=config.stale_after_min)

    # -- sources ---------------------------------------------------------------
    def incidents(self):
        return _read_json(self.incidents_path)

    def features(self):
        return _read_json(self.features_path)

    def journeys(self):
        memory = JourneyMemory(state_path=self.state_path, switch_log_path=self.switch_path)
        journeys = memory.all_journeys()
        for journey in journeys:
            journey["decision"] = memory.last_decision(journey.get("journey_id"))
        return journeys

    def journey(self, journey_id):
        memory = JourneyMemory(state_path=self.state_path, switch_log_path=self.switch_path)
        journey = memory.recall(journey_id)
        if journey is None:
            return None
        journey["decision"] = memory.last_decision(journey_id)
        return journey

    def switches(self, journey_id=None):
        memory = JourneyMemory(state_path=self.state_path, switch_log_path=self.switch_path)
        return memory.switches(journey_id)

    def journey_snapshot_availability(self):
        """Distinguish a missing/invalid state file from a valid empty producer snapshot."""
        if not os.path.exists(self.state_path):
            return "missing"
        try:
            with open(self.state_path, "r") as handle:
                state = json.load(handle)
        except (IOError, ValueError):
            return "unavailable"
        if not isinstance(state, dict) or not isinstance(state.get("journeys"), dict):
            return "unavailable"
        if not isinstance(state.get("decisions", {}), dict):
            return "unavailable"
        return "live" if state["journeys"] else "empty"

    def system_health(self):
        try:
            return self._health_fetcher()
        except Exception:
            return None

    def source_available(self, path):
        return os.path.exists(path)

    def _fetch_health(self):
        import requests
        response = requests.get(self.model_service_url + "/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()


# --------------------------------------------------------------------------- tool handlers

def get_incidents(sources):
    snapshot = sources.incidents()
    failure = _snapshot_failure(sources, sources.incidents_path, snapshot, "incident agent")
    if failure:
        return _result("get_incidents", failure[0], failure[1], data={"events": []})
    age = _age_minutes(snapshot.get("made_at") or snapshot.get("timestamp"))
    events = list(snapshot.get("active_events") or [])
    resolved = list(snapshot.get("recent_resolved") or [])
    if age is not None and age > sources.stale_after_min:
        availability = "stale"
    elif not events:
        availability = "empty"
    else:
        availability = "live"
    counts: Dict[str, int] = {}
    for event in events:
        status = str(event.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    if not events:
        headline = "No congestion incidents are being reported right now."
    else:
        headline = ("%d congestion event(s) are open: %s." % (
            len(events), ", ".join("%d %s" % (value, key) for key, value in sorted(counts.items()))))
    details = [
        _incident_line(event) + " (city-wide feed; route membership is not established)"
        for event in events[:5]
    ]
    data = {"poll_id": snapshot.get("poll_id"), "made_at": snapshot.get("made_at"),
            "age_min": age, "model_mode": snapshot.get("model_mode"),
            "sensor_coverage_pct": snapshot.get("sensor_coverage_pct"),
            "scope": "citywide",
            "event_count": len(events), "active_count": counts.get("active", 0),
            "status_counts": counts, "resolved_count": len(resolved),
            "events": [_incident_data(event) for event in events]}
    return _result("get_incidents", availability, headline, details, data)


def _incident_data(event):
    return {
        "event_id": event.get("event_id"), "event_type": event.get("event_type"),
        "status": event.get("status"), "severity": event.get("severity"),
        "area": event.get("area"), "anchor_junction_id": event.get("anchor_junction_id"),
        "anchor_name": event.get("anchor_name"),
        "affected_junction_count": event.get("affected_junction_count"),
        "avg_current_speed_kmh": event.get("avg_current_speed_kmh"),
        "avg_free_flow_speed_kmh": event.get("avg_free_flow_speed_kmh"),
        "avg_predicted_speed_kmh": event.get("avg_predicted_speed_kmh"),
        "forecast_horizon_min": event.get("forecast_horizon_min"),
        "speed_ratio_current": event.get("speed_ratio_current"),
        "evidence": event.get("evidence"), "signal_quality": event.get("signal_quality"),
        "model_mode": event.get("model_mode"), "warmup_steps": event.get("warmup_steps"),
        "forecast_used": event.get("forecast_used"), "data_quality": event.get("data_quality"),
        "first_seen": event.get("first_seen"), "updated_at": event.get("updated_at"),
    }


def _incident_line(event):
    area = event.get("area") or event.get("anchor_name") or "unknown area"
    parts = ["%s: %s, %s" % (area, event.get("status") or "?", event.get("severity") or "?")]
    if event.get("affected_junction_count"):
        parts.append("%d junction(s)" % int(event["affected_junction_count"]))
    if (event.get("avg_current_speed_kmh") is not None
            and event.get("avg_free_flow_speed_kmh") is not None):
        parts.append("observed TomTom speed %s km/h vs %s free-flow"
                     % (_num(event.get("avg_current_speed_kmh")),
                        _num(event.get("avg_free_flow_speed_kmh"))))
    if event.get("avg_predicted_speed_kmh") is not None:
        horizon = event.get("forecast_horizon_min")
        suffix = " at +%s min" % horizon if horizon is not None else ""
        parts.append("DCRNN forecast %s km/h%s"
                     % (_num(event.get("avg_predicted_speed_kmh")), suffix))
    if event.get("model_mode") == "pipeline_test":
        parts.append("pipeline-test forecast, not an accuracy claim")
    if event.get("warmup_steps"):
        parts.append("model warm-up: %s step(s)" % event["warmup_steps"])
    return ", ".join(parts)


def get_congestion_summary(sources):
    snapshot = sources.features()
    failure = _snapshot_failure(sources, sources.features_path, snapshot, "feature aggregator")
    if failure:
        return _result("get_congestion_summary", failure[0], failure[1], data={"areas": []})
    age = _age_minutes(snapshot.get("made_at") or snapshot.get("timestamp"))
    areas = list(snapshot.get("areas") or [])
    if age is not None and age > sources.stale_after_min:
        availability = "stale"
    elif not areas:
        availability = "empty"
    else:
        availability = "live"
    overall = _overall_level(areas)
    counts = {"free": 0, "moderate": 0, "heavy": 0}
    for area in areas:
        level = area.get("overall_congestion_level")
        if level in counts:
            counts[level] += 1
    if not areas:
        headline = "No area-wise congestion figures are available right now."
    else:
        headline = ("%d area(s) reporting; the overall DCRNN forecast is %s "
                    "(%d heavy, %d moderate, "
                    "%d free)." % (len(areas), overall or "unclassified", counts["heavy"],
                                   counts["moderate"], counts["free"]))
    worst = sorted(areas, key=lambda area: (-_SEVERITY_ORDER.get(
        area.get("overall_congestion_level"), -1), area.get("avg_current_speed_kmh") or 1e9))
    details = [_area_line(area) for area in worst[:5]]
    health_reader = getattr(sources, "system_health", None)
    health = health_reader() if callable(health_reader) else None
    forecast_context = None
    if isinstance(health, dict):
        forecast_context = {
            "mode": health.get("mode"), "ready": health.get("ready"),
            "horizons_min": health.get("horizons_min") or [],
            "warmup_steps": health.get("warmup_steps"),
            "not_ready_reason": health.get("not_ready_reason"),
            "warning": health.get("warning"), "warmup_warning": health.get("warmup_warning"),
        }
    data = {"scope": "area", "poll_id": snapshot.get("poll_id"),
            "made_at": snapshot.get("made_at"), "age_min": age,
            "sensor_coverage_pct": snapshot.get("sensor_coverage_pct"),
            "step_minutes": snapshot.get("step_minutes"), "data_age_min": snapshot.get("data_age_min"),
            "forecast_context": forecast_context,
            "area_count": len(areas), "overall_level": overall, "level_counts": counts,
            "areas": [_area_data(area) for area in areas]}
    return _result("get_congestion_summary", availability, headline, details, data)


def _overall_level(areas):
    """The level most areas are at; ties resolve to the more severe one. None when unknown."""
    counts = {"free": 0, "moderate": 0, "heavy": 0}
    for area in areas:
        level = area.get("overall_congestion_level")
        if level in counts:
            counts[level] += 1
    if not sum(counts.values()):
        return None
    return max(counts, key=lambda level: (counts[level], _SEVERITY_ORDER[level]))


def _area_data(area):
    return {
        "area": area.get("area"),
        "overall_congestion_level": area.get("overall_congestion_level"),
        "avg_current_speed_kmh": area.get("avg_current_speed_kmh"),
        "avg_free_flow_speed_kmh": area.get("avg_free_flow_speed_kmh"),
        "avg_predicted_speed_kmh": area.get("avg_predicted_speed_kmh"),
        "avg_delay_min_per_km": area.get("avg_delay_min_per_km"),
        "reporting_junction_count": area.get("reporting_junction_count"),
        "total_junction_count": area.get("total_junction_count"),
        "sensor_coverage_pct": area.get("sensor_coverage_pct"),
    }


def _area_line(area):
    observed = area.get("avg_current_speed_kmh")
    observed_text = ("observed TomTom speed %s km/h" % _num(observed) if observed is not None
                     else "no observed TomTom speed")
    return ("%s: %s; %s; %s km/h free-flow; DCRNN forecast %s km/h"
            % (area.get("area") or "unknown area",
               area.get("overall_congestion_level") or "unclassified", observed_text,
               _num(area.get("avg_free_flow_speed_kmh")),
               _num(area.get("avg_predicted_speed_kmh"))))


def get_journeys(sources):
    state = getattr(sources, "journey_snapshot_availability", None)
    state = state() if callable(state) else None
    if state in ("missing", "unavailable"):
        message = ("Journey data has not been published yet." if state == "missing" else
                   "Journey data is present but unavailable because its snapshot is invalid.")
        return _result("get_journeys", state, message,
                       data={"journey_count": 0, "journeys": []})
    journeys = sources.journeys()
    if not journeys:
        return _result("get_journeys", "empty",
                       "You are not watching any journey right now.",
                       data={"journey_count": 0, "journeys": []})
    summaries = [_journey_summary(journey, sources.stale_after_min) for journey in journeys]
    stale = all(summary["stale"] for summary in summaries)
    headline = "You have %d journey(s) being watched." % len(summaries)
    details = [_journey_line(summary) for summary in summaries[:5]]
    return _result("get_journeys", "stale" if stale else "live", headline, details,
                   {"journey_count": len(summaries), "journeys": summaries})


def get_journey(sources, journey_id):
    journey = sources.journey(journey_id)
    if journey is None:
        state = getattr(sources, "journey_snapshot_availability", None)
        state = state() if callable(state) else None
        if state in ("missing", "unavailable"):
            message = ("Journey data has not been published yet." if state == "missing" else
                       "Journey data is present but unavailable because its snapshot is invalid.")
            return _result("get_journey", state, message,
                           data={"requested_journey_id": journey_id})
        return _result("get_journey", "missing",
                       "I'm not watching a journey with id '%s'." % journey_id,
                       data={"requested_journey_id": journey_id,
                             "snapshot_availability": state or "live"})
    summary = _journey_summary(journey, sources.stale_after_min)
    return _result("get_journey", "stale" if summary["stale"] else "live",
                   "Here is the latest on journey %s." % (summary["label"] or summary["journey_id"]),
                   [_journey_line(summary)], {"journey": summary})


def _journey_summary(journey, stale_after_min):
    decision = journey.get("decision") or {}
    updated_at = decision.get("updated_at") or journey.get("updated_at")
    age = _age_minutes(updated_at)
    status = str(journey.get("status") or "").strip().lower()
    action = str(decision.get("action") or "").strip().lower()
    if status == "expired" or action == "expired":
        journey_state = "expired"
    elif status == "arrived" or action == "arrived":
        journey_state = "completed"
    elif status == "active" and action not in ("expired", "arrived"):
        journey_state = "active"
    else:
        journey_state = "unavailable"
    stale = age is None or age > stale_after_min
    trace = decision.get("tool_calls")
    route_check = next((call for call in trace if isinstance(call, dict)
                        and call.get("tool") == "check_incidents_on_route"),
                       None) if isinstance(trace, list) else None
    raw_incidents_ahead = decision.get("incidents_ahead")
    raw_incident_ids = decision.get("incident_ids")
    route_check_valid = (
        isinstance(raw_incidents_ahead, int) and not isinstance(raw_incidents_ahead, bool)
        and raw_incidents_ahead >= 0
        and isinstance(raw_incident_ids, list)
        and all(isinstance(item, str) and item for item in raw_incident_ids)
    )
    route_incident_state = (
        "current" if journey_state == "active" and not stale and route_check
        and route_check.get("ok") and route_check_valid else
        "stale" if journey_state == "active" and stale else
        "unavailable" if journey_state == "active" and route_check else
        "not_checked" if journey_state == "active" else
        "not_applicable"
    )
    route_incidents_ahead = (
        raw_incidents_ahead
        if route_incident_state == "current" else None
    )
    alternative = decision.get("alternative")
    if isinstance(alternative, dict):
        alternative = {
            "rank": alternative.get("rank"),
            "eta_min": alternative.get("eta_min"),
            "total_eta_min": alternative.get("eta_min"),
            "remaining_eta_min": alternative.get("remaining_eta_min"),
            "gain_min": alternative.get("gain_min"),
            "gain_pct": alternative.get("gain_pct"),
            "total_length_km": alternative.get("total_length_km"),
            "turn_off_after_km": alternative.get("turn_off_after_km"),
        }
    else:
        alternative = None
    trace = [
        {"tool": call.get("tool"), "ok": call.get("ok"),
         "skipped": call.get("skipped"), "summary": call.get("summary")}
        for call in trace
        if isinstance(call, dict)
        and call.get("tool") in ("score_current_route", "check_incidents_on_route",
                                 "find_alternatives")
    ] if isinstance(trace, list) else []
    return {
        "journey_id": journey.get("journey_id"),
        "label": journey.get("label"),
        "status": journey.get("status"),
        "journey_state": journey_state,
        "total_length_km": journey.get("total_length_km"),
        "progress_pct": decision.get("progress_pct"),
        "remaining_km": decision.get("remaining_km"),
        "eta_min": decision.get("eta_min"),
        "total_eta_min": decision.get("eta_min"),
        "remaining_eta_min": decision.get("remaining_eta_min"),
        "baseline_eta_min": decision.get("baseline_eta_min"),
        "eta_delta_min": decision.get("eta_delta_min"),
        "delay_min": decision.get("delay_min"),
        "achieved_total_min": decision.get("achieved_total_min"),
        "pct_heavy": decision.get("pct_heavy"),
        "route_coverage_pct": decision.get("route_coverage_pct"),
        "switch_count": decision.get("switch_count", journey.get("switch_count", 0)),
        "action": decision.get("action"),
        "confidence": decision.get("confidence"),
        "rationale": decision.get("rationale"),
        "evidence": list(decision.get("evidence") or []),
        "evaluation": trace,
        "alternative": alternative,
        "has_alternative": bool(alternative),
        "route_incident_state": route_incident_state,
        "incidents_ahead": route_incidents_ahead,
        "incident_ids": (list(raw_incident_ids)
                         if route_incident_state == "current" else []),
        "route_incident_summary": (route_check.get("summary")
                                   if route_incident_state == "current" else None),
        "updated_at": updated_at,
        "age_min": age,
        "freshness_known": age is not None,
        "stale": stale,
    }


def _journey_line(summary):
    state_text = summary["journey_state"]
    if summary["journey_state"] == "active" and summary["stale"]:
        state_text = "last known active; decision is stale"
    parts = ["%s (%s): %s" % (summary["label"] or summary["journey_id"],
                              summary["journey_id"], state_text)]
    last_known = summary["stale"] and summary["journey_state"] == "active"
    if summary["progress_pct"] is not None:
        progress_label = "last known progress" if last_known else "progress"
        parts.append("%s %s%%" % (progress_label, _num(summary["progress_pct"])))
    if summary["remaining_km"] is not None:
        distance_label = "last known remaining distance" if last_known else "remaining distance"
        parts.append("%s %s km" % (distance_label, _num(summary["remaining_km"])))
    if summary["journey_state"] == "active" and summary["eta_min"] is not None:
        eta_label = "last known total ETA" if summary["stale"] else "total ETA"
        parts.append("%s %s min" % (eta_label, _num(summary["total_eta_min"])))
        if summary["remaining_eta_min"] is not None:
            remaining_label = ("last known remaining ETA" if summary["stale"]
                               else "remaining ETA")
            parts.append("%s %s min" % (remaining_label, _num(summary["remaining_eta_min"])))
    if summary["route_incident_state"] == "current":
        if summary["incidents_ahead"] is not None:
            parts.append("%s active incident(s) reported ahead"
                         % summary["incidents_ahead"])
        if summary["route_incident_summary"]:
            parts.append("route check: %s" % summary["route_incident_summary"])
    elif summary["route_incident_state"] == "stale":
        parts.append("route incident decision is stale")
    if summary["journey_state"] == "completed":
        if summary["achieved_total_min"] is not None:
            parts.append("achieved duration %s min" % _num(summary["achieved_total_min"]))
        if summary["baseline_eta_min"] is not None:
            parts.append("baseline ETA %s min" % _num(summary["baseline_eta_min"]))
        if summary["switch_count"] is not None:
            parts.append("%s switch(es)" % summary["switch_count"])
    if summary["action"]:
        parts.append("action %s" % summary["action"])
    if summary["confidence"]:
        parts.append("confidence %s" % summary["confidence"])
    return ", ".join(parts)


def get_route_recommendations(sources):
    state = getattr(sources, "journey_snapshot_availability", None)
    state = state() if callable(state) else None
    if state in ("missing", "unavailable"):
        message = ("Journey data has not been published yet." if state == "missing" else
                   "Journey data is present but unavailable because its snapshot is invalid.")
        return _result("get_route_recommendations", state, message,
                       data={"recommendations": [], "recent_switches": [], "switch_count": 0})
    journeys = sources.journeys()
    recommendations = []
    decision_summaries = []
    stale_decisions = 0
    for journey in journeys:
        decision = journey.get("decision") or {}
        alternative = decision.get("alternative")
        summary = _journey_summary(journey, sources.stale_after_min)
        decision_summaries.append({
            "journey_id": summary["journey_id"], "label": summary["label"],
            "journey_state": summary["journey_state"], "stale": summary["stale"],
            "action": summary["action"], "confidence": summary["confidence"],
            "eta_min": summary["eta_min"], "total_eta_min": summary["total_eta_min"],
            "remaining_eta_min": summary["remaining_eta_min"],
            "baseline_eta_min": summary["baseline_eta_min"],
            "eta_delta_min": summary["eta_delta_min"],
            "route_incident_state": summary["route_incident_state"],
            "incidents_ahead": summary["incidents_ahead"],
            "incident_ids": summary["incident_ids"],
            "route_incident_summary": summary["route_incident_summary"],
            "rationale": summary["rationale"], "evidence": summary["evidence"],
            "evaluation": summary["evaluation"], "alternative": summary["alternative"],
            "updated_at": summary["updated_at"],
        })
        if summary["stale"]:
            stale_decisions += 1
        if (summary["journey_state"] == "active" and not summary["stale"]
                and decision.get("action") == "recommend_switch"
                and isinstance(alternative, dict)):
            recommendations.append({
                "journey_id": journey.get("journey_id"), "label": journey.get("label"),
                "journey_state": summary["journey_state"],
                "confidence": decision.get("confidence"), "rationale": decision.get("rationale"),
                "evidence": summary["evidence"], "evaluation": summary["evaluation"],
                "eta_min": decision.get("eta_min"),
                "total_eta_min": decision.get("eta_min"),
                "remaining_eta_min": decision.get("remaining_eta_min"),
                "baseline_eta_min": decision.get("baseline_eta_min"),
                "alternative": {
                    "rank": alternative.get("rank"), "eta_min": alternative.get("eta_min"),
                    "total_eta_min": alternative.get("eta_min"),
                    "remaining_eta_min": alternative.get("remaining_eta_min"),
                    "gain_min": alternative.get("gain_min"), "gain_pct": alternative.get("gain_pct"),
                    "total_length_km": alternative.get("total_length_km"),
                    "turn_off_after_km": alternative.get("turn_off_after_km"),
                },
                "updated_at": decision.get("updated_at"),
            })
    switch_rows = sources.switches() or []
    recent_switches = []
    for row in switch_rows[:5]:
        recent_switches.append({
            "journey_id": row.get("journey_id"), "switched_at": row.get("switched_at"),
            "gain_min": row.get("gain_min"), "gain_realised_min": row.get("gain_realised_min"),
            "reason": row.get("reason"),
        })
    if not journeys and state == "missing":
        return _result("get_route_recommendations", "missing",
                       "Journey data has not been published yet.",
                       data={"recommendations": [], "recent_switches": recent_switches,
                             "switch_count": len(switch_rows)})
    if not journeys and state == "unavailable":
        return _result("get_route_recommendations", "unavailable",
                       "Journey data is present but unavailable because its snapshot is invalid.",
                       data={"recommendations": [], "recent_switches": recent_switches,
                             "switch_count": len(switch_rows)})
    if not journeys:
        return _result("get_route_recommendations", "empty",
                       "You are not watching a journey, so there is no alternative route to "
                       "recommend.", data={"recommendations": [], "recent_switches": recent_switches,
                                           "switch_count": len(switch_rows)})
    if stale_decisions == len(journeys):
        return _result("get_route_recommendations", "stale",
                       "Journey decisions are stale, so I can't confirm a current route "
                       "recommendation.",
                       data={"recommendations": [], "recent_switches": recent_switches,
                             "switch_count": len(switch_rows)})
    if not recommendations:
        headline = "No route switch is being recommended right now."
    else:
        best = max(recommendations, key=lambda item: float(
            (item["alternative"].get("gain_min") or 0.0)))
        gain = best["alternative"].get("gain_min")
        headline = ("A faster route is recommended for %s, saving about %s minutes."
                    % (best["label"] or best["journey_id"], _num(gain)))
    details = [_recommendation_line(item) for item in recommendations[:5]]
    has_fresh_active_decision = any(
        item["journey_state"] == "active" and not item["stale"]
        for item in decision_summaries)
    availability = "live" if recommendations or has_fresh_active_decision else "empty"
    return _result("get_route_recommendations", availability, headline,
                   details, {"recommendations": recommendations,
                             "decisions": decision_summaries,
                             "recent_switches": recent_switches,
                             "switch_count": len(switch_rows)})


def _recommendation_line(item):
    alternative = item["alternative"]
    return ("%s: switch to alternative rank %s, total ETA %s min, remaining ETA %s min "
            "(save %s min / %s%%)"
            % (item["label"] or item["journey_id"], alternative.get("rank"),
               _num(alternative.get("total_eta_min")),
               _num(alternative.get("remaining_eta_min")), _num(alternative.get("gain_min")),
               _num(alternative.get("gain_pct"))))


def get_system_status(sources):
    health = sources.system_health()
    data = {"model_service_url": sources.model_service_url,
            "scope": "prediction_system",
            "incidents_published": sources.source_available(sources.incidents_path),
            "features_published": sources.source_available(sources.features_path)}
    if health is None:
        data.update({"reachable": False, "ready": False})
        return _result("get_system_status", "unavailable",
                       "I can't reach the prediction system at %s right now."
                       % sources.model_service_url, data=data)
    horizons = health.get("horizons_min") or []
    data.update({"reachable": True, "ready": bool(health.get("ready")),
                 "mode": health.get("mode"), "step_minutes": health.get("step_minutes"),
                 "horizons_min": horizons, "newest_slot": health.get("newest_slot"),
                 "latest_poll_id": health.get("latest_poll_id"),
                 "sensor_coverage_pct": health.get("sensor_coverage_pct"),
                 "not_ready_reason": health.get("not_ready_reason"),
                 "warmup_steps": health.get("warmup_steps"),
                 "input_window_steps": health.get("input_window_steps"),
                 "warning": health.get("warning"),
                 "warmup_warning": health.get("warmup_warning")})
    if health.get("ready"):
        if len(horizons) >= 2:
            span = "+%d to +%d minutes" % (int(horizons[0]), int(horizons[-1]))
        else:
            span = "the next few minutes"
        headline = "The prediction system is running (%s mode); forecasts cover %s." % (
            health.get("mode") or "unknown", span)
        if health.get("mode") == "pipeline_test":
            headline += " Pipeline-test output is not evidence of model accuracy."
        if health.get("warmup_steps"):
            headline += " The model is still warming up (%s step(s))." % health["warmup_steps"]
        if health.get("warmup_warning") or health.get("warning"):
            headline += " %s" % (health.get("warmup_warning") or health.get("warning"))
        availability = "live"
    else:
        headline = ("The prediction system is up but has no forecast yet (%s)."
                    % (health.get("not_ready_reason") or "still warming up"))
        if health.get("warmup_warning") or health.get("warning"):
            headline += " %s" % (health.get("warmup_warning") or health.get("warning"))
        availability = "unavailable"
    return _result("get_system_status", availability, headline, data=data)


# --------------------------------------------------------------------------- registry

@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Dict[str, Any]]


class ToolRegistry:
    """Holds the tools and calls them safely. ``specs()`` is Bedrock ``toolConfig`` shaped."""

    def __init__(self):
        self._tools: "OrderedDict[str, Tool]" = OrderedDict()

    def register(self, name, description, input_schema, handler):
        self._tools[name] = Tool(name=name, description=description,
                                 input_schema=input_schema or NO_ARGS_SCHEMA, handler=handler)
        return self

    def names(self) -> List[str]:
        return list(self._tools)

    def specs(self) -> List[Dict[str, Any]]:
        return [{"toolSpec": {"name": tool.name, "description": tool.description,
                              "inputSchema": {"json": tool.input_schema}}}
                for tool in self._tools.values()]

    def call(self, name, arguments=None):
        tool = self._tools.get(name)
        if tool is None:
            return _result(name, "unavailable", "Unknown tool '%s'." % name, ok=False,
                           error="unknown tool")
        try:
            return tool.handler(**(arguments or {}))
        except TypeError as exc:
            return _result(name, "unavailable",
                           "The '%s' tool could not be called with those arguments." % name,
                           ok=False, error=str(exc))
        except Exception as exc:                     # a tool must never take the turn down
            return _result(name, "unavailable", "The '%s' tool failed." % name, ok=False,
                           error=str(exc))


def build_default_registry(data_sources=None):
    """The read-only tools the interaction agent may use."""
    sources = data_sources or LiveDataSources.from_config()
    registry = ToolRegistry()
    registry.register(
        "get_system_status",
        "Is the traffic prediction system running? Reports reachability, readiness and which "
        "forecast horizons exist. Use for 'is it running' / 'system status' questions.",
        NO_ARGS_SCHEMA, lambda: get_system_status(sources))
    registry.register(
        "get_incidents",
        "Current city-wide congestion incidents (active, watch, recovering) from the incident "
        "monitoring agent. This source does not establish whether an event affects a particular "
        "journey or route. For route-specific incidents, also use get_journeys or get_journey and "
        "match event IDs only against the Journey Intelligence incident_ids.",
        NO_ARGS_SCHEMA, lambda: get_incidents(sources))
    registry.register(
        "get_congestion_summary",
        "Area-level congestion, observed speeds and coverage from the feature aggregator. This "
        "does not establish conditions on a particular route. Distinguishes observed TomTom "
        "readings from DCRNN forecasts.",
        NO_ARGS_SCHEMA, lambda: get_congestion_summary(sources))
    registry.register(
        "get_journeys",
        "Every journey currently being watched by the journey intelligence agent, with its latest "
        "decision, freshness, route incident count/IDs, total and remaining ETA, ETA delta, "
        "producer rationale, and recorded current-route/alternative evaluation summaries. Use for "
        "journey progress, whether a traveller is late, ETA, or why a route is being held; do not "
        "aggregate across multiple active journeys unless explicitly asked.",
        NO_ARGS_SCHEMA, lambda: get_journeys(sources))
    registry.register(
        "get_journey",
        "The latest decision for one journey, by journey id (e.g. jr-ff466f71c8), including route "
        "incident evidence when a fresh active-route check exists, distinct ETA fields, producer "
        "rationale, and recorded route-evaluation summaries.",
        {"type": "object",
         "properties": {"journey_id": {"type": "string",
                                       "description": "The journey id the page generated."}},
         "required": ["journey_id"]},
        lambda journey_id: get_journey(sources, journey_id))
    registry.register(
        "get_route_recommendations",
        "Alternative routes the journey agent is currently recommending with actual alternative "
        "data, plus each journey's current action, rationale, ETA comparison, recorded evaluation "
        "summary, and switches already taken. Use for route recommendations and questions about "
        "why the journey agent is holding or recommending a route. A hold/no-switch decision does "
        "not mean the current route is the fastest.",
        NO_ARGS_SCHEMA, lambda: get_route_recommendations(sources))
    return registry
