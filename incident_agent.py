"""Detect explainable traffic congestion events from live readings and forecasts.

This is a deterministic Kafka worker, not an LLM classifier. It consumes raw polls and forecast
envelopes on dedicated groups, joins them by poll_id, tracks event state, and publishes keyed
updates to traffic.congestion_events. TomTom free-flow is the primary speed baseline; the
forecaster's historical reference speed is deliberately not used as a substitute.
"""
from __future__ import absolute_import

import hashlib
import copy
import json
import logging
import math
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone

from kafka_topics import CONGESTION_EVENT_RECORD

LOG = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LIVE_DIR = os.path.join(PROJECT_ROOT, "data", "live")
SNAPSHOT_PATH = os.path.join(LIVE_DIR, "incidents.json")
STATE_PATH = os.path.join(LIVE_DIR, "incident_state.json")

# Initial, explicit operating thresholds. These are signal rules, not calibrated probabilities.
ENTRY_RATIO = 0.75
EXIT_RATIO = 0.82
TREND_DROP_RATIO = 0.10
FORECAST_WATCH_RATIO = 0.75
NEIGHBOR_RADIUS_KM = 2.5
PERSISTENCE_POLLS = 2
MAX_PENDING_POLLS = 8
HISTORY_POLLS = 8
MAX_RECENT_RESOLVED = 20


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _mean(values):
    clean = [value for value in values if value is not None]
    return round(sum(clean) / len(clean), 2) if clean else None


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _distance_km(a, b):
    if (a.get("lat") is None or a.get("lon") is None
            or b.get("lat") is None or b.get("lon") is None):
        return None
    lat1, lon1 = math.radians(a["lat"]), math.radians(a["lon"])
    lat2, lon2 = math.radians(b["lat"]), math.radians(b["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = (math.sin(dlat / 2.0) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2)
    return 6371.0 * 2.0 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1.0 - value)))


def _write_json_atomic(path, value):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, separators=(",", ":"), allow_nan=False)
    os.replace(temporary, path)


class IncidentAgent(object):
    """Join completed raw polls to forecasts and publish congestion event state updates.

    producer must provide send(value, key=None). Consumers are wired by start_pipeline.py on
    groups separate from the model and feature aggregator.
    """

    def __init__(self, producer, snapshot_path=SNAPSHOT_PATH, state_path=STATE_PATH,
                 max_pending=MAX_PENDING_POLLS):
        self.producer = producer
        self.snapshot_path = snapshot_path
        self.state_path = state_path
        self.max_pending = int(max_pending)
        self._lock = threading.RLock()
        self._raw_by_poll = OrderedDict()
        self._prediction_by_poll = OrderedDict()
        self._history = {}
        self._active = {}
        self._recent_resolved = []
        self._processed = []
        self._latest_snapshot = None
        self._load_state()

    def _load_state(self):
        try:
            with open(self.state_path, "r") as handle:
                state = json.load(handle)
        except (IOError, ValueError):
            return
        self._history = state.get("history_by_junction", {}) or {}
        self._active = state.get("active_events", {}) or {}
        self._recent_resolved = state.get("recent_resolved", []) or []
        self._processed = (state.get("processed_poll_ids", []) or [])[-128:]
        self._raw_by_poll = OrderedDict(state.get("pending_raw_by_poll", {}) or {})
        self._prediction_by_poll = OrderedDict(
            state.get("pending_prediction_by_poll", {}) or {})
        try:
            with open(self.snapshot_path, "r") as handle:
                self._latest_snapshot = json.load(handle)
        except (IOError, ValueError):
            self._latest_snapshot = None

    def handle_raw_batch(self, poll_id, timestamp, readings, marker=None):
        """Callback for RawPollAssembler; returns only after the raw side has been joined."""
        poll_id = str(poll_id)
        with self._lock:
            self._raw_by_poll[poll_id] = {
                "timestamp": timestamp,
                "readings": {str(row.get("junction_id")): row for row in (readings or [])
                             if row.get("junction_id")},
                "marker": marker or {},
            }
            self._persist_state()
            self._drain_pending()
            self._trim_pending()
        return True

    def handle_prediction(self, message):
        if message.get("record_type") != "forecast":
            return None
        poll_id = str(message.get("poll_id") or "")
        if not poll_id:
            LOG.warning("Ignoring forecast without poll_id in incident agent")
            return None
        with self._lock:
            self._prediction_by_poll[poll_id] = message
            self._persist_state()
            self._drain_pending()
            self._trim_pending()
        return True

    def _trim_pending(self):
        changed = False
        while len(self._raw_by_poll) > self.max_pending:
            self._raw_by_poll.popitem(last=False)
            changed = True
        while len(self._prediction_by_poll) > self.max_pending:
            self._prediction_by_poll.popitem(last=False)
            changed = True
        if changed:
            self._persist_state()

    def _try_process(self, poll_id):
        if poll_id in self._processed:
            self._raw_by_poll.pop(poll_id, None)
            self._prediction_by_poll.pop(poll_id, None)
            self._persist_state()
            return
        raw = self._raw_by_poll.get(poll_id)
        prediction = self._prediction_by_poll.get(poll_id)
        if raw is None or prediction is None:
            return
        before = (copy.deepcopy(self._history), copy.deepcopy(self._active),
                  copy.deepcopy(self._recent_resolved))
        try:
            self._process_poll(poll_id, raw, prediction)
        except Exception:
            self._history, self._active, self._recent_resolved = before
            raise
        self._raw_by_poll.pop(poll_id, None)
        self._prediction_by_poll.pop(poll_id, None)
        self._processed.append(poll_id)
        self._processed = self._processed[-128:]
        self._persist_state()

    def _drain_pending(self):
        for poll_id in list(self._raw_by_poll):
            if poll_id in self._prediction_by_poll:
                self._try_process(poll_id)

    @staticmethod
    def _forecast_views(message):
        forecasts = message.get("forecasts") or {}
        keys = sorted(forecasts, key=lambda key: int(key))
        if not keys:
            return None, None, None, None
        first = forecasts[keys[0]]
        later = None
        later_key = None
        for key in keys:
            if int(key) > int(keys[0]):
                later = forecasts[key]
                later_key = key
                break
        return first, later, int(keys[0]), int(later_key) if later_key is not None else None

    def _process_poll(self, poll_id, raw, message):
        first, later, forecast_horizon_min, next_forecast_horizon_min = self._forecast_views(message)
        if not first:
            LOG.warning("Forecast for poll %s has no horizons; skipping incidents", poll_id)
            return
        first_nodes = {str(node.get("junction_id")): node
                       for node in first.get("nodes", []) if node.get("junction_id")}
        later_nodes = {}
        if later:
            later_nodes = {str(node.get("junction_id")): node
                           for node in later.get("nodes", []) if node.get("junction_id")}

        timestamp = raw.get("timestamp") or message.get("timestamp") or _utc_now()
        made_at = _utc_now()
        step_minutes = int(first.get("step_minutes") or 0)
        model_mode = first.get("mode") or "unknown"
        coverage = _number(message.get("sensor_coverage_pct"))
        warmup_steps = int(first.get("warmup_steps") or 0)
        # Use every forecast the model emits, including padded warm-up forecasts and the
        # configured 2-minute polling mode.
        forecast_used = bool(first_nodes)
        data_age = _number(first.get("data_age_min"))
        readings = raw.get("readings", {})

        signals = {}
        for junction_id, node in first_nodes.items():
            row = readings.get(junction_id, {})
            current = _number(row.get("currentSpeed"))
            if current is not None and current < 0:
                current = None
            free_flow = _number(row.get("freeFlowSpeed"))
            if free_flow is not None and free_flow <= 0:
                free_flow = None
            current_ratio = (current / free_flow if current is not None and free_flow else None)
            predicted = _number(node.get("predicted_kmh"))
            predicted_ratio = (predicted / free_flow
                               if predicted is not None and free_flow else None)
            later_node = later_nodes.get(junction_id, {})
            predicted_later = _number(later_node.get("predicted_kmh"))
            predicted_later_ratio = (predicted_later / free_flow
                                     if predicted_later is not None and free_flow else None)
            history = self._history.get(junction_id, [])
            previous = history[-1] if history else {}
            previous_ratio = _number(previous.get("current_ratio"))
            trend_drop = (previous_ratio - current_ratio
                          if previous_ratio is not None and current_ratio is not None else None)
            consecutive_low = 1 if current_ratio is not None and current_ratio <= ENTRY_RATIO else 0
            for entry in reversed(history):
                old_ratio = _number(entry.get("current_ratio"))
                if old_ratio is not None and old_ratio <= ENTRY_RATIO:
                    consecutive_low += 1
                else:
                    break
            signals[junction_id] = {
                "junction_id": junction_id,
                "name": node.get("name") or row.get("junction_name") or junction_id,
                "area": node.get("area") or "Unknown",
                "lat": _number(node.get("lat")),
                "lon": _number(node.get("lon")),
                "current_kmh": current,
                "free_flow_kmh": free_flow,
                "current_ratio": current_ratio,
                "predicted_kmh": predicted,
                "predicted_ratio": predicted_ratio,
                "predicted_next_kmh": predicted_later,
                "predicted_next_ratio": predicted_later_ratio,
                "trend_drop_ratio": trend_drop,
                "consecutive_low_polls": consecutive_low,
                "confidence": _number(row.get("confidence")),
                "road_closure": row.get("roadClosure"),
                "reporting": current is not None,
            }

        for junction_id, signal in signals.items():
            history = self._history.setdefault(junction_id, [])
            history.append({"current_ratio": signal["current_ratio"], "timestamp": timestamp})
            self._history[junction_id] = history[-HISTORY_POLLS:]

        low_now = [signal for signal in signals.values()
                   if signal["current_ratio"] is not None
                   and signal["current_ratio"] <= ENTRY_RATIO]
        for signal in signals.values():
            nearby_low = []
            for other in low_now:
                if other["junction_id"] == signal["junction_id"]:
                    continue
                distance = _distance_km(signal, other)
                if distance is not None and distance <= NEIGHBOR_RADIUS_KM:
                    nearby_low.append(other["junction_id"])
            signal["nearby_slow_junction_ids"] = nearby_low
            signal["observed_candidate"] = bool(
                signal["current_ratio"] is not None
                and signal["current_ratio"] <= ENTRY_RATIO
                and (signal["consecutive_low_polls"] >= PERSISTENCE_POLLS
                     or len(nearby_low) >= 1))
            trend_supported = (signal["trend_drop_ratio"] is not None
                               and signal["trend_drop_ratio"] >= TREND_DROP_RATIO
                               and (len(nearby_low) >= 1
                                    or (forecast_used
                                        and signal["predicted_ratio"] is not None
                                        and signal["predicted_ratio"] <= 0.90)))
            forecast_candidate = bool(
                forecast_used and signal["predicted_ratio"] is not None
                and signal["predicted_ratio"] <= FORECAST_WATCH_RATIO)
            mild_forecast_candidate = bool(
                forecast_used and signal["current_ratio"] is not None
                and signal["predicted_ratio"] is not None
                and signal["current_ratio"] - signal["predicted_ratio"] >= 0.10
                and signal["predicted_ratio"] <= 0.85)
            signal["forecast_candidate"] = forecast_candidate or mild_forecast_candidate
            signal["trend_candidate"] = bool(trend_supported and not signal["observed_candidate"])

        candidates = {}
        for junction_id, signal in signals.items():
            if not (signal["observed_candidate"] or signal["forecast_candidate"]
                    or signal["trend_candidate"]):
                continue
            evidence = []
            ratio = signal["current_ratio"]
            if signal["observed_candidate"] and ratio is not None:
                evidence.append("observed_speed_below_tomtom_free_flow")
                if signal["consecutive_low_polls"] >= PERSISTENCE_POLLS:
                    evidence.append("slowdown_persisted_across_polls")
                if signal["nearby_slow_junction_ids"]:
                    evidence.append("nearby_junctions_also_slow")
            if signal["forecast_candidate"]:
                evidence.append("forecast_speed_declines_against_tomtom_free_flow")
            if signal["trend_candidate"]:
                evidence.append("current_speed_fell_since_previous_poll")
            signal["evidence"] = evidence
            candidates[junction_id] = signal

        components = self._components(candidates)
        now_active = []
        used_old_ids = set()
        matched_pairs = self._match_components(components)
        for component_index, member_ids in enumerate(components):
            old_id = matched_pairs.get(component_index)
            old = self._active.get(old_id, {}) if old_id else {}
            if old_id:
                used_old_ids.add(old_id)
            event = self._make_event(
                member_ids, candidates, signals, old, poll_id, timestamp, made_at,
                step_minutes, model_mode, coverage, data_age, forecast_used,
                forecast_horizon_min, next_forecast_horizon_min, warmup_steps)
            now_active.append(event)

        resolved_records = []
        for event_id, old in list(self._active.items()):
            if event_id in used_old_ids:
                continue
            affected_ids = old.get("affected_junction_ids", [])
            observed = [signals.get(junction_id) for junction_id in affected_ids]
            observed = [signal for signal in observed if signal]
            reporting = [signal for signal in observed if signal.get("current_ratio") is not None]
            if not reporting:
                stale = dict(old)
                stale.update({"data_quality": "stale", "updated_at": made_at,
                              "source_poll_id": poll_id})
                now_active.append(stale)
                continue
            if len(reporting) < len(affected_ids):
                stale = dict(old)
                stale.update({"data_quality": "stale", "updated_at": made_at,
                              "source_poll_id": poll_id})
                now_active.append(stale)
                continue
            still_slow = [signal for signal in reporting
                          if signal["current_ratio"] <= EXIT_RATIO]
            if still_slow:
                lingering = dict(old)
                lingering.update({"status": "active" if old.get("status") == "active"
                                  else "watch", "clear_polls": 0,
                                  "updated_at": made_at, "source_poll_id": poll_id,
                                  "data_quality": "limited" if len(reporting) < len(affected_ids)
                                  else "good"})
                now_active.append(lingering)
                continue
            clear_polls = int(old.get("clear_polls", 0)) + 1
            if clear_polls >= PERSISTENCE_POLLS:
                resolved = dict(old)
                resolved.update({"status": "resolved", "clear_polls": clear_polls,
                                 "updated_at": made_at, "source_poll_id": poll_id,
                                 "evidence": ["speeds_returned_above_recovery_threshold"]})
                resolved_records.append(resolved)
                self._recent_resolved.insert(0, resolved)
                self._recent_resolved = self._recent_resolved[:MAX_RECENT_RESOLVED]
            else:
                recovering = dict(old)
                recovering.update({"status": "recovering", "clear_polls": clear_polls,
                                   "updated_at": made_at, "source_poll_id": poll_id})
                now_active.append(recovering)

        active_map = {event["event_id"]: event for event in now_active}
        for event in now_active + resolved_records:
            outbound = dict(event)
            outbound.update({"record_type": CONGESTION_EVENT_RECORD, "schema_version": 1})
            self.producer.send(outbound, key=event["event_id"])

        snapshot = {
            "record_type": "incident_snapshot",
            "schema_version": 1,
            "poll_id": poll_id,
            "timestamp": timestamp,
            "made_at": made_at,
            "step_minutes": step_minutes,
            "model_mode": model_mode,
            "forecast_used": forecast_used,
            "sensor_coverage_pct": coverage,
            "data_age_min": data_age,
            "active_events": sorted(active_map.values(),
                                    key=lambda item: (item.get("status") != "active",
                                                      item.get("severity") != "high",
                                                      item.get("first_seen", ""))),
            "recent_resolved": self._recent_resolved[:MAX_RECENT_RESOLVED],
        }
        _write_json_atomic(self.snapshot_path, snapshot)
        self._active = active_map
        self._latest_snapshot = snapshot
        LOG.info("Incident agent processed poll %s: %d active/watch, %d resolved",
                 poll_id, len(active_map), len(resolved_records))

    def _components(self, candidates):
        remaining = set(candidates)
        components = []
        while remaining:
            seed = remaining.pop()
            component = {seed}
            frontier = [seed]
            while frontier:
                current_id = frontier.pop()
                current = candidates[current_id]
                adjacent = []
                for other_id in list(remaining):
                    other = candidates[other_id]
                    distance = _distance_km(current, other)
                    same_area = current.get("area") == other.get("area")
                    if ((distance is not None and distance <= NEIGHBOR_RADIUS_KM)
                            or (distance is None and same_area)):
                        adjacent.append(other_id)
                for other_id in adjacent:
                    remaining.remove(other_id)
                    component.add(other_id)
                    frontier.append(other_id)
            components.append(sorted(component))
        return components

    def _match_components(self, components):
        pairs = []
        for index, member_ids in enumerate(components):
            current_ids = set(member_ids)
            for event_id, old in self._active.items():
                old_ids = set(old.get("affected_junction_ids", []))
                overlap = len(current_ids & old_ids)
                if overlap:
                    pairs.append((overlap, float(overlap) / max(1, len(current_ids | old_ids)),
                                  index, event_id))
        pairs.sort(reverse=True)
        matched_components, matched_events = {}, set()
        for _, _, index, event_id in pairs:
            if index not in matched_components and event_id not in matched_events:
                matched_components[index] = event_id
                matched_events.add(event_id)
        return matched_components

    def _make_event(self, member_ids, candidates, signals, old, poll_id, timestamp,
                    made_at, step_minutes, model_mode, coverage, data_age,
                    forecast_used, forecast_horizon_min, next_forecast_horizon_min,
                    warmup_steps):
        members = [candidates[junction_id] for junction_id in member_ids]
        observed = [signal for signal in members if signal["observed_candidate"]]
        forecast = [signal for signal in members if signal["forecast_candidate"]]
        trend = [signal for signal in members if signal["trend_candidate"]]
        if observed and forecast:
            event_type = "congestion_build_up"
        elif observed:
            event_type = "observed_slowdown"
        elif trend:
            event_type = "congestion_build_up"
        else:
            event_type = "forecast_congestion"
        status = "active" if observed else "watch"
        current_ratios = [signal["current_ratio"] for signal in members
                          if signal["current_ratio"] is not None]
        predicted_ratios = [signal["predicted_ratio"] for signal in members
                            if signal["predicted_ratio"] is not None]
        low_ratio = min(current_ratios) if current_ratios else 1.0
        severity = ("high" if low_ratio <= 0.50 or len(member_ids) >= 4
                    else "moderate" if observed else "low")
        areas = {}
        for signal in members:
            area = signal.get("area") or "Unknown"
            areas[area] = areas.get(area, 0) + 1
        area = max(areas, key=areas.get) if areas else "Unknown"
        lat = _mean([signal.get("lat") for signal in members])
        lon = _mean([signal.get("lon") for signal in members])
        evidence = []
        for signal in members:
            for reason in signal.get("evidence", []):
                if reason not in evidence:
                    evidence.append(reason)
        if len(members) > 1:
            evidence.append("multiple_nearby_junctions_affected")
        if coverage is not None and coverage < 70:
            evidence.append("sensor_coverage_below_70_percent")
        if not old:
            anchor = member_ids[0]
            digest = hashlib.sha1((str(poll_id) + ":" + anchor).encode("utf-8")).hexdigest()[:10]
            event_id = "ce-%s-%s" % (anchor.lower(), digest)
            first_seen = timestamp
        else:
            event_id = old["event_id"]
            first_seen = old.get("first_seen") or timestamp
        observed_speeds = [signal["current_kmh"] for signal in members]
        free_flow_speeds = [signal["free_flow_kmh"] for signal in members]
        predicted_speeds = [signal["predicted_kmh"] for signal in members]
        predicted_next = [signal["predicted_next_kmh"] for signal in members]
        support_count = len(set().union(*[
            set(signal.get("nearby_slow_junction_ids", [])) for signal in members]))
        if coverage is not None and coverage >= 80 and (support_count > 0 or len(members) >= 2):
            signal_quality = "high"
        elif observed or trend:
            signal_quality = "medium"
        else:
            signal_quality = "low"
        affected = []
        for signal in members:
            affected.append({
                "junction_id": signal["junction_id"],
                "name": signal["name"],
                "area": signal["area"],
                "lat": signal["lat"],
                "lon": signal["lon"],
                "current_speed_kmh": signal["current_kmh"],
                "free_flow_speed_kmh": signal["free_flow_kmh"],
                "predicted_speed_kmh": signal["predicted_kmh"],
                "predicted_speed_horizon_min": forecast_horizon_min,
                "predicted_next_speed_kmh": signal["predicted_next_kmh"],
                "predicted_next_horizon_min": next_forecast_horizon_min,
                "speed_ratio_current": (round(signal["current_ratio"], 3)
                                        if signal["current_ratio"] is not None else None),
                "speed_ratio_predicted": (round(signal["predicted_ratio"], 3)
                                              if signal["predicted_ratio"] is not None else None),
                "road_closure_reported": signal["road_closure"],
            })
        return {
            "event_id": event_id,
            "event_type": event_type,
            "status": status,
            "severity": severity,
            "area": area,
            "anchor_junction_id": members[0]["junction_id"],
            "anchor_name": members[0]["name"],
            "lat": lat,
            "lon": lon,
            "affected_junction_ids": member_ids,
            "affected_junctions": affected,
            "affected_junction_count": len(member_ids),
            "avg_current_speed_kmh": _mean(observed_speeds),
            "avg_free_flow_speed_kmh": _mean(free_flow_speeds),
            "avg_predicted_speed_kmh": _mean(predicted_speeds),
            "forecast_horizon_min": forecast_horizon_min,
            "avg_predicted_next_speed_kmh": _mean(predicted_next),
            "next_forecast_horizon_min": next_forecast_horizon_min,
            "speed_ratio_current": (round(_mean(current_ratios), 3)
                                    if current_ratios else None),
            "speed_ratio_predicted": (round(_mean(predicted_ratios), 3)
                                          if predicted_ratios else None),
            "evidence": evidence,
            "signal_quality": signal_quality,
            "data_quality": "limited" if coverage is not None and coverage < 70 else "good",
            "sensor_coverage_pct": coverage,
            "data_age_min": data_age,
            "forecast_used": bool(forecast_used),
            "step_minutes": step_minutes,
            "warmup_steps": warmup_steps,
            "model_mode": model_mode,
            "first_seen": first_seen,
            "last_seen": timestamp,
            "updated_at": made_at,
            "source_poll_id": poll_id,
            "clear_polls": 0,
        }

    def _persist_state(self):
        state = {
            "schema_version": 1,
            "history_by_junction": self._history,
            "active_events": self._active,
            "recent_resolved": self._recent_resolved[:MAX_RECENT_RESOLVED],
            "processed_poll_ids": self._processed,
            "pending_raw_by_poll": self._raw_by_poll,
            "pending_prediction_by_poll": self._prediction_by_poll,
        }
        _write_json_atomic(self.state_path, state)
