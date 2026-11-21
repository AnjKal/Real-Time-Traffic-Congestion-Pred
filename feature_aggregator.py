"""Area-wise feature aggregation for the live traffic pipeline (traffic.features topic).

Runs inside the pipeline process (Python 3.7). It joins, per polling cycle:

  * traffic.raw         -> TomTom currentSpeed and freeFlowSpeed per junction (real readings)
  * traffic.predictions -> DCRNN predicted speed and congestion band per junction

by ``poll_id``, aggregates the junctions into their areas (from bangalore_traffic_spots.csv), and
publishes one ``area_features`` envelope to ``traffic.features``. The latest envelope is also
written to ``data/live/features.json`` so the web app (a separate process) can serve it without a
Kafka client.

Design notes / safety:
  * Dedicated consumer groups so the existing DCRNN raw consumer and dashboard consumer are never
    disturbed (see start_pipeline.py).
  * Readings and predictions are joined strictly on the same ``poll_id`` -- cycles are never mixed.
  * Duplicate predictions for a poll already processed are ignored.
  * Missing readings / incomplete polls only reduce an area's reporting count; they do not crash.
  * ``reference_kmh`` (the model's historical reference speed) is NEVER treated as free-flow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import threading
from collections import OrderedDict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SPOTS_PATH = os.path.join(PROJECT_ROOT, "bangalore_traffic_spots.csv")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "live", "features.json")

SCHEMA_VERSION = 1
BANDS = ("free", "moderate", "heavy")
_SEVERITY = {"free": 0, "moderate": 1, "heavy": 2}

LOG = logging.getLogger(__name__)


def build_total_by_area(spots_path=SPOTS_PATH):
    """{area_name: number of modeled junctions in that area} from the spots file."""
    import pandas as pd
    spots = pd.read_csv(spots_path)
    counts = spots["area"].astype(str).value_counts()
    return {str(area): int(n) for area, n in counts.items()}


def _mean(values):
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 2) if valid else None


def _num(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):   # NaN / inf
        return None
    return result


def _delay_min_per_km(predicted_kmh, free_flow_kmh):
    """Extra minutes to drive 1 km at the predicted speed versus TomTom free-flow. None if either
    speed is missing/non-positive; never negative. Mirrors the segment hover-card formula per km."""
    if predicted_kmh is None or free_flow_kmh is None or predicted_kmh <= 0 or free_flow_kmh <= 0:
        return None
    return round(max(0.0, 60.0 / predicted_kmh - 60.0 / free_flow_kmh), 3)


def aggregate_area_features(poll_id, timestamp, junctions, total_by_area, meta=None):
    """Aggregate already-joined per-junction rows into the traffic.features envelope.

    junctions: list of dicts with keys ``junction_id``, ``area``, ``current_kmh``,
        ``free_flow_kmh``, ``predicted_kmh``, ``band``, ``reporting`` (bool: had a real TomTom
        currentSpeed this poll).
    total_by_area: {area: total modeled junction count}.
    meta: optional {made_at, step_minutes, data_age_min} passed through from the prediction envelope.
    """
    meta = meta or {}
    by_area = OrderedDict()
    for row in junctions:
        area = str(row.get("area") or "Unknown")
        by_area.setdefault(area, []).append(row)

    areas = []
    total_junctions = 0
    total_reporting = 0
    for area in sorted(by_area):
        rows = by_area[area]
        reporting = [r for r in rows if r.get("reporting")]
        distribution = {band: 0 for band in BANDS}
        for r in rows:
            band = r.get("band")
            if band in distribution:
                distribution[band] += 1
        total = int(total_by_area.get(area, len(rows)))
        total_junctions += total
        total_reporting += len(reporting)

        # Overall level: the band with the most junctions; ties resolve to the more severe band.
        if any(distribution.values()):
            level = max(BANDS, key=lambda b: (distribution[b], _SEVERITY[b]))
        else:
            level = "unknown"

        delays = [_delay_min_per_km(_num(r.get("predicted_kmh")), _num(r.get("free_flow_kmh")))
                  for r in rows]
        areas.append({
            "area": area,
            "total_junction_count": total,
            "reporting_junction_count": len(reporting),
            "sensor_coverage_pct": round(100.0 * len(reporting) / total, 1) if total else 0.0,
            "avg_current_speed_kmh": _mean([_num(r.get("current_kmh")) for r in reporting]),
            "avg_free_flow_speed_kmh": _mean([_num(r.get("free_flow_kmh")) for r in rows]),
            "avg_predicted_speed_kmh": _mean([_num(r.get("predicted_kmh")) for r in rows]),
            "congestion_distribution": distribution,
            "overall_congestion_level": level,
            "avg_delay_min_per_km": _mean(delays),
        })

    coverage = round(100.0 * total_reporting / total_junctions, 1) if total_junctions else 0.0
    return {
        "record_type": "area_features",
        "schema_version": SCHEMA_VERSION,
        "poll_id": str(poll_id),
        "timestamp": timestamp,
        "made_at": meta.get("made_at") or datetime.now().isoformat(timespec="seconds"),
        "step_minutes": meta.get("step_minutes"),
        "data_age_min": meta.get("data_age_min"),
        "sensor_coverage_pct": coverage,
        "area_count": len(areas),
        "areas": areas,
    }


class FeatureAggregator(object):
    """Joins raw readings and predictions per poll and emits area_features.

    producer: an object with ``send(value, key=None)`` (the pipeline's JsonProducer for
        traffic.features). May be None in tests, in which case only features.json is written.
    """

    def __init__(self, total_by_area, producer=None, out_path=FEATURES_PATH, max_tracked=64):
        self.total_by_area = total_by_area
        self.producer = producer
        self.out_path = out_path
        self.max_tracked = int(max_tracked)
        self._lock = threading.RLock()
        self._raw_by_poll = OrderedDict()      # poll_id -> {junction_id: {current, free_flow}}
        self._processed = OrderedDict()         # poll_id -> True (dedup)

    # -- traffic.raw: one completed poll (reuses RawPollAssembler in start_pipeline) ----------
    def handle_raw_batch(self, poll_id, timestamp, readings, marker=None):
        poll_id = str(poll_id)
        per_junction = {}
        for row in readings or []:
            junction_id = str(row.get("junction_id", ""))
            if not junction_id:
                continue
            per_junction[junction_id] = {
                "current": _num(row.get("currentSpeed")),
                "free_flow": _num(row.get("freeFlowSpeed")),
            }
        with self._lock:
            self._raw_by_poll[poll_id] = per_junction
            while len(self._raw_by_poll) > self.max_tracked:
                self._raw_by_poll.popitem(last=False)
        return True

    # -- traffic.predictions: a forecast envelope -> aggregate + emit -------------------------
    def handle_prediction(self, message):
        if message.get("record_type") != "forecast":
            return None
        poll_id = str(message.get("poll_id", ""))
        if not poll_id:
            LOG.warning("Ignoring prediction without a poll_id for feature aggregation")
            return None
        with self._lock:
            if poll_id in self._processed:
                return None                      # duplicate prediction for this poll
            raw = self._raw_by_poll.get(poll_id, {})

        nodes = self._first_horizon_nodes(message)
        if not nodes:
            LOG.warning("Prediction for poll %s has no nodes; skipping features", poll_id)
            return None

        junctions = []
        for node in nodes:
            junction_id = str(node.get("junction_id", ""))
            raw_row = raw.get(junction_id, {})
            current_raw = raw_row.get("current")
            junctions.append({
                "junction_id": junction_id,
                "area": node.get("area"),
                # Prefer the real TomTom reading; fall back to the model's current_kmh if the
                # junction had no reading this poll, but mark it as not reporting.
                "current_kmh": current_raw if current_raw is not None else _num(node.get("current_kmh")),
                "free_flow_kmh": raw_row.get("free_flow"),
                "predicted_kmh": _num(node.get("predicted_kmh")),
                "band": node.get("band"),
                "reporting": current_raw is not None,
            })

        meta = {"made_at": message.get("made_at"),
                "step_minutes": message.get("forecasts", {}).get(
                    self._first_horizon_key(message), {}).get("step_minutes"),
                "data_age_min": message.get("forecasts", {}).get(
                    self._first_horizon_key(message), {}).get("data_age_min")}
        envelope = aggregate_area_features(poll_id, message.get("timestamp"), junctions,
                                           self.total_by_area, meta)
        with self._lock:
            self._processed[poll_id] = True
            while len(self._processed) > self.max_tracked:
                self._processed.popitem(last=False)
            self._raw_by_poll.pop(poll_id, None)
        self._emit(envelope)
        return envelope

    @staticmethod
    def _first_horizon_key(message):
        forecasts = message.get("forecasts", {})
        if not forecasts:
            return None
        return sorted(forecasts, key=lambda k: int(k))[0]

    def _first_horizon_nodes(self, message):
        key = self._first_horizon_key(message)
        if key is None:
            return []
        return message["forecasts"][key].get("nodes", [])

    def _emit(self, envelope):
        if self.producer is not None:
            try:
                self.producer.send(envelope, key=envelope["poll_id"])
            except Exception:
                LOG.exception("Failed to publish area features for poll %s", envelope["poll_id"])
        self._write_latest(envelope)

    def _write_latest(self, envelope):
        folder = os.path.dirname(self.out_path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        tmp = self.out_path + ".tmp"
        try:
            with open(tmp, "w") as handle:
                json.dump(envelope, handle, separators=(",", ":"), allow_nan=False)
            os.replace(tmp, self.out_path)        # atomic so the web app never reads a half file
        except OSError:
            LOG.exception("Could not write %s", self.out_path)
