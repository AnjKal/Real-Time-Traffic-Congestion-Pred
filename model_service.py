"""HTTP model service used by the Kafka-backed live pipeline.

The raw-topic consumer calls ``ForecastEngine.process_raw_batch`` after receiving a completed
poll. The resulting forecast is produced to ``traffic.predictions``. A separate consumer calls
``accept_prediction``; the HTTP API serves only forecasts received from that topic.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import copy
import os
import sys
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from collect_tomtom import COLUMNS
from kafka_topics import FORECAST_RECORD
from predict_bangalore import (BangaloreForecaster, build_window_from_readings, LOCAL_TZ,
                               STEP_MINUTES)

LIVE_DIR = os.path.join(PROJECT_ROOT, "data", "live")
READINGS_PATH = os.path.join(LIVE_DIR, "readings.csv")
FORECAST_DIR = os.path.join(LIVE_DIR, "forecasts")

TEST_MODE_WARNING = ("PIPELINE TEST: the model was trained on readings 15 minutes apart but is "
                     "being fed readings %d minutes apart. The plumbing is real, the numbers "
                     "are not a measure of model accuracy.")


class ForecastEngine(object):
    """Runs inference only for poll batches delivered by the raw-topic consumer."""

    def __init__(self, forecaster, readings_path=READINGS_PATH, forecast_dir=FORECAST_DIR):
        self.fc = forecaster
        self.readings_path = readings_path
        self.forecast_dir = forecast_dir
        self._lock = threading.RLock()
        self._window = None          # (slot_key, speeds, times, pred_all, warmup_steps)
        self._session_readings = []
        self._session_batches = []
        self._last_speed_by_junction = {}
        self._processed = OrderedDict()
        self._latest_prediction = None

    @property
    def step(self):
        return self.fc.step_minutes

    def current_window(self):
        """Return the last inference window; only ``process_raw_batch`` creates one."""
        with self._lock:
            if self._window is None:
                raise LookupError("waiting for a completed poll consumed from traffic.raw")
            return self._window[1:4]

    def process_raw_batch(self, poll_id, timestamp, readings, marker=None):
        """Archive one consumed raw poll, run inference, and return a Kafka forecast envelope."""
        poll_id = str(poll_id)
        with self._lock:
            if poll_id in self._processed:
                return copy.deepcopy(self._processed[poll_id])
            if not timestamp:
                timestamp = datetime.now(tz=timezone.utc).isoformat()
            actual = {}
            for row in readings:
                junction_id = str(row.get("junction_id", ""))
                if junction_id in self.fc.sensor_ids:
                    actual[junction_id] = row

            batch_rows = []
            imputed = 0
            for index, junction_id in enumerate(self.fc.sensor_ids):
                row = actual.get(junction_id)
                speed = None
                if row is not None:
                    try:
                        speed = float(row.get("currentSpeed"))
                    except (TypeError, ValueError):
                        speed = None
                if speed is None:
                    speed = self._last_speed_by_junction.get(
                        junction_id, float(self.fc.reference_kmh[index]))
                    imputed += 1
                else:
                    self._last_speed_by_junction[junction_id] = speed
                completed_row = {column: (row.get(column) if row else None)
                                 for column in COLUMNS}
                completed_row.update({"timestamp": timestamp, "junction_id": junction_id,
                                      "currentSpeed": speed})
                batch_rows.append(completed_row)
            self._archive_batch(batch_rows)
            self._session_readings.extend(batch_rows)
            max_rows = self.fc.seq_len * len(self.fc.sensor_ids)
            self._session_readings = self._session_readings[-max_rows:]
            self._session_batches.append(poll_id)
            self._session_batches = self._session_batches[-self.fc.seq_len:]
            self._window = None

            try:
                speeds, times, warmup_steps = build_window_from_readings(
                    pd.DataFrame(self._session_readings), self.fc.sensor_ids,
                    seq_len=self.fc.seq_len, step_minutes=self.step, allow_warmup=True,
                    fallback_speeds=self.fc.reference_kmh, return_warmup=True)
            except ValueError as exc:
                raise LookupError(str(exc))
            pred_all = self.fc.predict(speeds, times)
            self._window = (times[-1].isoformat(), speeds, times, pred_all, warmup_steps)
            self._save(times, speeds, pred_all)
            message = self._prediction_message(
                poll_id, timestamp, readings, marker or {}, imputed, speeds, times, pred_all,
                warmup_steps)
            self._processed[poll_id] = copy.deepcopy(message)
            while len(self._processed) > 24:
                self._processed.popitem(last=False)
            return message

    def _archive_batch(self, rows):
        folder = os.path.dirname(self.readings_path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        new_file = not os.path.exists(self.readings_path)
        with open(self.readings_path, "a", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=COLUMNS, extrasaction="ignore",
                                    lineterminator="\n")
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

    def _prediction_message(self, poll_id, timestamp, readings, marker, imputed,
                            speeds, times, pred_all, warmup_steps):
        made_at = datetime.now().isoformat(timespec="seconds")
        expected = max(1, len(self.fc.sensor_ids))
        actual_count = len(set(str(row.get("junction_id")) for row in readings
                                if row.get("currentSpeed") is not None))
        coverage = round(100.0 * actual_count / expected, 1)
        forecasts = {}
        for horizon_min in self.fc.horizon_minutes:
            result = self.fc.forecast_from_prediction(pred_all, speeds, times, horizon_min)
            result.update({"made_at": made_at, "step_minutes": self.step,
                           "horizons_min": self.fc.horizon_minutes,
                           "mode": "live" if self.step == STEP_MINUTES else "pipeline_test",
                           "input_window_steps": self.fc.seq_len,
                           "warmup_steps": int(warmup_steps),
                           "input_batches": len(self._session_batches),
                           "sensor_coverage_pct": coverage,
                           "imputed_junctions": int(imputed)})
            slot_end = pd.Timestamp(times[-1]) + pd.Timedelta(minutes=self.step)
            now = pd.Timestamp.now(tz=LOCAL_TZ)
            result["data_age_min"] = round(max(0.0, (now - slot_end).total_seconds() / 60.0), 1)
            if self.step != STEP_MINUTES:
                result["warning"] = TEST_MODE_WARNING % self.step
            if warmup_steps:
                result["warmup_warning"] = (
                    "Warm-up forecast: %d of %d model input steps are padded from the first "
                    "available readings." % (warmup_steps, self.fc.seq_len))
            forecasts[str(horizon_min)] = result
        return {"record_type": FORECAST_RECORD, "schema_version": 1,
                "poll_id": poll_id, "timestamp": timestamp, "made_at": made_at,
                "successful_count": actual_count,
                "expected_count": int(marker.get("expected_count", expected)),
                "sensor_coverage_pct": coverage, "imputed_junctions": int(imputed),
                "warmup_steps": int(warmup_steps),
                "input_batches": len(self._session_batches), "forecasts": forecasts}

    def accept_prediction(self, message):
        """Cache a forecast only after the predictions-topic consumer receives it."""
        if message.get("record_type") != FORECAST_RECORD:
            return
        with self._lock:
            self._latest_prediction = copy.deepcopy(message)

    def _latest(self):
        with self._lock:
            if self._latest_prediction is None:
                raise LookupError("waiting for a forecast consumed from traffic.predictions")
            return copy.deepcopy(self._latest_prediction)

    def _save(self, times, speeds, pred_all):
        if not os.path.isdir(self.forecast_dir):
            os.makedirs(self.forecast_dir)
        stamp = times[-1].strftime("%Y%m%dT%H%M%S%z")
        np.savez(os.path.join(self.forecast_dir, "forecast_%s_step%d.npz" % (stamp, self.step)),
                 pred=pred_all.astype(np.float32), speeds_in=speeds.astype(np.float32),
                 last_slot=np.array(times[-1].isoformat()), step_minutes=np.array(self.step),
                 horizon_minutes=np.array(self.fc.horizon_minutes))

    def forecast(self, horizon_min):
        message = self._latest()
        result = message.get("forecasts", {}).get(str(int(horizon_min)))
        if result is None:
            raise LookupError("no forecast for horizon %s in traffic.predictions" % horizon_min)
        return result

    def find_junction(self, query):
        """Row index of a junction given its id (J001) or a part of its name.
        Raises KeyError (no match) or ValueError (several matches)."""
        q = str(query).strip().lower()
        ids = [x.lower() for x in self.fc.sensor_ids]
        if q in ids:
            return ids.index(q)
        names = ([str(n).lower() for n in self.fc.spots["name"]]
                 if self.fc.spots is not None else [])
        exact = [i for i, n in enumerate(names) if n == q]
        if exact:
            return exact[0]
        hits = [i for i, n in enumerate(names) if q and q in n]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise KeyError("no junction matches '%s' (type 'list' to see them)" % query)
        raise ValueError("'%s' matches several junctions: %s" % (
            query, "; ".join(str(self.fc.spots.iloc[i]["name"]) for i in hits[:8])))

    def node_forecast(self, query):
        """Current forecast for ONE junction, all horizons, from the newest window.
        Raises LookupError if there is not enough live data yet."""
        i = self.find_junction(query)
        message = self._latest()
        first_horizon = str(self.fc.horizon_minutes[0])
        first = message["forecasts"][first_horizon]
        node = next((dict(item) for item in first["nodes"]
                     if item["junction_id"] == self.fc.sensor_ids[i]), None)
        if node is None:
            raise LookupError("junction missing from the latest prediction")
        node.update({"input_last_slot": first["input_last_slot"],
                     "data_age_min": first.get("data_age_min"),
                     "step_minutes": self.step, "mode": first.get("mode"),
                     "warmup_steps": first.get("warmup_steps"),
                     "sensor_coverage_pct": first.get("sensor_coverage_pct"),
                     "horizons": []})
        for horizon_min in self.fc.horizon_minutes:
            forecast = message["forecasts"][str(horizon_min)]
            result_node = next(item for item in forecast["nodes"]
                               if item["junction_id"] == self.fc.sensor_ids[i])
            node["horizons"].append({"minutes_ahead": horizon_min,
                                     "forecast_for": forecast["forecast_for"],
                                     "predicted_kmh": result_node["predicted_kmh"],
                                     "band": result_node["band"],
                                     "color": result_node["color"]})
        if self.fc.spots is not None:
            row = self.fc.spots.iloc[i]
            node.update({"name": str(row["name"]), "area": str(row["area"]),
                         "lat": float(row["lat"]), "lon": float(row["lon"])})
        if first.get("warning"):
            node["warning"] = first["warning"]
        if first.get("warmup_warning"):
            node["warmup_warning"] = first["warmup_warning"]
        return node

    def list_junctions(self):
        names = (list(self.fc.spots["name"]) if self.fc.spots is not None
                 else [None] * len(self.fc.sensor_ids))
        return [{"junction_id": j, "name": None if n is None else str(n)}
                for j, n in zip(self.fc.sensor_ids, names)]

    def health(self):
        info = {"status": "ok", "step_minutes": self.step,
                "mode": "live" if self.step == STEP_MINUTES else "pipeline_test",
                "horizons_min": self.fc.horizon_minutes, "ready": False,
                "slots_needed": 1}
        try:
            latest = self._latest()
            info.update({"ready": True, "newest_slot": latest.get("timestamp"),
                         "latest_poll_id": latest.get("poll_id"),
                         "warmup_steps": latest.get("warmup_steps"),
                         "input_batches": latest.get("input_batches"),
                         "sensor_coverage_pct": latest.get("sensor_coverage_pct")})
        except LookupError as exc:
            info["not_ready_reason"] = str(exc)
        return info


def create_app(engine=None, step_minutes=STEP_MINUTES, readings_path=READINGS_PATH,
               forecast_dir=FORECAST_DIR):
    """engine: a ready ForecastEngine (tests); otherwise the model is loaded on startup."""
    state = {"engine": engine}

    @asynccontextmanager
    async def lifespan(app):
        if state["engine"] is None:
            print("Loading model (about a minute)...", flush=True)
            state["engine"] = ForecastEngine(BangaloreForecaster(step_minutes=step_minutes),
                                             readings_path, forecast_dir)
            print("Model ready.", flush=True)
        yield

    app = FastAPI(title="Bangalore DCRNN model service", lifespan=lifespan)

    @app.get("/health")
    def health():
        return state["engine"].health()

    @app.get("/forecast")
    def forecast(horizon_min=None):
        eng = state["engine"]
        horizons = eng.fc.horizon_minutes
        if horizon_min is None:
            horizon_min = horizons[0]
        try:
            horizon_min = int(horizon_min)
        except ValueError:
            raise HTTPException(400, "horizon_min must be a whole number")
        if horizon_min not in horizons:
            raise HTTPException(400, "horizon_min must be one of %s" % horizons)
        try:
            return eng.forecast(horizon_min)
        except LookupError as e:
            raise HTTPException(503, "Not enough live data yet: %s" % e)

    @app.get("/nodes")
    def nodes():
        return state["engine"].list_junctions()

    @app.get("/node")
    def node(name):
        """Current forecast for one junction: /node?name=J001 or /node?name=silk board"""
        try:
            return state["engine"].node_forecast(name)
        except KeyError as e:
            raise HTTPException(404, e.args[0])
        except ValueError as e:
            raise HTTPException(400, str(e))
        except LookupError as e:
            raise HTTPException(503, "Not enough live data yet: %s" % e)

    return app


def main():
    from start_pipeline import main as start_live_pipeline
    return start_live_pipeline()


if __name__ == "__main__":
    main()
