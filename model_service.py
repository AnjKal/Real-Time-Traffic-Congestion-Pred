"""Model service (Python 3.7 / TensorFlow 1.15): live readings in, forecasts out.

    python model_service.py                       # real mode: readings 15 minutes apart
    python model_service.py --step-min 2          # PIPELINE TEST: readings 2 minutes apart

It reads the readings file written by collect_tomtom.py, builds the model's 12-step input
window from the newest complete slots, runs the DCRNN once per new window, and serves:

    GET /health                       service state and how much history exists
    GET /forecast?horizon_min=30      JSON forecast for all 100 junctions, one horizon

Every new window's full 12-step forecast is also saved to data/live/forecasts/ so that
evaluate_live.py can later compare it with what TomTom actually reported.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from predict_bangalore import (BangaloreForecaster, build_window_from_readings, LOCAL_TZ,
                               STEP_MINUTES, BAND_COLORS, SPEED_SCALE_KMH, speed_band)

LIVE_DIR = os.path.join(PROJECT_ROOT, "data", "live")
READINGS_PATH = os.path.join(LIVE_DIR, "readings.csv")
FORECAST_DIR = os.path.join(LIVE_DIR, "forecasts")

TEST_MODE_WARNING = ("PIPELINE TEST: the model was trained on readings 15 minutes apart but is "
                     "being fed readings %d minutes apart. The plumbing is real, the numbers "
                     "are not a measure of model accuracy.")


class ForecastEngine(object):
    """Turns the readings file into (cached) forecasts. Separate from FastAPI so it is testable."""

    def __init__(self, forecaster, readings_path=READINGS_PATH, forecast_dir=FORECAST_DIR):
        self.fc = forecaster
        self.readings_path = readings_path
        self.forecast_dir = forecast_dir
        self._lock = threading.Lock()
        self._window = None          # (slot_key, speeds, times, pred_all)
        self._file_sig = None

    @property
    def step(self):
        return self.fc.step_minutes

    def _read_readings(self):
        if not os.path.exists(self.readings_path):
            raise LookupError("no readings yet (%s does not exist). Start collect_tomtom.py."
                              % self.readings_path)
        return pd.read_csv(self.readings_path, usecols=["timestamp", "junction_id", "currentSpeed"])

    def current_window(self):
        """(speeds, times, pred_all) for the newest complete window; recomputed only when the
        readings file changed. Raises LookupError with a human-readable reason if not ready."""
        with self._lock:
            sig = None
            if os.path.exists(self.readings_path):
                st = os.stat(self.readings_path)
                sig = (st.st_mtime, st.st_size)
            if self._window is not None and sig == self._file_sig:
                return self._window[1:]
            readings = self._read_readings()
            try:
                speeds, times = build_window_from_readings(
                    readings, self.fc.sensor_ids, seq_len=self.fc.seq_len,
                    step_minutes=self.step)
            except ValueError as e:
                raise LookupError(str(e))
            key = times[-1].isoformat()
            if self._window is None or self._window[0] != key:
                pred_all = self.fc.predict(speeds, times)
                self._window = (key, speeds, times, pred_all)
                self._save(times, speeds, pred_all)
            self._file_sig = sig
            return self._window[1:]

    def _save(self, times, speeds, pred_all):
        if not os.path.isdir(self.forecast_dir):
            os.makedirs(self.forecast_dir)
        stamp = times[-1].strftime("%Y%m%dT%H%M%S%z")
        np.savez(os.path.join(self.forecast_dir, "forecast_%s_step%d.npz" % (stamp, self.step)),
                 pred=pred_all.astype(np.float32), speeds_in=speeds.astype(np.float32),
                 last_slot=np.array(times[-1].isoformat()), step_minutes=np.array(self.step),
                 horizon_minutes=np.array(self.fc.horizon_minutes))

    def forecast(self, horizon_min):
        speeds, times, pred_all = self.current_window()
        result = self.fc.forecast_from_prediction(pred_all, speeds, times, horizon_min)
        result["made_at"] = datetime.now().isoformat(timespec="seconds")
        result["step_minutes"] = self.step
        result["horizons_min"] = self.fc.horizon_minutes
        result["mode"] = "live" if self.step == STEP_MINUTES else "pipeline_test"
        if self.step != STEP_MINUTES:
            result["warning"] = TEST_MODE_WARNING % self.step
        slot_end = pd.Timestamp(times[-1]) + pd.Timedelta(minutes=self.step)
        now = pd.Timestamp.now(tz=LOCAL_TZ)
        result["data_age_min"] = round(max(0.0, (now - slot_end).total_seconds() / 60.0), 1)
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
        speeds, times, pred_all = self.current_window()
        ref = float(self.fc.reference_kmh[i])
        last = pd.Timestamp(times[-1])
        horizons = []
        for k, minutes in enumerate(self.fc.horizon_minutes):
            p = float(pred_all[k, i])
            band = speed_band(p / ref, self.fc.band_thresholds)
            horizons.append({"minutes_ahead": minutes,
                             "forecast_for": (last + pd.Timedelta(minutes=minutes)).isoformat(),
                             "predicted_kmh": round(p, 2), "band": band,
                             "color": BAND_COLORS[band]})
        node = {"junction_id": self.fc.sensor_ids[i],
                "current_kmh": round(float(np.clip(speeds[-1, i], 0.0, SPEED_SCALE_KMH)), 2),
                "reference_kmh": round(ref, 2), "input_last_slot": last.isoformat(),
                "data_age_min": round(max(0.0, (pd.Timestamp.now(tz=LOCAL_TZ)
                                                - (last + pd.Timedelta(minutes=self.step))
                                                ).total_seconds() / 60.0), 1),
                "step_minutes": self.step,
                "mode": "live" if self.step == STEP_MINUTES else "pipeline_test",
                "horizons": horizons}
        if self.fc.spots is not None:
            row = self.fc.spots.iloc[i]
            node.update({"name": str(row["name"]), "area": str(row["area"]),
                         "lat": float(row["lat"]), "lon": float(row["lon"])})
        if self.step != STEP_MINUTES:
            node["warning"] = TEST_MODE_WARNING % self.step
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
                "slots_needed": self.fc.seq_len}
        if os.path.exists(self.readings_path):
            df = self._read_readings()
            info["reading_rows"] = int(len(df))
            info["distinct_batches"] = int(df["timestamp"].nunique())
        try:
            _, times, _ = self.current_window()
            info["ready"] = True
            info["newest_slot"] = times[-1].isoformat()
        except LookupError as e:
            info["not_ready_reason"] = str(e)
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step-min", type=int, default=STEP_MINUTES,
                        help="minutes between readings (15 = real; 2 = pipeline test)")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--readings", default=READINGS_PATH, help="readings CSV to read")
    parser.add_argument("--forecast-dir", default=FORECAST_DIR, help="where forecasts are saved")
    args = parser.parse_args()
    import uvicorn
    if args.step_min != STEP_MINUTES:
        print("*** PIPELINE TEST MODE (%d-minute readings): not a measure of model accuracy ***"
              % args.step_min)
    app = create_app(step_minutes=args.step_min, readings_path=args.readings,
                     forecast_dir=args.forecast_dir)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
