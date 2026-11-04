"""Live prediction wrapper around the trained Bangalore DCRNN checkpoint.

Python 3.7 / TensorFlow 1.15 (same environment as dcrnn_train.py).

What it does
------------
1. Restores the trained weights (steps 4-5: the same DCRNNSupervisor.load call that
   evaluate_bangalore.py uses) and keeps the TensorFlow session open.
2. Accepts the LAST 12 FIFTEEN-MINUTE readings for the 100 junctions (speed in km/h, as
   returned by the TomTom Flow Segment API's currentSpeed) and returns the forecast for the
   next 12 fifteen-minute steps (+15, +30, ... +180 minutes).

The model was trained with:
  * input  (12 steps, 100 junctions, 2 features): [ (speed_kmh / 120) standardised with the
    training mean/std , time-of-day fraction (minutes since local midnight / 1440) ]
  * output (12 steps, 100 junctions): the same standardised speed.
This module applies exactly that pre- and post-processing, using the scaler the supervisor
built from train.npz, so the numbers are identical to the ones evaluate_bangalore.py sees.

Typical use
-----------
    from predict_bangalore import BangaloreForecaster, build_window_from_readings

    fc = BangaloreForecaster()                        # loads the checkpoint once
    speeds, times = build_window_from_readings(readings_df, fc.sensor_ids, now=pd.Timestamp.now(tz="Asia/Kolkata"))
    result = fc.forecast(speeds, times, horizon_min=30)   # JSON-ready dict for the front end
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)            # so `lib` and `model` import from anywhere

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor

CONFIG_PATH = os.path.join(PROJECT_ROOT, "data", "model", "dcrnn_bangalore_15min_smoke.yaml")
CHECKPOINT = os.path.join(
    PROJECT_ROOT, "data", "model", "bangalore_15min_smoke",
    "dcrnn_DR_2_h_12_64-64_lr_0.01_bs_16_1103202622", "models-0.0246-114")
SPOTS_PATH = os.path.join(PROJECT_ROOT, "bangalore_traffic_spots.csv")
SERVING_LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "serving_bangalore")

SPEED_SCALE_KMH = 120.0        # generate_bangalore_15min_data.py divides speeds by 120
STEP_MINUTES = 15
LOCAL_TZ = "Asia/Kolkata"

BAND_COLORS = {"free": "#2ecc71", "moderate": "#f1c40f", "heavy": "#e74c3c"}


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _time_fractions(times):
    """Time-of-day fractions (minutes since local midnight / 1440) for datetimes or floats."""
    out = []
    for t in times:
        if hasattr(t, "hour"):
            out.append((t.hour * 60 + t.minute) / 1440.0)
        else:
            out.append(float(t))
    return np.asarray(out, dtype=np.float32)


def speed_band(ratio, thresholds=(0.75, 0.40)):
    """free / moderate / heavy from predicted speed relative to the junction's usual free speed."""
    free_above, moderate_from = thresholds
    if ratio > free_above:
        return "free"
    if ratio >= moderate_from:
        return "moderate"
    return "heavy"


class BangaloreForecaster(object):
    """Holds the restored DCRNN and turns 12 recent readings into 12 future steps."""

    def __init__(self, config_path=CONFIG_PATH, checkpoint=CHECKPOINT, use_cpu=True,
                 band_thresholds=(0.75, 0.40), step_minutes=STEP_MINUTES):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Make every path absolute so this works from any working directory.
        config["base_dir"] = _abs(config["base_dir"])
        config["data"]["dataset_dir"] = _abs(config["data"]["dataset_dir"])
        config["data"]["graph_pkl_filename"] = _abs(config["data"]["graph_pkl_filename"])
        # Without log_dir the supervisor creates a NEW run folder inside data/model/... on
        # every load. Point it at a fixed logs folder instead.
        config["train"]["log_dir"] = SERVING_LOG_DIR

        model_cfg = config["model"]
        self.num_nodes = int(model_cfg["num_nodes"])
        self.seq_len = int(model_cfg["seq_len"])
        self.horizon = int(model_cfg["horizon"])
        self.input_dim = int(model_cfg["input_dim"])
        # step_minutes is the spacing of the readings you feed in. The model was TRAINED on 15;
        # any other value (e.g. 2 for a quick pipeline test) only relabels the time axis and
        # puts the model outside what it learned, so its accuracy cannot be trusted.
        self.step_minutes = int(step_minutes)
        self.horizon_minutes = [self.step_minutes * (k + 1) for k in range(self.horizon)]
        self.band_thresholds = band_thresholds
        self.config = config

        checkpoint = os.path.normpath(checkpoint)
        if not os.path.exists(checkpoint + ".index"):
            raise IOError(
                "Checkpoint not found: %s(.index / .data-00000-of-00001). "
                "Check CHECKPOINT in predict_bangalore.py." % checkpoint)

        sensor_ids, _, adj_mx = load_graph_data(config["data"]["graph_pkl_filename"])
        self.sensor_ids = [str(s) for s in sensor_ids]
        if len(self.sensor_ids) != self.num_nodes:
            raise ValueError("graph has %d nodes but config says %d"
                             % (len(self.sensor_ids), self.num_nodes))

        # Own graph + session so several forecasters (or other TF code) cannot collide.
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
            tf_config = tf.ConfigProto(device_count={"GPU": 0}) if use_cpu else tf.ConfigProto()
            self._sess = tf.Session(graph=self._graph, config=tf_config)
            self._sess.run(tf.global_variables_initializer())
            self._supervisor.load(self._sess, checkpoint)

        self._model = self._supervisor._test_model
        self._scaler = self._supervisor._data["scaler"]
        self._batch_size = int(config["data"]["test_batch_size"])   # fixed by the test model
        self._lock = threading.Lock()

        # A per-junction "usual free speed" (90th percentile of training speeds) so the front
        # end can show free / moderate / heavy. TomTom's own free-flow speed is not part of
        # this dataset. x_train's speed channel was standardised in place by load_dataset.
        train_speed = self._scaler.inverse_transform(
            self._supervisor._data["x_train"][..., 0]) * SPEED_SCALE_KMH
        self.reference_kmh = np.percentile(train_speed.reshape(-1, self.num_nodes), 90, axis=0)

        self.spots = None
        if os.path.exists(SPOTS_PATH):
            spots = pd.read_csv(SPOTS_PATH)
            # ASSUMPTION: row i of the spots file is graph node i (J001 = first row, ...).
            if len(spots) == self.num_nodes:
                self.spots = spots

    # ------------------------------------------------------------------ pre-processing
    def make_input_batch(self, speeds_kmh, time_fractions):
        """speeds_kmh (n, 12, 100) km/h, time_fractions (n, 12) -> model input (n, 12, 100, 2)."""
        speeds = np.asarray(speeds_kmh, dtype=np.float32)
        tod = np.asarray(time_fractions, dtype=np.float32)
        if speeds.ndim != 3 or speeds.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError("speeds must have shape (n, %d, %d), got %s"
                             % (self.seq_len, self.num_nodes, speeds.shape))
        if tod.shape != speeds.shape[:2]:
            raise ValueError("time_fractions must have shape %s, got %s"
                             % (speeds.shape[:2], tod.shape))
        if not np.isfinite(speeds).all():
            raise ValueError("speeds contain NaN/inf; every junction needs a value for all "
                             "%d steps" % self.seq_len)
        x = np.zeros(speeds.shape + (self.input_dim,), dtype=np.float32)
        x[..., 0] = self._scaler.transform(np.clip(speeds, 0.0, SPEED_SCALE_KMH) / SPEED_SCALE_KMH)
        x[..., 1] = tod[:, :, None]                      # broadcast over junctions, NOT scaled
        return x

    def make_input(self, speeds_kmh, times):
        """One window: speeds_kmh (12, 100), times = 12 datetimes (or 12 time-of-day floats)."""
        return self.make_input_batch(np.asarray(speeds_kmh)[None], _time_fractions(times)[None])

    # ------------------------------------------------------------------ inference
    def predict_windows(self, x):
        """Model-ready x (n, 12, 100, 2) -> forecast in km/h, shape (n, 12, 100)."""
        x = np.asarray(x, dtype=np.float32)
        n, b = x.shape[0], self._batch_size
        labels = np.zeros((b, self.horizon, self.num_nodes, self.input_dim), dtype=np.float32)
        chunks = []
        with self._lock:
            for start in range(0, n, b):
                chunk = x[start:start + b]
                m = chunk.shape[0]
                if m < b:                                # the test model has a fixed batch size
                    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], b - m, axis=0)])
                out = self._sess.run(
                    self._model.outputs,
                    feed_dict={self._model.inputs: chunk, self._model.labels: labels})
                chunks.append(out[:m, :, :, 0])
        scaled = np.concatenate(chunks, axis=0)
        kmh = self._scaler.inverse_transform(scaled) * SPEED_SCALE_KMH
        return np.clip(kmh, 0.0, SPEED_SCALE_KMH)

    def predict(self, speeds_kmh, times):
        """12 recent readings -> forecast (12, 100) km/h; row k is +15*(k+1) minutes."""
        return self.predict_windows(self.make_input(speeds_kmh, times))[0]

    # ------------------------------------------------------------------ front-end format
    def forecast(self, speeds_kmh, times, horizon_min=15):
        """JSON-ready forecast for ONE horizon (15, 30, ... 180 minutes) for all 100 junctions."""
        return self.forecast_from_prediction(self.predict(speeds_kmh, times), speeds_kmh, times,
                                             horizon_min)

    def forecast_from_prediction(self, pred_all, speeds_kmh, times, horizon_min=15):
        """Same output as forecast(), reusing an already computed predict() result (12, 100)."""
        if horizon_min not in self.horizon_minutes:
            raise ValueError("horizon_min must be one of %s" % self.horizon_minutes)
        k = horizon_min // self.step_minutes - 1
        pred = pred_all[k]
        current = np.clip(np.asarray(speeds_kmh, dtype=np.float64)[-1], 0.0, SPEED_SCALE_KMH)
        # Real timestamps give real forecast times; bare time-of-day floats do not.
        last_slot = pd.Timestamp(times[-1]) if hasattr(times[-1], "hour") else None

        nodes = []
        for i in range(self.num_nodes):
            ratio = float(pred[i] / self.reference_kmh[i])
            band = speed_band(ratio, self.band_thresholds)
            node = {
                "index": i,
                "junction_id": self.sensor_ids[i],
                "current_kmh": round(float(current[i]), 2),
                "predicted_kmh": round(float(pred[i]), 2),
                "reference_kmh": round(float(self.reference_kmh[i]), 2),
                "speed_ratio": round(ratio, 3),
                "band": band,
                "color": BAND_COLORS[band],
            }
            if self.spots is not None:
                row = self.spots.iloc[i]
                node.update({"name": str(row["name"]), "area": str(row["area"]),
                             "lat": float(row["lat"]), "lon": float(row["lon"])})
            nodes.append(node)

        return {
            "horizon_min": int(horizon_min),
            "input_last_slot": last_slot.isoformat() if last_slot is not None else None,
            "forecast_for": ((last_slot + pd.Timedelta(minutes=horizon_min)).isoformat()
                             if last_slot is not None else None),
            "nodes": nodes,
        }


def build_window_from_readings(readings, sensor_ids, seq_len=12, now=None,
                               step_minutes=STEP_MINUTES):
    """Turn raw TomTom readings into the model's 12-step input window.

    readings: DataFrame with columns timestamp, junction_id, currentSpeed (the same layout as
        data/raw/tomtom_traffic_raw.csv). Several readings per 15-minute slot are fine: like
        generate_bangalore_15min_data.py they are averaged per slot, timestamps are converted
        to Asia/Kolkata (naive timestamps are treated as UTC, as in that script), and a single
        missing slot per junction is interpolated.
    now: optional current time. Slots that have not finished yet (slot start + 15 min > now)
        are ignored, because a half-filled slot would give a misleading average.

    Returns (speeds_kmh (seq_len, n_junctions) float32, times: list of tz-aware Timestamps =
    the START of each 15-minute slot, oldest first). Raises ValueError, naming what is
    missing, if there are not seq_len consecutive slots with data for every junction.
    """
    df = readings.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(LOCAL_TZ)
    df["junction_id"] = df["junction_id"].astype(str)
    df["currentSpeed"] = pd.to_numeric(df["currentSpeed"], errors="coerce")
    sensor_ids = [str(s) for s in sensor_ids]
    df = df[df["junction_id"].isin(set(sensor_ids))]
    if df.empty:
        raise ValueError("no readings for any of the model's junctions")

    pivot = df.pivot_table(index="timestamp", columns="junction_id", values="currentSpeed",
                           aggfunc="mean")
    pivot = pivot.reindex(columns=sensor_ids)
    pivot = pivot.sort_index().resample("%dmin" % step_minutes).mean()
    pivot = pivot.interpolate(method="time", limit=1, limit_area="inside")

    if now is not None:
        now = pd.Timestamp(now)
        now = now.tz_localize(LOCAL_TZ) if now.tzinfo is None else now.tz_convert(LOCAL_TZ)
        pivot = pivot[pivot.index + pd.Timedelta(minutes=step_minutes) <= now]

    complete = pivot.dropna(how="any")
    if complete.empty:
        missing_junctions = [c for c in pivot.columns if pivot[c].isna().all()]
        raise ValueError("no 15-minute slot has data for all %d junctions (junctions with no "
                         "data at all: %s)" % (len(sensor_ids), missing_junctions[:10]))

    wanted = pd.date_range(end=complete.index[-1], periods=seq_len,
                           freq="%dmin" % step_minutes)
    missing = [t for t in wanted if t not in complete.index]
    if missing:
        raise ValueError(
            "need %d consecutive %d-minute slots with all junctions, newest complete slot is "
            "%s but these are missing/incomplete: %s"
            % (seq_len, step_minutes, complete.index[-1],
               [t.strftime("%H:%M") for t in missing]))

    window = pivot.loc[wanted]
    return window.to_numpy(dtype=np.float32), list(wanted)


if __name__ == "__main__":
    # Quick demo: forecast from the first window of the saved test set.
    fc = BangaloreForecaster()
    test = np.load(os.path.join(PROJECT_ROOT, "data", "BANGLORE_15MIN", "test.npz"))
    x0 = test["x"][0]                                     # raw: speed/120 and time fraction
    result = fc.forecast(x0[:, :, 0] * SPEED_SCALE_KMH, x0[:, 0, 1], horizon_min=30)
    worst = sorted(result["nodes"], key=lambda n: n["speed_ratio"])[:5]
    print("Forecast for +%d min, 5 slowest junctions:" % result["horizon_min"])
    for n in worst:
        print("  %s %-34s now %5.1f -> %5.1f km/h  %s" % (
            n["junction_id"], n.get("name", ""), n["current_kmh"], n["predicted_kmh"], n["band"]))
