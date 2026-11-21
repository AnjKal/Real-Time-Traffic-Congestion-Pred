"""Self-check for predict_bangalore.py. Run from the project folder inside the Python 3.7 env:

    python test_predict_bangalore.py

It prints PASS/FAIL per check and ends with "ALL CHECKS PASSED" (exit code 0) or
"SOME CHECKS FAILED" (exit code 1). It makes no TomTom calls and only reads the test set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from predict_bangalore import (BangaloreForecaster, build_window_from_readings,
                               SPEED_SCALE_KMH, STEP_MINUTES)

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append(True)
        print("PASS  %s%s" % (name, ("  -> " + str(detail)) if detail else ""))
    except Exception as e:                                   # noqa: BLE001
        RESULTS.append(False)
        print("FAIL  %s: %s" % (name, e))
        traceback.print_exc()


def masked_mae(truth, pred):
    mask = truth != 0
    return float(np.abs(pred - truth)[mask].mean())


def main():
    print("Loading model (takes ~30-60 s on CPU)...")
    fc = BangaloreForecaster()
    test = np.load(os.path.join(PROJECT_ROOT, "data", "BANGLORE_15MIN", "test.npz"))
    x_raw, y_raw = test["x"], test["y"]            # raw: speed/120 (unscaled) + time fraction
    n = len(x_raw)
    speeds = x_raw[..., 0] * SPEED_SCALE_KMH       # (n, 12, 100) km/h
    tod = x_raw[:, :, 0, 1]                        # (n, 12)
    truth = y_raw[..., 0] * SPEED_SCALE_KMH        # (n, 12, 100) km/h

    check("model loaded: 100 junctions, 12 horizons of 15 min",
          lambda: _assert(fc.num_nodes == 100 and fc.horizon_minutes == list(range(15, 181, 15)),
                          "nodes=%d horizons=%s" % (fc.num_nodes, fc.horizon_minutes)))

    # 1. My pre/post-processing must reproduce the supervisor's own evaluation path exactly.
    def same_as_supervisor():
        loader = fc._supervisor._data["test_loader"]
        x_loader, y_loader = next(loader.get_iterator())       # standardised, batch of 16
        b = x_loader.shape[0]
        with fc._graph.as_default():
            ref = fc._sess.run(fc._model.outputs,
                               feed_dict={fc._model.inputs: x_loader,
                                          fc._model.labels: y_loader})
        ref_kmh = np.clip(fc._scaler.inverse_transform(ref[..., 0]) * SPEED_SCALE_KMH,
                          0, SPEED_SCALE_KMH)
        mine = fc.predict_windows(fc.make_input_batch(speeds[:b], tod[:b]))
        diff = float(np.abs(ref_kmh - mine).max())
        _assert(diff < 1e-2, "max difference %.5f km/h" % diff)
        return "max difference vs evaluate path = %.6f km/h" % diff

    check("input/output scaling identical to the training/evaluation pipeline", same_as_supervisor)

    # 2. Accuracy on the whole untouched test set vs 'same speed as now'.
    pred = {}

    def accuracy():
        pred["all"] = fc.predict_windows(fc.make_input_batch(speeds, tod))
        _assert(pred["all"].shape == (n, 12, 100), "bad shape %s" % (pred["all"].shape,))
        _assert(np.isfinite(pred["all"]).all(), "non-finite predictions")
        persistence = np.repeat(speeds[:, -1:, :], 12, axis=1)
        rows = ["   horizon | DCRNN MAE | same-as-now MAE   (km/h, %d test windows)" % n]
        for k in (0, 1, 3, 5, 11):
            rows.append("   +%3d min |   %5.2f   |   %5.2f" % (
                STEP_MINUTES * (k + 1), masked_mae(truth[:, k], pred["all"][:, k]),
                masked_mae(truth[:, k], persistence[:, k])))
        print("\n".join(rows))
        overall = masked_mae(truth, pred["all"])
        _assert(overall < 10.0, "overall MAE %.2f km/h is far above the ~3 km/h expected" % overall)
        return "overall MAE %.2f km/h (training log: val ~2.95 km/h)" % overall

    check("forecast accuracy on test.npz is sensible", accuracy)

    # 3. Raw TomTom-style readings -> window -> identical to the stored window.
    def readings_roundtrip():
        sensor_ids = fc.sensor_ids
        start_utc = pd.Timestamp("2026-11-02 04:30:00", tz="UTC")           # 10:00 IST
        rows = []
        for t in range(12):
            slot_start = start_utc + pd.Timedelta(minutes=STEP_MINUTES * t)
            for j, sid in enumerate(sensor_ids):
                for offset in (2, 8):                          # two readings per slot, same value
                    rows.append((slot_start + pd.Timedelta(minutes=offset), sid,
                                 float(speeds[0, t, j])))
        df = pd.DataFrame(rows, columns=["timestamp", "junction_id", "currentSpeed"])
        w, times = build_window_from_readings(df, sensor_ids)
        _assert(w.shape == (12, 100), "shape %s" % (w.shape,))
        _assert(np.allclose(w, speeds[0], atol=1e-3), "speeds differ from the source window")
        _assert(str(times[0].tz) == "Asia/Kolkata" and times[0].hour == 10 and times[0].minute == 0,
                "first slot should be 10:00 Asia/Kolkata, got %s" % times[0])
        res = fc.forecast(w, times, horizon_min=30)
        _assert(res["input_last_slot"].startswith("2026-11-02T12:45"), res["input_last_slot"])
        _assert(res["forecast_for"].startswith("2026-11-02T13:15"), res["forecast_for"])
        # ONE missing slot is interpolated (same rule as the training data script)...
        def without(minutes_from, minutes_to):
            return df[~((df["timestamp"] >= start_utc + pd.Timedelta(minutes=minutes_from)) &
                        (df["timestamp"] < start_utc + pd.Timedelta(minutes=minutes_to)))]

        w1, _ = build_window_from_readings(without(60, 75), sensor_ids)       # slot 4 removed
        _assert(np.allclose(w1[4], (speeds[0, 3] + speeds[0, 5]) / 2, atol=1e-3),
                "single missing slot was not interpolated")
        # ...but TWO consecutive missing slots must be rejected with a clear message.
        try:
            build_window_from_readings(without(60, 90), sensor_ids)           # slots 4 and 5
        except ValueError as e:
            return "1-slot gap interpolated, 2-slot gap rejected (%s...)" % str(e)[:50]
        raise AssertionError("a window with two missing slots was accepted")

    check("raw readings -> 12-slot window (averaging, timezone, gap detection)", readings_roundtrip)

    # 4. Front-end output.
    def front_end_format():
        res = fc.forecast(speeds[0], tod[0], horizon_min=60)
        json.dumps(res)                                        # must be JSON-serialisable
        _assert(len(res["nodes"]) == 100, "expected 100 nodes")
        need = {"index", "junction_id", "current_kmh", "predicted_kmh", "reference_kmh",
                "speed_ratio", "band", "color"}
        _assert(all(need <= set(nd) for nd in res["nodes"]), "missing keys")
        _assert({nd["band"] for nd in res["nodes"]} <= {"free", "moderate", "heavy"}, "bad band")
        _assert(all(0 <= nd["predicted_kmh"] <= 120 for nd in res["nodes"]), "speed out of range")
        counts = {b: sum(1 for nd in res["nodes"] if nd["band"] == b)
                  for b in ("free", "moderate", "heavy")}
        for bad_h in (5, 10, 17, 195):
            try:
                fc.forecast(speeds[0], tod[0], horizon_min=bad_h)
                raise AssertionError("horizon %d should be rejected" % bad_h)
            except ValueError:
                pass
        return "bands at +60 min: %s; named=%s" % (counts, "name" in res["nodes"][0])

    check("front-end JSON format and horizon validation", front_end_format)

    print("\n" + ("ALL CHECKS PASSED" if all(RESULTS) else "SOME CHECKS FAILED"))
    return 0 if all(RESULTS) else 1


def _assert(cond, message):
    if not cond:
        raise AssertionError(message)


if __name__ == "__main__":
    sys.exit(main())
