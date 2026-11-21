"""Predicted-vs-actual graphs and metrics on the held-out test set (real TomTom data).

    python backtest_bangalore.py

The test windows were never used for training or for choosing the checkpoint's epoch, and they
use the correct 15-minute spacing, so these are the trustworthy accuracy numbers.
Outputs go to reports/backtest/ (PNG graphs, metrics.csv, metrics.json).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

import eval_plots as ep
from predict_bangalore import BangaloreForecaster, SPEED_SCALE_KMH

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "backtest")


def main():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    fc = BangaloreForecaster()
    test = np.load(os.path.join(PROJECT_ROOT, "data", "BANGLORE_15MIN", "test.npz"))
    x_raw, y_raw = test["x"], test["y"]
    speeds = x_raw[..., 0] * SPEED_SCALE_KMH               # (n, 12, 100) km/h inputs
    tod = x_raw[:, :, 0, 1]
    truth = y_raw[..., 0] * SPEED_SCALE_KMH                # (n, 12, 100) km/h actual
    n, horizon = truth.shape[0], truth.shape[1]
    minutes = fc.horizon_minutes

    pred = fc.predict_windows(fc.make_input_batch(speeds, tod))
    base = ep.persistence(speeds[:, -1, :], horizon)

    m_model = ep.metrics_by_horizon(truth, pred)
    m_base = ep.metrics_by_horizon(truth, base)

    rows = []
    for k in range(horizon):
        rows.append({"minutes_ahead": minutes[k],
                     "dcrnn_mae": m_model["mae"][k], "dcrnn_rmse": m_model["rmse"][k],
                     "dcrnn_mape_pct": m_model["mape"][k],
                     "same_as_now_mae": m_base["mae"][k], "same_as_now_rmse": m_base["rmse"][k],
                     "same_as_now_mape_pct": m_base["mape"][k]})
    table = pd.DataFrame(rows).round(3)
    table.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    summary = {"test_windows": int(n), "dcrnn": ep.overall(truth, pred),
               "same_as_now": ep.overall(truth, base)}
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    title = "DCRNN on held-out real TomTom data (%d windows, 100 junctions)" % n
    ep.save_scatter(truth, pred, minutes, os.path.join(OUT_DIR, "predicted_vs_actual_scatter.png"),
                    title)
    ep.save_horizon_curve(minutes, {"DCRNN": m_model["mae"], "same speed as now": m_base["mae"]},
                          os.path.join(OUT_DIR, "error_by_horizon.png"),
                          "Error grows with forecast distance")

    # Time series for the 3 most variable junctions: windows are consecutive 15-minute steps, so
    # the +15 min forecast of window i is compared with the actual speed 15 minutes later.
    spots = pd.read_csv(os.path.join(PROJECT_ROOT, "bangalore_traffic_spots.csv"))
    picks = list(np.argsort(truth[:, 0].std(axis=0))[::-1][:3])
    names = ["%s (%s)" % (spots.iloc[i]["name"], fc.sensor_ids[i]) for i in picks]
    xs = np.arange(n) * fc.step_minutes / 60.0
    for k in (0, 3):
        ep.save_timeseries(xs, truth[:, k][:, picks], pred[:, k][:, picks], names,
                           os.path.join(OUT_DIR, "timeseries_plus%dmin.png" % minutes[k]),
                           "Actual vs predicted, +%d min ahead (x axis: hours into the test period)"
                           % minutes[k])

    print("\nTest windows: %d\n" % n)
    print(table[["minutes_ahead", "dcrnn_mae", "same_as_now_mae", "dcrnn_rmse",
                 "dcrnn_mape_pct"]].to_string(index=False))
    o, b = summary["dcrnn"], summary["same_as_now"]
    print("\nOverall MAE: DCRNN %.2f km/h vs same-as-now %.2f km/h" % (o["mae"], b["mae"]))
    better = int((m_model["mae"] < m_base["mae"]).sum())
    print("DCRNN beats 'same as now' at %d of %d forecast distances" % (better, horizon))
    print("\nGraphs and tables saved in: %s" % OUT_DIR)


if __name__ == "__main__":
    main()
