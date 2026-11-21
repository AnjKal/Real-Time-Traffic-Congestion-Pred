"""Compare the model's saved live forecasts with what TomTom actually reported afterwards.

    python evaluate_live.py            # Python 3.7 env (uses matplotlib)

Reads data/live/forecasts/*.npz (written by model_service.py) and data/live/readings.csv
(written by collect_tomtom.py), looks up the actual speed at every forecast's target time, and
writes predicted-vs-actual graphs plus MAE/RMSE/MAPE to reports/live/.
A forecast for horizon k made from the window ending at slot T targets slot T + k * step.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

import eval_plots as ep

LIVE_DIR = os.path.join(PROJECT_ROOT, "data", "live")
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "live")
GRAPH_PATH = os.path.join(PROJECT_ROOT, "data", "sensor_graph", "adj_mx_bangalore.pkl")
SPOTS_PATH = os.path.join(PROJECT_ROOT, "bangalore_traffic_spots.csv")
LOCAL_TZ = "Asia/Kolkata"


def load_actuals(readings_path, sensor_ids, step_minutes):
    """DataFrame indexed by local slot start, one column per junction (mean speed per slot)."""
    df = pd.read_csv(readings_path, usecols=["timestamp", "junction_id", "currentSpeed"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(LOCAL_TZ)
    df["junction_id"] = df["junction_id"].astype(str)
    pivot = df.pivot_table(index="timestamp", columns="junction_id", values="currentSpeed",
                           aggfunc="mean").reindex(columns=sensor_ids)
    return pivot.sort_index().resample("%dmin" % step_minutes).mean()


def collect(forecast_dir, actuals, step_minutes):
    """-> truth, pred, base (n, 12, 100) with NaN truth where TomTom data is missing, plus the
    first-horizon target times of each forecast."""
    truth, pred, base, target0 = [], [], [], []
    for path in sorted(glob.glob(os.path.join(forecast_dir, "forecast_*_step%d.npz" % step_minutes))):
        f = np.load(path)
        last = pd.Timestamp(str(f["last_slot"]))
        horizon = f["pred"].shape[0]
        t_arr = np.full(f["pred"].shape, np.nan, dtype=np.float64)
        for k in range(horizon):
            slot = last + pd.Timedelta(minutes=step_minutes * (k + 1))
            if slot in actuals.index:
                t_arr[k] = actuals.loc[slot].to_numpy(dtype=np.float64)
        if np.isfinite(t_arr).any():
            truth.append(t_arr)
            pred.append(f["pred"].astype(np.float64))
            base.append(np.repeat(f["speeds_in"][-1:].astype(np.float64), horizon, axis=0))
            target0.append(last + pd.Timedelta(minutes=step_minutes))
    return np.array(truth), np.array(pred), np.array(base), target0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-min", type=int, default=15, help="reading spacing used (2 = pipeline test)")
    parser.add_argument("--readings", default=os.path.join(LIVE_DIR, "readings.csv"))
    parser.add_argument("--forecast-dir", default=os.path.join(LIVE_DIR, "forecasts"))
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    step = args.step_min

    with open(GRAPH_PATH, "rb") as f:
        sensor_ids = [str(s) for s in pickle.load(f, encoding="latin1")[0]]
    actuals = load_actuals(args.readings, sensor_ids, step)
    truth, pred, base, target0 = collect(args.forecast_dir, actuals, step)
    if len(truth) == 0:
        print("No forecast has a matching TomTom reading yet. A forecast made at slot T can only "
              "be checked once the collector has polled slot T + %d minutes." % step)
        return 1

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    horizon = truth.shape[1]
    minutes = [step * (k + 1) for k in range(horizon)]
    m_model, m_base = ep.metrics_by_horizon(truth, pred), ep.metrics_by_horizon(truth, base)
    n_per_h = [int((np.isfinite(truth[:, k]) & (truth[:, k] > 0)).sum()) for k in range(horizon)]
    table = pd.DataFrame({"minutes_ahead": minutes, "values_compared": n_per_h,
                          "dcrnn_mae": m_model["mae"], "dcrnn_rmse": m_model["rmse"],
                          "dcrnn_mape_pct": m_model["mape"], "same_as_now_mae": m_base["mae"],
                          "same_as_now_rmse": m_base["rmse"]}).round(3)
    table.to_csv(os.path.join(args.out, "live_metrics.csv"), index=False)

    test_mode = step != 15
    tag = ("PIPELINE TEST (%d-min readings; model trained on 15) - plumbing check, NOT model accuracy"
           % step if test_mode else "live data")
    title = "DCRNN vs actual TomTom: %d forecasts, %s" % (len(truth), tag)
    have = [k for k in range(horizon) if n_per_h[k] > 0]
    picks = sorted({have[0], have[len(have) // 2], have[-1]})
    ep.save_scatter(truth, pred, minutes, os.path.join(args.out, "live_predicted_vs_actual.png"),
                    title, picks=tuple(picks))
    ep.save_horizon_curve(minutes, {"DCRNN": m_model["mae"], "same speed as now": m_base["mae"]},
                          os.path.join(args.out, "live_error_by_horizon.png"),
                          "Live error by forecast distance" + (" (PIPELINE TEST)" if test_mode else ""))

    spots = pd.read_csv(SPOTS_PATH)
    first = np.where(np.isfinite(truth[:, 0]).any(axis=1))[0]
    if len(first) >= 2:
        spread = np.nanstd(truth[first, 0], axis=0)
        top = list(np.argsort(np.nan_to_num(spread))[::-1][:3])
        ep.save_timeseries([t.strftime("%H:%M") for t in np.array(target0)[first]],
                           truth[first, 0][:, top], pred[first, 0][:, top],
                           ["%s (%s)" % (spots.iloc[i]["name"], sensor_ids[i]) for i in top],
                           os.path.join(args.out, "live_timeseries.png"),
                           "Actual vs predicted one step ahead, one point per forecast - " + tag)

    print("\n" + title + "\n")
    print(table[["minutes_ahead", "values_compared", "dcrnn_mae", "same_as_now_mae",
                 "dcrnn_rmse"]].to_string(index=False))
    if test_mode:
        print("\n*** PIPELINE TEST: these errors do NOT measure the model. Use "
              "backtest_bangalore.py for accuracy. ***")
    print("\nSaved to %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
