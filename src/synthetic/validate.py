"""Six validation gates on the generated data -> data/synthetic/validation_report.json."""
import json
import sys

import numpy as np
import pandas as pd

from src.config import NUM_NODES, SYNTHETIC_DIR
from src.synthetic.generator import PARQUET_PATH, load_inputs
from src.synthetic.scenarios import TARGET_PROPS

REPORT_PATH = SYNTHETIC_DIR / "validation_report.json"


def load_grids():
    df = pd.read_parquet(PARQUET_PATH)
    T = len(df) // NUM_NODES
    assert (df["node_id"].to_numpy().reshape(T, NUM_NODES) == np.arange(NUM_NODES)).all()
    speed = df["speed_kmh"].to_numpy(dtype=float).reshape(T, NUM_NODES)
    labels = df["scenario_id"].to_numpy().reshape(T, NUM_NODES)[:, 0]
    weekend = df["is_weekend"].to_numpy().reshape(T, NUM_NODES)[:, 0]
    return speed, labels, weekend


def _gate(passed, value, threshold, **detail):
    return {"pass": bool(passed), "value": float(value), "threshold": threshold, **detail}


def gate1_scenario_proportions(labels):
    actual = {s: float((labels == s).mean()) for s in TARGET_PROPS}
    worst = max(abs(actual[s] - TARGET_PROPS[s]) for s in TARGET_PROPS)
    return _gate(worst < 0.02, worst, 0.02, actual=actual, target=TARGET_PROPS)


def gate2_per_node_mean_speed(speed, ff):
    ratio = speed.mean(axis=0) / ff
    margin = min(ratio.min() - 0.4, 0.9 - ratio.max())   # >= 0 means every node is in range
    return _gate(margin >= 0, margin, 0.0, range=[0.4, 0.9],
                 min_ratio=float(ratio.min()), max_ratio=float(ratio.max()),
                 per_node=[float(r) for r in ratio])


def gate3_lag1_autocorrelation(speed):
    x = speed - speed.mean(axis=0)
    acf = (x[1:] * x[:-1]).sum(axis=0) / (x * x).sum(axis=0)
    return _gate(acf.mean() >= 0.7, acf.mean(), 0.7, min_node_acf=float(acf.min()))


def gate4_neighbor_crosscorr(speed, W):
    corr = np.corrcoef(speed.T)
    vals = []
    for i in range(NUM_NODES):
        w = W[i].copy()
        w[i] = -1
        vals += [corr[i, j] for j in np.argsort(-w)[:3]]
    return _gate(np.mean(vals) >= 0.5, np.mean(vals), 0.5)


def gate5_weekday_weekend_delta(speed, weekend):
    # Spec wrote (weekday - weekend) / weekday >= 0.10, which would demand slower weekends.
    # Intent is "weekdays are more congested", so measure how much faster weekends are.
    wd, we = speed[~weekend].mean(), speed[weekend].mean()
    delta = (we - wd) / we
    return _gate(delta >= 0.10, delta, 0.10, weekday_mean=float(wd), weekend_mean=float(we))


def gate6_rush_vs_night_delta(speed, labels):
    night = speed[labels == 1].mean()
    rush = speed[np.isin(labels, (3, 6))].mean()
    delta = (night - rush) / night
    return _gate(delta >= 0.25, delta, 0.25, night_mean=float(night), rush_mean=float(rush))


def run() -> tuple[dict, bool]:
    ff, W = load_inputs()
    speed, labels, weekend = load_grids()
    report = {
        "gate1_scenario_proportions": gate1_scenario_proportions(labels),
        "gate2_per_node_mean_speed": gate2_per_node_mean_speed(speed, ff),
        "gate3_lag1_autocorrelation": gate3_lag1_autocorrelation(speed),
        "gate4_neighbor_crosscorr": gate4_neighbor_crosscorr(speed, W),
        "gate5_weekday_weekend_delta": gate5_weekday_weekend_delta(speed, weekend),
        "gate6_rush_vs_night_delta": gate6_rush_vs_night_delta(speed, labels),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    ok = all(g["pass"] for g in report.values())
    for name, g in report.items():
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {name}: value={g['value']:.3f} "
              f"threshold={g['threshold']}")
    if not ok:
        failed = [n for n, g in report.items() if not g["pass"]]
        print(f"VALIDATION FAILED: {failed}")
    return report, ok


if __name__ == "__main__":
    sys.exit(0 if run()[1] else 1)
