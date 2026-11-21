"""Metrics and predicted-vs-actual plots, shared by the backtest and the live evaluation.

All arrays are in km/h. truth/pred shapes: (windows, horizon, junctions). Python 3.7 safe.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use("Agg")                      # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np


def metrics_by_horizon(truth, pred):
    """Masked (truth > 0) MAE / RMSE / MAPE per horizon step -> dict of arrays (horizon,)."""
    mae, rmse, mape = [], [], []
    for k in range(truth.shape[1]):
        t, p = truth[:, k], pred[:, k]
        m = np.isfinite(t) & (t > 0)             # NaN = no actual reading for that cell
        if not m.any():
            mae.append(np.nan)
            rmse.append(np.nan)
            mape.append(np.nan)
            continue
        err = (p - t)[m]
        mae.append(np.abs(err).mean())
        rmse.append(np.sqrt((err ** 2).mean()))
        mape.append(100.0 * (np.abs(err) / t[m]).mean())
    return {"mae": np.array(mae), "rmse": np.array(rmse), "mape": np.array(mape)}


def overall(truth, pred):
    m = metrics_by_horizon(truth, pred)
    return {k: float(v.mean()) for k, v in m.items()}


def persistence(speeds_last, horizon):
    """'Same speed as now' baseline: (windows, junctions) -> (windows, horizon, junctions)."""
    return np.repeat(np.asarray(speeds_last)[:, None, :], horizon, axis=1)


def save_scatter(truth, pred, minutes, path, title, picks=(0, 3, 11)):
    """Predicted vs actual, one panel per chosen horizon index, with the ideal diagonal."""
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4.8))
    rng = np.random.RandomState(0)
    for ax, k in zip(np.atleast_1d(axes), picks):
        t, p = truth[:, k].ravel(), pred[:, k].ravel()
        ok = np.isfinite(t) & (t > 0)
        t, p = t[ok], p[ok]
        if len(t) == 0:
            ax.set_title("+%d min: no actual readings yet" % minutes[k])
            continue
        keep = rng.choice(len(t), size=min(4000, len(t)), replace=False)
        ax.scatter(t[keep], p[keep], s=6, alpha=0.35)
        hi = max(t.max(), p.max()) * 1.02
        ax.plot([0, hi], [0, hi], "r--", lw=1, label="perfect")
        mae = np.abs(p - t).mean()
        ax.set_title("+%d min   (MAE %.2f km/h)" % (minutes[k], mae))
        ax.set_xlabel("actual speed (km/h)")
        ax.set_ylabel("predicted speed (km/h)")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_horizon_curve(minutes, series, path, title):
    """MAE versus forecast distance for several models. series: {label: mae_array}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, mae in series.items():
        ax.plot(minutes, mae, marker="o", label=label)
    ax.set_xlabel("forecast distance (minutes ahead)")
    ax.set_ylabel("mean absolute error (km/h)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_timeseries(x_axis, truth_series, pred_series, names, path, title, ylabel="speed (km/h)"):
    """Actual vs predicted over time for a few junctions (one panel each)."""
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 3.2 * len(names)), sharex=True)
    for ax, i, name in zip(np.atleast_1d(axes), range(len(names)), names):
        ax.plot(x_axis, truth_series[:, i], label="actual", lw=1.6)
        ax.plot(x_axis, pred_series[:, i], label="predicted", lw=1.6, ls="--")
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
