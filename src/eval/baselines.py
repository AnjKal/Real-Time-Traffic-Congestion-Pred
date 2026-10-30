"""Three honest baselines on NORMALIZED speeds (speed / free-flow).

Inputs x: (N, seq_len, nodes, 2) with channels (speed_norm, time_of_day_fraction).
Outputs:  (N, horizon, nodes, 1).
"""
import numpy as np
import pandas as pd

from src.config import HORIZON, POLL_CADENCE_MIN

BUCKET_MIN = 15
N_BUCKETS = 24 * 60 // BUCKET_MIN        # 96


def persistence(x, horizon: int = HORIZON):
    """Repeat the last observed speed for every future step."""
    return np.repeat(x[:, -1:, :, 0:1], horizon, axis=1)


def bucket_table(speeds, timestamps):
    """Mean speed per (15-min time-of-day bucket, node) -> (96, nodes)."""
    ts = pd.DatetimeIndex(timestamps)
    bucket = (ts.hour.to_numpy() * 60 + ts.minute.to_numpy()) // BUCKET_MIN
    table = np.zeros((N_BUCKETS, speeds.shape[1]))
    for b in range(N_BUCKETS):
        table[b] = speeds[bucket == b].mean(axis=0)
    return table


def _target_buckets(x, horizon):
    """Forecast time-of-day buckets (N, horizon) recovered from the tod channel of x."""
    last_min = np.rint(x[:, -1, 0, 1].astype(np.float64) * 24 * 60).astype(int)
    minutes = last_min[:, None] + POLL_CADENCE_MIN * np.arange(1, horizon + 1)[None, :]
    return (minutes % (24 * 60)) // BUCKET_MIN


def historical_average(x, train_speeds, train_timestamps, horizon: int = HORIZON):
    """Mean train-split speed for the (node, 15-min bucket) of each forecast step."""
    table = bucket_table(train_speeds, train_timestamps)
    return table[_target_buckets(x, horizon)][..., None]           # (N, H, nodes, 1)


def last_week_same_time(x, train_speeds, train_timestamps, test_timestamps,
                        history_speeds=None, history_timestamps=None, horizon: int = HORIZON):
    """Speed exactly 7 days before each forecast timestamp; falls back to historical average.

    test_timestamps: (N, horizon) datetime64 forecast times.
    history_*: the stream to look back into (defaults to the train stream). The evaluator
    passes the full observed stream: a value 7 days before the forecast time is always in
    the past at prediction time, so this is not leakage.
    """
    hs = train_speeds if history_speeds is None else history_speeds
    ht = pd.DatetimeIndex(train_timestamps if history_timestamps is None else history_timestamps)
    step = np.timedelta64(POLL_CADENCE_MIN, "m")
    idx = ((np.asarray(test_timestamps) - np.timedelta64(7, "D") - ht[0].to_datetime64())
           // step).astype(int)
    valid = (idx >= 0) & (idx < len(hs))
    fallback = historical_average(x, train_speeds, train_timestamps, horizon)
    looked_up = hs[np.clip(idx, 0, len(hs) - 1)][..., None]        # (N, H, nodes, 1)
    return np.where(valid[..., None, None], looked_up, fallback)
