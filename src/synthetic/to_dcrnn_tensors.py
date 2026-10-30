"""raw_speeds.parquet -> chronological train/val/test .npz in the liyaguang/DCRNN layout."""
import pickle

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.config import HORIZON, NUM_NODES, PROCESSED_DIR, SEQ_LEN
from src.synthetic.generator import PARQUET_PATH, load_inputs

TRAIN_FRAC, VAL_FRAC = 0.7, 0.1


def run() -> dict:
    ff, _ = load_inputs()
    df = pd.read_parquet(PARQUET_PATH, columns=["timestamp", "node_id", "speed_kmh"])
    T = len(df) // NUM_NODES
    speed_norm = (df["speed_kmh"].to_numpy(dtype=np.float32).reshape(T, NUM_NODES)
                  / ff.astype(np.float32))
    ts = pd.DatetimeIndex(df["timestamp"].to_numpy().reshape(T, NUM_NODES)[:, 0])
    tod = ((ts.hour.to_numpy() + ts.minute.to_numpy() / 60) / 24).astype(np.float32)

    features = np.stack([speed_norm, np.broadcast_to(tod[:, None], speed_norm.shape)], axis=-1)
    n_win = T - SEQ_LEN - HORIZON + 1

    # x: (n, seq_len, N, 2) from t..t+seq_len-1; y: (n, horizon, N, 1) from t+seq_len..
    x = sliding_window_view(features, SEQ_LEN, axis=0).transpose(0, 3, 1, 2)[:n_win]
    y = sliding_window_view(speed_norm[SEQ_LEN:, :, None], HORIZON, axis=0).transpose(0, 3, 1, 2)
    assert x.shape[0] == y.shape[0] == n_win

    x_offsets = np.arange(-(SEQ_LEN - 1), 1).reshape(-1, 1)     # relative to last input step
    y_offsets = np.arange(1, HORIZON + 1).reshape(-1, 1)

    n_train, n_val = int(n_win * TRAIN_FRAC), int(n_win * VAL_FRAC)
    bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val),
              "test": (n_train + n_val, n_win)}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, (a, b) in bounds.items():
        np.savez(PROCESSED_DIR / f"{name}.npz", x=np.ascontiguousarray(x[a:b]),
                 y=np.ascontiguousarray(y[a:b]), x_offsets=x_offsets, y_offsets=y_offsets)
        counts[name] = b - a
        print(f"Wrote {name}.npz  windows={b - a:,}")

    with open(PROCESSED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump({"free_flow_kmh": ff.astype(np.float32),
                     "node_ids": list(range(NUM_NODES))}, f)
    return counts


if __name__ == "__main__":
    run()
