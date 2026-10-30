"""Generate 90 days of 3-min synthetic speeds for all nodes -> data/synthetic/raw_speeds.parquet."""
import pickle

import numpy as np
import pandas as pd

from src.config import GRAPH_DIR, NUM_NODES, POLL_CADENCE_MIN, RANDOM_SEED, SYNTHETIC_DAYS, SYNTHETIC_DIR
from src.synthetic.scenarios import SCENARIOS, base_labels, hotspot_multipliers

START = "2026-08-01 00:00"
N_INCIDENTS = 104        # ~12.5 steps each -> ~3% of 43,200 steps; recovery ~2/3 of that -> ~2%
PARQUET_PATH = SYNTHETIC_DIR / "raw_speeds.parquet"


def load_inputs():
    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv").sort_values("node_id")
    ff = nodes["free_flow_speed_kmh"].to_numpy(dtype=float)
    if len(ff) != NUM_NODES or np.isnan(ff).any():
        raise RuntimeError("nodes.csv must hold a free_flow_speed_kmh for every node")
    with open(GRAPH_DIR / "adj_mx.pkl", "rb") as f:
        _, _, W = pickle.load(f)
    return ff, np.asarray(W, dtype=float)


def build_timestamps() -> pd.DatetimeIndex:
    steps = SYNTHETIC_DAYS * 24 * 60 // POLL_CADENCE_MIN
    return pd.date_range(START, periods=steps, freq=f"{POLL_CADENCE_MIN}min")


def inject_incidents(labels: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Overwrite labels with s8 (incident) then s9 (recovery); return labels + (T, N) severity."""
    T = len(labels)
    labels = labels.copy()
    severity = np.zeros((T, NUM_NODES), dtype=np.float32)
    slot = T // N_INCIDENTS          # one event per slot -> events never overlap
    for e in range(N_INCIDENTS):
        length = int(rng.integers(9, 17))
        recovery = int(round(length * 0.67))
        start = e * slot + int(rng.integers(0, slot - length - recovery))
        affected = rng.choice(NUM_NODES, size=int(rng.integers(1, 3)), replace=False)
        labels[start:start + length] = 8
        labels[start + length:start + length + recovery] = 9
        ramp = np.minimum(1.0, (np.arange(length) + 1) / 2)
        decay = 1.0 - (np.arange(recovery) + 1) / (recovery + 1)
        for nd in affected:
            severity[start:start + length, nd] = ramp
            severity[start + length:start + length + recovery, nd] = decay
    return labels, severity


def run() -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    ff, W = load_inputs()
    hotspot = hotspot_multipliers(NUM_NODES)
    ts = build_timestamps()
    T = len(ts)

    labels, severity = inject_incidents(base_labels(ts), rng)

    stack = np.stack([s.fn(ts, ff, W, hotspot, rng, severity=severity) for s in SCENARIOS])
    speed = stack[labels - 1, np.arange(T)]            # (T, N): row from each step's scenario

    confidence = rng.beta(9, 1, size=(T, NUM_NODES))
    incident = (labels[:, None] == 8) & (severity > 0)
    confidence[incident] = rng.beta(3, 2, size=int(incident.sum()))

    weekend = ts.dayofweek.to_numpy() >= 5
    df = pd.DataFrame({
        "timestamp": np.repeat(ts.to_numpy(), NUM_NODES),
        "node_id": np.tile(np.arange(NUM_NODES, dtype=np.int16), T),
        "speed_kmh": speed.astype(np.float32).ravel(),
        "confidence": confidence.astype(np.float32).ravel(),
        "scenario_id": np.repeat(labels, NUM_NODES),
        "is_weekend": np.repeat(weekend, NUM_NODES),
    })
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Wrote {PARQUET_PATH} ({len(df):,} rows)")
    return {"path": PARQUET_PATH, "rows": len(df), "timesteps": T}


if __name__ == "__main__":
    run()
