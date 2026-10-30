"""Step 5: evaluate the trained DCRNN and three baselines on the untouched test split."""
import json
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from src.config import GRAPH_DIR, NUM_NODES, PROCESSED_DIR, PROJECT_ROOT
from src.eval import baselines as bl
from src.eval import metrics as mt
from src.model.dcrnn.dcrnn_model import forward_batch
from src.model.dcrnn.dcrnn_supervisor import device, load_checkpoint
from src.synthetic.generator import PARQUET_PATH, build_timestamps
from scripts.step4_train import CONFIG_PATH, load_config

REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "bangalore_dcrnn" / "best_model.pt"
KEY_HORIZONS = {"5min": 1, "10min": 3, "20min": 6}     # 0-based step idx -> ~6 / 12 / 21 min
MODELS = ["DCRNN", "Persistence", "Hist. Average", "Last-Week"]
STEP_MIN = 3


def predict_dcrnn(x, ckpt, model, batch_size=32):
    mean, std = ckpt["scaler"]["mean"], ckpt["scaler"]["std"]
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].copy()
            xb[..., 0] = (xb[..., 0] - mean) / std
            pred = forward_batch(model, torch.from_numpy(xb).float()).cpu().numpy()
            out.append(pred * std + mean)             # back to normalized speed
    return np.concatenate(out)                         # (N, 7, 20, 1)


def metric_block(yt, yp, where=None):
    return {"mae": mt.masked_mae(yt, yp, where), "rmse": mt.masked_rmse(yt, yp, where),
            "mape": mt.masked_mape(yt, yp, where)}


def model_report(yt, yp, rush, weekend):
    """All inputs in km/h, shape (N, H, nodes); rush/weekend: (N, H) bool on target time."""
    r3 = lambda m: m[:, :, None]                       # broadcast over nodes
    per_h = {name: getattr(mt, f"masked_{name}_per_horizon")(yt, yp) for name in
             ("mae", "rmse", "mape")}
    return {
        "overall": metric_block(yt, yp),
        "per_horizon": [{"horizon_idx": h, "minutes": STEP_MIN * (h + 1),
                         **{k: float(v[h]) for k, v in per_h.items()}}
                        for h in range(yt.shape[1])],
        "key_horizons": {label: {k: float(v[h]) for k, v in per_h.items()}
                         for label, h in KEY_HORIZONS.items()},
        "per_node_mae": [float(v) for v in mt.masked_mae_per_node(yt, yp)],
        "rush_vs_offpeak_mae": {"rush": mt.masked_mae(yt, yp, r3(rush)),
                                "off_peak": mt.masked_mae(yt, yp, r3(~rush))},
        "weekday_vs_weekend_mae": {"weekday": mt.masked_mae(yt, yp, r3(~weekend)),
                                   "weekend": mt.masked_mae(yt, yp, r3(weekend))},
    }


def main() -> int:
    cfg = load_config(CONFIG_PATH)
    with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
        ff = np.asarray(pickle.load(f)["free_flow_kmh"], dtype=np.float64)

    test = np.load(PROCESSED_DIR / "test.npz")
    x, y = test["x"], test["y"]                        # raw normalized speeds, never scaled
    m = cfg["model"]
    assert x.shape[1:] == (m["seq_len"], m["num_nodes"], m["input_dim"]), x.shape
    assert y.shape[1:] == (m["horizon"], m["num_nodes"], m["output_dim"]), y.shape
    N, H = len(x), m["horizon"]

    # Re-attach timestamps: windows are consecutive, test = the last N windows.
    grid = pd.read_parquet(PARQUET_PATH, columns=["speed_kmh"])["speed_kmh"].to_numpy()
    T = len(grid) // NUM_NODES
    stream = grid.reshape(T, NUM_NODES) / ff           # normalized speed stream (T, 20)
    ts = build_timestamps()
    w0 = (T - m["seq_len"] - H + 1) - N
    assert np.allclose(x[0, :, :, 0], stream[w0:w0 + m["seq_len"]], atol=1e-5), "misaligned"
    assert np.allclose(y[-1, :, :, 0],
                       stream[w0 + N - 1 + m["seq_len"]: w0 + N - 1 + m["seq_len"] + H], atol=1e-5)
    target_idx = w0 + np.arange(N)[:, None] + m["seq_len"] + np.arange(H)[None, :]
    target_ts = ts.to_numpy()[target_idx]                                   # (N, H)
    tt = pd.DatetimeIndex(target_ts.ravel())
    hour = tt.hour.to_numpy().reshape(N, H)
    rush = ((hour >= 7) & (hour < 10)) | ((hour >= 17) & (hour < 21))
    weekend = (tt.dayofweek.to_numpy() >= 5).reshape(N, H)

    print(f"Device: {device} | test windows: {N:,} (stream idx {w0}..{w0 + N - 1})")
    model, ckpt = load_checkpoint(CHECKPOINT)
    print(f"Loaded checkpoint epoch {ckpt['epoch']} (val_mae {ckpt['val_mae']:.4f})")

    # train stream = every step the train windows touched (windows 0..n_train-1)
    n_train = len(np.load(PROCESSED_DIR / "train.npz")["x"])
    tr_end = n_train + m["seq_len"] + H - 1
    tr_speeds, tr_ts = stream[:tr_end], ts[:tr_end]

    preds = {
        "DCRNN": predict_dcrnn(x, ckpt, model),
        "Persistence": bl.persistence(x, H),
        "Hist. Average": bl.historical_average(x, tr_speeds, tr_ts, H),
        "Last-Week": bl.last_week_same_time(x, tr_speeds, tr_ts, target_ts,
                                            history_speeds=stream, history_timestamps=ts,
                                            horizon=H),
    }
    lw_fallback = float((np.asarray(target_ts) - np.timedelta64(7, "D") < ts[0].to_datetime64()).mean())

    yt = y[..., 0].astype(np.float64) * ff                                   # km/h
    kmh = {k: v[..., 0].astype(np.float64) * ff for k, v in preds.items()}
    report = {"meta": {"n_test": N, "checkpoint_epoch": int(ckpt["epoch"]),
                       "checkpoint_val_mae_norm": float(ckpt["val_mae"]),
                       "horizon_idx": KEY_HORIZONS, "units": "km/h",
                       "last_week_fallback_share": lw_fallback},
              "models": {k: model_report(yt, v, rush, weekend) for k, v in kmh.items()}}

    # verdict: DCRNN must beat every baseline at every key horizon (MAE)
    losses = [(b, lab) for b in MODELS[1:] for lab in KEY_HORIZONS
              if report["models"]["DCRNN"]["key_horizons"][lab]["mae"]
              >= report["models"][b]["key_horizons"][lab]["mae"]]
    report["verdict"] = {"pass": not losses, "losses": [f"{b}@{lab}" for b, lab in losses]}

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "step5_metrics.json").write_text(json.dumps(report, indent=2))
    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv").sort_values("node_id")
    pd.DataFrame({"node_id": nodes["node_id"], "name": nodes["name"],
                  **{f"{k}_mae_kmh": report["models"][k]["per_node_mae"] for k in MODELS}}
                 ).to_csv(REPORTS_DIR / "step5_per_node.csv", index=False)
    pd.DataFrame([{"model": k, **row} for k in MODELS
                  for row in report["models"][k]["per_horizon"]]
                 ).to_csv(REPORTS_DIR / "step5_per_horizon.csv", index=False)
    np.savez_compressed(REPORTS_DIR / "step5_predictions.npz", truth=yt.astype(np.float32),
                        dcrnn=kmh["DCRNN"].astype(np.float32),
                        target_ts=target_ts.astype("datetime64[s]").astype(np.int64))

    lines = ["| Model | 5-min MAE | 10-min MAE | 20-min MAE |", "|---|---|---|---|"]
    for k in MODELS:
        kh = report["models"][k]["key_horizons"]
        lines.append(f"| {k} | " + " | ".join(f"{kh[l]['mae']:.2f}" for l in KEY_HORIZONS) + " |")
    verdict = ("DCRNN beats all baselines at all 3 horizons -> PASS" if not losses else
               "DCRNN loses to " + ", ".join(f"{b} at {lab}" for b, lab in losses)
               + " -> INVESTIGATE")
    d = report["models"]["DCRNN"]
    md = ["# Step 5 - test-set evaluation", "", f"Test windows: {N:,}. Units: km/h (MAE).", "",
          *lines, "", f"**Verdict:** {verdict}", "",
          f"DCRNN overall: MAE {d['overall']['mae']:.2f}, RMSE {d['overall']['rmse']:.2f}, "
          f"MAPE {d['overall']['mape']:.1f}%",
          f"Rush MAE {d['rush_vs_offpeak_mae']['rush']:.2f} vs off-peak "
          f"{d['rush_vs_offpeak_mae']['off_peak']:.2f}; weekday "
          f"{d['weekday_vs_weekend_mae']['weekday']:.2f} vs weekend "
          f"{d['weekday_vs_weekend_mae']['weekend']:.2f}", ""]
    (REPORTS_DIR / "step5_summary.md").write_text("\n".join(md))

    print("\n" + "=" * 56)
    print(f"{'Model':<16}| 5-min MAE | 10-min MAE | 20-min MAE")
    print("-" * 16 + "|-----------|------------|------------")
    for k in MODELS:
        kh = report["models"][k]["key_horizons"]
        print(f"{k:<16}|   {kh['5min']['mae']:5.2f}   |   {kh['10min']['mae']:5.2f}    |"
              f"   {kh['20min']['mae']:5.2f}")
    print("=" * 56)
    print(f"Last-Week fell back to historical average for {lw_fallback:.1%} of forecasts")
    print("Verdict:", verdict)
    print(f"Reports in {REPORTS_DIR}")
    return 0 if not losses else 1


if __name__ == "__main__":
    sys.exit(main())
