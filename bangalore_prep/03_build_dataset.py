"""STEP 3 - build the DCRNN training files with MANY input features and speed as the target.

    python bangalore_prep/03_build_dataset.py --csv bangalore_prep/data/bangalore_traffic.csv --graph-dir bangalore_prep/data/BANGALORE

Writes
  <graph-dir>/bangalore_features.h5      multi-key HDF5: one (days x nodes) table per feature
  <graph-dir>/bangalore_speed.h5         speed only, in the exact single-table layout DCRNN's own scripts use
  <graph-dir>/{train,val,test}.npz        x,y with F channels (channel 0 = speed, the prediction target)
  <speed-dir>/{train,val,test}.npz        baseline: speed + day-of-week only (same as METR-LA's setup)
  feature_stats.json                     the exact normalisation, needed again when streaming live data
  dcrnn_bangalore_multi.yaml / dcrnn_bangalore_speed.yaml   training configs with num_nodes / input_dim filled in
Windows are 12 steps in, 12 steps out. The data is DAILY, so a "step" is one day.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import common


def make_windows(X, Y, seq, horizon):
    """X, Y: (T, N, F). Returns x (S, seq, N, F) and y (S, horizon, N, F), same sampling as DCRNN."""
    from numpy.lib.stride_tricks import sliding_window_view
    S = X.shape[0] - seq - horizon + 1
    xw = sliding_window_view(X, seq, axis=0)[:S]                 # (S, N, F, seq)
    yw = sliding_window_view(Y, horizon, axis=0)[seq:seq + S]    # (S, N, F, horizon)
    return xw.transpose(0, 3, 1, 2).copy(), yw.transpose(0, 3, 1, 2).copy()


def save_split(out_dir, x, y, info, seq, horizon):
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = {"train": (0, info["n_train"]),
              "val": (info["n_train"], info["n_train"] + info["n_val"]),
              "test": (info["n_train"] + info["n_val"], info["S"])}
    x_off = np.arange(-(seq - 1), 1).reshape(-1, 1)
    y_off = np.arange(1, horizon + 1).reshape(-1, 1)
    for name, (a, b) in bounds.items():
        np.savez_compressed(out_dir / ("%s.npz" % name), x=x[a:b].astype(np.float32), y=y[a:b].astype(np.float32),
                            x_offsets=x_off, y_offsets=y_off)
        print("  %s: x %s  y %s" % (name, x[a:b].shape, y[a:b].shape))


def write_config(path, dataset_dir, graph_pkl, N, F, batch_size, tag):
    cfg = {
        "base_dir": "data/model/bangalore_%s" % tag,
        "log_level": "INFO",
        "data": {"batch_size": batch_size, "dataset_dir": str(dataset_dir).replace("\\", "/"),
                 "graph_pkl_filename": str(graph_pkl).replace("\\", "/"),
                 "test_batch_size": batch_size, "val_batch_size": batch_size},
        "model": {"cl_decay_steps": 2000, "filter_type": "dual_random_walk", "horizon": 12, "input_dim": F,
                  "l1_decay": 0, "max_diffusion_step": 2, "num_nodes": N, "num_rnn_layers": 2,
                  "output_dim": 1, "rnn_units": 64, "seq_len": 12, "use_curriculum_learning": True},
        "train": {"base_lr": 0.01, "dropout": 0, "epoch": 0, "epochs": 100, "epsilon": 0.001,
                  "global_step": 0, "lr_decay_ratio": 0.1, "max_grad_norm": 5, "max_to_keep": 100,
                  "min_learning_rate": 2.0e-06, "optimizer": "adam", "patience": 50,
                  "steps": [20, 30, 40, 50], "test_every_n_epochs": 10},
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--graph-dir", default="bangalore_prep/data/BANGALORE", help="folder written by step 2")
    ap.add_argument("--speed-dir", default="bangalore_prep/data/BANGALORE_SPEED")
    ap.add_argument("--fill", choices=["none", "ffill"], default="none",
                    help="none: missing speed stays 0 (masked in the loss, like METR-LA). ffill: carry the last value "
                         "forward into INPUTS only (never into targets)")
    ap.add_argument("--ffill-limit", type=int, default=3)
    ap.add_argument("--exclude", nargs="*", default=[], help="normalised column names to leave out, e.g. congestion_level")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()

    gdir = Path(args.graph_dir)
    nodes = pd.read_csv(gdir / "nodes.csv")
    ids = (gdir / "graph_sensor_ids.txt").read_text().strip().split(",")
    assert list(nodes["node_id"]) == ids, "nodes.csv and graph_sensor_ids.txt disagree on node order"
    df = common.load_raw(args.csv)
    panel, kinds = common.build_panel(df, nodes[["node_id", "name", "area", "road"]], exclude=set(args.exclude))
    dates = panel["speed"].index
    T, N = panel["speed"].shape
    info = common.split_info(T, args.seq, args.horizon)
    train_end = info["stats_end"]
    print("days %d, nodes %d, samples %d (train %d / val %d / test %d)" %
          (T, N, info["S"], info["n_train"], info["n_val"], info["n_test"]))

    # ---- channel order: speed first, then numeric inputs, binary inputs, observed-flag, calendar
    numeric = [k for k, v in kinds.items() if v == "numeric" and k != "speed"]
    binary = [k for k, v in kinds.items() if v == "binary"]
    observed = panel["speed"].notna().astype(float)

    def causal_fill(p):
        return p.ffill(limit=args.ffill_limit) if args.fill == "ffill" else p

    chans, names, stats = [], [], {}
    sp_x = causal_fill(panel["speed"]).fillna(0.0).values          # input speed (raw units, 0 = missing)
    sp_y = panel["speed"].fillna(0.0).values                        # target speed: NEVER filled
    names.append("speed")
    for k in numeric:
        p = causal_fill(panel[k])
        mu = np.nanmean(p.iloc[:train_end].values)
        sd = np.nanstd(p.iloc[:train_end].values) or 1.0
        stats[k] = {"mean": float(mu), "std": float(sd)}
        chans.append(((p - mu) / sd).fillna(0.0).values)
        names.append(k)
    for k in binary:
        chans.append(causal_fill(panel[k]).fillna(0.0).values)
        names.append(k)
    chans.append(observed.values)
    names.append("observed")
    dow = np.repeat((dates.dayofweek.values / 7.0)[:, None], N, axis=1)
    wk = np.repeat((dates.dayofweek.values >= 5).astype(float)[:, None], N, axis=1)
    chans += [dow, wk]
    names += ["day_of_week", "is_weekend"]

    X = np.stack([sp_x] + chans, axis=-1).astype(np.float32)        # (T, N, F)
    Y = X.copy()
    Y[..., 0] = sp_y
    F = X.shape[-1]
    print("input channels (F=%d): %s" % (F, names))

    # ---- multi-feature dataset
    xw, yw = make_windows(X, Y, args.seq, args.horizon)
    print("multi-feature dataset ->", gdir)
    save_split(gdir, xw, yw, info, args.seq, args.horizon)

    # ---- speed-only baseline (channels: speed, day_of_week) = DCRNN's original input layout
    keep = [0, names.index("day_of_week")]
    xb, yb = make_windows(X[..., keep], Y[..., keep], args.seq, args.horizon)
    print("speed-only baseline dataset ->", args.speed_dir)
    save_split(Path(args.speed_dir), xb, yb, info, args.seq, args.horizon)

    # ---- h5 files
    panel_h5 = gdir / "bangalore_features.h5"
    if panel_h5.exists():
        panel_h5.unlink()
    for k, p in panel.items():
        p.to_hdf(panel_h5, key=k.replace("=", "_"), mode="a")
    observed.to_hdf(panel_h5, key="observed", mode="a")
    panel["speed"].fillna(0.0).to_hdf(gdir / "bangalore_speed.h5", key="df", mode="w")
    print("wrote %s and %s" % (panel_h5, gdir / "bangalore_speed.h5"))

    stats["_channels"] = names
    stats["_train_end_index"] = int(train_end)
    stats["_fill"] = args.fill
    (gdir / "feature_stats.json").write_text(json.dumps(stats, indent=2))

    pkl = gdir / "adj_bangalore.pkl"
    write_config(gdir / "dcrnn_bangalore_multi.yaml", gdir, pkl, N, F, args.batch_size, "multi")
    write_config(gdir / "dcrnn_bangalore_speed.yaml", args.speed_dir, pkl, N, 2, args.batch_size, "speed")
    print("configs written next to the graph files. Missing speed (0) share of targets: %.1f%%" % (100 * (sp_y == 0).mean()))


if __name__ == "__main__":
    main()
