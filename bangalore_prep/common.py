"""Shared helpers for the Bangalore Traffic Pulse -> DCRNN pipeline."""
import re
import sys

import numpy as np
import pandas as pd

# column aliases (after lower-casing and replacing non-alphanumerics with "_")
ALIASES = {
    "date": ["date", "timestamp", "datetime", "time"],
    "area": ["area_name", "area", "locality"],
    "road": ["road_intersection_name", "road_intersection", "road_name", "intersection", "road"],
    "speed": ["average_speed", "avg_speed", "speed"],
}
RESERVED = {"date", "area", "road", "node_id"}
BINARY_WORDS = {"yes", "no", "true", "false", "0", "1"}
YES_WORDS = {"yes", "true", "1"}


def norm(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def load_raw(csv_path):
    """Read the CSV, normalise column names, map date/area/road/speed. Returns a DataFrame."""
    df = pd.read_csv(csv_path)
    df.columns = [norm(c) for c in df.columns]
    rename = {}
    for key, options in ALIASES.items():
        hit = next((o for o in options if o in df.columns), None)
        if hit is None:
            sys.exit("Could not find a '%s' column. Columns in your file: %s" % (key, list(df.columns)))
        rename[hit] = key
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    bad = int(df["date"].isna().sum())
    if bad:
        print("WARNING: dropping %d rows with an unreadable date" % bad)
        df = df.dropna(subset=["date"])
    for c in ("area", "road"):
        df[c] = df[c].astype(str).str.strip()
    return df


def build_nodes(df):
    """One node per (area, road) pair, in a fixed sorted order."""
    nodes = (df[["area", "road"]].drop_duplicates().sort_values(["area", "road"]).reset_index(drop=True))
    nodes["node_id"] = ["n%02d" % (i + 1) for i in range(len(nodes))]
    nodes["name"] = nodes["road"] + " (" + nodes["area"] + ")"
    return nodes[["node_id", "name", "area", "road"]]


def build_panel(df, nodes, exclude=()):
    """Return (panel, kinds): panel[feature] is a DataFrame (daily dates x node_id) with NaN where unobserved.

    kinds[feature] is 'numeric' or 'binary' (one-hot / yes-no columns are 'binary').
    Duplicate (date, node) rows are averaged.
    """
    df = df.merge(nodes[["area", "road", "node_id"]], on=["area", "road"], how="inner")
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    cols = list(nodes["node_id"])
    panel, kinds = {}, {}

    def pivot(series_name, values):
        tmp = pd.DataFrame({"date": df["date"], "node_id": df["node_id"], "v": values})
        p = tmp.pivot_table(index="date", columns="node_id", values="v", aggfunc="mean")
        return p.reindex(index=dates, columns=cols)

    panel["speed"] = pivot("speed", pd.to_numeric(df["speed"], errors="coerce"))
    kinds["speed"] = "numeric"
    for c in df.columns:
        if c in RESERVED or c == "speed" or c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            p = pivot(c, s.astype(float))
            if np.nanstd(p.values) > 0:
                panel[c], kinds[c] = p, "numeric"
            continue
        low = s.astype(str).str.strip().str.lower()
        vals = sorted(low.unique())
        if set(vals) <= BINARY_WORDS:
            panel[c], kinds[c] = pivot(c, low.isin(YES_WORDS).astype(float)), "binary"
        elif len(vals) <= 8:
            for v in vals:
                name = "%s=%s" % (c, norm(v))
                panel[name], kinds[name] = pivot(name, (low == v).astype(float)), "binary"
        else:
            print("NOTE: skipping text column '%s' (%d distinct values)" % (c, len(vals)))
    return panel, kinds


def split_info(T, seq=12, horizon=12, train=0.7, test=0.2):
    """Same sample split rule as DCRNN's generate_training_data.py (chronological)."""
    S = T - seq - horizon + 1
    n_test = int(round(S * test))
    n_train = int(round(S * train))
    n_val = S - n_train - n_test
    return {"S": S, "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "stats_end": n_train + seq - 1}   # first timestep index NOT seen by any training input window
