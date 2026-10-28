"""STEP 1 - look at the raw CSV before building anything.

    python 01_inspect_data.py --csv data/bangalore_traffic.csv --out data/BANGALORE

Prints the schema, how many nodes (area+road) exist, how complete the daily data is, and two
sanity checks that decide whether DCRNN can learn anything at all (day-to-day autocorrelation of
speed, and how correlated different roads are). Also writes nodes.csv (lat/lon blank) for step 2.
Please paste the printed output back if anything looks odd.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/BANGALORE")
    args = ap.parse_args()

    df = common.load_raw(args.csv)
    nodes = common.build_nodes(df)
    panel, kinds = common.build_panel(df, nodes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== FILE ===")
    print("rows: %d   columns: %s" % (len(df), list(df.columns)))
    print("dates: %s -> %s  (%d calendar days)" % (df.date.min().date(), df.date.max().date(), len(panel["speed"])))
    print("nodes (area + road): %d" % len(nodes))
    print(nodes.to_string(index=False))
    dup = int(df.duplicated(["date", "area", "road"]).sum())
    print("duplicate (date, node) rows: %d  (averaged if any)" % dup)
    per_road_areas = df.groupby("road")["area"].nunique()
    if (per_road_areas > 1).any():
        print("NOTE: these road names appear in more than one area: %s" % list(per_road_areas[per_road_areas > 1].index))

    print("\n=== FEATURES THAT WILL BE USED AS INPUTS ===")
    for k, v in kinds.items():
        tag = "TARGET" if k == "speed" else v
        obs = panel[k].notna().values.mean() * 100
        print("  %-42s %-8s observed %.0f%% of node-days" % (k, tag, obs))

    sp = panel["speed"]
    obs = sp.notna()
    print("\n=== COVERAGE (speed) ===")
    print("overall observed node-days: %.1f%%" % (obs.values.mean() * 100))
    print("rows per calendar day: min %d / median %d / max %d (of %d nodes)" %
          (obs.sum(1).min(), obs.sum(1).median(), obs.sum(1).max(), len(nodes)))
    cov = obs.mean().rename("observed_fraction")
    print(cov.round(2).to_string())
    longest = []
    for c in sp.columns:
        run = best = 0
        for v in obs[c].values:
            run = run + 1 if v else 0
            best = max(best, run)
        longest.append(best)
    print("longest run of consecutive observed days per node: min %d / median %d / max %d" %
          (min(longest), int(np.median(longest)), max(longest)))
    print("(DCRNN needs 12 days of history; the longer the runs, the more usable windows)")

    print("\n=== CAN A FORECASTER LEARN ANYTHING? ===")
    ac = []
    for c in sp.columns:
        s = sp[c]
        pair = pd.concat([s, s.shift(1)], axis=1).dropna()
        ac.append(pair.corr().iloc[0, 1] if len(pair) > 30 else np.nan)
    ac = pd.Series(ac, index=sp.columns)
    print("lag-1 (day-to-day) autocorrelation of speed: median %.2f  (min %.2f, max %.2f)" %
          (ac.median(), ac.min(), ac.max()))
    dow = sp.groupby(sp.index.dayofweek).mean().mean(axis=1)
    print("mean speed by weekday (Mon..Sun): %s" % np.round(dow.values, 1).tolist())
    cc = sp.corr(min_periods=60).values
    off = cc[~np.eye(len(cc), dtype=bool)]
    print("cross-road correlation of speed: median %.2f, 90th pct %.2f" % (np.nanmedian(off), np.nanpercentile(off, 90)))
    print("Reading: autocorrelation near 0 and cross-road correlation near 0 means the speed series is close to")
    print("noise at daily resolution. No graph or model can forecast that, so tell me before training.")

    nf = out / "nodes.csv"
    if nf.exists():
        print("\n%s already exists, left untouched." % nf)
    else:
        nodes = nodes.assign(lat=np.nan, lon=np.nan, geocode_note="")
        nodes.to_csv(nf, index=False)
        print("\nwrote %s (lat/lon empty; step 2 fills them or you can type them in)" % nf)


if __name__ == "__main__":
    main()
