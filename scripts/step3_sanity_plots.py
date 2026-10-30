"""Step 3.6: notebooks/step3_data_sanity.html (Plotly.js loaded from CDN, data embedded)."""
import json
import webbrowser

import numpy as np
import pandas as pd

from src.config import GRAPH_DIR, NUM_NODES, PROJECT_ROOT
from src.synthetic.generator import PARQUET_PATH, load_inputs
from src.synthetic.scenarios import SCENARIOS
from src.synthetic.validate import load_grids

OUT = PROJECT_ROOT / "notebooks" / "step3_data_sanity.html"
REP_NODES = [10, 4, 0]     # Silk Board, M.G. Road, Hebbal


def main() -> None:
    ff, W = load_inputs()
    names = pd.read_csv(GRAPH_DIR / "nodes.csv").sort_values("node_id")["name"].tolist()
    speed, labels, weekend = load_grids()
    ts = pd.DatetimeIndex(pd.read_parquet(PARQUET_PATH, columns=["timestamp"])["timestamp"]
                          .to_numpy()[::NUM_NODES])
    plots = []

    # 1. one week (Mon 2026-08-03 .. Sun 2026-08-09) for three nodes
    wk = (ts >= "2026-08-03") & (ts < "2026-08-10")
    plots.append(("Speed over one week (Mon-Sun)", [
        {"x": ts[wk].strftime("%Y-%m-%d %H:%M").tolist(), "y": speed[wk, n].round(1).tolist(),
         "name": names[n], "type": "scatter", "mode": "lines"} for n in REP_NODES],
        {"yaxis": {"title": "km/h"}}))

    # 2. weekday 24h average per node
    hod = ts.hour.to_numpy() * 60 + ts.minute.to_numpy()
    wd = ~weekend
    mins = np.unique(hod)
    prof = np.stack([speed[wd & (hod == m)].mean(axis=0) for m in mins])
    plots.append(("Weekday 24-hour average speed per node", [
        {"x": (mins / 60).tolist(), "y": prof[:, n].round(1).tolist(), "name": names[n],
         "type": "scatter", "mode": "lines"} for n in range(NUM_NODES)],
        {"xaxis": {"title": "hour of day"}, "yaxis": {"title": "km/h"}}))

    # 3. scenario proportions
    plots.append(("Scenario proportions", [
        {"x": [s.name for s in SCENARIOS], "y": [s.target_prop for s in SCENARIOS],
         "name": "target", "type": "bar"},
        {"x": [s.name for s in SCENARIOS],
         "y": [round(float((labels == s.id).mean()), 4) for s in SCENARIOS],
         "name": "actual", "type": "bar"}], {"barmode": "group"}))

    # 4. cross-correlation heatmap
    plots.append(("Cross-correlation between nodes", [
        {"z": np.corrcoef(speed.T).round(3).tolist(), "x": names, "y": names,
         "type": "heatmap", "colorscale": "Viridis"}], {"height": 650}))

    # 5. one incident with downstream propagation
    idx = np.flatnonzero(labels == 8)
    start = int(idx[0])
    end = start + int(np.flatnonzero(np.diff(idx) > 1)[0] + 1) if (np.diff(idx) > 1).any() else start
    ratio = speed[start:end + 1] / ff
    hit = int(np.argmin(ratio.mean(axis=0)))
    w = W[hit].copy()
    w[hit] = -1
    nbrs = [int(j) for j in np.argsort(-w)[:2]]
    lo, hi = max(0, start - 20), min(len(ts), end + 30)
    plots.append((f"Incident at {names[hit]} ({ts[start]:%Y-%m-%d %H:%M}) and two neighbors", [
        {"x": ts[lo:hi].strftime("%H:%M").tolist(), "y": (speed[lo:hi, n] / ff[n]).round(3).tolist(),
         "name": names[n] + (" (incident)" if n == hit else ""), "type": "scatter",
         "mode": "lines"} for n in [hit] + nbrs],
        {"yaxis": {"title": "speed / free-flow"}}))

    divs = "\n".join(f'<h3>{t}</h3><div id="p{i}"></div>' for i, (t, _, _) in enumerate(plots))
    calls = "\n".join(
        f"Plotly.newPlot('p{i}', {json.dumps(d)}, Object.assign({{margin:{{t:20}}}}, "
        f"{json.dumps(lay)}));" for i, (_, d, lay) in enumerate(plots))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Step 3 data sanity</title>'
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script></head>'
        f'<body style="font-family:sans-serif;max-width:1200px;margin:auto">{divs}'
        f"<script>{calls}</script></body></html>", encoding="utf-8")
    print(f"Saved {OUT.resolve()}")
    webbrowser.open(OUT.resolve().as_uri())


if __name__ == "__main__":
    main()
