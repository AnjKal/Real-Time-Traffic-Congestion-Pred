"""Step 5.4: reports/step5_report.html (Plotly.js from CDN, data embedded)."""
import json
import webbrowser

import numpy as np
import pandas as pd

from src.config import GRAPH_DIR, PROJECT_ROOT
from scripts.step5_evaluate import KEY_HORIZONS, MODELS, REPORTS_DIR

OUT = REPORTS_DIR / "step5_report.html"
REP_NODES = [(4, "CBD: M.G. Road"), (15, "ORR: Sarjapur"), (0, "Fringe: Hebbal")]


def main() -> None:
    metrics = json.loads((REPORTS_DIR / "step5_metrics.json").read_text())
    d = np.load(REPORTS_DIR / "step5_predictions.npz")
    truth, pred = d["truth"], d["dcrnn"]                                  # (N, H, nodes) km/h
    ts = pd.to_datetime(d["target_ts"], unit="s")                         # (N, H)
    names = pd.read_csv(GRAPH_DIR / "nodes.csv").sort_values("node_id")["name"].tolist()
    rng = np.random.default_rng(0)
    plots = []

    # A: predicted vs actual, one subplot per horizon, coloured by node
    n_pts = 1500
    traces, layout = [], {"height": 420, "showlegend": False}
    for i, (label, h) in enumerate(KEY_HORIZONS.items()):
        sel = rng.choice(truth.shape[0] * truth.shape[2], n_pts, replace=False)
        w, n = np.divmod(sel, truth.shape[2])
        ax = "" if i == 0 else str(i + 1)
        traces.append({"x": truth[w, h, n].round(2).tolist(), "y": pred[w, h, n].round(2).tolist(),
                       "mode": "markers", "marker": {"size": 3, "color": n.tolist(),
                                                     "colorscale": "Turbo", "opacity": 0.6},
                       "xaxis": f"x{ax}", "yaxis": f"y{ax}", "text": [names[k] for k in n]})
        hi = float(max(truth[:, h].max(), pred[:, h].max()))
        traces.append({"x": [0, hi], "y": [0, hi], "mode": "lines", "line": {"color": "black"},
                       "xaxis": f"x{ax}", "yaxis": f"y{ax}"})
        lo, hi_d = i / 3 + 0.02, (i + 1) / 3 - 0.02
        layout[f"xaxis{ax}"] = {"domain": [lo, hi_d], "title": f"actual km/h ({label})"}
        layout[f"yaxis{ax}"] = {"domain": [0, 1], "anchor": f"x{ax}",
                                **({"title": "predicted km/h"} if i == 0 else {})}
    plots.append(("A. Predicted vs actual", traces, layout))

    # B: MAE over time (hourly buckets of the target time)
    hours = ts[:, 0].floor("h")
    traces = []
    for label, h in KEY_HORIZONS.items():
        err = np.abs(pred[:, h] - truth[:, h]).mean(axis=1)
        s = pd.Series(err).groupby(hours.to_numpy()).mean()
        traces.append({"x": s.index.strftime("%Y-%m-%d %H:%M").tolist(), "y": s.round(3).tolist(),
                       "name": label, "mode": "lines"})
    plots.append(("B. Test MAE over time", traces, {"yaxis": {"title": "MAE km/h"}}))

    # C: per-node MAE
    node_mae = metrics["models"]["DCRNN"]["per_node_mae"]
    plots.append(("C. DCRNN MAE per node", [
        {"x": names, "y": [round(v, 3) for v in node_mae], "type": "bar",
         "marker": {"color": list(range(len(names))), "colorscale": "Turbo"}}],
        {"yaxis": {"title": "MAE km/h"}, "margin": {"b": 140}}))

    # D: residual histograms
    traces = []
    for label, h in KEY_HORIZONS.items():
        counts, edges = np.histogram((pred[:, h] - truth[:, h]).ravel(), bins=80, range=(-15, 15))
        traces.append({"x": ((edges[:-1] + edges[1:]) / 2).round(2).tolist(),
                       "y": counts.tolist(), "type": "bar", "name": label, "opacity": 0.6})
    plots.append(("D. Residuals (pred - actual)", traces,
                  {"barmode": "overlay", "xaxis": {"title": "km/h"}}))

    # E: a 24-hour weekday slice at the ~10-min horizon for three nodes
    h = KEY_HORIZONS["10min"]
    days = ts[:, h].normalize()
    day = next(dd for dd in sorted(days.unique()) if dd.dayofweek < 5 and (days == dd).sum() >= 470)
    sel = np.flatnonzero(days == day)
    traces = []
    for node, label in REP_NODES:
        x = ts[sel, h].strftime("%H:%M").tolist()
        traces += [{"x": x, "y": truth[sel, h, node].round(2).tolist(), "name": f"{label} actual",
                    "mode": "lines"},
                   {"x": x, "y": pred[sel, h, node].round(2).tolist(),
                    "name": f"{label} DCRNN", "mode": "lines", "line": {"dash": "dot"}}]
    plots.append((f"E. 24-hour slice, 10-min horizon ({day:%Y-%m-%d})", traces,
                  {"yaxis": {"title": "km/h"}}))

    # F: baseline comparison
    traces = [{"x": list(KEY_HORIZONS), "type": "bar", "name": m,
               "y": [round(metrics["models"][m]["key_horizons"][k]["mae"], 3) for k in KEY_HORIZONS]}
              for m in MODELS]
    plots.append(("F. MAE by model and horizon", traces,
                  {"barmode": "group", "yaxis": {"title": "MAE km/h"}}))

    divs = "\n".join(f'<h3>{t}</h3><div id="p{i}"></div>' for i, (t, _, _) in enumerate(plots))
    calls = "\n".join(
        f"Plotly.newPlot('p{i}', {json.dumps(tr)}, Object.assign({{margin:{{t:20}}}}, "
        f"{json.dumps(lay)}));" for i, (_, tr, lay) in enumerate(plots))
    OUT.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Step 5 report</title>'
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script></head>'
        f'<body style="font-family:sans-serif;max-width:1200px;margin:auto">{divs}'
        f"<script>{calls}</script></body></html>", encoding="utf-8")
    print(f"Saved {OUT.resolve()}")
    webbrowser.open(OUT.resolve().as_uri())


if __name__ == "__main__":
    main()
