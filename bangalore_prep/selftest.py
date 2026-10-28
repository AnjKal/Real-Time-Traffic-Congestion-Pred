"""Dry run on SYNTHETIC data (random numbers, fake road grid) to check your environment and the
pipeline end to end. Nothing here is real Bangalore data.

    python selftest.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
work = Path(tempfile.mkdtemp(prefix="blr_selftest_"))
print("working in", work)

# ---- fake road grid (30 x 30 junctions, ~550 m apart, every third row one-way)
G = nx.DiGraph()
lat0, lon0, step = 12.90, 77.55, 0.005
n = 30
for r in range(n):
    for c in range(n):
        G.add_node(r * n + c, x=lon0 + c * step, y=lat0 + r * step)


def hav(a, b):
    p1, p2 = np.radians(G.nodes[a]["y"]), np.radians(G.nodes[b]["y"])
    dl = np.radians(G.nodes[b]["x"] - G.nodes[a]["x"])
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * 6371008.8 * np.arcsin(np.sqrt(h)))


for r in range(n):
    for c in range(n):
        u = r * n + c
        for dr, dc in ((0, 1), (1, 0)):
            rr, cc = r + dr, c + dc
            if rr < n and cc < n:
                v = rr * n + cc
                G.add_edge(u, v, length=hav(u, v))
                if not (dr == 0 and r % 3 == 0):          # horizontal roads in every 3rd row are one-way
                    G.add_edge(v, u, length=hav(v, u))
nx.write_graphml(G, work / "fake_roads.graphml")

# ---- 16 fake nodes in 8 areas
cells = rng.choice(n * n, 16, replace=False)
rows = []
for k, cell in enumerate(cells):
    rows.append({"node": k, "area": "Area %d" % (k // 2 + 1), "road": "Road %d" % (k + 1),
                 "lat": G.nodes[cell]["y"] + 0.0004, "lon": G.nodes[cell]["x"] - 0.0003})
fake = pd.DataFrame(rows)

# ---- fake daily table with the Kaggle column names, ~60% of node-days present
dates = pd.date_range("2022-01-01", "2024-08-09", freq="D")
shared = np.zeros(len(dates))
for t in range(1, len(dates)):
    shared[t] = 0.8 * shared[t - 1] + rng.normal(0, 1)
recs = []
for k, r in fake.iterrows():
    own = np.zeros(len(dates))
    for t in range(1, len(dates)):
        own[t] = 0.6 * own[t - 1] + rng.normal(0, 1)
    dow = np.where(dates.dayofweek >= 5, 6, 0)
    speed = 35 + 3 * shared + 2 * own + dow
    volume = 30000 - 700 * speed + rng.normal(0, 1500, len(dates))
    for t, d in enumerate(dates):
        if rng.random() < 0.6:
            recs.append({"Date": d.strftime("%Y-%m-%d"), "Area Name": r["area"], "Road/Intersection Name": r["road"],
                         "Traffic Volume": volume[t], "Average Speed": speed[t],
                         "Travel Time Index": 1.2 + (45 - speed[t]) / 40, "Congestion Level": min(100, max(0, 120 - 2 * speed[t])),
                         "Road Capacity Utilization": 60 + (45 - speed[t]),
                         "Incident Reports": int(rng.integers(0, 4)), "Environmental Impact": 90 + rng.normal(0, 5),
                         "Public Transport Usage": 40 + rng.normal(0, 5), "Traffic Signal Compliance": 80 + rng.normal(0, 5),
                         "Parking Usage": 70 + rng.normal(0, 5), "Pedestrian and Cyclist Count": int(rng.integers(50, 200)),
                         "Weather Conditions": rng.choice(["Clear", "Overcast", "Rain", "Fog"]),
                         "Roadwork and Construction Activity": rng.choice(["Yes", "No"])})
csv = work / "synthetic_traffic.csv"
pd.DataFrame(recs).to_csv(csv, index=False)

out = work / "data" / "BANGALORE"
out.mkdir(parents=True)


def run(*a):
    print("\n$ python " + " ".join(str(x) for x in a))
    subprocess.check_call([sys.executable, *map(str, a)], cwd=work)


run(HERE / "01_inspect_data.py", "--csv", csv, "--out", out)
nodes = pd.read_csv(out / "nodes.csv")
nodes["lat"] = fake["lat"].values
nodes["lon"] = fake["lon"].values
nodes.to_csv(out / "nodes.csv", index=False)           # pretend geocoding is done
run(HERE / "02_build_graph.py", "--csv", csv, "--out", out, "--graphml", work / "fake_roads.graphml")
run(HERE / "03_build_dataset.py", "--csv", csv, "--graph-dir", out)

# ---- checks
tr = np.load(work / "data/BANGALORE/train.npz")
te = np.load(work / "data/BANGALORE/test.npz")
sp = np.load(work / "data/BANGALORE_SPEED/train.npz")
A = np.load(out / "adj_mx.npy")
assert tr["x"].shape[1:3] == (12, 16) and tr["y"].shape[1:3] == (12, 16), tr["x"].shape
assert sp["x"].shape[-1] == 2 and tr["x"].shape[-1] > 2
assert A.shape == (16, 16) and np.allclose(np.diag(A), 1)
assert np.isfinite(tr["x"]).all() and np.isfinite(tr["y"]).all()
assert (tr["y"][..., 0] >= 0).all()
print("\nSELFTEST OK  x%s  y%s  test x%s  adjacency edges (no self): %d" %
      (tr["x"].shape, tr["y"].shape, te["x"].shape, int((A > 0).sum() - 16)))
