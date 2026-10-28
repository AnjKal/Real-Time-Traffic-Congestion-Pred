"""Write adj_bangalore.pkl in DCRNN's format. RUN THIS INSIDE YOUR TF1 ('dcrnn') ENVIRONMENT
so the pickle is readable by its old NumPy:   python npy_to_pkl.py --graph-dir data/BANGALORE
"""
import argparse
import os
import pickle

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--graph-dir", default="data/BANGALORE")
ap.add_argument("--adj", default="adj_mx.npy")
ap.add_argument("--out", default="adj_bangalore.pkl")
a = ap.parse_args()

with open(os.path.join(a.graph_dir, "graph_sensor_ids.txt")) as f:
    ids = [s.strip() for s in f.read().strip().split(",")]
adj = np.load(os.path.join(a.graph_dir, a.adj)).astype(np.float32)
assert adj.shape == (len(ids), len(ids)), (adj.shape, len(ids))
with open(os.path.join(a.graph_dir, a.out), "wb") as f:
    pickle.dump((ids, {s: i for i, s in enumerate(ids)}, adj), f, protocol=2)
print("wrote %s  (%d nodes, numpy %s)" % (os.path.join(a.graph_dir, a.out), len(ids), np.__version__))
