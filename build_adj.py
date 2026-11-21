"""
Build DCRNN adjacency matrix (adj_mx.pkl) for the Bangalore junctions in the Excel file (20 or 100 or any N) using OSRM.

Usage:
    python build_adj.py dcrnn_20_nodes.xlsx          # or a .csv: python build_adj.py bangalore_traffic_spots.csv
    python build_adj.py dcrnn_20_nodes.xlsx --threshold 0.05 --k 5
    python build_adj.py dcrnn_20_nodes.xlsx --offline      # no internet: haversine x1.3 estimate (testing only)
"""
import argparse, json, pickle, sys, time, urllib.request
import numpy as np
import pandas as pd

# ---------- STEP 1: nodes come from the Excel file (order = teammate column order) ----------
OSRM_URLS = [   # tried in order; if one is overloaded the next is used
    "https://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance",
    "https://routing.openstreetmap.de/routed-car/table/v1/driving/{coords}?annotations=distance",
]


def load_coordinates(path):
    """STEP 2: read the nodes (row order = node order). Accepts .xlsx (needs openpyxl) or .csv (no extra install).
    Excel columns: Node ID, Name, Latitude, Longitude.  CSV columns: name, lat, lon (IDs J001.. are generated)."""
    if path.lower().endswith(".csv"):
        d = pd.read_csv(path)
        df = pd.DataFrame({"id": [f"J{i:03d}" for i in range(1, len(d) + 1)],
                           "name": d["name"], "lat": d["lat"], "lon": d["lon"]})
    else:
        df = pd.read_excel(path)[["Node ID", "Name", "Latitude", "Longitude"]]
        df.columns = ["id", "name", "lat", "lon"]
    return df


def _osrm_call(coords_list, tries=4):
    """One OSRM Table request for the given (lat, lon) list. Retries, then falls back to the 2nd server."""
    coords = ";".join(f"{lon},{lat}" for lat, lon in coords_list)                 # OSRM wants lon,lat
    for url in OSRM_URLS:
        for attempt in range(tries):
            try:
                req = urllib.request.Request(url.format(coords=coords),
                                             headers={"User-Agent": "rvce-dcrnn-student-project"})
                data = json.load(urllib.request.urlopen(req, timeout=60))
                if data.get("code") == "Ok" and data.get("distances"):
                    return np.array([[np.inf if v is None else v for v in row] for row in data["distances"]], dtype=float)
                print("OSRM replied:", data.get("code"), data.get("message", ""))
            except Exception as e:
                print(f"OSRM attempt {attempt+1}/{tries} failed ({e}); waiting {5*2**attempt}s")
            time.sleep(5 * 2 ** attempt)
    sys.exit("All OSRM servers busy. Try again in a few minutes, or ask a teammate who succeeded to share dist_m.npy.")


def osrm_distance_matrix(nodes, block=25):
    """STEP 3: road distance (metres) between every pair of nodes.
    Big requests overload the free server, so we cut the N nodes into blocks of `block`
    and ask for two blocks at a time (<= 2*block points per request), then stitch the answers."""
    pts = list(zip(nodes.lat, nodes.lon)); n = len(pts)
    cuts = [range(i, min(i + block, n)) for i in range(0, n, block)]
    D = np.zeros((n, n)); calls = 0; total = len(cuts) * (len(cuts) + 1) // 2
    for bi in range(len(cuts)):
        for bj in range(bi, len(cuts)):
            A, B = list(cuts[bi]), list(cuts[bj])
            idx = A if bi == bj else A + B
            M = _osrm_call([pts[i] for i in idx])
            for r, i in enumerate(idx):
                for c, j in enumerate(idx):
                    D[i, j] = M[r, c]                      # fills both directions (roads can be one-way)
            calls += 1; print(f"OSRM request {calls}/{total} done", flush=True)
            time.sleep(1.2)                                # stay polite: ~1 request per second
    np.fill_diagonal(D, 0)
    return D


def haversine_matrix(nodes, detour=1.3):
    """Offline stand-in: straight-line distance x 1.3 (roads are longer than straight lines)."""
    lat, lon = np.radians(nodes.lat.values), np.radians(nodes.lon.values)
    dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a)) * detour


def to_weights(dist, threshold=0.1, k=None):
    """STEP 4: turn distances into 0-1 closeness weights (Gaussian kernel)."""
    sigma = dist[np.isfinite(dist)].std()           # how fast closeness falls with distance
    adj = np.exp(-np.square(dist / sigma))          # near -> ~1, far -> ~0; diagonal = exp(0) = 1
    adj[adj < threshold] = 0                        # drop weak links = "not directly connected"
    if k is not None:                               # optional: keep only the k strongest links per node
        for i in range(len(adj)):
            off = np.delete(np.arange(len(adj)), i)
            weakest = off[np.argsort(adj[i, off])[:-k]]
            adj[i, weakest] = 0
    return adj.astype(np.float32), sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", help="nodes file: .xlsx or .csv")
    ap.add_argument("--k", type=int, default=None, help="keep only k nearest neighbours per node")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    nodes = load_coordinates(a.xlsx)
    dist = haversine_matrix(nodes) if a.offline else osrm_distance_matrix(nodes)
    adj, sigma = to_weights(dist, a.threshold, a.k)

    # STEP 5: sparsity report -> is the graph too dense / too empty?
    nbrs = (adj > 0).sum(1) - 1
    print(f"sigma = {sigma/1000:.1f} km | neighbours per node: min {nbrs.min()}, "
          f"avg {nbrs.mean():.1f}, max {nbrs.max()} (of {len(adj)-1} possible)")

    # STEP 6: save. pkl = what DCRNN reads; csv files = for teammates to open/inspect.
    ids = nodes.id.tolist()
    pickle.dump([ids, {s: i for i, s in enumerate(ids)}, adj], open("adj_mx.pkl", "wb"))
    nodes.to_csv("nodes.csv", index=False)
    np.save("dist_m.npy", dist)                      # raw metres, used by tune_graph.py
    pd.DataFrame(dist / 1000, index=ids, columns=ids).round(2).to_csv("distance_km.csv")
    pd.DataFrame(adj, index=ids, columns=ids).round(3).to_csv("adj_mx.csv")

    # STEP 7: reload exactly the way DCRNN does, to prove the file is valid.
    with open("adj_mx.pkl", "rb") as f:
        sid, s2i, m = pickle.load(f, encoding="latin1")
    N = len(nodes)
    assert m.shape == (N, N) and m.dtype == np.float32 and len(sid) == N
    print("adj_mx.pkl OK:", m.shape, m.dtype, f"| set num_nodes: {N} in the DCRNN config")


if __name__ == "__main__":
    main()