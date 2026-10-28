"""STEP 2 - build the DCRNN adjacency matrix from the REAL road network.

    python 02_build_graph.py --csv data/bangalore_traffic.csv --out data/BANGALORE

What it does
  1. Gives every node (area + road/intersection) a latitude/longitude. Missing ones are looked up on
     OpenStreetMap (Nominatim); results are cached in nodes.csv, which you can edit by hand
     (right-click a place in Google Maps -> copy the coordinates).
  2. Downloads the OpenStreetMap drivable road network once (cached as .graphml) and snaps each node to it.
  3. Computes DIRECTED driving distances between every pair of nodes along real roads (one-way streets
     make d[i,j] != d[j,i]).
  4. Decides which pairs are DIRECTLY connected: the shortest driving route i -> j must not pass
     within --pass-m metres of any other node, and must be shorter than --max-direct-km.
  5. Weights direct edges with the DCRNN Gaussian kernel exp(-(d/sigma)^2), threshold 0.1.
  6. Optionally blends in a traffic-similarity graph (computed from TRAINING days only).
Everything is saved as separate component files so you can run ablations.
"""
import argparse
import json
import math
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import common

NOMINATIM = "https://nominatim.openstreetmap.org/search"
# loose box around urban Bengaluru; a geocoded point outside it is almost certainly the wrong place
BLR_BBOX = {"south": 12.80, "north": 13.20, "west": 77.40, "east": 77.85}


# ----------------------------------------------------------------------------- geometry
def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres; broadcasts over numpy arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------- geocoding
def geocode(road, area):
    import requests
    headers = {"User-Agent": "dcrnn-bangalore-student-project/1.0"}
    view = "%f,%f,%f,%f" % (BLR_BBOX["west"], BLR_BBOX["north"], BLR_BBOX["east"], BLR_BBOX["south"])
    for q in ("%s, %s, Bengaluru, Karnataka, India" % (road, area), "%s, Bengaluru, Karnataka, India" % road):
        r = requests.get(NOMINATIM, headers=headers, timeout=30,
                         params={"q": q, "format": "jsonv2", "limit": 1, "viewbox": view, "bounded": 1})
        time.sleep(1.1)                                   # Nominatim usage policy: max 1 request/second
        r.raise_for_status()
        hits = r.json()
        if hits:
            return float(hits[0]["lat"]), float(hits[0]["lon"]), hits[0].get("display_name", "")[:120]
    return None


def fill_coordinates(nodes, nodes_csv, allow_flagged):
    nodes = nodes.copy()
    for i, r in nodes.iterrows():
        if pd.notna(r["lat"]) and pd.notna(r["lon"]):
            continue
        print("geocoding: %s" % r["name"])
        try:
            hit = geocode(r["road"], r["area"])
        except Exception as exc:                          # offline, blocked, rate-limited ...
            print("   lookup failed (%s)" % exc)
            hit = None
        if hit:
            nodes.loc[i, ["lat", "lon", "geocode_note"]] = [hit[0], hit[1], "OSM: " + hit[2]]
        else:
            nodes.loc[i, "geocode_note"] = "NOT FOUND - type lat/lon by hand"
    nodes.to_csv(nodes_csv, index=False)

    missing = nodes[nodes["lat"].isna() | nodes["lon"].isna()]
    out_of_box = nodes[(nodes["lat"] < BLR_BBOX["south"]) | (nodes["lat"] > BLR_BBOX["north"]) |
                       (nodes["lon"] < BLR_BBOX["west"]) | (nodes["lon"] > BLR_BBOX["east"])]
    dup = nodes[nodes.duplicated(["lat", "lon"], keep=False) & nodes["lat"].notna()]
    if len(missing):
        print("\nThese nodes have no coordinates yet:\n" + missing[["node_id", "name"]].to_string(index=False))
        raise SystemExit("Open %s, fill the empty lat/lon cells (decimal degrees), save, and rerun." % nodes_csv)
    problems = pd.concat([out_of_box, dup]).drop_duplicates("node_id")
    if len(problems):
        print("\nCHECK THESE coordinates (outside Bengaluru, or identical to another node):")
        print(problems[["node_id", "name", "lat", "lon", "geocode_note"]].to_string(index=False))
        if not allow_flagged:
            raise SystemExit("Fix them in %s, or rerun with --accept-flagged if they are correct." % nodes_csv)
    return nodes


# ----------------------------------------------------------------------------- road network
def download_graph(nodes, path, buffer_km):
    import osmnx as ox
    dlat = buffer_km / 111.0
    dlon = buffer_km / (111.0 * math.cos(math.radians(nodes["lat"].mean())))
    west, east = nodes["lon"].min() - dlon, nodes["lon"].max() + dlon
    south, north = nodes["lat"].min() - dlat, nodes["lat"].max() + dlat
    print("downloading the OSM drive network for the area around your nodes (a few minutes the first time)...")
    try:
        G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")   # osmnx >= 2
    except TypeError:
        G = ox.graph_from_bbox(north, south, east, west, network_type="drive")          # osmnx 1.x
    ox.save_graphml(G, path)


def load_graph(path):
    raw = nx.read_graphml(path)
    D = nx.DiGraph()
    for n, d in raw.nodes(data=True):
        D.add_node(n, x=float(d["x"]), y=float(d["y"]))
    for u, v, d in raw.edges(data=True):
        L = float(d.get("length", 0.0))
        geometry = d.get("geometry")
        if geometry:
            try:
                from shapely import wkt
                geometry = list(wkt.loads(geometry).coords) if isinstance(geometry, str) else list(geometry.coords)
            except (TypeError, ValueError):
                geometry = None
        if D.has_edge(u, v) and D[u][v]["length"] <= L:
            continue
        D.add_edge(u, v, length=L, geometry=geometry)
    biggest = max(nx.strongly_connected_components(D), key=len)   # drop dead-end islands
    return D.subgraph(biggest).copy()


def road_path_geometry(D, path, start, target):
    """Build a simplified [lat, lon] road geometry for one sensor-to-sensor path."""
    from shapely.geometry import LineString

    points = [(float(start["lon"]), float(start["lat"]))]
    for u, v in zip(path, path[1:]):
        edge = D[u][v]
        piece = edge.get("geometry")
        if not piece:
            piece = [(D.nodes[u]["x"], D.nodes[u]["y"]), (D.nodes[v]["x"], D.nodes[v]["y"])]
        piece = [(float(x), float(y)) for x, y in piece]
        u_xy = (D.nodes[u]["x"], D.nodes[u]["y"])
        v_xy = (D.nodes[v]["x"], D.nodes[v]["y"])
        if math.dist(piece[0], v_xy) < math.dist(piece[-1], v_xy):
            piece.reverse()
        if math.dist(points[-1], piece[0]) > 1e-12:
            points.append(piece[0])
        points.extend(piece[1:])
    target_xy = (float(target["lon"]), float(target["lat"]))
    if math.dist(points[-1], target_xy) > 1e-12:
        points.append(target_xy)
    simplified = LineString(points).simplify(0.00003, preserve_topology=True)
    if len(simplified.coords) < 3 and len(points) >= 3:
        simplified = LineString([points[0], points[len(points) // 2], points[-1]])
    coords = [[float(y), float(x)] for x, y in simplified.coords]
    assert len(coords) >= 2
    assert float(haversine_m(coords[0][0], coords[0][1], start["lat"], start["lon"])) <= 500
    assert float(haversine_m(coords[-1][0], coords[-1][1], target["lat"], target["lon"])) <= 500
    return coords


def write_edge_geometry(D, nodes, ids, A, dist, direct, snap_node, out):
    """Write one GeoJSON feature per undirected kept road edge."""
    features = []
    seen = set()
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j or not A[i, j] or not np.isfinite(dist[i, j]):
                continue
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            try:
                path = nx.dijkstra_path(D, snap_node[i], snap_node[j], weight="length")
            except nx.NetworkXNoPath:
                continue
            coords = road_path_geometry(D, path, nodes.iloc[i], nodes.iloc[j])
            seen.add(key)
            features.append({
                "type": "Feature",
                "properties": {
                    "source": ids[i],
                    "target": ids[j],
                    "distance_km": round(float(min(dist[i, j], dist[j, i])), 3),
                    "weight": round(float(max(A[i, j], A[j, i])), 4),
                },
                "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in coords]},
            })
    payload = {"type": "FeatureCollection", "features": features}
    (out / "edge_geometry.geojson").write_text(json.dumps(payload), encoding="utf-8")
    print("wrote %s (%d edge geometries)" % (out / "edge_geometry.geojson", len(features)))


def road_distances(D, nodes, max_direct_km, pass_m, search_km):
    """Directed driving distances (m) and the 'direct connection' mask."""
    gid = np.array(list(D.nodes))
    gy = np.array([D.nodes[n]["y"] for n in gid])
    gx = np.array([D.nodes[n]["x"] for n in gid])
    lat, lon = nodes["lat"].values, nodes["lon"].values
    N = len(nodes)

    snap_node, snap_m = [], []
    for i in range(N):
        d = haversine_m(lat[i], lon[i], gy, gx)
        k = int(np.argmin(d))
        snap_node.append(gid[k])
        snap_m.append(float(d[k]))
    snap_m = np.array(snap_m)

    dist = np.full((N, N), np.inf)
    direct = np.zeros((N, N), dtype=bool)
    near = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) < pass_m   # co-located sites
    for i in range(N):
        lengths, paths = nx.single_source_dijkstra(D, snap_node[i], cutoff=search_km * 1000.0, weight="length")
        for j in range(N):
            if i == j:
                dist[i, j] = 0.0
                continue
            if snap_node[j] not in lengths:
                continue
            d = lengths[snap_node[j]] + snap_m[i] + snap_m[j]
            dist[i, j] = d
            if d > max_direct_km * 1000.0:
                continue
            p = paths[snap_node[j]]
            py = np.array([D.nodes[n]["y"] for n in p])
            px = np.array([D.nodes[n]["x"] for n in p])
            blocked = False
            for k in range(N):
                if k in (i, j) or near[k, i] or near[k, j]:
                    continue
                if haversine_m(lat[k], lon[k], py, px).min() < pass_m:
                    blocked = True
                    break
            direct[i, j] = not blocked
    return dist, direct, np.array(snap_node), snap_m


# ----------------------------------------------------------------------------- weights
def gaussian_adj(dist_m, direct, thr=0.1):
    finite = dist_m[np.isfinite(dist_m) & (dist_m > 0)]
    sigma = finite.std()
    W = np.where(np.isfinite(dist_m), np.exp(-np.square(dist_m / sigma)), 0.0)
    A = np.where(direct & (W >= thr), W, 0.0)
    np.fill_diagonal(A, 1.0)
    return A, W, sigma


def rescue_isolated(A, W, dist_m, thr):
    """Every node must have at least one outgoing and one incoming edge besides itself."""
    fixed = []
    N = len(A)
    off = ~np.eye(N, dtype=bool)
    for i in range(N):
        if not (A[i] * off[i]).any():
            cand = np.where(np.isfinite(dist_m[i]) & off[i], dist_m[i], np.inf)
            j = int(np.argmin(cand))
            if np.isfinite(cand[j]):
                A[i, j] = max(W[i, j], thr)
                fixed.append(("out", i, j))
        if not (A[:, i] * off[:, i]).any():
            cand = np.where(np.isfinite(dist_m[:, i]) & off[:, i], dist_m[:, i], np.inf)
            j = int(np.argmin(cand))
            if np.isfinite(cand[j]):
                A[j, i] = max(W[j, i], thr)
                fixed.append(("in", i, j))
    return A, fixed


def similarity_adj(panel, train_end, min_overlap=60):
    """Correlation of day-of-week-adjusted speed (and volume if present) over TRAINING days only."""
    mats = []
    for key in ("speed", "traffic_volume"):
        if key not in panel:
            continue
        x = panel[key].iloc[:train_end]
        resid = x - x.groupby(x.index.dayofweek).transform("mean")
        c = resid.corr(min_periods=min_overlap).values
        mats.append(np.nan_to_num(np.clip(c, 0, None)))
    S = np.mean(mats, axis=0)
    np.fill_diagonal(S, 1.0)
    return S


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/BANGALORE")
    ap.add_argument("--graphml", default=None, help="cache file for the OSM network (default: <out>/bangalore_drive.graphml)")
    ap.add_argument("--buffer-km", type=float, default=3.0)
    ap.add_argument("--max-direct-km", type=float, default=12.0, help="longest route that can count as a direct link")
    ap.add_argument("--search-km", type=float, default=40.0, help="stop route search beyond this distance")
    ap.add_argument("--pass-m", type=float, default=250.0, help="a route passing this close to another node is not direct")
    ap.add_argument("--kernel-thr", type=float, default=0.1)
    ap.add_argument("--sim-mode", choices=["off", "reweight", "augment"], default="off",
                    help="off = pure road graph; reweight = blend similarity into road edges; augment = also add similarity-only edges")
    ap.add_argument("--beta", type=float, default=0.3, help="weight of the similarity graph when sim-mode is not off")
    ap.add_argument("--sim-thr", type=float, default=0.3)
    ap.add_argument("--accept-flagged", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = common.load_raw(args.csv)
    nodes_csv = out / "nodes.csv"
    nodes = pd.read_csv(nodes_csv) if nodes_csv.exists() else common.build_nodes(df).assign(lat=np.nan, lon=np.nan, geocode_note="")
    for c in ("lat", "lon"):
        nodes[c] = pd.to_numeric(nodes[c], errors="coerce")
    if "geocode_note" not in nodes:
        nodes["geocode_note"] = ""
    nodes["geocode_note"] = nodes["geocode_note"].fillna("").astype(str)
    nodes = fill_coordinates(nodes, nodes_csv, args.accept_flagged)
    N = len(nodes)

    gpath = Path(args.graphml) if args.graphml else out / "bangalore_drive.graphml"
    if not gpath.exists():
        download_graph(nodes, str(gpath), args.buffer_km)
    D = load_graph(str(gpath))
    print("road network: %d junctions, %d directed road segments" % (D.number_of_nodes(), D.number_of_edges()))

    dist, direct, snap_node, snap_m = road_distances(D, nodes, args.max_direct_km, args.pass_m, args.search_km)
    A_road, W, sigma = gaussian_adj(dist, direct, args.kernel_thr)
    A_road, fixed = rescue_isolated(A_road, W, dist, args.kernel_thr)

    panel, _ = common.build_panel(df, nodes[["node_id", "name", "area", "road"]])
    T = len(panel["speed"])
    info = common.split_info(T)
    S = similarity_adj(panel, info["stats_end"])
    A_sim = np.where(S >= args.sim_thr, S, 0.0)
    np.fill_diagonal(A_sim, 1.0)

    if args.sim_mode == "off":
        A = A_road.copy()
    elif args.sim_mode == "reweight":
        A = (1 - args.beta) * A_road + args.beta * A_sim * (A_road > 0)
    else:
        A = (1 - args.beta) * A_road + args.beta * A_sim
    np.fill_diagonal(A, 1.0)

    # ---- save components + final
    km = dist / 1000.0
    ids = list(nodes["node_id"])
    pd.DataFrame(km, index=ids, columns=ids).to_csv(out / "distances_km.csv", float_format="%.3f")
    pd.DataFrame(direct.astype(int), index=ids, columns=ids).to_csv(out / "direct_connection.csv")
    np.save(out / "adj_road.npy", A_road.astype(np.float32))
    np.save(out / "adj_similarity.npy", A_sim.astype(np.float32))
    np.save(out / "adj_mx.npy", A.astype(np.float32))
    (out / "graph_sensor_ids.txt").write_text(",".join(ids))
    loc = pd.DataFrame({"index": range(N), "sensor_id": ids, "latitude": nodes["lat"], "longitude": nodes["lon"]})
    loc.to_csv(out / "graph_sensor_locations.csv", index=False)
    write_edge_geometry(D, nodes, ids, A, dist, direct, snap_node, out)
    edges = [{"from": ids[i], "to": ids[j], "km": round(km[i, j], 3), "direct": bool(direct[i, j]), "weight": round(float(A[i, j]), 4)}
             for i in range(N) for j in range(N) if i != j and A[i, j] > 0]
    pd.DataFrame(edges).to_csv(out / "edges.csv", index=False)

    off = ~np.eye(N, dtype=bool)
    asym = int(((A_road > 0) != (A_road.T > 0))[off].sum() // 2)
    report = {
        "nodes": N, "days": T, "gaussian_sigma_km": round(float(sigma) / 1000, 3),
        "direct_pairs": int(direct.sum()), "final_edges_excluding_self": int(((A > 0) & off).sum()),
        "one_way_asymmetric_pairs": asym,
        "snap_distance_m": {"median": round(float(np.median(snap_m)), 1), "max": round(float(snap_m.max()), 1)},
        "nodes_snapped_over_300m": [ids[i] for i in range(N) if snap_m[i] > 300],
        "nodes_sharing_a_junction": int(N - len(set(snap_node))),
        "unreachable_pairs": int((~np.isfinite(dist) & off).sum()),
        "rescued_for_isolation": [(w, ids[i], ids[j]) for w, i, j in fixed],
        "sim_mode": args.sim_mode, "beta": args.beta,
    }
    (out / "graph_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(N):
            for j in range(N):
                if i != j and A[i, j] > 0:
                    ax.annotate("", xy=(nodes.lon[j], nodes.lat[j]), xytext=(nodes.lon[i], nodes.lat[i]),
                                arrowprops=dict(arrowstyle="->", color="#4477aa", alpha=0.35 + 0.5 * float(A[i, j]), lw=1))
        ax.scatter(nodes.lon, nodes.lat, c="#c8323c", zorder=3)
        for i in range(N):
            ax.annotate(ids[i], (nodes.lon[i], nodes.lat[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title("Adjacency used by DCRNN (arrow i -> j means i influences j)")
        fig.savefig(out / "graph_check.png", dpi=130, bbox_inches="tight")
        print("wrote %s" % (out / "graph_check.png"))
    except Exception as exc:
        print("plot skipped: %s" % exc)

    if int(np.__version__.split(".")[0]) < 2:
        import subprocess, sys
        subprocess.check_call([sys.executable, str(Path(__file__).with_name("npy_to_pkl.py")), "--graph-dir", str(out)])
    else:
        print("\nNumPy >= 2 here, so I did not write the .pkl (DCRNN's old NumPy cannot read it).")
        print("Run this in your TF1 'dcrnn' environment:  python npy_to_pkl.py --graph-dir %s" % out)


if __name__ == "__main__":
    main()
