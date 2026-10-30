"""Step 2.4: road-network distances + Gaussian-kernel adjacency (DCRNN adj_mx.pkl format)."""
import csv
import json
import pickle

import networkx as nx
import numpy as np
import osmnx as ox
from geopy.distance import geodesic

from src.config import ADJ_EPSILON, BBOX, GRAPH_DIR, RAW_DIR
from src.nodes import NODES

GRAPHML = GRAPH_DIR / "bangalore_drive.graphml"


def load_graph():
    if GRAPHML.exists():
        print(f"Loading cached graph {GRAPHML}")
        return ox.load_graphml(GRAPHML)
    print("Downloading OSM drive network (one-time)...")
    G = ox.graph_from_bbox(
    bbox=(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]),
    network_type="drive",
)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, GRAPHML)
    return G


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    G_proj = ox.project_graph(load_graph())

    # nearest_nodes on a projected graph needs projected X/Y, so project the points first
    import geopandas as gpd

    pts = gpd.GeoSeries(gpd.points_from_xy([n.lon for n in NODES], [n.lat for n in NODES]),
                        crs="EPSG:4326").to_crs(G_proj.graph["crs"])
    osm_ids = list(ox.distance.nearest_nodes(G_proj, X=pts.x.values, Y=pts.y.values))

    n = len(NODES)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                D[i, j] = nx.shortest_path_length(G_proj, osm_ids[i], osm_ids[j], weight="length")
            except nx.NetworkXNoPath:
                D[i, j] = np.inf
            if not np.isfinite(D[i, j]):
                D[i, j] = geodesic((NODES[i].lat, NODES[i].lon),
                                   (NODES[j].lat, NODES[j].lon)).meters
                print(f"WARNING: no path {i}->{j}, using geodesic {D[i, j]:.0f} m")
    with open(GRAPH_DIR / "distance_matrix.pkl", "wb") as f:
        pickle.dump(D, f)

    mask = np.isfinite(D) & (D > 0)
    sigma = np.std(D[mask])
    W = np.exp(-(D ** 2) / (sigma ** 2))
    np.fill_diagonal(W, 0)
    W[W < ADJ_EPSILON] = 0

    sensor_ids = [str(nd.node_id) for nd in NODES]
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}
    with open(GRAPH_DIR / "adj_mx.pkl", "wb") as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, W), f)

    check = json.loads((RAW_DIR / "step2_tomtom_check.json").read_text())
    by_id = {c["node_id"]: c for c in check}
    with open(GRAPH_DIR / "nodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "name", "lat", "lon", "frc", "free_flow_speed_kmh"])
        for nd in NODES:
            c = by_id.get(nd.node_id, {})
            w.writerow([nd.node_id, nd.name, nd.lat, nd.lon, c.get("frc"), c.get("freeFlowSpeed")])

    off = D.copy()
    np.fill_diagonal(off, np.inf)
    print(f"\nsigma = {sigma:.1f} m")
    print(f"non-zero adjacency entries: {int((W > 0).sum())} (expect 60-180)")
    print(f"mean nearest-neighbor distance: {off.min(axis=1).mean():.0f} m")
    print("\nAdjacency heatmap (' '=0, '.'<0.3, ':'<0.6, '#'>=0.6):")
    for i in range(n):
        row = "".join(" " if v == 0 else "." if v < 0.3 else ":" if v < 0.6 else "#" for v in W[i])
        print(f"{i:>2} |{row}|")


if __name__ == "__main__":
    main()
