"""Road network for routing (runs in the modern Python, not the TensorFlow 3.7 env).

    python -m webapp.osm_graph        # one-time download + build (a few minutes, needs internet)

Builds a main-roads network (motorway ... tertiary) around all 100 junctions, keeps the largest
strongly connected part so every pair of points has a route, collapses parallel roads to the
shortest one, and caches the result in data/graph/bangalore_core.pkl.
"""
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPOTS_PATH = os.path.join(PROJECT_ROOT, "bangalore_traffic_spots.csv")
GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "graph")
GRAPHML = os.path.join(GRAPH_DIR, "bangalore_major.graphml")
CORE_PKL = os.path.join(GRAPH_DIR, "bangalore_core.pkl")
PAD_DEG = 0.03                      # ~3 km of road around the outermost junctions
MAJOR = ("motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|"
         "secondary_link|tertiary_link")
CUSTOM_FILTER = '["highway"~"%s"]' % MAJOR

_G = None
_index = {}


def coverage_bbox():
    """(west, south, east, north) around all junctions."""
    s = pd.read_csv(SPOTS_PATH)
    return (s.lon.min() - PAD_DEG, s.lat.min() - PAD_DEG, s.lon.max() + PAD_DEG,
            s.lat.max() + PAD_DEG)


OVERPASS_MIRRORS = ["https://overpass-api.de/api/interpreter",
                    "https://overpass.kumi.systems/api/interpreter",
                    "https://overpass.private.coffee/api/interpreter"]


def _download(bbox):
    """Try each Overpass server in turn, backing off between failed attempts (public servers
    are often at capacity)."""
    import time
    import osmnx as ox
    ox.settings.requests_timeout = 600
    last = None
    for attempt in range(6):
        url = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
        ox.settings.overpass_url = url
        print("Download attempt %d via %s ..." % (attempt + 1, url), flush=True)
        try:
            return ox.graph_from_bbox(bbox=bbox, custom_filter=CUSTOM_FILTER, retain_all=False,
                                      simplify=True)
        except Exception as e:                       # noqa: BLE001 - network/server errors vary
            last = e
            wait = 20 * (attempt + 1)
            reason = " ".join(str(e).split())[:90]
            print("  failed (%s: %s); waiting %ds" % (type(e).__name__, reason, wait), flush=True)
            time.sleep(wait)
    raise last


def build():
    import osmnx as ox
    west, south, east, north = coverage_bbox()
    print("Downloading main roads for bbox W%.3f S%.3f E%.3f N%.3f ..." % (west, south, east, north),
          flush=True)
    multi = _download((west, south, east, north))
    print("Downloaded: %d nodes, %d edges" % (multi.number_of_nodes(), multi.number_of_edges()),
          flush=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    ox.save_graphml(multi, GRAPHML)

    digraph = ox.convert.to_digraph(multi, weight="length")
    core = max(nx.strongly_connected_components(digraph), key=len)
    slim = nx.DiGraph()
    for n in core:
        slim.add_node(n, x=digraph.nodes[n]["x"], y=digraph.nodes[n]["y"])
    for u, v, d in digraph.subgraph(core).edges(data=True):
        name = d.get("name")
        name = name[0] if isinstance(name, list) and name else name
        hw = d.get("highway")
        hw = hw[0] if isinstance(hw, list) and hw else hw
        slim.add_edge(u, v, length=float(d["length"]), name=str(name or ""), highway=str(hw or ""),
                      geometry=d.get("geometry"))
    with open(CORE_PKL, "wb") as f:
        pickle.dump(slim, f)
    print("Saved %s: %d nodes, %d edges" % (CORE_PKL, slim.number_of_nodes(), slim.number_of_edges()))


def get_graph():
    """Cached singleton DiGraph (lat/lon in node attrs y/x, meters in edge attr length)."""
    global _G
    if _G is None:
        if not os.path.exists(CORE_PKL):
            raise FileNotFoundError("%s missing: run  python -m webapp.osm_graph  first" % CORE_PKL)
        with open(CORE_PKL, "rb") as f:
            _G = pickle.load(f)                       # our own cache file
    return _G


def node_index(G):
    """(ids, lats, lons, BallTree) built once per graph."""
    if id(G) not in _index:
        ids = np.fromiter(G.nodes, dtype=np.int64)
        lats = np.array([G.nodes[n]["y"] for n in ids])
        lons = np.array([G.nodes[n]["x"] for n in ids])
        tree = BallTree(np.radians(np.column_stack([lats, lons])), metric="haversine")
        _index[id(G)] = (ids, lats, lons, tree)
    return _index[id(G)]


def nearest_node(G, lat, lon):
    """(osm node id, distance in meters) of the road node closest to a point."""
    ids, _, _, tree = node_index(G)
    dist, idx = tree.query(np.radians([[lat, lon]]), k=1)
    return int(ids[idx[0, 0]]), float(dist[0, 0] * 6371000.0)


if __name__ == "__main__":
    build()
