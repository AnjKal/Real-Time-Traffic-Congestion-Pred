"""Load the cached OSM drive network (Step 2) once and answer nearest-node queries."""
import pickle

import networkx as nx
import numpy as np
import osmnx as ox
from sklearn.neighbors import BallTree

from src.config import GRAPH_DIR

GRAPHML = GRAPH_DIR / "bangalore_drive.graphml"
CORE_CACHE = GRAPH_DIR / "bangalore_drive_core.pkl"     # derived from the graphml; safe to delete
_G = None
_indexes: dict[int, tuple] = {}


def _to_digraph(multi):
    """Collapse parallel edges to the shortest one (keeps length/name/highway/geometry)."""
    if hasattr(ox, "convert") and hasattr(ox.convert, "to_digraph"):
        return ox.convert.to_digraph(multi, weight="length")
    return ox.utils_graph.get_digraph(multi, weight="length")


def get_osm_graph() -> nx.DiGraph:
    """Cached singleton: lat/lon DiGraph of the largest strongly connected component.

    Restricting to the strongly connected core guarantees a route exists between any two
    nodes we snap to (dead-end one-way fragments would otherwise give "no path").
    """
    global _G
    if _G is None:
        if CORE_CACHE.exists() and CORE_CACHE.stat().st_mtime >= GRAPHML.stat().st_mtime:
            with open(CORE_CACHE, "rb") as f:
                _G = pickle.load(f)                  # our own cache file; ~10x faster than graphml
        else:
            digraph = _to_digraph(ox.load_graphml(GRAPHML))
            core = max(nx.strongly_connected_components(digraph), key=len)
            _G = digraph.subgraph(core).copy()
            with open(CORE_CACHE, "wb") as f:
                pickle.dump(_G, f)
    return _G


def node_index(G: nx.DiGraph):
    """(ids, lats, lons, BallTree) for G, built once per graph."""
    if id(G) not in _indexes:
        ids = np.fromiter(G.nodes, dtype=np.int64)
        lats = np.array([G.nodes[n]["y"] for n in ids])
        lons = np.array([G.nodes[n]["x"] for n in ids])
        tree = BallTree(np.radians(np.column_stack([lats, lons])), metric="haversine")
        _indexes[id(G)] = (ids, lats, lons, tree)
    return _indexes[id(G)]


def nearest_osm_node(G: nx.DiGraph, lat: float, lon: float) -> int:
    ids, _, _, tree = node_index(G)
    _, idx = tree.query(np.radians([[lat, lon]]), k=1)
    return int(ids[idx[0, 0]])
