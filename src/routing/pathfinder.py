"""Candidate routes between two points on the OSM network."""
from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.routing.osm_loader import nearest_osm_node, node_index

PENALTY = 2.5            # multiply the cost of edges already used by an accepted route
MAX_ATTEMPTS = 12        # penalised re-runs tried while looking for k distinct routes
MAX_OVERLAP = 0.80       # reject a route sharing more than this share of length with another
MAX_DETOUR = 2.0         # ... or longer than this multiple of the shortest route
CORRIDOR_PAD = 0.35      # search area: src/dst bounding box padded by this fraction (min 2 km)


@dataclass
class Path:
    osm_node_ids: list
    edges: list              # dicts: osm_u, osm_v, length_m, name, highway, geometry [(lat, lon)]
    total_length_m: float


def _first(value):
    return value[0] if isinstance(value, list) and value else value


def _edge_record(G, u, v) -> dict:
    d = G.edges[u, v]
    geom = d.get("geometry")
    if geom is not None:
        coords = [(lat, lon) for lon, lat in geom.coords]
    else:
        coords = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
    return {"osm_u": u, "osm_v": v, "length_m": float(d["length"]),
            "name": str(_first(d.get("name")) or ""), "highway": str(_first(d.get("highway")) or ""),
            "geometry": coords}


def _corridor(G, s, t):
    ids, lats, lons, _ = node_index(G)
    ys, xs = [G.nodes[s]["y"], G.nodes[t]["y"]], [G.nodes[s]["x"], G.nodes[t]["x"]]
    pad_y = max(0.02, CORRIDOR_PAD * (max(ys) - min(ys)))
    pad_x = max(0.02, CORRIDOR_PAD * (max(xs) - min(xs)))
    mask = ((lats >= min(ys) - pad_y) & (lats <= max(ys) + pad_y)
            & (lons >= min(xs) - pad_x) & (lons <= max(xs) + pad_x))
    return ids[mask].tolist()


def _overlap(a: list, b: list, G) -> float:
    """Shared length / length of the shorter route (edges treated as undirected)."""
    eb = {frozenset(e) for e in zip(b, b[1:])}
    length = lambda nodes: sum(G.edges[u, v]["length"] for u, v in zip(nodes, nodes[1:]))
    shared = 0.0
    for u, v in zip(a, a[1:]):
        if frozenset((u, v)) in eb:
            shared += G.edges[u, v]["length"]
    return shared / max(1.0, min(length(a), length(b)))


def find_candidate_paths(G, src_latlon, dst_latlon, k: int = 3) -> list[Path]:
    """Up to k distinct low-cost routes, shortest first.

    The first route is the shortest by length. Further routes come from re-running the search
    with the edges of accepted routes made 2.5x more expensive, and are kept only if they are
    genuinely different (see MAX_OVERLAP) and not absurdly long. Returns [] if no path exists.
    """
    s, t = nearest_osm_node(G, *src_latlon), nearest_osm_node(G, *dst_latlon)
    if s == t:
        return []
    sub = G.subgraph(_corridor(G, s, t))
    penalties: dict = {}
    accepted: list[list] = []
    for _ in range(MAX_ATTEMPTS):
        weight = lambda u, v, d: d["length"] * penalties.get((u, v), 1.0)
        try:
            nodes = nx.shortest_path(sub, s, t, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            break
        length = sum(G.edges[u, v]["length"] for u, v in zip(nodes, nodes[1:]))
        best_len = sum(G.edges[u, v]["length"] for u, v in zip(accepted[0], accepted[0][1:])) \
            if accepted else length
        distinct = all(_overlap(nodes, other, G) <= MAX_OVERLAP for other in accepted)
        if not accepted or (distinct and length <= MAX_DETOUR * best_len):
            accepted.append(nodes)
            if len(accepted) == k:
                break
        for u, v in zip(nodes, nodes[1:]):
            penalties[(u, v)] = penalties.get((u, v), 1.0) * PENALTY
    paths = []
    for nodes in accepted:
        edges = [_edge_record(G, u, v) for u, v in zip(nodes, nodes[1:])]
        paths.append(Path(nodes, edges, sum(e["length_m"] for e in edges)))
    return paths
