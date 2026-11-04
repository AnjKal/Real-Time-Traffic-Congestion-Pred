"""Candidate routes between two points, coloured by the model's per-junction forecast."""
import networkx as nx
import numpy as np

from webapp.osm_graph import nearest_node, node_index

PENALTY = 2.5            # cost multiplier on edges already used by an accepted route
MAX_ATTEMPTS = 12        # search re-runs while looking for k distinct routes
MAX_OVERLAP = 0.80       # reject a route sharing more than this share of length with another
MAX_DETOUR = 2.0         # ... or longer than this multiple of the shortest route
CORRIDOR_PAD = 0.35      # search area: src/dst bounding box padded by this fraction (min 2 km)
MAX_SNAP_M = 3000.0      # a clicked point farther than this from any mapped road is rejected
FAR_JUNCTION_M = 3000.0  # beyond this, a colour is an extrapolation from a distant junction
MIN_SPEED_KMH = 2.0
EARTH_RADIUS_M = 6371000.0


class RoutingError(Exception):
    """User-facing routing problem (point off the map, no path, ...)."""


def _first(value):
    return value[0] if isinstance(value, list) and value else value


def _edge_record(G, u, v):
    d = G.edges[u, v]
    geom = d.get("geometry")
    if geom is not None:
        coords = [(lat, lon) for lon, lat in geom.coords]
    else:
        coords = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
    return {"length_m": float(d["length"]), "name": str(_first(d.get("name")) or ""),
            "highway": str(_first(d.get("highway")) or ""), "geometry": coords}


def _corridor(G, s, t):
    ids, lats, lons, _ = node_index(G)
    ys = [G.nodes[s]["y"], G.nodes[t]["y"]]
    xs = [G.nodes[s]["x"], G.nodes[t]["x"]]
    pad_y = max(0.02, CORRIDOR_PAD * (max(ys) - min(ys)))
    pad_x = max(0.02, CORRIDOR_PAD * (max(xs) - min(xs)))
    mask = ((lats >= min(ys) - pad_y) & (lats <= max(ys) + pad_y)
            & (lons >= min(xs) - pad_x) & (lons <= max(xs) + pad_x))
    return ids[mask].tolist()


def _length(G, nodes):
    return sum(G.edges[u, v]["length"] for u, v in zip(nodes, nodes[1:]))


def _overlap(G, a, b):
    """Shared length / length of the shorter route (roads treated as undirected)."""
    eb = {frozenset(e) for e in zip(b, b[1:])}
    shared = sum(G.edges[u, v]["length"] for u, v in zip(a, a[1:]) if frozenset((u, v)) in eb)
    return shared / max(1.0, min(_length(G, a), _length(G, b)))


def find_candidate_paths(G, src_latlon, dst_latlon, k=3):
    """Up to k distinct routes as lists of edge records, shortest first.

    The first route is the shortest by length; further routes re-run the search with the edges
    of accepted routes made 2.5x costlier and are kept only if clearly different.
    Raises RoutingError for points too far from the mapped roads or when no path exists.
    """
    (s, s_dist), (t, t_dist) = nearest_node(G, *src_latlon), nearest_node(G, *dst_latlon)
    for label, dist in (("source", s_dist), ("destination", t_dist)):
        if dist > MAX_SNAP_M:
            raise RoutingError("the %s is %.1f km from the nearest mapped main road; pick a "
                               "point closer to a main road" % (label, dist / 1000.0))
    if s == t:
        raise RoutingError("source and destination snap to the same road node")
    sub = G.subgraph(_corridor(G, s, t))
    penalties, accepted = {}, []
    for _ in range(MAX_ATTEMPTS):
        weight = lambda u, v, d: d["length"] * penalties.get((u, v), 1.0)
        try:
            nodes = nx.shortest_path(sub, s, t, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            break
        length = _length(G, nodes)
        best = _length(G, accepted[0]) if accepted else length
        if not accepted or (all(_overlap(G, nodes, o) <= MAX_OVERLAP for o in accepted)
                            and length <= MAX_DETOUR * best):
            accepted.append(nodes)
            if len(accepted) == k:
                break
        for u, v in zip(nodes, nodes[1:]):
            penalties[(u, v)] = penalties.get((u, v), 1.0) * PENALTY
    if not accepted:
        raise RoutingError("no drivable route between those points on the mapped roads")
    return [[_edge_record(G, u, v) for u, v in zip(nodes, nodes[1:])] for nodes in accepted]


def _haversine_m(lat, lon, lats, lons):
    lat, lon, lats, lons = map(np.radians, (lat, lon, np.asarray(lats), np.asarray(lons)))
    a = np.sin((lats - lat) / 2) ** 2 + np.cos(lat) * np.cos(lats) * np.sin((lons - lon) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def _midpoint(geometry):
    return geometry[len(geometry) // 2]


def colorize_path(edges, junctions):
    """One segment per road edge, taking speed/band/color from the NEAREST junction.

    junctions: the forecast's node dicts (need lat, lon, predicted_kmh, band, color, name,
    junction_id).
    """
    jl = np.array([j["lat"] for j in junctions])
    jn = np.array([j["lon"] for j in junctions])
    segments = []
    for e in edges:
        mlat, mlon = _midpoint(e["geometry"])
        d = _haversine_m(mlat, mlon, jl, jn)
        i = int(np.argmin(d))
        j = junctions[i]
        segments.append({
            "geometry": [list(p) for p in e["geometry"]], "name": e["name"],
            "highway": e["highway"], "length_m": e["length_m"],
            "junction_id": j["junction_id"], "junction_name": j.get("name", j["junction_id"]),
            "distance_to_junction_m": int(round(d[i])),
            "predicted_speed_kmh": j["predicted_kmh"], "band": j["band"], "color": j["color"],
            "extrapolated": bool(d[i] > FAR_JUNCTION_M)})
    return segments


def path_score(segments):
    """Travel time from predicted speeds (length-weighted) and band shares by length."""
    total_m = sum(s["length_m"] for s in segments)
    seconds = sum(s["length_m"] / (max(s["predicted_speed_kmh"], MIN_SPEED_KMH) * 1000 / 3600)
                  for s in segments)
    share = lambda band: 100.0 * sum(s["length_m"] for s in segments
                                     if s["band"] == band) / max(total_m, 1.0)
    return {"total_length_km": round(total_m / 1000, 2), "total_time_min": round(seconds / 60, 1),
            "mean_speed_kmh": round((total_m / 1000) / (seconds / 3600), 1) if seconds else 0.0,
            "pct_heavy": round(share("heavy"), 1), "pct_moderate": round(share("moderate"), 1),
            "pct_free": round(share("free"), 1)}
