"""Candidate routes between two points, coloured by the model's per-junction forecast."""
import math

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
    # u/v are the graph node ids this edge runs between. The journey agent needs them to tell
    # whether a candidate route is really the road the traveller is already on, and to find the
    # point where a genuine alternative leaves the current route.
    return {"u": u, "v": v,
            "length_m": float(d["length"]), "name": str(_first(d.get("name")) or ""),
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


def _num(value):
    """A finite float, or None if the value is missing/invalid (so a bad reading becomes a
    null metric on the segment instead of crashing the route response)."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _segment_delay_min(length_m, predicted_kmh, baseline_kmh):
    """Extra minutes to drive this segment at the predicted speed versus a baseline (free-flow)
    speed. None if either speed is missing/non-positive; never negative.

        max(0, (km / predicted - km / baseline) * 60)

    Stage 2 passes TomTom free-flow as the baseline (see colorize_path). Both denominators are
    guarded so a zero/negative/imputed speed yields None instead of dividing by zero.
    """
    if predicted_kmh is None or baseline_kmh is None:
        return None
    length_km = _num(length_m)
    if length_km is None or predicted_kmh <= 0 or baseline_kmh <= 0:
        return None
    length_km /= 1000.0
    at_predicted = length_km / predicted_kmh
    at_baseline = length_km / baseline_kmh
    return round(max(0.0, (at_predicted - at_baseline) * 60.0), 1)


def colorize_path(edges, junctions, free_flow_by_junction=None):
    """One segment per road edge, taking speed/band/color from the NEAREST junction.

    junctions: the forecast's node dicts (need lat, lon, predicted_kmh, band, color, name,
        junction_id; optionally current_kmh, reference_kmh).
    free_flow_by_junction: optional {junction_id: TomTom freeFlowSpeed km/h} from the latest raw
        readings. Used for free_flow_speed_kmh and the delay baseline. reference_kmh is the model's
        historical reference speed and is NEVER used as free-flow.
    """
    free_flow_by_junction = free_flow_by_junction or {}
    jl = np.array([j["lat"] for j in junctions])
    jn = np.array([j["lon"] for j in junctions])
    segments = []
    for e in edges:
        mlat, mlon = _midpoint(e["geometry"])
        d = _haversine_m(mlat, mlon, jl, jn)
        i = int(np.argmin(d))
        j = junctions[i]
        junction_id = j["junction_id"]
        # current_kmh: last observed TomTom currentSpeed (may be imputed upstream).
        # reference_kmh: model's historical reference speed (NOT free-flow).
        # free_flow: actual TomTom freeFlowSpeed for this junction, latest valid reading.
        current_speed = _num(j.get("current_kmh"))
        reference_speed = _num(j.get("reference_kmh"))
        predicted_speed = _num(j.get("predicted_kmh"))
        free_flow_speed = _num(free_flow_by_junction.get(junction_id, j.get("free_flow_kmh")))
        segments.append({
            "geometry": [list(p) for p in e["geometry"]], "name": e["name"],
            "highway": e["highway"], "length_m": e["length_m"],
            # Carried through for the journey agent; the map ignores them.
            "u": e.get("u"), "v": e.get("v"),
            "junction_id": junction_id, "junction_name": j.get("name", junction_id),
            "distance_to_junction_m": int(round(d[i])),
            "predicted_speed_kmh": j["predicted_kmh"], "band": j["band"], "color": j["color"],
            "current_speed_kmh": current_speed,
            "reference_speed_kmh": reference_speed,
            "free_flow_speed_kmh": free_flow_speed,
            # Delay baseline is TomTom free-flow (Stage 2); None when free-flow is unavailable.
            "delay_min": _segment_delay_min(e["length_m"], predicted_speed, free_flow_speed),
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


# --------------------------------------------------------------------------- journey re-scoring
# The journey agent must re-score a route the user already chose, every time a new forecast
# arrives, WITHOUT re-running the graph search. These helpers compress a scored route into the
# minimum state needed for that (route_spine) plus a bounded polyline for progress estimation,
# and re-score the spine against fresh per-junction predictions (score_spine). They are additive:
# find_candidate_paths, colorize_path and path_score are unchanged.

MAX_POLYLINE_POINTS = 400   # cap on stored route geometry; enough to place a point on the route


def route_spine(segments):
    """Compact per-junction summary of a coloured route, in travel order.

    Consecutive segments that borrow from the same junction are merged, so a route collapses to
    at most one entry per modelled junction. ``start_m``/``cum_m`` are distances from the source
    to the start/end of the entry, so a journey's progress along the route maps straight onto it.
    This is what the journey agent stores and re-scores; it needs no road graph to replay.
    """
    spine = []
    cum = 0.0
    for segment in segments:
        length = _num(segment.get("length_m")) or 0.0
        start = cum
        cum += length
        junction_id = segment.get("junction_id")
        if spine and spine[-1]["junction_id"] == junction_id:
            spine[-1]["length_m"] = round(spine[-1]["length_m"] + length, 1)
            spine[-1]["cum_m"] = round(cum, 1)
            spine[-1]["extrapolated"] = (spine[-1]["extrapolated"]
                                         or bool(segment.get("extrapolated")))
            continue
        spine.append({"junction_id": junction_id,
                      "junction_name": segment.get("junction_name") or junction_id,
                      "length_m": round(length, 1), "start_m": round(start, 1),
                      "cum_m": round(cum, 1),
                      "extrapolated": bool(segment.get("extrapolated"))})
    return spine


def route_polyline(segments, max_points=MAX_POLYLINE_POINTS):
    """Route geometry as ``[[lat, lon, cum_m], ...]``, downsampled to at most max_points.

    Used to place an estimated current position on the route (see position_at). The first and
    last points are always kept so the source and destination stay exact.
    """
    points = []
    if segments and segments[0].get("geometry"):
        first = segments[0]["geometry"][0]
        points.append([round(float(first[0]), 6), round(float(first[1]), 6), 0.0])
    cum = 0.0
    for segment in segments:
        cum += _num(segment.get("length_m")) or 0.0
        geometry = segment.get("geometry") or []
        if not geometry:
            continue
        last = geometry[-1]
        points.append([round(float(last[0]), 6), round(float(last[1]), 6), round(cum, 1)])
    if len(points) <= max_points or len(points) < 3:
        return points
    step = int(math.ceil((len(points) - 2) / float(max_points - 2)))
    kept = [points[0]] + points[1:-1:step] + [points[-1]]
    return kept


def position_at(polyline, cum_m):
    """(lat, lon) at ``cum_m`` metres along a stored polyline, interpolated between points.

    Returns None when the polyline is empty. Values before the start or past the end clamp to
    the first/last point, so an over-run estimate never falls off the route.
    """
    if not polyline:
        return None
    distance = max(0.0, float(cum_m or 0.0))
    previous = polyline[0]
    for point in polyline:
        if point[2] >= distance:
            span = point[2] - previous[2]
            if span <= 0:
                return (point[0], point[1])
            share = (distance - previous[2]) / span
            return (previous[0] + (point[0] - previous[0]) * share,
                    previous[1] + (point[1] - previous[1]) * share)
        previous = point
    last = polyline[-1]
    return (last[0], last[1])


def _node_lookup(nodes):
    """Accept either {junction_id: node} or a callable(junction_id, offset_min) -> node.

    The callable form lets the journey agent pick a DIFFERENT forecast horizon per spine entry,
    based on when the driver is expected to reach it.
    """
    if callable(nodes):
        return nodes
    table = nodes or {}
    return lambda junction_id, offset_min: table.get(junction_id)


def score_spine(spine, nodes, free_flow_by_junction=None, from_cum_m=0.0):
    """Travel time, band shares and delay for a stored route, from fresh forecast nodes.

    spine: route_spine output. nodes: {junction_id: forecast node} or a callable (see
    _node_lookup). free_flow_by_junction: TomTom free-flow per junction for the delay baseline;
    reference_kmh is never substituted. from_cum_m: skip the part of the route already driven, so
    the agent scores only what is left.

    Speeds use the same MIN_SPEED_KMH floor as path_score, so a full-route score matches
    path_score for the same route. Entries whose junction is missing from the forecast are
    excluded from the timing and reported through ``route_coverage_pct``; their length is still
    counted in ``total_length_km`` and priced at the mean speed of the scored entries, so the ETA
    never silently loses part of the route.
    """
    free_flow_by_junction = free_flow_by_junction or {}
    pick = _node_lookup(nodes)
    start = max(0.0, float(from_cum_m or 0.0))
    remaining_m, scored_m, seconds, delay_min = 0.0, 0.0, 0.0, 0.0
    by_band = {}
    has_delay = False
    offset_min = 0.0
    unscored_m = 0.0
    for entry in spine:
        length = max(0.0, float(entry.get("cum_m", 0.0)) - max(start, float(entry.get("start_m", 0.0))))
        if length <= 0:
            continue
        remaining_m += length
        node = pick(entry.get("junction_id"), offset_min)
        predicted = _num((node or {}).get("predicted_kmh"))
        if predicted is None:
            unscored_m += length
            continue
        speed = max(predicted, MIN_SPEED_KMH)
        entry_seconds = length / (speed * 1000.0 / 3600.0)
        seconds += entry_seconds
        offset_min += entry_seconds / 60.0
        scored_m += length
        band = (node or {}).get("band") or "unknown"
        by_band[band] = by_band.get(band, 0.0) + length
        baseline = _num(free_flow_by_junction.get(entry.get("junction_id"),
                                                  (node or {}).get("free_flow_kmh")))
        entry_delay = _segment_delay_min(length, predicted, baseline)
        if entry_delay is not None:
            delay_min += entry_delay
            has_delay = True
    if unscored_m > 0 and scored_m > 0:
        seconds += unscored_m / (scored_m / seconds) if seconds else 0.0
    share = lambda band: round(100.0 * by_band.get(band, 0.0) / max(scored_m, 1.0), 1)
    return {"total_length_km": round(remaining_m / 1000.0, 2),
            "total_time_min": round(seconds / 60.0, 1),
            "mean_speed_kmh": (round((remaining_m / 1000.0) / (seconds / 3600.0), 1)
                               if seconds else 0.0),
            "pct_heavy": share("heavy"), "pct_moderate": share("moderate"),
            "pct_free": share("free"),
            "delay_min": round(delay_min, 1) if has_delay else None,
            "route_coverage_pct": round(100.0 * scored_m / max(remaining_m, 1.0), 1),
            "scored_length_km": round(scored_m / 1000.0, 2)}


def route_node_chain(segments):
    """The graph nodes a coloured route runs through, in travel order, with distance along.

    Returns ``(nodes, cum_m)`` where ``cum_m[i]`` is the distance from the source to ``nodes[i]``,
    so ``nodes[0]`` sits at 0 and the last node sits at the full route length. The journey agent
    uses this to start an alternatives search from the next junction *ahead* of the car (never
    from a node behind it, which is how naive re-routing ends up asking for a U-turn).

    Returns ``([], [])`` for a route whose segments carry no node ids.
    """
    usable = [s for s in segments if s.get("u") is not None and s.get("v") is not None]
    if not usable or len(usable) != len(segments):
        return [], []
    nodes = [usable[0]["u"]]
    cum = [0.0]
    running = 0.0
    for segment in usable:
        running += _num(segment.get("length_m")) or 0.0
        nodes.append(segment["v"])
        cum.append(round(running, 1))
    return nodes, cum


def route_edge_pairs(segments):
    """A route as an ordered list of ``(u, v)`` node pairs, for prefix comparison."""
    return [(s.get("u"), s.get("v")) for s in segments]


def shared_fraction(G, nodes_a, nodes_b):
    """Share of the shorter of two node-chain routes that the two have in common (0..1).

    Thin public wrapper over the search's own overlap test, so the journey agent can ask "is this
    candidate really the road I am already on?" with exactly the same measure the search uses to
    decide two routes are distinct.
    """
    if len(nodes_a) < 2 or len(nodes_b) < 2:
        return 0.0
    return _overlap(G, nodes_a, nodes_b)


def cum_at_position(polyline, lat, lon):
    """Distance along a stored route of the polyline point nearest to (lat, lon).

    Lets a real position feed (POST /api/journey/{id}/position) replace the agent's elapsed-time
    progress estimate without the agent's policy changing at all. Returns None for an empty
    polyline.
    """
    if not polyline:
        return None
    distances = _haversine_m(lat, lon, [point[0] for point in polyline],
                             [point[1] for point in polyline])
    return float(polyline[int(np.argmin(distances))][2])
