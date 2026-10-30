"""Spread the 20 sensors' forecasts onto road edges, and score whole routes."""
import numpy as np
import pandas as pd

from src.app.map_builder import node_color

EARTH_RADIUS_M = 6_371_000.0
FAR_SENSOR_M = 2500.0          # beyond this, a colour is an extrapolation from a distant sensor
MIN_SPEED_KMH = 2.0


def _haversine_m(lat, lon, lats, lons):
    lat, lon, lats, lons = map(np.radians, (lat, lon, np.asarray(lats), np.asarray(lons)))
    a = (np.sin((lats - lat) / 2) ** 2
         + np.cos(lat) * np.cos(lats) * np.sin((lons - lon) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def nearest_sensor(midpoint_latlon, nodes_df: pd.DataFrame) -> tuple[int, float]:
    """(node_id, distance_m) of the closest sensor. Great-circle distance, not road distance."""
    d = _haversine_m(midpoint_latlon[0], midpoint_latlon[1], nodes_df["lat"], nodes_df["lon"])
    i = int(np.argmin(d))
    return int(nodes_df["node_id"].iloc[i]), float(d[i])


def assign_sensor_to_edge(edge_midpoint_latlon, nodes_df: pd.DataFrame, k_nearest: int = 1) -> int:
    """node_id of the nearest sensor (only k_nearest=1 is supported)."""
    if k_nearest != 1:
        raise NotImplementedError("only the single nearest sensor is used")
    return nearest_sensor(edge_midpoint_latlon, nodes_df)[0]


def _midpoint(geometry):
    """Point halfway along the polyline's vertex list (good enough for short edges)."""
    return geometry[len(geometry) // 2] if len(geometry) % 2 else tuple(
        np.mean([geometry[len(geometry) // 2 - 1], geometry[len(geometry) // 2]], axis=0))


def colorize_path(path, predictions: dict, nodes_df: pd.DataFrame) -> list[dict]:
    """One segment per edge, coloured by the nearest sensor's forecast.

    predictions: {"nodes": [{"node_id", "pred_kmh", "band", ...}]} for ONE horizon (the shape
    the server builds from predict_once's output).
    """
    by_id = {n["node_id"]: n for n in predictions["nodes"]}
    names = dict(zip(nodes_df["node_id"], nodes_df["name"]))
    segments = []
    for e in path.edges:
        sensor_id, dist = nearest_sensor(_midpoint(e["geometry"]), nodes_df)
        node = by_id[sensor_id]
        segments.append({
            "geometry": [list(p) for p in e["geometry"]], "name": e["name"],
            "highway": e["highway"], "length_m": e["length_m"], "sensor_id": sensor_id,
            "sensor_name": names[sensor_id], "distance_to_sensor_m": round(dist),
            "predicted_speed_kmh": node["pred_kmh"], "band": node["band"],
            "color": node_color(node["band"]), "extrapolated": dist > FAR_SENSOR_M})
    return segments


def path_score(segments: list[dict], free_flow_fallback_kmh: float = 30.0) -> dict:
    """Travel time from predicted speeds (length-weighted), plus band shares by length."""
    total_m = sum(s["length_m"] for s in segments)
    seconds = 0.0
    for s in segments:
        speed = s["predicted_speed_kmh"]
        if speed is None or not np.isfinite(speed):
            speed = free_flow_fallback_kmh
        seconds += s["length_m"] / (max(speed, MIN_SPEED_KMH) * 1000 / 3600)
    share = lambda band: 100 * sum(s["length_m"] for s in segments if s["band"] == band) / max(total_m, 1)
    return {"total_length_km": round(total_m / 1000, 2), "total_time_min": round(seconds / 60, 1),
            "mean_speed_kmh": round((total_m / 1000) / (seconds / 3600), 1) if seconds else 0.0,
            "pct_heavy": round(share("heavy"), 1), "pct_moderate": round(share("moderate"), 1),
            "pct_free": round(share("free"), 1)}
