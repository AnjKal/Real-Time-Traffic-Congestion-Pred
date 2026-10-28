"""Dashboard backend: tails traffic.predictions and serves the latest forecast + sensor locations.

Runs fine in the same environment as the rest (Flask + kafka-python only).
"""
import json
import os
import pickle
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory
from kafka import KafkaConsumer, TopicPartition

import settings
from create_topics import ensure_topics

app = Flask(__name__, static_folder="static")
LOCK = threading.Lock()
STATE = {"latest": None, "received_at": None}


def load_sensors():
    ids = settings.load_sensor_ids()
    loc = pd.read_csv(settings.SENSOR_LOC_CSV, dtype={"sensor_id": str})
    loc.columns = [c.strip().lower() for c in loc.columns]
    id_column = "sensor_id" if "sensor_id" in loc.columns else "node_id"
    lat_column = "latitude" if "latitude" in loc.columns else "lat"
    lon_column = "longitude" if "longitude" in loc.columns else "lon"
    by_id = {
        str(r[id_column]).strip(): (float(r[lat_column]), float(r[lon_column]))
        for _, r in loc.iterrows()
    }
    sensors = []
    for i, sid in enumerate(ids):
        lat, lon = by_id.get(sid, (None, None))
        row = loc[loc[id_column].astype(str).str.strip() == sid]
        sensors.append({
            "index": i,
            "id": sid,
            "lat": lat,
            "lon": lon,
            "name": row.iloc[0].get("name", sid) if not row.empty else sid,
            "area": row.iloc[0].get("area", sid) if not row.empty else sid,
            "road": row.iloc[0].get("road", sid) if not row.empty else sid,
        })
    return sensors


SENSORS = load_sensors()
CARTO_KEY = os.environ.get("CARTO_API_KEY", settings.CARTO_API_KEY).strip()


def load_road_segments():
    """Return graph-connected sensor pairs for coloring the road corridors."""
    graph_file = os.path.join(settings.DATASET_DIR, "adj_bangalore.pkl")
    try:
        with open(graph_file, "rb") as f:
            _, _, adjacency = pickle.load(f)
    except (OSError, pickle.PickleError, ValueError) as exc:
        print("dashboard road graph error: %s" % exc)
        return []

    segments = []
    for i in range(len(SENSORS)):
        for j in range(i + 1, len(SENSORS)):
            if adjacency[i, j] > 0 or adjacency[j, i] > 0:
                segments.append({
                    "from": i,
                    "to": j,
                    "weight": round(float(max(adjacency[i, j], adjacency[j, i])), 4),
                })
    return segments


ROAD_SEGMENTS = load_road_segments()


def load_edge_geometry():
    path = settings.EDGE_GEOMETRY_FILE
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("type") != "FeatureCollection":
            raise ValueError("edge geometry is not a FeatureCollection")
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print("dashboard edge geometry unavailable (%s); using straight-line fallback" % exc)
        return {"type": "FeatureCollection", "features": []}


EDGE_GEOMETRY = load_edge_geometry()
EDGE_GEOMETRY_LOCK = threading.Lock()


def route_edge_with_osrm(road):
    """Fetch curved driving geometry for one graph edge from OSRM."""
    source = SENSORS[road["from"]]
    target = SENSORS[road["to"]]
    coordinates = "%s,%s;%s,%s" % (source["lon"], source["lat"], target["lon"], target["lat"])
    url = settings.ROAD_ROUTER_URL.rstrip("/") + "/route/v1/driving/" + coordinates
    query = urllib.parse.urlencode({"overview": "full", "geometries": "geojson", "steps": "false"})
    request = urllib.request.Request(url + "?" + query, headers={"User-Agent": "DCRNN-dashboard/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        route = payload["routes"][0]
        return {
            "type": "Feature",
            "properties": {
                "source": source["id"],
                "target": target["id"],
                "distance_km": round(float(route["distance"]) / 1000.0, 3),
                "weight": road["weight"],
            },
            "geometry": route["geometry"],
        }
    except (KeyError, IndexError, TypeError, ValueError, OSError, urllib.error.URLError) as exc:
        print("road routing failed for %s -> %s: %s" % (source["id"], target["id"], exc))
        return None


def build_edge_geometry_from_router():
    """Route graph edges on demand when the offline geometry artifact is absent."""
    global EDGE_GEOMETRY
    with EDGE_GEOMETRY_LOCK:
        existing = {
            tuple(sorted((feature.get("properties", {}).get("source"),
                          feature.get("properties", {}).get("target"))))
            for feature in EDGE_GEOMETRY["features"]
        }
        pending = [
            road for road in ROAD_SEGMENTS
            if tuple(sorted((SENSORS[road["from"]]["id"], SENSORS[road["to"]]["id"]))) not in existing
        ]
        if not pending:
            return EDGE_GEOMETRY
        print("Requesting %d missing road geometries from OSRM..." % len(pending))
        features = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(route_edge_with_osrm, road) for road in pending]
            for future in as_completed(futures):
                feature = future.result()
                if feature:
                    features.append(feature)
        EDGE_GEOMETRY["features"].extend(features)
        if features:
            try:
                with open(settings.EDGE_GEOMETRY_FILE, "w", encoding="utf-8") as f:
                    json.dump(EDGE_GEOMETRY, f)
            except OSError as exc:
                print("could not cache routed edge geometry: %s" % exc)
        return EDGE_GEOMETRY


def consume_loop():
    """Keep the newest forecast in memory; reconnect automatically if Kafka drops."""
    while True:
        try:
            consumer = KafkaConsumer(
                bootstrap_servers=settings.BOOTSTRAP,
                enable_auto_commit=False,
                value_deserializer=lambda b: json.loads(b.decode()),
            )
            tp = TopicPartition(settings.TOPIC_PRED, 0)
            consumer.assign([tp])
            end = consumer.end_offsets([tp])[tp]
            consumer.seek(tp, max(0, end - 1))          # start from the newest forecast
            while True:
                for recs in consumer.poll(timeout_ms=1000).values():
                    for r in recs:
                        with LOCK:
                            STATE["latest"] = r.value
                            STATE["received_at"] = time.time()
        except Exception as exc:                          # broker restart, topic not yet created...
            print("dashboard consumer error: %s (retrying in 3 s)" % exc)
            time.sleep(3)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/config")
def api_config():
    # The browser needs the key to request map tiles, so it is sent to the page (it is visible in the
    # browser's network tab either way). Restrict the key to your domain/localhost in the CARTO dashboard.
    return jsonify({"carto_key": CARTO_KEY})


@app.route("/api/sensors")
def api_sensors():
    return jsonify(SENSORS)


@app.route("/api/roads")
def api_roads():
    return jsonify(ROAD_SEGMENTS)


@app.route("/api/edges")
def api_edges():
    return jsonify(build_edge_geometry_from_router())


@app.route("/api/state")
def api_state():
    with LOCK:
        latest, received = STATE["latest"], STATE["received_at"]
    if latest is None:
        return jsonify({"ready": False})
    payload = dict(latest)
    payload["ready"] = True
    payload["age_s"] = round(time.time() - received, 1)
    payload["level_names"] = settings.LEVEL_NAMES
    payload["thresholds"] = list(settings.LEVEL_THRESHOLDS)
    return jsonify(payload)


if __name__ == "__main__":
    try:
        ensure_topics()
    except SystemExit as e:
        print(e)
    threading.Thread(target=consume_loop, daemon=True).start()
    if not CARTO_KEY:
        print("No CARTO key set (CARTO_API_KEY): using the keyless Esri gray basemap instead.")
    print("Dashboard on http://%s:%d" % (settings.DASHBOARD_HOST, settings.DASHBOARD_PORT))
    app.run(host=settings.DASHBOARD_HOST, port=settings.DASHBOARD_PORT, threaded=True)