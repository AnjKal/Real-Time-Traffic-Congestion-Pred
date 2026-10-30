from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app import server
from src.inference.predict import HORIZONS
from src.nodes import NODES
from src.routing.spreading import colorize_path, path_score
from src.tomtom.buffer import RollingBuffer

FF = np.full(20, 40.0)
BY_NAME = {n.name: n for n in NODES}
PRESET_LABELS = {"Marathahalli -> Silk Board (ORR lunch)", "Hebbal -> M.G. Road (morning rush)",
                 "Indiranagar -> Jayanagar (CBD)", "Whitefield -> Majestic (east-west)",
                 "Silk Board -> Marathahalli (evening ORR)"}


def _fake_result(last_poll):
    nodes = []
    for node in NODES:
        n = {"node_id": node.node_id, "name": node.name, "lat": node.lat, "lon": node.lon,
             "current_kmh": 30.0, "free_flow_kmh": 40.0, "congestion_now": 0.25}
        heavy = node.node_id == 10                       # Silk Board is jammed
        for label in HORIZONS:
            n[f"pred_{label}_kmh"] = 8.0 if heavy else 32.0
            n[f"congestion_{label}"] = 0.8 if heavy else 0.2
            n[f"band_{label}"] = "heavy" if heavy else "free"
        nodes.append(n)
    return {"forecast_made_at": datetime.now().isoformat(timespec="seconds"),
            "input_last_poll_at": last_poll.isoformat(), "warnings": [], "nodes": nodes}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("step8")
    buf = RollingBuffer()
    start = datetime.now().replace(microsecond=0) - timedelta(minutes=33)
    for i in range(12):
        buf.push(start + timedelta(minutes=3 * i), FF * 0.7, FF)
    buf.save(tmp / "buffer.pkl")

    mp = pytest.MonkeyPatch()
    mp.setattr(server, "BUFFER_PATH", tmp / "buffer.pkl")
    mp.setattr(server, "FORECAST_DIR", tmp / "forecasts")
    mp.setattr(server, "load_model", lambda: (object(), {}))
    mp.setattr(server, "load_reference_free_flow", lambda: FF)
    mp.setattr(server, "predict_once",
               lambda model, buffer, ff, *a, **k: _fake_result(buffer.snapshot()[1][-1]))
    with TestClient(server.app) as c:                   # loads the real OSM graph (cached)
        yield c
    mp.undo()


def _route(client, a, b, **extra):
    body = {"src_lat": BY_NAME[a].lat, "src_lon": BY_NAME[a].lon,
            "dst_lat": BY_NAME[b].lat, "dst_lon": BY_NAME[b].lon, "horizon_min": 5,
            "k_paths": 3, **extra}
    return client.post("/api/route", json=body)


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "DCRNN" in r.text and "Compute Routes" in r.text


def test_route_marathahalli_to_silk_board(client):
    r = _route(client, "Marathahalli Bridge", "Silk Board")
    assert r.status_code == 200
    body = r.json()
    assert 1 <= len(body["paths"]) <= 3
    times = [p["total_time_min"] for p in body["paths"]]
    assert times == sorted(times) and [p["rank"] for p in body["paths"]] == list(
        range(1, len(times) + 1))
    for p in body["paths"]:
        assert 5 < p["total_length_km"] < 40
        assert abs(p["pct_heavy"] + p["pct_moderate"] + p["pct_free"] - 100) < 0.5
        for s in p["segments"]:
            assert s["sensor_name"] and s["band"] in {"free", "moderate", "heavy"}
            assert s["color"] in {"#2ecc71", "#f1c40f", "#e74c3c"} and len(s["geometry"]) >= 2
    # the jam at Silk Board must colour at least some of the route near its destination
    assert any(s["band"] == "heavy" and s["sensor_name"] == "Silk Board"
               for s in body["paths"][0]["segments"])


def test_route_horizon_and_k(client):
    r = _route(client, "Hebbal Flyover", "M.G. Road Trinity", horizon_min=20, k_paths=1)
    assert r.status_code == 200 and len(r.json()["paths"]) == 1 and r.json()["horizon_min"] == 20


def test_route_outside_bbox_is_400(client):
    r = client.post("/api/route", json={"src_lat": 20.0, "src_lon": 77.6, "dst_lat": 12.9,
                                        "dst_lon": 77.6, "horizon_min": 5, "k_paths": 3})
    assert r.status_code == 400 and "outside" in r.json()["detail"]


def test_route_bad_horizon_is_400(client):
    assert _route(client, "Hebbal Flyover", "Majestic", horizon_min=99).status_code == 400


def test_presets(client):
    presets = client.get("/api/presets").json()
    assert {p["label"] for p in presets} == PRESET_LABELS
    assert all({"lat", "lon"} <= set(p["src"]) and {"lat", "lon"} <= set(p["dst"]) for p in presets)
    first = next(p for p in presets if p["label"].startswith("Marathahalli"))
    assert first["src"]["lat"] == BY_NAME["Marathahalli Bridge"].lat


def test_predict_endpoint(client):
    body = client.get("/api/predict?horizon=10").json()
    assert body["horizon_min"] == 10 and len(body["nodes"]) == 20


def test_path_score_math():
    segs = [{"length_m": 1000, "predicted_speed_kmh": 30.0, "band": "free"},
            {"length_m": 1000, "predicted_speed_kmh": 10.0, "band": "heavy"}]
    s = path_score(segs)
    assert s["total_length_km"] == 2.0 and s["total_time_min"] == 8.0      # 2 min + 6 min
    assert s["pct_heavy"] == 50.0 and s["pct_free"] == 50.0 and s["mean_speed_kmh"] == 15.0
    assert path_score([{"length_m": 1000, "predicted_speed_kmh": float("nan"), "band": "free"}]
                      )["total_time_min"] == 2.0                           # 30 km/h fallback


def test_colorize_uses_nearest_sensor(client):
    from src.routing.pathfinder import Path

    n = BY_NAME["Silk Board"]
    edge = {"osm_u": 1, "osm_v": 2, "length_m": 100.0, "name": "x", "highway": "primary",
            "geometry": [(n.lat, n.lon), (n.lat + 0.0005, n.lon)]}
    preds = server.shape_for_horizon(_fake_result(datetime.now()), 5)
    seg = colorize_path(Path([1, 2], [edge], 100.0), preds, server.app.state.nodes_df)[0]
    assert seg["sensor_name"] == "Silk Board" and seg["band"] == "heavy"
    assert seg["distance_to_sensor_m"] < 100 and seg["extrapolated"] is False
