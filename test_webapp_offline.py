"""Offline checks for the web app + routing (modern Python, no model service, no internet).

    python test_webapp_offline.py
"""
import math
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import networkx as nx
import pandas as pd
import requests
from fastapi.testclient import TestClient

from webapp import routing
from webapp import server as srv
from webapp.osm_graph import SPOTS_PATH

RESULTS = []
SPOTS = pd.read_csv(SPOTS_PATH)


def grid_graph():
    """Synthetic road grid (0.01 deg ~ 1.1 km) over the middle of Bangalore, two-way roads."""
    G = nx.DiGraph()
    n = 16
    for i in range(n):
        for j in range(n):
            G.add_node(i * n + j, y=12.90 + i * 0.01, x=77.50 + j * 0.01)
    def metres(a, b):
        dy = (G.nodes[a]["y"] - G.nodes[b]["y"]) * 111000
        dx = (G.nodes[a]["x"] - G.nodes[b]["x"]) * 108000
        return math.hypot(dx, dy)
    for i in range(n):
        for j in range(n):
            a = i * n + j
            for di, dj in ((0, 1), (1, 0)):
                if i + di < n and j + dj < n:
                    b = (i + di) * n + j + dj
                    for u, v in ((a, b), (b, a)):
                        G.add_edge(u, v, length=metres(a, b), name="Road %d-%d" % (a, b),
                                   highway="primary", geometry=None)
    return G


def fake_forecast(horizon):
    nodes = []
    for i, row in SPOTS.iterrows():
        heavy = i % 5 == 0
        nodes.append({"index": i, "junction_id": "J%03d" % (i + 1), "name": row["name"],
                      "lat": float(row.lat), "lon": float(row.lon), "current_kmh": 30.0,
                      "predicted_kmh": 10.0 if heavy else 40.0,
                      "band": "heavy" if heavy else "free",
                      "color": "#e74c3c" if heavy else "#2ecc71"})
    return {"horizon_min": horizon, "input_last_slot": "2026-11-04T12:00:00+05:30",
            "forecast_for": "2026-11-04T12:%02d:00+05:30" % horizon, "mode": "pipeline_test",
            "warning": "PIPELINE TEST", "data_age_min": 1.5, "step_minutes": 2, "nodes": nodes}


class FakeResp(object):
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


def with_model_service(handler):
    original = requests.get
    requests.get = handler
    return original


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append(True)
        print("PASS  %s%s" % (name, ("  -> %s" % detail) if detail else ""))
    except Exception as e:                                          # noqa: BLE001
        RESULTS.append(False)
        print("FAIL  %s: %s" % (name, e))
        traceback.print_exc()


def make_client():
    app = srv.create_app(model_service_url="http://fake", graph=grid_graph(), spots=SPOTS)
    return TestClient(app)


def t_pages_and_presets():
    def handler(url, params=None, timeout=None):
        if url.endswith("/health"):
            return FakeResp(200, {"status": "ok", "ready": True, "mode": "pipeline_test",
                                  "step_minutes": 2, "horizons_min": [2, 4, 6]})
        return FakeResp(200, fake_forecast(int(params["horizon_min"])))
    orig = with_model_service(handler)
    try:
        c = make_client()
        page = c.get("/")
        assert page.status_code == 200 and "Compute Routes" in page.text and "circle_marker" in page.text
        assert c.get("/api/status").json()["horizons_min"] == [2, 4, 6]
        assert c.get("/api/forecast?horizon_min=4").json()["horizon_min"] == 4
        presets = c.get("/api/presets").json()
        assert len(presets) == 5 and all({"lat", "lon", "name"} <= set(p["src"]) for p in presets)
        return "page, status, forecast proxy, %d presets" % len(presets)
    finally:
        requests.get = orig


def t_route():
    orig = with_model_service(lambda url, params=None, timeout=None:
                              FakeResp(200, fake_forecast(int(params["horizon_min"]))))
    try:
        c = make_client()
        body = {"src_lat": 12.915, "src_lon": 77.515, "dst_lat": 13.035, "dst_lon": 77.635,
                "horizon_min": 6, "k_paths": 3}
        r = c.post("/api/route", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        paths = data["paths"]
        assert 1 <= len(paths) <= 3
        times = [p["total_time_min"] for p in paths]
        assert times == sorted(times) and [p["rank"] for p in paths] == list(range(1, len(paths) + 1))
        for p in paths:
            assert abs(p["pct_heavy"] + p["pct_moderate"] + p["pct_free"] - 100) < 0.6
            for s in p["segments"]:
                assert s["band"] in ("free", "heavy") and s["junction_name"] and len(s["geometry"]) >= 2
        assert data["mode"] == "pipeline_test" and data["warning"]
        # distinct alternatives must differ materially
        if len(paths) > 1:
            a = {tuple(map(tuple, s["geometry"])) for s in paths[0]["segments"]}
            b = {tuple(map(tuple, s["geometry"])) for s in paths[1]["segments"]}
            assert len(a & b) / float(min(len(a), len(b))) <= 0.81, "alternatives overlap too much"
        # heavy junction (every 5th) must slow a route that passes near it
        assert any(s["band"] == "heavy" for p in paths for s in p["segments"])
        # a point miles from any road is rejected with a clear message
        far = dict(body, src_lat=13.4, src_lon=78.2)
        rf = c.post("/api/route", json=far)
        assert rf.status_code == 400 and "main road" in rf.json()["detail"], rf.text
        return "%d paths, times %s min" % (len(paths), times)
    finally:
        requests.get = orig


def t_model_service_errors():
    def down(url, params=None, timeout=None):
        raise requests.ConnectionError("refused")
    orig = with_model_service(down)
    try:
        c = make_client()
        r = c.get("/api/forecast?horizon_min=2")
        assert r.status_code == 502 and "model_service.py" in r.json()["detail"], r.text
    finally:
        requests.get = orig

    def not_ready(url, params=None, timeout=None):
        return FakeResp(503, {"detail": "Not enough live data yet: need 12 slots"})
    orig = with_model_service(not_ready)
    try:
        c = make_client()
        r = c.post("/api/route", json={"src_lat": 12.915, "src_lon": 77.515, "dst_lat": 13.0,
                                       "dst_lon": 77.6, "horizon_min": 2})
        assert r.status_code == 503 and "Not enough live data" in r.json()["detail"], r.text
        return "502 when the service is down, 503 passed through when not ready"
    finally:
        requests.get = orig


def t_colorize_math():
    heavy_row, free_row = SPOTS.iloc[0], SPOTS.iloc[7]     # fake data: every 5th junction is heavy
    edges = [{"length_m": 1000.0, "name": "A", "highway": "primary",
              "geometry": [(heavy_row.lat, heavy_row.lon), (heavy_row.lat + 1e-4, heavy_row.lon)]},
             {"length_m": 1000.0, "name": "B", "highway": "primary",
              "geometry": [(free_row.lat, free_row.lon), (free_row.lat + 1e-4, free_row.lon)]}]
    segs = routing.colorize_path(edges, fake_forecast(2)["nodes"])
    assert segs[0]["junction_name"] == heavy_row["name"] and segs[0]["band"] == "heavy"
    assert segs[1]["junction_name"] == free_row["name"] and segs[1]["band"] == "free"
    score = routing.path_score(segs)
    assert score["total_length_km"] == 2.0
    assert abs(score["total_time_min"] - (1 / 10.0 + 1 / 40.0) * 60) < 0.1      # 6 + 1.5 min
    assert score["pct_heavy"] == 50.0 and score["pct_free"] == 50.0


def main():
    check("web app pages, status, forecast proxy and presets", t_pages_and_presets)
    check("route API: ranked distinct colored paths, far points rejected", t_route)
    check("route API: model service down (502) / not ready (503)", t_model_service_errors)
    check("segment coloring and travel-time math", t_colorize_math)
    print("\n" + ("ALL CHECKS PASSED" if all(RESULTS) else "SOME CHECKS FAILED"))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
