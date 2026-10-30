from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app import server
from src.inference.predict import HORIZONS
from src.nodes import NODES
from src.tomtom.buffer import RollingBuffer

FF = np.full(20, 40.0)


def _fake_result(last_poll):
    nodes = []
    for node in NODES:
        n = {"node_id": node.node_id, "name": node.name, "lat": node.lat, "lon": node.lon,
             "current_kmh": 30.0, "free_flow_kmh": 40.0, "congestion_now": 0.25}
        for label in HORIZONS:
            n[f"pred_{label}_kmh"] = 12.0 if node.node_id == 10 else 36.0
            n[f"congestion_{label}"] = 0.7 if node.node_id == 10 else 0.1
            n[f"band_{label}"] = "heavy" if node.node_id == 10 else "free"
        nodes.append(n)
    return {"forecast_made_at": datetime.now().isoformat(timespec="seconds"),
            "input_last_poll_at": last_poll.isoformat(), "warnings": [], "nodes": nodes}


@pytest.fixture
def client(tmp_path, monkeypatch):
    # warm buffer whose newest poll is "now"
    buf = RollingBuffer()
    start = datetime.now().replace(microsecond=0) - timedelta(minutes=33)
    for i in range(12):
        buf.push(start + timedelta(minutes=3 * i), FF * 0.7, FF)
    buf.save(tmp_path / "buffer.pkl")
    calls = []

    def fake_predict(model, buffer, free_flows, *a, **k):
        calls.append(1)
        return _fake_result(buffer.snapshot()[1][-1])

    monkeypatch.setattr(server, "BUFFER_PATH", tmp_path / "buffer.pkl")
    monkeypatch.setattr(server, "FORECAST_DIR", tmp_path / "forecasts")
    monkeypatch.setattr(server, "load_model", lambda: (object(), {}))
    monkeypatch.setattr(server, "get_osm_graph", lambda: None)
    monkeypatch.setattr(server, "load_reference_free_flow", lambda: FF)
    monkeypatch.setattr(server, "predict_once", fake_predict)
    with TestClient(server.app) as c:
        c.calls, c.tmp = calls, tmp_path
        yield c


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "DCRNN" in r.text and "node-0" in r.text and "google.com/maps" in r.text


def test_forecast_endpoint_shape(client):
    r = client.get("/api/forecast?horizon=5")
    assert r.status_code == 200
    body = r.json()
    assert body["horizon_min"] == 5 and len(body["nodes"]) == 20
    required = {"node_id", "name", "lat", "lon", "current_kmh", "pred_kmh", "free_flow_kmh",
                "congestion_pct", "band", "color"}
    assert all(required <= set(n) for n in body["nodes"])
    silk = next(n for n in body["nodes"] if n["node_id"] == 10)
    assert (silk["band"], silk["color"], silk["congestion_pct"]) == ("heavy", "#e74c3c", 70.0)
    other = next(n for n in body["nodes"] if n["node_id"] == 0)
    assert (other["band"], other["color"]) == ("free", "#2ecc71")


@pytest.mark.parametrize("h", [5, 10, 20])
def test_all_horizons_ok(client, h):
    assert client.get(f"/api/forecast?horizon={h}").json()["horizon_min"] == h


def test_invalid_horizon_is_400(client):
    assert client.get("/api/forecast?horizon=99").status_code == 400


def test_forecast_is_reused_for_same_poll(client):
    client.get("/api/forecast?horizon=5")
    client.get("/api/forecast?horizon=10")
    client.get("/api/forecast?horizon=20")
    assert len(client.calls) == 1                         # one inference, then cached JSON
    assert len(list((client.tmp / "forecasts").glob("forecast_*.json"))) == 1


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["buffer_ready"] is True
    assert body["latest_forecast_at"] is None
    client.get("/api/forecast?horizon=5")
    assert client.get("/api/health").json()["latest_forecast_at"] is not None


def test_503_when_no_buffer_and_no_forecast(client):
    (client.tmp / "buffer.pkl").unlink()
    assert client.get("/api/forecast?horizon=5").status_code == 503
    assert client.get("/api/health").json()["buffer_ready"] is False


def test_stale_buffer_adds_warning(client):
    buf = RollingBuffer()
    old = datetime.now() - timedelta(hours=3)
    for i in range(12):
        buf.push(old + timedelta(minutes=3 * i), FF * 0.7, FF)
    buf.save(client.tmp / "buffer.pkl")
    body = client.get("/api/forecast?horizon=5").json()
    assert any("stale" in w for w in body["warnings"])
