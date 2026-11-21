"""Offline checks for the live pipeline (collector, window builder, model service).

    python test_pipeline_offline.py

No TomTom calls and no model loading. Run it inside the Python 3.7 environment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

import collect_tomtom as ct
from predict_bangalore import build_window_from_readings

RESULTS = []
JUNCTIONS = [("J%03d" % (i + 1), "Junction %d" % (i + 1), 12.9 + i * 0.001, 77.6 + i * 0.001)
             for i in range(100)]


def run(name, fn):
    tmp = tempfile.mkdtemp()
    try:
        detail = fn(tmp)
        RESULTS.append(True)
        print("PASS  %s%s" % (name, ("  -> %s" % detail) if detail else ""))
    except Exception as e:                                           # noqa: BLE001
        RESULTS.append(False)
        print("FAIL  %s: %s" % (name, e))
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class Clock(object):
    def __init__(self, start):
        self.t = float(start)

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


def speed_of(lat, lon):
    """Deterministic fake speed per junction (km/h) so rows can be checked."""
    return round(20 + (lat - 12.9) * 1000 % 40, 3)


def good_fetch(lat, lon):
    return {"currentSpeed": speed_of(lat, lon), "freeFlowSpeed": 50, "confidence": 0.9,
            "frc": "FRC2", "roadClosure": False, "currentTravelTime": 120, "freeFlowTravelTime": 90,
            "coordinates": {"coordinate": [{"latitude": lat, "longitude": lon}]}}


def make_collector(tmp, fetch, **kw):
    clock = kw.pop("clock", Clock(1700000000))
    kw.setdefault("daily_quota", 2400)
    return ct.Collector(fetch, JUNCTIONS, out_path=os.path.join(tmp, "readings.csv"),
                        quota=ct.QuotaTracker(os.path.join(tmp, "quota.json")), max_qps=None,
                        sleep_fn=clock.sleep, now_fn=clock.time, **kw), clock


def t_poll_ok(tmp):
    c, _ = make_collector(tmp, good_fetch, interval_min=2)
    s = c.poll_once(1700000040)
    assert (s["ok"], s["failed"], s["calls"]) == (100, 0, 100), s
    df = pd.read_csv(c.out_path)
    assert list(df.columns) == ct.COLUMNS and len(df) == 100
    assert df["junction_name"].iloc[0] == "Junction 1" and df["frc"].iloc[0] == "FRC2"
    assert df["currentTravelTime"].iloc[0] == 120 and "latitude" in df["coordinates"].iloc[0]
    assert df["timestamp"].nunique() == 1 and df["timestamp"].iloc[0].startswith("2023-11-14T22:")
    assert df["junction_id"].tolist() == [j[0] for j in JUNCTIONS]
    assert c.quota.used_today() == 100


def t_retry_and_permanent_failure(tmp):
    seen = {}

    def fetch(lat, lon):
        n = seen[lat] = seen.get(lat, 0) + 1
        if abs(lat - JUNCTIONS[2][2]) < 1e-9 and n == 1:
            raise ct.TransientError("HTTP 500")                      # recovers on retry
        if abs(lat - JUNCTIONS[5][2]) < 1e-9:
            raise ct.ClientError("HTTP 404")                         # permanent: no retry
        if abs(lat - JUNCTIONS[7][2]) < 1e-9:
            raise ct.TransientError("HTTP 503")                      # never recovers
        return good_fetch(lat, lon)

    c, _ = make_collector(tmp, fetch, interval_min=2, max_retries=2)
    s = c.poll_once(1700000040)
    assert (s["ok"], s["failed"]) == (98, 2), s
    assert s["calls"] == 98 + 1 + 1 + 3, s            # 98 ok + retry + 404 once + 503 three times
    ids = set(pd.read_csv(c.out_path)["junction_id"])
    assert "J006" not in ids and "J008" not in ids and "J003" in ids


def t_quota_budget(tmp):
    c, _ = make_collector(tmp, good_fetch, interval_min=2, daily_quota=60)
    s = c.poll_once(1700000040)
    assert s["calls"] == 60 and s["ok"] == 60, s       # never exceeds the daily budget
    logs = []
    assert c.run(max_polls=3, start_now=True, log=logs.append) == 0       # nothing left to poll
    assert any("quota" in l.lower() for l in logs), logs


def t_fatal_key(tmp):
    def fetch(lat, lon):
        raise ct.FatalError("HTTP 403: the TomTom API key was rejected")

    c, _ = make_collector(tmp, fetch, interval_min=2)
    try:
        c.poll_once(1700000040)
    except ct.FatalError:
        assert not os.path.exists(c.out_path), "must not write rows after a rejected key"
        return "stops on rejected key"
    raise AssertionError("FatalError not raised")


def t_schedule_aligned(tmp):
    c, clock = make_collector(tmp, good_fetch, interval_min=2, clock=Clock(1700000050))
    logs = []
    assert c.run(max_polls=4, log=logs.append) == 4
    df = pd.read_csv(c.out_path)
    stamps = sorted(pd.to_datetime(df["timestamp"].unique(), utc=True))
    gaps = {(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])}
    assert len(stamps) == 4 and gaps == {120.0}, (len(stamps), gaps)
    assert all(int(s.timestamp()) % 120 == 0 for s in stamps), "slots must sit on 2-minute marks"
    assert int(stamps[0].timestamp()) == 1700000160                  # first boundary after start
    assert "poll 4/4" in logs[-1]


def t_window_from_collector_output(tmp):
    c, clock = make_collector(tmp, good_fetch, interval_min=2, clock=Clock(1700000050))
    c.run(max_polls=12)
    df = pd.read_csv(c.out_path)
    w, times = build_window_from_readings(df, [j[0] for j in JUNCTIONS], step_minutes=2)
    assert w.shape == (12, 100)
    assert np.allclose(w[0], [speed_of(j[2], j[3]) for j in JUNCTIONS], atol=1e-3)
    assert [(b - a).total_seconds() for a, b in zip(times, times[1:])] == [120.0] * 11
    # 11 slots is NOT enough
    short = df[df["timestamp"] != df["timestamp"].iloc[-1]]
    try:
        build_window_from_readings(short, [j[0] for j in JUNCTIONS], step_minutes=2)
    except ValueError as e:
        return "12 slots accepted, 11 rejected (%s...)" % str(e)[:45]
    raise AssertionError("11 slots were accepted")


class StubForecaster(object):
    """Same interface the service needs from BangaloreForecaster, without TensorFlow."""
    seq_len, num_nodes, step_minutes = 12, 100, 2
    sensor_ids = [j[0] for j in JUNCTIONS]
    horizon_minutes = [2 * (k + 1) for k in range(12)]
    reference_kmh = np.full(100, 40.0)
    band_thresholds = (0.75, 0.40)
    spots = pd.DataFrame({"name": ["Junction %d" % (i + 1) for i in range(100)],
                          "area": ["Area"] * 100, "lat": [12.9] * 100, "lon": [77.6] * 100})
    calls = 0

    def predict(self, speeds, times):
        self.calls += 1
        return np.repeat(np.asarray(speeds)[-1:], 12, axis=0) * 0.9

    def forecast_from_prediction(self, pred_all, speeds, times, horizon_min):
        k = horizon_min // 2 - 1
        return {"horizon_min": horizon_min, "input_last_slot": times[-1].isoformat(),
                "forecast_for": (times[-1] + pd.Timedelta(minutes=horizon_min)).isoformat(),
                "nodes": [{"index": i, "junction_id": "J%03d" % (i + 1),
                           "predicted_kmh": float(pred_all[k, i])} for i in range(100)]}


def t_service_engine_and_api(tmp):
    from fastapi.testclient import TestClient
    import model_service as ms

    readings = os.path.join(tmp, "readings.csv")
    fdir = os.path.join(tmp, "forecasts")
    stub = StubForecaster()
    engine = ms.ForecastEngine(stub, readings, fdir)
    client = TestClient(ms.create_app(engine=engine))

    h = client.get("/health").json()
    assert h["ready"] is False and "no readings" in h["not_ready_reason"], h
    r = client.get("/forecast?horizon_min=4")
    assert r.status_code == 503 and "Not enough live data" in r.json()["detail"], r.text

    # 12 polls through the real collector code, stamped to END a minute ago
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = int((now - timedelta(minutes=2 * 12 + 1)).timestamp()) // 120 * 120
    c = ct.Collector(good_fetch, JUNCTIONS, out_path=readings, interval_min=2, max_qps=None,
                     quota=ct.QuotaTracker(os.path.join(tmp, "q.json")))
    for i in range(11):
        c.poll_once(start + 120 * i)
    assert client.get("/health").json()["ready"] is False             # 11 slots: still waiting
    c.poll_once(start + 120 * 11)

    h = client.get("/health").json()
    assert h["ready"] is True and h["mode"] == "pipeline_test", h
    r = client.get("/forecast?horizon_min=6")
    body = r.json()
    assert r.status_code == 200 and body["horizon_min"] == 6 and len(body["nodes"]) == 100, r.text
    assert body["mode"] == "pipeline_test" and "PIPELINE TEST" in body["warning"], body
    assert abs(body["nodes"][0]["predicted_kmh"] - 0.9 * speed_of(*JUNCTIONS[0][2:])) < 1e-3
    client.get("/forecast?horizon_min=24")
    assert stub.calls == 1, "model must run once per new window, not once per request"
    assert len(os.listdir(fdir)) == 1
    assert client.get("/forecast?horizon_min=15").status_code == 400
    assert client.get("/forecast?horizon_min=abc").status_code == 400
    # a new poll => a new window => a second model run and a second saved forecast
    c.poll_once(start + 120 * 12)
    client.get("/forecast?horizon_min=2")
    assert stub.calls == 2 and len(os.listdir(fdir)) == 2
    return "model runs once per new window, 2 forecasts saved"


def t_node_lookup(tmp):
    from fastapi.testclient import TestClient
    import model_service as ms

    readings = os.path.join(tmp, "readings.csv")
    engine = ms.ForecastEngine(StubForecaster(), readings, os.path.join(tmp, "f"))
    client = TestClient(ms.create_app(engine=engine))
    assert client.get("/node?name=J001").status_code == 503             # no readings yet
    try:
        engine.node_forecast("nothing")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = int((now - timedelta(minutes=26)).timestamp()) // 120 * 120
    c = ct.Collector(good_fetch, JUNCTIONS, out_path=readings, interval_min=2, max_qps=None,
                     quota=ct.QuotaTracker(os.path.join(tmp, "q.json")))
    for i in range(12):
        c.poll_once(start + 120 * i)
    assert client.get("/node?name=junction").status_code == 400, "ambiguous names are rejected"
    r = client.get("/node?name=Junction 7")                 # exact name (any case) wins
    body = r.json()
    assert r.status_code == 200 and body["junction_id"] == "J007", r.text
    assert len(body["horizons"]) == 12 and body["mode"] == "pipeline_test"
    assert [h["minutes_ahead"] for h in body["horizons"]] == list(range(2, 26, 2))
    assert abs(body["current_kmh"] - speed_of(*JUNCTIONS[6][2:])) < 1e-2
    assert abs(body["horizons"][0]["predicted_kmh"] - 0.9 * speed_of(*JUNCTIONS[6][2:])) < 1e-2
    assert client.get("/node?name=j007").json()["junction_id"] == "J007"            # id, any case
    assert client.get("/node?name=zzz").status_code == 404
    assert len(client.get("/nodes").json()) == 100
    return "id / name lookup, ambiguity and unknown names handled"


def main():
    run("collector: one poll writes 100 stamped rows and counts quota", t_poll_ok)
    run("collector: retries transient errors, skips failed junctions", t_retry_and_permanent_failure)
    run("collector: never exceeds the daily budget", t_quota_budget)
    run("collector: stops immediately on a rejected API key", t_fatal_key)
    run("collector: polls sit on exact 2-minute marks", t_schedule_aligned)
    run("collector output -> 12-slot model window (and 11 slots rejected)",
        t_window_from_collector_output)
    run("model service API (stub model): 503 -> ready -> forecasts, caching, validation",
        t_service_engine_and_api)
    run("single-node lookup (what you get when you type a junction name)", t_node_lookup)
    print("\n" + ("ALL CHECKS PASSED" if all(RESULTS) else "SOME CHECKS FAILED"))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
