"""Tests for Stage 2 (free-flow route hover cards) and the traffic.features aggregation.

Pure-logic tests that run in the web-app Python (3.10+), no Kafka, model, or Docker required:

    python -m pytest test_stage2_features.py -v

They cover: free-flow propagation onto route segments, the delay-vs-free-flow formula and its
safe handling of missing/zero/invalid speeds, latest-free-flow extraction from the readings CSV,
junction-to-area mapping, and area feature aggregation (averages, congestion distribution, overall
level, coverage, missing readings, duplicate predictions).
"""
import json
import os

import pytest

import feature_aggregator as fa
from webapp import live_readings, routing

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- fixtures / helpers
def _edge(length_m=1000.0, mid=(12.917, 77.608)):
    return {"geometry": [(mid[0] - 0.001, mid[1] - 0.001), mid, (mid[0] + 0.001, mid[1] + 0.001)],
            "name": "Hosur Road", "highway": "trunk", "length_m": length_m}


def _junction(jid="J001", lat=12.917, lon=77.623, **kw):
    node = {"junction_id": jid, "name": kw.get("name", jid), "area": kw.get("area", "HSR Layout"),
            "lat": lat, "lon": lon, "current_kmh": kw.get("current_kmh", 21.0),
            "reference_kmh": kw.get("reference_kmh", 42.0), "predicted_kmh": kw.get("predicted_kmh", 18.0),
            "band": kw.get("band", "moderate"), "color": kw.get("color", "#f1c40f")}
    return node


# --------------------------------------------------------------------------- Stage 2: routing
def test_segment_carries_free_flow_and_delay():
    segs = routing.colorize_path([_edge()], [_junction()], free_flow_by_junction={"J001": 26.0})
    s = segs[0]
    assert s["free_flow_speed_kmh"] == 26.0
    assert s["current_speed_kmh"] == 21.0
    assert s["reference_speed_kmh"] == 42.0           # kept, never used as free-flow
    # 1 km at 18 vs 26 km/h -> (1/18 - 1/26) * 60 == 1.03 min
    assert s["delay_min"] == pytest.approx(1.0, abs=0.1)


def test_delay_is_null_without_free_flow():
    segs = routing.colorize_path([_edge()], [_junction()], free_flow_by_junction={})
    assert segs[0]["free_flow_speed_kmh"] is None
    assert segs[0]["delay_min"] is None               # never substitutes reference_kmh


def test_delay_floored_at_zero_when_faster_than_free_flow():
    segs = routing.colorize_path([_edge()], [_junction(predicted_kmh=50.0)],
                                 free_flow_by_junction={"J001": 40.0})
    assert segs[0]["delay_min"] == 0.0


@pytest.mark.parametrize("pred,free", [(0, 26), (18, 0), (-5, 26), (18, None), (None, 26)])
def test_delay_guards_bad_speeds(pred, free):
    assert routing._segment_delay_min(1000.0, pred, free) is None


def test_extrapolation_flag_when_junction_far():
    segs = routing.colorize_path([_edge(mid=(12.917, 77.608))],
                                 [_junction(lat=13.9, lon=78.9)],       # ~160 km away
                                 free_flow_by_junction={"J001": 26.0})
    assert segs[0]["extrapolated"] is True


def test_existing_fields_preserved():
    segs = routing.colorize_path([_edge()], [_junction()], free_flow_by_junction={"J001": 26.0})
    for key in ("predicted_speed_kmh", "band", "color", "junction_id", "distance_to_junction_m",
                "geometry", "name", "highway", "length_m"):
        assert key in segs[0]
    # path scoring still works on the enriched segments
    score = routing.path_score(segs)
    assert score["total_length_km"] > 0 and "pct_heavy" in score


# --------------------------------------------------------------------------- live free-flow CSV
def test_latest_free_flow_from_csv(tmp_path):
    csv = tmp_path / "readings.csv"
    csv.write_text(
        "timestamp,junction_id,freeFlowSpeed\n"
        "2026-11-04T08:00:00+00:00,J001,25\n"
        "2026-11-04T08:02:00+00:00,J001,27\n"       # newer -> wins
        "2026-11-04T08:02:00+00:00,J002,\n"          # missing -> excluded
        "2026-11-04T08:02:00+00:00,J003,40\n")
    ff, ts = live_readings.latest_free_flow_by_junction(str(csv))
    assert ff["J001"] == 27.0
    assert "J002" not in ff
    assert ff["J003"] == 40.0
    assert ts.startswith("2026-11-04T08:02")


def test_latest_free_flow_missing_file():
    assert live_readings.latest_free_flow_by_junction("/no/such/file.csv") == ({}, None)


def test_free_flow_from_real_archive_if_present():
    path = os.path.join(HERE, "data", "live", "readings.csv")
    if not os.path.exists(path):
        pytest.skip("no live readings archive in this checkout")
    ff, ts = live_readings.latest_free_flow_by_junction(path)
    assert len(ff) > 0 and all(v > 0 for v in ff.values())


# --------------------------------------------------------------------------- area aggregation
def test_build_total_by_area_real_spots():
    total = fa.build_total_by_area()
    assert sum(total.values()) == 100                 # the 100 modeled junctions
    assert all(n >= 1 for n in total.values())


def _prediction(poll_id="P1", nodes=None):
    nodes = nodes or [
        {"junction_id": "J001", "area": "HSR Layout", "predicted_kmh": 18.0, "band": "moderate", "current_kmh": 21.0},
        {"junction_id": "J002", "area": "HSR Layout", "predicted_kmh": 10.0, "band": "heavy", "current_kmh": 12.0},
        {"junction_id": "J018", "area": "MG Road", "predicted_kmh": 40.0, "band": "free", "current_kmh": 38.0}]
    return {"record_type": "forecast", "poll_id": poll_id, "timestamp": "2026-11-04T08:00:00+00:00",
            "made_at": "2026-11-04T13:40:00",
            "forecasts": {"15": {"step_minutes": 15, "data_age_min": 1.0, "nodes": nodes}}}


def _raw():
    return [{"junction_id": "J001", "currentSpeed": 21.0, "freeFlowSpeed": 26.0},
            {"junction_id": "J002", "currentSpeed": 12.0, "freeFlowSpeed": 30.0}]   # J018 missing


def _agg(tmp_path):
    out = str(tmp_path / "features.json")
    total = {"HSR Layout": 2, "MG Road": 1}
    return fa.FeatureAggregator(total, producer=None, out_path=out), out


def test_aggregation_basic(tmp_path):
    agg, out = _agg(tmp_path)
    agg.handle_raw_batch("P1", "2026-11-04T08:00:00+00:00", _raw())
    env = agg.handle_prediction(_prediction())
    assert env["area_count"] == 2
    hsr = next(a for a in env["areas"] if a["area"] == "HSR Layout")
    assert hsr["reporting_junction_count"] == 2
    assert hsr["avg_current_speed_kmh"] == pytest.approx(16.5)
    assert hsr["avg_free_flow_speed_kmh"] == pytest.approx(28.0)
    assert hsr["avg_predicted_speed_kmh"] == pytest.approx(14.0)
    assert hsr["congestion_distribution"] == {"free": 0, "moderate": 1, "heavy": 1}
    assert hsr["overall_congestion_level"] == "heavy"         # tie resolves to more severe
    assert hsr["avg_delay_min_per_km"] > 0
    assert os.path.exists(out)                                # latest written to shared volume
    with open(out) as fh:
        assert json.load(fh)["poll_id"] == "P1"


def test_missing_reading_is_not_counted_as_reporting(tmp_path):
    agg, _ = _agg(tmp_path)
    agg.handle_raw_batch("P1", "2026-11-04T08:00:00+00:00", _raw())   # no J018 reading
    env = agg.handle_prediction(_prediction())
    mg = next(a for a in env["areas"] if a["area"] == "MG Road")
    assert mg["reporting_junction_count"] == 0
    assert mg["avg_current_speed_kmh"] is None
    assert mg["avg_free_flow_speed_kmh"] is None
    assert mg["avg_predicted_speed_kmh"] == 40.0              # model still predicts it
    assert mg["sensor_coverage_pct"] == 0.0


def test_poll_ids_never_mixed(tmp_path):
    agg, _ = _agg(tmp_path)
    # raw for a DIFFERENT poll must not feed this prediction's free-flow
    agg.handle_raw_batch("P0", "t", _raw())
    env = agg.handle_prediction(_prediction(poll_id="P1"))
    hsr = next(a for a in env["areas"] if a["area"] == "HSR Layout")
    assert hsr["avg_free_flow_speed_kmh"] is None             # no raw for P1


def test_duplicate_prediction_ignored(tmp_path):
    agg, _ = _agg(tmp_path)
    agg.handle_raw_batch("P1", "t", _raw())
    assert agg.handle_prediction(_prediction()) is not None
    assert agg.handle_prediction(_prediction()) is None       # dedup


def test_overall_coverage(tmp_path):
    agg, _ = _agg(tmp_path)
    agg.handle_raw_batch("P1", "t", _raw())
    env = agg.handle_prediction(_prediction())
    assert env["sensor_coverage_pct"] == pytest.approx(66.7, abs=0.1)   # 2 of 3 reporting


def test_delay_per_km_formula():
    assert fa._delay_min_per_km(18.0, 26.0) == pytest.approx((60 / 18 - 60 / 26), abs=0.01)
    assert fa._delay_min_per_km(50.0, 40.0) == 0.0            # faster than free-flow -> 0
    assert fa._delay_min_per_km(0, 26) is None
    assert fa._delay_min_per_km(18, None) is None
