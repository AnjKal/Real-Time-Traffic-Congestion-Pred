import json
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from src.config import PROCESSED_DIR
from src.inference import forecast_io
from src.inference.loader import DEFAULT_CHECKPOINT
from src.inference.predict import HORIZONS, congestion_band, format_table, predict_once
from src.tomtom import storage
from src.tomtom.buffer import RollingBuffer

FF = np.linspace(20, 55, 20)


class ConstModel(torch.nn.Module):
    """Mimics the DCRNN call: (seq, batch, nodes*dim) -> (horizon, batch, nodes)."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, inputs):
        return torch.full((7, inputs.shape[1], 20), float(self.value))


def _buffer(start=datetime(2026, 10, 30, 12, 0), step_min=3, speed_ratio=0.6):
    b = RollingBuffer()
    for i in range(12):
        b.push(start + timedelta(minutes=step_min * i), FF * speed_ratio, FF)
    return b


def test_predict_once_structure():
    r = predict_once(ConstModel(0.5), _buffer(), FF, now=datetime(2026, 10, 30, 12, 35))
    assert len(r["nodes"]) == 20
    assert {"forecast_made_at", "input_last_poll_at", "nodes"} <= set(r)
    n = r["nodes"][0]
    for label in HORIZONS:
        assert {f"pred_{label}_kmh", f"congestion_{label}", f"band_{label}"} <= set(n)
    assert {"node_id", "name", "lat", "lon", "current_kmh", "free_flow_kmh",
            "congestion_now"} <= set(n)
    assert n["current_kmh"] == pytest.approx(FF[0] * 0.6, abs=0.01)
    assert r["warnings"] == []
    assert "Forecast made at" in format_table(r)


@pytest.mark.parametrize("value,lo_hi", [(50.0, "hi"), (-5.0, "lo")])
def test_predictions_clipped(value, lo_hi):
    r = predict_once(ConstModel(value), _buffer(), FF, now=datetime(2026, 10, 30, 12, 35))
    for i, n in enumerate(r["nodes"]):
        for label in HORIZONS:
            p = n[f"pred_{label}_kmh"]
            assert 2.0 <= p <= 1.2 * FF[i] + 0.01          # outputs are rounded to 2 dp
            assert p == pytest.approx(1.2 * FF[i] if lo_hi == "hi" else 2.0, abs=0.01)


def test_congestion_band_boundaries():
    assert congestion_band(0.74) == "moderate"
    assert congestion_band(0.75) == "moderate"
    assert congestion_band(0.751) == "free"
    assert congestion_band(0.40) == "moderate"
    assert congestion_band(0.399) == "heavy"


def test_not_ready_buffer_raises():
    with pytest.raises(ValueError):
        predict_once(ConstModel(0.5), RollingBuffer(), FF)


def test_warns_on_stale_or_irregular_buffer():
    r = predict_once(ConstModel(0.5), _buffer(step_min=7), FF, now=datetime(2026, 10, 30, 14, 0))
    assert any("stale" in w for w in r["warnings"])
    assert any("not 3 min apart" in w for w in r["warnings"])


def test_compare_with_actual_abs_error(tmp_path):
    base = datetime(2026, 10, 30, 12, 0, 0)
    result = predict_once(ConstModel(0.5), _buffer(start=base - timedelta(minutes=33)), FF,
                          now=base + timedelta(seconds=20))
    fdir, sdir = tmp_path / "forecasts", tmp_path / "obs"
    forecast_io.save_forecast(result, fdir)

    # real observations at the target times (+6 / +12 / +21 min after the last poll), 4 km/h
    # above whatever was predicted for node 3, node 4 missing at +6 min
    rows = []
    for label, minutes in {"plus_5min": 6, "plus_10min": 12, "plus_20min": 21}.items():
        for n in result["nodes"]:
            if n["node_id"] == 4 and minutes == 6:
                continue
            actual = n[f"pred_{label}_kmh"] + (4.0 if n["node_id"] == 3 else 0.0)
            rows.append(storage.make_row(base + timedelta(minutes=minutes, seconds=5),
                                         n["node_id"], n["lat"], n["lon"], actual, 40.0, 0.9,
                                         False, "FRC2", "{}"))
    storage.append_rows(rows, sdir)

    df = forecast_io.build_accuracy_rows(fdir, sdir, now=base + timedelta(minutes=30))
    assert set(df["horizon_min"]) == {5, 10, 20}
    assert len(df) == 20 * 3 - 1
    n3 = df[df["node_id"] == 3]
    assert (n3["abs_error_kmh"] == 4.0).all() and len(n3) == 3
    assert (df[df["node_id"] != 3]["abs_error_kmh"] < 0.02).all()

    log = forecast_io.update_accuracy_log(df, fdir)
    assert len(forecast_io.update_accuracy_log(df, fdir)) == len(log)      # rerun: no duplicates
    mae = forecast_io.rolling_mae(log, now=base + timedelta(minutes=30))
    assert mae.loc[10] == pytest.approx(4.0 / 20, abs=0.02)


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists() or not (PROCESSED_DIR / "test.npz").exists(),
                    reason="needs the trained checkpoint and test data")
def test_real_model_pipeline_sanity():
    """A historical window pushed through the buffer must give sensible km/h forecasts."""
    import pickle

    from src.inference.loader import load_model

    ff = np.asarray(pickle.load(open(PROCESSED_DIR / "scaler.pkl", "rb"))["free_flow_kmh"])
    test = np.load(PROCESSED_DIR / "test.npz")
    x, y = test["x"][0], test["y"][0, :, :, 0]                  # (12, 20, 2), (7, 20)
    b = RollingBuffer()
    for t in range(12):
        minutes = int(round(x[t, 0, 1] * 1440))
        b.push(datetime(2026, 10, 30, minutes // 60, minutes % 60), x[t, :, 0] * ff, ff)
    model, _ = load_model()
    r = predict_once(model, b, ff, now=datetime(2026, 10, 30, 23, 59))
    pred20 = np.array([n["pred_plus_20min_kmh"] for n in r["nodes"]])
    assert np.abs(pred20 - y[6] * ff).mean() < 6.0               # unscaled inputs would be wildly off
    assert (pred20 >= 2).all() and (pred20 <= 1.2 * ff).all()
