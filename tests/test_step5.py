import json

import numpy as np
import pytest

from src.config import PROJECT_ROOT

METRICS = PROJECT_ROOT / "reports" / "step5_metrics.json"
MODELS = ["DCRNN", "Persistence", "Hist. Average", "Last-Week"]
LABELS = ["5min", "10min", "20min"]


def _load():
    if not METRICS.exists():
        pytest.skip("run python -m scripts.step5_evaluate first")
    return json.loads(METRICS.read_text())


def test_metrics_file_exists_with_keys():
    assert METRICS.exists(), "run python -m scripts.step5_evaluate first"
    r = _load()
    assert {"meta", "models", "verdict"} <= set(r)
    assert set(MODELS) <= set(r["models"])
    for m in MODELS:
        assert {"overall", "per_horizon", "key_horizons", "per_node_mae",
                "rush_vs_offpeak_mae", "weekday_vs_weekend_mae"} <= set(r["models"][m])


def test_dcrnn_mae_positive_finite():
    for lab in LABELS:
        mae = _load()["models"]["DCRNN"]["key_horizons"][lab]["mae"]
        assert np.isfinite(mae) and mae > 0, (lab, mae)


def test_dcrnn_beats_persistence_20min_by_10pct():
    r = _load()["models"]
    d, p = r["DCRNN"]["key_horizons"]["20min"]["mae"], r["Persistence"]["key_horizons"]["20min"]["mae"]
    assert (p - d) / p >= 0.10, f"DCRNN {d:.2f} vs persistence {p:.2f}: model is barely learning"


def test_baseline_maes_finite():
    r = _load()["models"]
    for m in MODELS[1:]:
        for lab in LABELS:
            assert np.isfinite(r[m]["key_horizons"][lab]["mae"]), (m, lab)
