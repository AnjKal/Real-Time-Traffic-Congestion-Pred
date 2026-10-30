import json

import numpy as np
import pandas as pd
import pytest

from src.config import GRAPH_DIR, NUM_NODES, PROCESSED_DIR, SYNTHETIC_DIR

PARQUET = SYNTHETIC_DIR / "raw_speeds.parquet"
pytestmark = pytest.mark.skipif(not (PROCESSED_DIR / "train.npz").exists(),
                                reason="Step 3 artifacts not built yet")


def test_parquet_rows():
    df = pd.read_parquet(PARQUET)
    assert len(df) == 43200 * 20
    assert list(df.columns) == ["timestamp", "node_id", "speed_kmh", "confidence",
                                "scenario_id", "is_weekend"]


@pytest.mark.parametrize("name", ["train", "val", "test"])
def test_npz_shapes_and_nans(name):
    d = np.load(PROCESSED_DIR / f"{name}.npz")
    assert set(d.files) == {"x", "y", "x_offsets", "y_offsets"}
    assert d["x"].shape[1:] == (12, NUM_NODES, 2)
    assert d["y"].shape[1:] == (7, NUM_NODES, 1)
    for k in d.files:
        assert not np.isnan(d[k]).any(), k


def test_validation_report_all_pass():
    report = json.loads((SYNTHETIC_DIR / "validation_report.json").read_text())
    assert len(report) == 6
    assert all(g["pass"] for g in report.values()), report


def test_per_node_mean_speed_ratio():
    ff = pd.read_csv(GRAPH_DIR / "nodes.csv").sort_values("node_id")["free_flow_speed_kmh"]
    mean = pd.read_parquet(PARQUET, columns=["node_id", "speed_kmh"]).groupby("node_id").mean()
    ratio = mean["speed_kmh"].to_numpy() / ff.to_numpy()
    assert ((ratio >= 0.4) & (ratio <= 0.9)).all(), ratio
