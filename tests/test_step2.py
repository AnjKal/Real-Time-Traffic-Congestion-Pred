import csv
import pickle

import numpy as np
import pytest

from src.config import GRAPH_DIR

pytestmark = pytest.mark.skipif(not (GRAPH_DIR / "adj_mx.pkl").exists(),
                                reason="Step 2 artifacts not built yet")


def test_nodes_csv():
    with open(GRAPH_DIR / "nodes.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == ["node_id", "name", "lat", "lon", "frc", "free_flow_speed_kmh"]
    assert len(rows) == 20


def test_distance_matrix():
    with open(GRAPH_DIR / "distance_matrix.pkl", "rb") as f:
        D = pickle.load(f)
    assert D.shape == (20, 20)
    assert np.all(np.diag(D) == 0)


def test_adj_mx():
    with open(GRAPH_DIR / "adj_mx.pkl", "rb") as f:
        obj = pickle.load(f)
    assert isinstance(obj, tuple) and len(obj) == 3
    sensor_ids, sensor_id_to_ind, W = obj
    assert len(sensor_ids) == 20
    assert sensor_id_to_ind == {s: i for i, s in enumerate(sensor_ids)}
    assert W.shape == (20, 20)
    assert W.min() >= 0 and W.max() <= 1
    assert 60 <= np.count_nonzero(W) <= 300
