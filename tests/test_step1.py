import importlib
import sys

import pytest

from scripts.verify_step1 import PACKAGES


def test_python_version():
    assert sys.version_info >= (3, 10)


@pytest.mark.parametrize("mod,dist", [p for p in PACKAGES if p[0] != "ruff"])
def test_imports(mod, dist):
    importlib.import_module(mod)


def test_directories_exist():
    from src import config

    for name in ("DATA_DIR", "RAW_DIR", "SYNTHETIC_DIR", "PROCESSED_DIR", "GRAPH_DIR",
                 "CHECKPOINT_DIR", "LOG_DIR"):
        assert getattr(config, name).is_dir(), name


def test_nodes():
    from src.config import NUM_NODES
    from src.nodes import NODES

    assert len(NODES) == NUM_NODES
    assert [n.node_id for n in NODES] == list(range(NUM_NODES))
