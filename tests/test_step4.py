import logging

import numpy as np
import pytest
import torch
import yaml

from src.config import GRAPH_DIR, PROJECT_ROOT
from src.model.dcrnn import utils
from src.model.dcrnn.dcrnn_model import forward_batch
from src.model.dcrnn.dcrnn_supervisor import build_model, load_checkpoint

CONFIG_PATH = PROJECT_ROOT / "src" / "model" / "dcrnn" / "config_bangalore.yaml"
BEST = PROJECT_ROOT / "checkpoints" / "bangalore_dcrnn" / "best_model.pt"


def _cfg():
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_config_has_required_keys():
    cfg = _cfg()
    assert {"base_dir", "log_level", "data", "model", "train"} <= set(cfg)
    assert {"batch_size", "dataset_dir", "graph_pkl_filename"} <= set(cfg["data"])
    assert {"horizon", "input_dim", "output_dim", "num_nodes", "seq_len", "rnn_units",
            "filter_type", "max_diffusion_step", "num_rnn_layers"} <= set(cfg["model"])
    assert {"base_lr", "epochs", "patience", "steps", "epsilon", "max_grad_norm",
            "lr_decay_ratio"} <= set(cfg["train"])
    assert cfg["model"]["num_nodes"] == 20 and cfg["model"]["horizon"] == 7


def test_adj_mx_loads():
    _, _, W = utils.load_graph_data(GRAPH_DIR / "adj_mx.pkl")
    assert W.shape == (20, 20)


def test_forward_pass_shape():
    cfg = _cfg()
    W = utils.load_graph_data(GRAPH_DIR / "adj_mx.pkl")[2]
    model = build_model(W, logging.getLogger("test"), **cfg["model"]).eval()
    x = torch.randn(2, 12, 20, 2)
    with torch.no_grad():
        out = forward_batch(model, x)
    assert out.shape == (2, 7, 20, 1)
    assert torch.isfinite(out).all()


def test_best_model_after_training():
    assert BEST.exists(), "run python -m scripts.step4_train first"
    model, ckpt = load_checkpoint(BEST)
    assert np.isfinite(ckpt["val_mae"])
