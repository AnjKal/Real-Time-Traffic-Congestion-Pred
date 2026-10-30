"""Load the trained DCRNN for inference."""
import logging
from pathlib import Path

import torch
import yaml

from src.config import PROJECT_ROOT
from src.model.dcrnn import utils
from src.model.dcrnn.dcrnn_supervisor import build_model

DEFAULT_CONFIG = PROJECT_ROOT / "src" / "model" / "dcrnn" / "config_bangalore.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "bangalore_dcrnn" / "best_model.pt"
ARCH_KEYS = ("num_nodes", "input_dim", "output_dim", "horizon", "seq_len", "rnn_units",
             "num_rnn_layers", "max_diffusion_step", "filter_type")


def _abs(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model(config_path=DEFAULT_CONFIG, checkpoint_path=DEFAULT_CHECKPOINT, device="cpu"):
    """Returns (model in eval mode, config dict).

    The model expects the speed channel standardised with the TRAINING mean/std; those are
    stored in the checkpoint and attached as model.speed_scaler = (mean, std).
    Note: the vendored DCRNN code places tensors on cuda if available, else cpu, regardless
    of `device`; the argument only controls where the checkpoint is mapped while loading.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
    trained = ckpt["config"]["model"]
    mismatch = {k: (config["model"][k], trained[k]) for k in ARCH_KEYS
                if config["model"][k] != trained[k]}
    if mismatch:
        raise ValueError(f"config differs from the trained model (yaml, checkpoint): {mismatch}")

    adj_mx = utils.load_graph_data(_abs(config["data"]["graph_pkl_filename"]))[2]
    model = build_model(adj_mx, logging.getLogger("dcrnn"), **config["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.speed_scaler = (float(ckpt["scaler"]["mean"]), float(ckpt["scaler"]["std"]))
    return model, config
