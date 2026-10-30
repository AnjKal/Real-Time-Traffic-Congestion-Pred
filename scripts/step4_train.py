"""Step 4: train DCRNN on the synthetic data. Evaluates on VAL only (test is for Step 5)."""
import random
import sys

import numpy as np
import torch
import yaml

from src.config import PROJECT_ROOT, RANDOM_SEED
from src.model.dcrnn.dcrnn_supervisor import DCRNNSupervisor, device

CONFIG_PATH = PROJECT_ROOT / "src" / "model" / "dcrnn" / "config_bangalore.yaml"


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path=CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def print_horizon_table(rows) -> None:
    print("\nHorizon index | approx minutes | val MAE (normalized) | val MAE (km/h)")
    for r in rows:
        note = {2: "  (~5 min horizon)", 4: "  (~10 min horizon)", 7: "  (~20 min horizon)"}
        kmh = f"{r['mae_kmh']:.2f}" if r["mae_kmh"] is not None else "n/a"
        print(f"{r['horizon_idx']:<13} | {r['minutes']:>3} min        | "
              f"{r['mae_norm']:<20.3f} | {kmh}{note.get(r['horizon_idx'], '')}")


def main() -> int:
    seed_everything()
    print(f"Device: {device}")
    sup = DCRNNSupervisor(**load_config())
    history = sup.train()

    best = min(history, key=lambda h: h["val_mae"])
    print(f"\nBest val MAE (normalized): {best['val_mae']:.4f} at epoch {best['epoch']}")
    gap = best["val_mae"] - best["train_mae"]
    print(f"Train/val at best epoch: train_mae {best['train_mae']:.4f}, val_mae "
          f"{best['val_mae']:.4f}, gap {gap:+.4f}")
    if best["val_mae"] > 1.5 * best["train_mae"]:
        print("WARNING: val MAE is >1.5x train MAE - likely overfitting (see README of step 4)")
    print_horizon_table(sup.horizon_report("val"))
    if best["val_mae"] > 0.3:
        print("WARNING: val MAE > 0.3 - something is wrong, debug before continuing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
