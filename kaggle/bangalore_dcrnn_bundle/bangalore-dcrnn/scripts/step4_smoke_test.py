"""Step 4 pre-flight: 2 epochs, 500 training samples, batch 4. Exit 0 iff loss is finite and falls."""
import sys

import numpy as np

from scripts.step4_train import load_config, seed_everything
from src.model.dcrnn.dcrnn_model import forward_batch
from src.model.dcrnn.dcrnn_supervisor import DCRNNSupervisor, load_checkpoint


def main() -> int:
    seed_everything()
    cfg = load_config()
    cfg["base_dir"] = "checkpoints/bangalore_dcrnn_smoke"     # keep the real run's dir clean
    cfg["train"]["epochs"] = 2
    cfg["data"].update(batch_size=4, val_batch_size=4, test_batch_size=4,
                       max_train_samples=500, max_eval_samples=100)

    sup = DCRNNSupervisor(**cfg)
    history = sup.train()
    l1, l2 = history[0]["train_mae"], history[-1]["train_mae"]
    print(f"\ntrain_mae epoch1={l1:.4f} epoch2={l2:.4f}")

    ok = len(history) == 2 and np.isfinite([l1, l2, history[-1]["val_mae"]]).all() and l2 < l1
    if not ok:
        print("SMOKE FAIL: loss not finite or did not decrease")
        return 1

    model, ckpt = load_checkpoint(sup.best_path)
    x, _ = next(iter(sup._data["val_loader"].get_iterator()))
    import torch

    out = forward_batch(model, torch.from_numpy(x).float())
    if not torch.isfinite(out).all() or tuple(out.shape[1:]) != (7, 20, 1):
        print(f"SMOKE FAIL: bad reloaded output {tuple(out.shape)}")
        return 1
    print(f"Checkpoint saved + reloaded OK (epoch {ckpt['epoch']}, val_mae {ckpt['val_mae']:.4f})")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
