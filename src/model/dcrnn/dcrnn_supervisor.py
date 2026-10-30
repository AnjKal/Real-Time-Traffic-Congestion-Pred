"""Training loop wrapper, adapted from chnsh/DCRNN_PyTorch's DCRNNSupervisor.

MOD summary vs upstream: TensorBoard/TF removed; outputs go straight to base_dir
(best_model.pt, training_log.csv, config_used.yaml, info.log); parameters are materialised
before the optimizer/scheduler are built (upstream rebuilt the optimizer after step 0, which
left the LR scheduler attached to a dead optimizer); evaluation drops loader padding and
reports per-horizon MAE; min_learning_rate is honoured.
"""
import copy
import csv
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.config import PROJECT_ROOT
from src.model.dcrnn import utils
from src.model.dcrnn.dcrnn_model import DCRNNModel
from src.model.dcrnn.loss import masked_mae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _abs(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_model(adj_mx, logger, **model_kwargs) -> DCRNNModel:
    """DCGRUCell registers its weights lazily on first forward, so run one dummy pass."""
    model = DCRNNModel(adj_mx, logger, **model_kwargs).to(device)
    n, d, s = (int(model_kwargs["num_nodes"]), int(model_kwargs["input_dim"]),
               int(model_kwargs["seq_len"]))
    with torch.no_grad():
        model.eval()(torch.zeros(s, 1, n * d, device=device))
    return model


def load_checkpoint(path, logger=None):
    """Rebuild a model from a best_model.pt written by the supervisor."""
    import logging

    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    adj_mx = utils.load_graph_data(_abs(cfg["data"]["graph_pkl_filename"]))[2]
    model = build_model(adj_mx, logger or logging.getLogger("dcrnn"), **cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), ckpt


class DCRNNSupervisor:
    def __init__(self, **kwargs):
        self._kwargs = copy.deepcopy(kwargs)
        self._data_kwargs = kwargs["data"]
        self._model_kwargs = kwargs["model"]
        self._train_kwargs = kwargs["train"]

        self._log_dir = _abs(kwargs["base_dir"])
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._logger = utils.get_logger(str(self._log_dir), __name__, "info.log",
                                        level=kwargs.get("log_level", "INFO"))
        with open(self._log_dir / "config_used.yaml", "w") as f:
            yaml.safe_dump(self._kwargs, f, sort_keys=False)

        dataset_dir = _abs(self._data_kwargs["dataset_dir"])
        self._data = utils.load_dataset(**{**self._data_kwargs, "dataset_dir": dataset_dir})
        self.standard_scaler = self._data["scaler"]
        scaler_pkl = dataset_dir / "scaler.pkl"
        self._free_flow = None
        if scaler_pkl.exists():
            with open(scaler_pkl, "rb") as f:
                self._free_flow = np.asarray(pickle.load(f)["free_flow_kmh"], dtype=np.float32)

        self.num_nodes = int(self._model_kwargs["num_nodes"])
        self.input_dim = int(self._model_kwargs["input_dim"])
        self.seq_len = int(self._model_kwargs["seq_len"])
        self.output_dim = int(self._model_kwargs["output_dim"])
        self.horizon = int(self._model_kwargs["horizon"])

        adj_mx = utils.load_graph_data(_abs(self._data_kwargs["graph_pkl_filename"]))[2]
        self.dcrnn_model = build_model(adj_mx, self._logger, **self._model_kwargs)
        self.history = []
        self.best_val_loss = float("inf")
        self.best_path = self._log_dir / "best_model.pt"
        self._logger.info("Model created on %s", device)

    # ------------------------------------------------------------------ data helpers
    def _prepare_data(self, x, y):
        """x (B, T, N, D_in), y (B, H, N, D) numpy -> time-major flat tensors on device."""
        x = torch.from_numpy(x).float().permute(1, 0, 2, 3)
        y = torch.from_numpy(y).float().permute(1, 0, 2, 3)
        b = x.size(1)
        x = x.reshape(self.seq_len, b, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].reshape(self.horizon, b, self.num_nodes * self.output_dim)
        return x.to(device), y.to(device)

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, dataset="val"):
        """Returns (mean MAE in free-flow-normalised units, preds, truths); arrays are (H, n, N)."""
        loader = self._data[f"{dataset}_loader"]
        self.dcrnn_model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x, y in loader.get_iterator():
                x, y = self._prepare_data(x, y)
                out = self.dcrnn_model(x)
                preds.append(out.cpu())
                truths.append(y.cpu())
        preds = torch.cat(preds, dim=1)[:, :loader.n_real]      # drop padded duplicates
        truths = torch.cat(truths, dim=1)[:, :loader.n_real]
        loss = self._compute_loss(truths, preds).item()
        h, n = preds.shape[0], preds.shape[1]
        shape = (h, n, self.num_nodes, self.output_dim)
        inv = self.standard_scaler.inverse_transform
        return loss, inv(preds).reshape(shape).numpy(), inv(truths).reshape(shape).numpy()

    def horizon_report(self, dataset="val"):
        """Per-horizon MAE: normalised units and km/h (needs scaler.pkl for km/h)."""
        _, preds, truths = self.evaluate(dataset)
        err = np.abs(preds - truths)[..., 0]                     # (H, n, N)
        rows = []
        for h in range(err.shape[0]):
            kmh = float((err[h] * self._free_flow).mean()) if self._free_flow is not None else None
            rows.append({"horizon_idx": h + 1, "minutes": 3 * (h + 1),
                         "mae_norm": float(err[h].mean()), "mae_kmh": kmh})
        return rows

    # ------------------------------------------------------------------ training
    def _save_best(self, epoch, val_loss):
        torch.save({"config": self._kwargs, "model_state_dict": self.dcrnn_model.state_dict(),
                    "epoch": epoch, "val_mae": val_loss,
                    "scaler": {"mean": self.standard_scaler.mean, "std": self.standard_scaler.std}},
                   self.best_path)

    def train(self):
        t = self._train_kwargs
        model = self.dcrnn_model
        optimizer = torch.optim.Adam(model.parameters(), lr=t["base_lr"], eps=t["epsilon"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=t["steps"], gamma=t["lr_decay_ratio"])
        min_lr = t.get("min_learning_rate", 0.0)
        train_loader = self._data["train_loader"]
        self._logger.info("Start training: %d batches/epoch", train_loader.num_batch)

        log_file = open(self._log_dir / "training_log.csv", "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_mae", "val_mae", "lr"])
        batches_seen, wait = 0, 0
        for epoch in range(t["epochs"]):
            start = time.time()
            model.train()
            lr = optimizer.param_groups[0]["lr"]
            losses = []
            for x, y in train_loader.get_iterator():
                optimizer.zero_grad()
                x, y = self._prepare_data(x, y)
                output = model(x, y, batches_seen)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())
                batches_seen += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
                optimizer.step()
            scheduler.step()
            for g in optimizer.param_groups:
                g["lr"] = max(g["lr"], min_lr)

            val_loss, _, _ = self.evaluate("val")
            train_loss = float(np.mean(losses))
            self.history.append({"epoch": epoch + 1, "train_mae": train_loss,
                                 "val_mae": val_loss, "lr": lr})
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.3e}"])
            log_file.flush()
            line = (f"epoch {epoch + 1:02d} | train_mae {train_loss:.3f} | val_mae {val_loss:.3f}"
                    f" | lr {lr:.1e} | elapsed {time.time() - start:.0f}s")
            print(line, flush=True)
            self._logger.debug(line)

            if not np.isfinite(train_loss):
                self._logger.error("non-finite training loss, stopping")
                break
            if val_loss < self.best_val_loss:
                self.best_val_loss, wait = val_loss, 0
                self._save_best(epoch + 1, val_loss)
            else:
                wait += 1
                if wait >= t["patience"]:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        log_file.close()

        if self.best_path.exists():       # report on the best weights, not the last
            model.load_state_dict(torch.load(self.best_path, map_location=device)
                                  ["model_state_dict"])
        return self.history
