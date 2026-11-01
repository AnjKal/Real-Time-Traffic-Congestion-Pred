"""Paired-seed DCRNN vs temporal-GRU ablation on Bangalore data.

Each learned model is selected using validation SmoothL1 only. Test data is
loaded and scored only after every run has completed model selection.
"""

import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_dcrnn import BangaloreDCRNN, FORECAST_MODE
from train_bangalore import make_correlation_supports


class TemporalGRU(nn.Module):
    """Shared per-road GRU without graph message passing."""

    def __init__(self, input_dim, hidden_dim, horizon):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.temporal = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x):
        batch, steps, nodes, features = x.shape
        per_road = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, features)
        encoded, _ = self.temporal(per_road)
        residual = self.output(encoded[:, -1]).reshape(batch, nodes, self.horizon)
        residual = residual.permute(0, 2, 1).unsqueeze(-1)
        last_speed = x[:, -1, :, 0].unsqueeze(1).unsqueeze(-1)
        return last_speed + residual


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_val(path):
    with np.load(path) as data:
        x = np.nan_to_num(
            data["x"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        y = data["y"].astype(np.float32)
        mask = data["mask"] > 0.5 if "mask" in data else np.isfinite(y)
        mask &= np.isfinite(y)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(mask.astype(np.float32)),
    )


def evaluate_validation(model, loader, device):
    model.eval()
    loss_sum = 0.0
    count_sum = 0
    with torch.inference_mode():
        for x, y, mask in loader:
            x, y = x.to(device), y.to(device)
            valid = (mask.to(device) > 0.5) & torch.isfinite(y)
            prediction = model(x)
            valid &= torch.isfinite(prediction)
            count = int(valid.sum().item())
            if count:
                loss_sum += nn.functional.smooth_l1_loss(
                    prediction[valid], y[valid], reduction="sum"
                ).item()
                count_sum += count
    if not count_sum:
        raise ValueError("Validation split contains no observed targets.")
    return loss_sum / count_sum


def state_to_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_one(architecture, seed, train_data, val_data, supports, args, device):
    seed_everything(seed)
    x_train, y_train, mask_train = train_data
    x_val, y_val, mask_val = val_data
    train_loader = DataLoader(
        TensorDataset(x_train, y_train, mask_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val, mask_val),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if architecture == "DCRNN":
        model = BangaloreDCRNN(
            input_dim=x_train.shape[-1],
            hidden_dim=args.hidden_dim,
            horizon=y_train.shape[1],
            supports=supports.to(device),
        ).to(device)
    elif architecture == "GRU":
        model = TemporalGRU(
            input_dim=x_train.shape[-1],
            hidden_dim=args.hidden_dim,
            horizon=y_train.shape[1],
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    best_val_loss = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, mask in train_loader:
            x, y = x.to(device), y.to(device)
            valid = (mask.to(device) > 0.5) & torch.isfinite(y)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            valid &= torch.isfinite(prediction)
            if not valid.any():
                continue
            loss = nn.functional.smooth_l1_loss(prediction[valid], y[valid])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_loss = evaluate_validation(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = state_to_cpu(model)
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError(f"{architecture} seed {seed} produced no checkpoint.")
    return best_state, best_val_loss, epoch


def predictions_from_state(architecture, state, x, supports, args, device):
    if architecture == "DCRNN":
        model = BangaloreDCRNN(
            input_dim=x.shape[-1],
            hidden_dim=args.hidden_dim,
            horizon=args.horizon,
            supports=supports.to(device),
        )
    else:
        model = TemporalGRU(
            input_dim=x.shape[-1],
            hidden_dim=args.hidden_dim,
            horizon=args.horizon,
        )
    model.load_state_dict(state)
    model.to(device).eval()
    x_tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x_tensor), args.batch_size):
            outputs.append(
                model(x_tensor[start : start + args.batch_size].to(device)).cpu().numpy()
            )
    return np.concatenate(outputs, axis=0)


def metrics(prediction, target, valid):
    count = int(valid.sum())
    if count == 0:
        return count, float("nan"), float("nan")
    errors = prediction[valid].astype(np.float64) - target[valid].astype(np.float64)
    return count, float(np.abs(errors).mean()), float(np.sqrt(np.square(errors).mean()))


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/BANGALORE"))
    parser.add_argument(
        "--current-checkpoint",
        type=Path,
        default=Path("data/BANGALORE/pytorch_model/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/BANGALORE/pytorch_model/ablation"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = load_train_val(args.data_dir / "train.npz")
    val_data = load_train_val(args.data_dir / "val.npz")
    x_train, y_train, mask_train = train_data
    feature_names = (args.data_dir / "feature_names.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    node_ids = (args.data_dir / "node_ids.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if not feature_names or feature_names[0] != "Average Speed":
        raise ValueError("Both architectures require Average Speed at input feature 0.")

    supports, _ = make_correlation_supports(x_train)
    args.horizon = y_train.shape[1]
    selected_runs = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Complete every validation-based selection before loading the test split.
    for seed in args.seeds:
        for architecture in ("DCRNN", "GRU"):
            state, best_val_loss, epochs_run = train_one(
                architecture, seed, train_data, val_data, supports, args, device
            )
            checkpoint_path = args.output_dir / f"{architecture.lower()}_seed_{seed}.pt"
            torch.save(
                {
                    "architecture": architecture,
                    "forecast_mode": FORECAST_MODE,
                    "seed": seed,
                    "best_validation_smooth_l1": best_val_loss,
                    "epochs_run": epochs_run,
                    "input_dim": x_train.shape[-1],
                    "hidden_dim": args.hidden_dim,
                    "horizon": args.horizon,
                    "supports": supports.cpu() if architecture == "DCRNN" else None,
                    "model_state_dict": state,
                    "feature_names": feature_names,
                    "node_ids": node_ids,
                },
                checkpoint_path,
            )
            selected_runs.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "state": state,
                    "val_loss": best_val_loss,
                    "epochs_run": epochs_run,
                }
            )
            print(
                f"Selected {architecture} seed={seed}: "
                f"val_smooth_l1={best_val_loss:.4f}, epochs={epochs_run}"
            )

    # The existing current checkpoint is included as the single-run reference.
    current_checkpoint = torch.load(
        args.current_checkpoint, map_location="cpu", weights_only=False
    )
    if current_checkpoint.get("forecast_mode") != FORECAST_MODE:
        raise ValueError(
            "Current checkpoint is stale or missing forecast_mode; retrain it "
            "with the corrected train_bangalore.py first."
        )
    if current_checkpoint.get("node_ids") != node_ids:
        raise ValueError("Current checkpoint node order does not match the data.")

    # First and only access to test data: all learned models are already selected.
    with np.load(args.data_dir / "test.npz") as test_data:
        x_test = np.nan_to_num(
            test_data["x"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        y_test = test_data["y"].astype(np.float32)
        test_mask = test_data["mask"] > 0.5 if "mask" in test_data else np.isfinite(y_test)
        test_mask &= np.isfinite(y_test)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

    test_mask_t = torch.from_numpy(test_mask.astype(np.float32))
    test_predictions = []
    for run in selected_runs:
        run["prediction"] = predictions_from_state(
            run["architecture"], run["state"], x_test, supports, args, device
        )
        test_predictions.append(run)

    checkpoint_model = BangaloreDCRNN(
        input_dim=current_checkpoint["input_dim"],
        hidden_dim=current_checkpoint["hidden_dim"],
        horizon=current_checkpoint["horizon"],
        supports=current_checkpoint["supports"],
    )
    checkpoint_model.load_state_dict(current_checkpoint["model_state_dict"])
    checkpoint_model.to(device).eval()
    current_outputs = []
    with torch.inference_mode():
        for start in range(0, len(x_test), args.batch_size):
            batch = torch.from_numpy(x_test[start : start + args.batch_size]).to(device)
            current_outputs.append(checkpoint_model(batch).cpu().numpy())
    current_prediction = np.concatenate(current_outputs, axis=0)
    test_predictions.append(
        {
            "architecture": "DCRNN current checkpoint",
            "seed": "reference",
            "prediction": current_prediction,
            "val_loss": current_checkpoint.get("best_validation_loss", ""),
            "epochs_run": "",
        }
    )

    train_mean = float(y_train[mask_train.numpy().astype(bool)].mean())
    speed_index = feature_names.index("Average Speed")
    persistence = np.repeat(
        x_test[:, -1, :, speed_index][:, None, :, None],
        y_test.shape[1],
        axis=1,
    )
    fixed_predictions = {
        "Training mean": np.full_like(y_test, train_mean),
        "Persistence": persistence,
    }

    overall_rows = []
    horizon_rows = []
    road_horizon_rows = []
    all_predictions = [(run["architecture"], run["seed"], run["prediction"], run["val_loss"], run["epochs_run"]) for run in test_predictions]
    all_predictions.extend(
        (name, "baseline", prediction, "", "")
        for name, prediction in fixed_predictions.items()
    )

    for architecture, seed, prediction, val_loss, epochs_run in all_predictions:
        count, mae, rmse = metrics(prediction, y_test, test_mask)
        overall_rows.append(
            {
                "model": architecture,
                "seed": seed,
                "best_validation_smooth_l1": val_loss,
                "epochs_run": epochs_run,
                "test_targets": count,
                "test_mae_kmh": f"{mae:.6f}",
                "test_rmse_kmh": f"{rmse:.6f}",
            }
        )
        for horizon in range(y_test.shape[1]):
            valid_h = test_mask[:, horizon, :, :]
            count_h, mae_h, rmse_h = metrics(
                prediction[:, horizon, :, :], y_test[:, horizon, :, :], valid_h
            )
            horizon_rows.append(
                {
                    "model": architecture,
                    "seed": seed,
                    "horizon_days": horizon + 1,
                    "targets": count_h,
                    "mae_kmh": f"{mae_h:.6f}",
                    "rmse_kmh": f"{rmse_h:.6f}",
                }
            )
            for node_index, node_id in enumerate(node_ids):
                valid_road = valid_h[:, node_index, :]
                count_road, mae_road, rmse_road = metrics(
                    prediction[:, horizon, node_index, :],
                    y_test[:, horizon, node_index, :],
                    valid_road,
                )
                road_horizon_rows.append(
                    {
                        "model": architecture,
                        "seed": seed,
                        "road": node_id,
                        "horizon_days": horizon + 1,
                        "targets": count_road,
                        "mae_kmh": f"{mae_road:.6f}",
                        "rmse_kmh": f"{rmse_road:.6f}",
                    }
                )

    seed_summary_rows = []
    for architecture in ("DCRNN", "GRU"):
        rows = [r for r in overall_rows if r["model"] == architecture]
        maes = np.asarray([float(r["test_mae_kmh"]) for r in rows])
        rmses = np.asarray([float(r["test_rmse_kmh"]) for r in rows])
        seed_summary_rows.append(
            {
                "model": architecture,
                "runs": len(rows),
                "mean_seed_mae_kmh": f"{maes.mean():.6f}",
                "std_seed_mae_kmh": f"{maes.std(ddof=1) if len(maes) > 1 else 0.0:.6f}",
                "mean_seed_rmse_kmh": f"{rmses.mean():.6f}",
                "std_seed_rmse_kmh": f"{rmses.std(ddof=1) if len(rmses) > 1 else 0.0:.6f}",
            }
        )

    # Paired seed-wise delta summaries identify where graph structure helps.
    paired_delta_rows = []
    for horizon in range(y_test.shape[1]):
        for node_index, node_id in enumerate(node_ids):
            deltas = []
            valid = test_mask[:, horizon, node_index, :]
            for seed in args.seeds:
                dcrnn_run = next(
                    r for r in selected_runs if r["architecture"] == "DCRNN" and r["seed"] == seed
                )
                gru_run = next(
                    r for r in selected_runs if r["architecture"] == "GRU" and r["seed"] == seed
                )
                d_mae = metrics(
                    dcrnn_run["prediction"][:, horizon, node_index, :],
                    y_test[:, horizon, node_index, :],
                    valid,
                )[1]
                g_mae = metrics(
                    gru_run["prediction"][:, horizon, node_index, :],
                    y_test[:, horizon, node_index, :],
                    valid,
                )[1]
                deltas.append(d_mae - g_mae)
            paired_delta_rows.append(
                {
                    "road": node_id,
                    "horizon_days": horizon + 1,
                    "runs": len(deltas),
                    "mean_dcrnn_minus_gru_mae_kmh": f"{np.mean(deltas):.6f}",
                    "std_paired_delta_kmh": f"{np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0:.6f}",
                }
            )

    write_csv(
        args.output_dir / "test_overall.csv",
        overall_rows,
        ["model", "seed", "best_validation_smooth_l1", "epochs_run", "test_targets", "test_mae_kmh", "test_rmse_kmh"],
    )
    write_csv(
        args.output_dir / "test_by_horizon.csv",
        horizon_rows,
        ["model", "seed", "horizon_days", "targets", "mae_kmh", "rmse_kmh"],
    )
    write_csv(
        args.output_dir / "test_by_road_horizon.csv",
        road_horizon_rows,
        ["model", "seed", "road", "horizon_days", "targets", "mae_kmh", "rmse_kmh"],
    )
    write_csv(
        args.output_dir / "seed_summary.csv",
        seed_summary_rows,
        ["model", "runs", "mean_seed_mae_kmh", "std_seed_mae_kmh", "mean_seed_rmse_kmh", "std_seed_rmse_kmh"],
    )
    write_csv(
        args.output_dir / "paired_dcrnn_minus_gru_by_road_horizon.csv",
        paired_delta_rows,
        ["road", "horizon_days", "runs", "mean_dcrnn_minus_gru_mae_kmh", "std_paired_delta_kmh"],
    )

    print(f"Device: {device}; seeds: {args.seeds}")
    print("Final test comparison (selection used validation set only)")
    for row in overall_rows:
        print(
            f"{row['model']:<26} seed={str(row['seed']):<9} "
            f"MAE={float(row['test_mae_kmh']):.3f} "
            f"RMSE={float(row['test_rmse_kmh']):.3f}"
        )
    print("\nMulti-seed mean ± sample standard deviation")
    for row in seed_summary_rows:
        print(
            f"{row['model']:<8} MAE={float(row['mean_seed_mae_kmh']):.3f} "
            f"± {float(row['std_seed_mae_kmh']):.3f}; "
            f"RMSE={float(row['mean_seed_rmse_kmh']):.3f} "
            f"± {float(row['std_seed_rmse_kmh']):.3f}"
        )
    mean_delta = np.mean(
        [float(r["mean_dcrnn_minus_gru_mae_kmh"]) for r in paired_delta_rows]
    )
    print(
        "\nRoad/horizon paired delta: mean DCRNN MAE - GRU MAE = "
        f"{mean_delta:+.3f} km/h (negative favors DCRNN)."
    )
    print(f"Reports and selected run checkpoints: {args.output_dir}")


if __name__ == "__main__":
    main()
