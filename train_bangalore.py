import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_dcrnn import BangaloreDCRNN, FORECAST_MODE


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(path):
    with np.load(path) as data:
        x = np.nan_to_num(data["x"], nan=0.0, posinf=0.0, neginf=0.0)
        y = data["y"].astype(np.float32)
        mask = data["mask"].astype(np.float32) if "mask" in data else np.isfinite(y).astype(np.float32)
        mask *= np.isfinite(y).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
    )


def make_correlation_supports(train_x, top_k=4):
    speed = train_x[:, :, :, 0].numpy()
    series = speed.transpose(2, 0, 1).reshape(speed.shape[2], -1)
    corr = np.corrcoef(series)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    weights = np.abs(corr)
    adjacency = np.zeros_like(weights, dtype=np.float32)
    for i in range(weights.shape[0]):
        neighbours = np.argsort(weights[i])[-top_k:]
        adjacency[i, neighbours] = weights[i, neighbours]
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    degree = adjacency.sum(axis=1)
    inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    forward = inv_degree[:, None] * adjacency
    reverse_adj = adjacency.T
    reverse_degree = reverse_adj.sum(axis=1)
    reverse_inv = np.divide(1.0, reverse_degree, out=np.zeros_like(reverse_degree), where=reverse_degree > 0)
    reverse = reverse_inv[:, None] * reverse_adj
    return torch.tensor(np.stack([forward, reverse]), dtype=torch.float32), torch.tensor(adjacency)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y, target_mask in loader:
            x = x.to(device)
            y = y.to(device)
            target_mask = target_mask.to(device) > 0.5

            pred = model(x)
            mask = target_mask & torch.isfinite(y) & torch.isfinite(pred)

            if not mask.any():
                continue

            valid_pred = pred[mask]
            valid_y = y[mask]

            loss = criterion(valid_pred, valid_y)
            total_loss += loss.item() * valid_y.numel()
            total_abs += torch.abs(valid_pred - valid_y).sum().item()
            total_sq += torch.square(valid_pred - valid_y).sum().item()
            total_count += valid_y.numel()

    if total_count == 0:
        raise RuntimeError("No valid targets found during evaluation.")

    avg_loss = total_loss / total_count
    mae = total_abs / total_count
    rmse = (total_sq / total_count) ** 0.5

    return avg_loss, mae, rmse




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/BANGALORE")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = data_dir / "pytorch_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, mask_train = load_split(data_dir / "train.npz")
    x_val, y_val, mask_val = load_split(data_dir / "val.npz")
    x_test, y_test, mask_test = load_split(data_dir / "test.npz")

    supports, adjacency = make_correlation_supports(x_train)
    np.save(output_dir / "correlation_adjacency.npy", adjacency.numpy())

    train_loader = DataLoader(TensorDataset(x_train, y_train, mask_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val, mask_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test, mask_test), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BangaloreDCRNN(
        input_dim=x_train.shape[-1],
        hidden_dim=args.hidden_dim,
        horizon=y_train.shape[1],
        supports=supports.to(device)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()
    best_val = float("inf")
    best_state = None
    patience_left = args.patience

    print(f"Device: {device}")
    print(f"Train: x={tuple(x_train.shape)}, y={tuple(y_train.shape)}")
    print(f"Validation: x={tuple(x_val.shape)}, y={tuple(y_val.shape)}")
    print(f"Test: x={tuple(x_test.shape)}, y={tuple(y_test.shape)}")
    print(f"Observed target fraction: train={mask_train.mean().item():.3f}, val={mask_val.mean().item():.3f}, test={mask_test.mean().item():.3f}")
    print("Loss and metrics ignore targets that were not observed.")
    print("Graph: top-k absolute correlations from training-period speed series; not verified road connectivity.")
    print("Target: Average Speed in the dataset's original units (expected km/h).")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y, target_mask in train_loader:
            x, y = x.to(device), y.to(device)
            target_mask = target_mask.to(device) > 0.5
            optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            mask = target_mask & torch.isfinite(y) & torch.isfinite(pred)

            if not mask.any():
                continue

            loss = criterion(pred[mask], y[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * y[mask].numel()
            seen += y[mask].numel()

        if seen == 0:
            raise RuntimeError("Training split contains no observed targets.")

        train_loss = running_loss / max(seen, 1)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_MAE={val_mae:.3f} | val_RMSE={val_rmse:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, criterion, device)
    checkpoint = {
        "forecast_mode": FORECAST_MODE,
        "model_state_dict": model.state_dict(),
        "input_dim": x_train.shape[-1],
        "hidden_dim": args.hidden_dim,
        "horizon": y_train.shape[1],
        "supports": supports.cpu(),
        "feature_names": (data_dir / "feature_names.txt").read_text(encoding="utf-8").splitlines(),
        "node_ids": (data_dir / "node_ids.txt").read_text(encoding="utf-8").splitlines(),
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_observed_targets": int(mask_test.sum().item()),
        "graph_type": "top-k absolute training-period speed correlation; not road topology"
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    print(f"Best validation loss: {best_val:.4f}")
    print(f"TEST | loss={test_loss:.4f} | MAE={test_mae:.3f} | RMSE={test_rmse:.3f}")
    print(f"Saved checkpoint: {output_dir / 'best_model.pt'}")
    print(f"Saved correlation adjacency: {output_dir / 'correlation_adjacency.npy'}")


if __name__ == "__main__":
    main()