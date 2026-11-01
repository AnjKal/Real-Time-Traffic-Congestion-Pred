"""Compare DCRNN with matched training-mean and persistence baselines."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from pytorch_dcrnn import BangaloreDCRNN, FORECAST_MODE


def load_split(path):
    with np.load(path) as data:
        x = np.nan_to_num(
            data["x"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        y = data["y"].astype(np.float32)
        observed = data["mask"] > 0.5 if "mask" in data else np.isfinite(y)
        observed &= np.isfinite(y)
    return x, y, observed


def masked_metrics(prediction, target, valid):
    count = int(valid.sum())
    if count == 0:
        return count, float("nan"), float("nan")
    errors = prediction[valid].astype(np.float64) - target[valid].astype(np.float64)
    return count, float(np.abs(errors).mean()), float(np.sqrt(np.square(errors).mean()))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/BANGALORE"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/BANGALORE/pytorch_model/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/BANGALORE/pytorch_model")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("forecast_mode") != FORECAST_MODE:
        saved_mode = checkpoint.get("forecast_mode", "missing (legacy checkpoint)")
        raise SystemExit(
            f"Checkpoint forecast mode is {saved_mode!r}; expected {FORECAST_MODE!r}. "
            "Retrain with train_bangalore.py before evaluating the current model."
        )

    x_train, y_train, mask_train = load_split(args.data_dir / "train.npz")
    x_test, y_test, mask_test = load_split(args.data_dir / "test.npz")
    observed_train = mask_train
    if not observed_train.any():
        raise ValueError("Training split has no observed targets for the mean baseline.")
    training_mean = float(y_train[observed_train].mean())

    feature_names = (args.data_dir / "feature_names.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    node_ids = (args.data_dir / "node_ids.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    speed_index = feature_names.index("Average Speed")
    if speed_index != 0:
        raise ValueError(
            "The current persistence-residual model requires Average Speed "
            "to be input feature 0."
        )
    if len(node_ids) != x_test.shape[2]:
        raise ValueError("node_ids.txt does not match the test tensor's node dimension.")
    if checkpoint["input_dim"] != x_test.shape[-1]:
        raise ValueError("Checkpoint input dimension does not match the test data.")
    if checkpoint["horizon"] != y_test.shape[1]:
        raise ValueError("Checkpoint horizon does not match the test target horizon.")
    if checkpoint.get("node_ids") != node_ids:
        raise ValueError("Checkpoint node order does not match node_ids.txt.")

    model = BangaloreDCRNN(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        horizon=checkpoint["horizon"],
        supports=checkpoint["supports"].to(device),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(x_test), args.batch_size):
            batch = torch.from_numpy(x_test[start : start + args.batch_size]).to(device)
            predictions.append(model(batch).cpu().numpy())
    dcrnn = np.concatenate(predictions, axis=0)
    if dcrnn.shape != y_test.shape:
        raise ValueError(
            f"Model output shape {dcrnn.shape} does not match targets {y_test.shape}."
        )
    if not np.isfinite(dcrnn[mask_test]).all():
        raise ValueError("Model produced non-finite predictions for observed test targets.")

    persistence = np.repeat(
        x_test[:, -1, :, speed_index][:, None, :, None], y_test.shape[1], axis=1
    )
    training_mean_pred = np.full_like(y_test, training_mean)
    predictions_by_model = {
        "DCRNN": dcrnn,
        "Persistence": persistence,
        "Training mean": training_mean_pred,
    }

    horizon_rows = []
    road_rows = []
    for horizon in range(y_test.shape[1]):
        valid = mask_test[:, horizon, :, :]
        for model_name, prediction in predictions_by_model.items():
            count, mae, rmse = masked_metrics(
                prediction[:, horizon, :, :], y_test[:, horizon, :, :], valid
            )
            horizon_rows.append(
                {
                    "horizon_days": horizon + 1,
                    "model": model_name,
                    "targets": count,
                    "mae_kmh": f"{mae:.6f}",
                    "rmse_kmh": f"{rmse:.6f}",
                }
            )

        for node_index, node_id in enumerate(node_ids):
            road_valid = valid[:, node_index, :]
            for model_name, prediction in predictions_by_model.items():
                count, mae, rmse = masked_metrics(
                    prediction[:, horizon, node_index, :],
                    y_test[:, horizon, node_index, :],
                    road_valid,
                )
                road_rows.append(
                    {
                        "road": node_id,
                        "horizon_days": horizon + 1,
                        "model": model_name,
                        "targets": count,
                        "mae_kmh": f"{mae:.6f}",
                        "rmse_kmh": f"{rmse:.6f}",
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizon_path = args.output_dir / "baseline_comparison_by_horizon.csv"
    road_path = args.output_dir / "baseline_comparison_by_road.csv"
    write_csv(
        horizon_path,
        horizon_rows,
        ["horizon_days", "model", "targets", "mae_kmh", "rmse_kmh"],
    )
    write_csv(
        road_path,
        road_rows,
        ["road", "horizon_days", "model", "targets", "mae_kmh", "rmse_kmh"],
    )

    print(f"Checkpoint verified: forecast_mode={FORECAST_MODE}")
    print(f"Training-mean baseline: {training_mean:.3f} km/h")
    print("\nOverall metrics by horizon (all methods use the same observed targets)")
    print("Days | Method        | Targets | MAE km/h | RMSE km/h")
    for row in horizon_rows:
        print(
            f"{row['horizon_days']:4} | {row['model']:<13} | "
            f"{row['targets']:7} | {float(row['mae_kmh']):8.3f} | "
            f"{float(row['rmse_kmh']):9.3f}"
        )
    print(f"\nSaved horizon report: {horizon_path}")
    print(f"Saved road-by-horizon report: {road_path}")


if __name__ == "__main__":
    main()
