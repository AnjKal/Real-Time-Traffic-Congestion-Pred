import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class TrafficGRU(nn.Module):
    def __init__(self, input_size, hidden_size=96, layers=2, dropout=0.15, horizon=12):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=layers,
                          batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/BANGALORE/traffic_5mins/bengaluru_traffic_5min_synthetic.csv")
    parser.add_argument("--model", default="models/traffic_gru_best.pt")
    parser.add_argument("--road", default=None)
    parser.add_argument("--output-dir", default="predictions")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        raise FileNotFoundError(f"Dataset not found: {Path(args.csv).resolve()}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Checkpoint not found: {Path(args.model).resolve()}")

    try:
        ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.model, map_location="cpu")

    df = pd.read_csv(args.csv, parse_dates=["Timestamp"])
    roads = ckpt["roads"]

    if args.road is None:
        print("Available roads:")
        for road in roads:
            print(" -", road)
        print("\nRun again with --road \"Exact Road Name\"")
        return
    if args.road not in roads:
        raise ValueError("Unknown road. Available roads: " + ", ".join(roads))

    part = df[df["Road/Intersection Name"].astype(str) == args.road].copy()
    part = part.sort_values("Timestamp").tail(int(ckpt["seq_len"])).reset_index(drop=True)
    if len(part) < int(ckpt["seq_len"]):
        raise ValueError("Not enough observations for this road.")

    gaps = part["Timestamp"].diff().dropna()
    if not (gaps == pd.Timedelta(minutes=5)).all():
        raise ValueError("Latest observations are not exactly 5 minutes apart.")

    numeric_features = ckpt["numeric_features"]
    numeric = part[numeric_features].astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    weather = pd.get_dummies(part["Weather Conditions"].fillna("Unknown"))
    weather = weather.reindex(columns=ckpt["weather_categories"], fill_value=0)

    road_features = np.zeros((len(part), len(roads)), dtype=np.float32)
    road_features[:, roads.index(args.road)] = 1.0

    features = np.concatenate([
        numeric.to_numpy(dtype=np.float32),
        weather.to_numpy(dtype=np.float32),
        road_features
    ], axis=1)

    if features.shape[1] != int(ckpt["input_size"]):
        raise ValueError("Feature count does not match checkpoint. Use the matching training dataset.")

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    x_std[x_std < 1e-6] = 1.0
    features = (features - x_mean) / x_std
    x = torch.tensor(features[None], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficGRU(
        input_size=int(ckpt["input_size"]),
        hidden_size=int(ckpt["hidden_size"]),
        layers=int(ckpt["layers"]),
        dropout=float(ckpt["dropout"]),
        horizon=int(ckpt["horizon"])
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        scaled = model(x.to(device)).cpu().numpy()[0]

    speeds = np.clip(scaled * float(ckpt["y_std"]) + float(ckpt["y_mean"]), 0, 150)
    last_time = part["Timestamp"].iloc[-1]
    horizons = np.arange(1, len(speeds) + 1) * 5
    result = pd.DataFrame({
        "Road/Intersection Name": args.road,
        "Forecast Timestamp": [last_time + pd.Timedelta(minutes=int(m)) for m in horizons],
        "Horizon_Minutes": horizons,
        "Predicted_Average_Speed_kmh": np.round(speeds, 2),
        "DataType": ckpt.get("data_type", "UNKNOWN")
    })

    print(f"\nRoad: {args.road}")
    print(f"Last observed timestamp: {last_time}")
    print(result[["Horizon_Minutes", "Forecast Timestamp", "Predicted_Average_Speed_kmh"]].to_string(index=False))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", args.road).strip("_")
    out_path = out_dir / f"forecast_{safe_name}.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    if ckpt.get("data_type") == "SYNTHETIC":
        print("\nWARNING: These are prototype forecasts from synthetic data, not validated real-world traffic predictions.")


if __name__ == "__main__":
    main()
