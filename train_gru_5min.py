import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

FEATURE_NUMERIC = [
    "Traffic Volume", "Travel Time Index", "Congestion Level",
    "Road Capacity Utilization", "Incident Reports", "Environmental Impact",
    "Public Transport Usage", "Traffic Signal Compliance", "Parking Usage",
    "Pedestrian and Cyclist Count", "Hour", "Minute", "DayOfWeek",
    "IsWeekend", "IsHoliday", "Rainfall_mm", "Temperature_C",
    "Visibility_km", "IncidentSeverity"
]
TARGET = "Average Speed"
SEQ_LEN = 12
HORIZON = 12

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class WindowDataset(Dataset):
    def __init__(self, groups, x_mean, x_std, y_mean, y_std, start_frac, end_frac):
        self.samples = []
        self.groups = groups
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        for g in groups:
            n = len(g["x"])
            start = int(n * start_frac)
            end = int(n * end_frac)
            first = max(SEQ_LEN, start)
            last = min(n - HORIZON + 1, end)
            for target_start in range(first, last):
                self.samples.append((g, target_start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        g, t = self.samples[idx]
        x = (g["x"][t-SEQ_LEN:t] - self.x_mean) / self.x_std
        y = (g["y"][t:t+HORIZON] - self.y_mean) / self.y_std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class TrafficGRU(nn.Module):
    def __init__(self, input_size, hidden_size=96, layers=2, dropout=0.15, horizon=HORIZON):
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

def build_groups(df):
    weather = sorted(df["Weather Conditions"].fillna("Unknown").unique())
    roads = sorted(df["Road/Intersection Name"].astype(str).unique())
    groups = []
    feature_names = FEATURE_NUMERIC + [f"weather={w}" for w in weather] + [f"road={r}" for r in roads]
    for road in roads:
        part = df[df["Road/Intersection Name"].astype(str) == road].sort_values("Timestamp").copy()
        numeric = part[FEATURE_NUMERIC].astype(float).replace([np.inf, -np.inf], np.nan)
        numeric = numeric.ffill().bfill().fillna(0)
        w = pd.get_dummies(part["Weather Conditions"].fillna("Unknown")).reindex(columns=weather, fill_value=0)
        r = np.zeros((len(part), len(roads)), dtype=np.float32)
        r[:, roads.index(road)] = 1.0
        x = np.concatenate([numeric.to_numpy(dtype=np.float32), w.to_numpy(dtype=np.float32), r], axis=1)
        y = part[TARGET].to_numpy(dtype=np.float32)
        groups.append({"road": road, "x": x, "y": y})
    return groups, feature_names

def fit_scalers(groups, train_end):
    xs, ys = [], []
    for g in groups:
        cut = int(len(g["x"]) * train_end)
        xs.append(g["x"][:cut])
        ys.append(g["y"][:cut])
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    x_mean = x_all.mean(axis=0)
    x_std = x_all.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = float(y_all.mean())
    y_std = float(y_all.std())
    if y_std < 1e-6:
        y_std = 1.0
    return x_mean.astype(np.float32), x_std.astype(np.float32), y_mean, y_std

def metrics(pred, actual):
    err = pred - actual
    return {"MAE_kmh": float(np.mean(np.abs(err))),
            "RMSE_kmh": float(np.sqrt(np.mean(err ** 2)))}

def evaluate(model, groups, x_mean, x_std, y_mean, y_std, device):
    model.eval()
    results = []
    with torch.no_grad():
        for g in groups:
            n = len(g["x"])
            test_start = int(n * 0.85)
            for t in range(max(SEQ_LEN, test_start), n - HORIZON + 1, HORIZON):
                xin = (g["x"][t-SEQ_LEN:t] - x_mean) / x_std
                xb = torch.tensor(xin[None], dtype=torch.float32, device=device)
                pred = model(xb).cpu().numpy()[0] * y_std + y_mean
                actual = g["y"][t:t+HORIZON]
                persistence = np.repeat(g["y"][t-1], HORIZON)
                results.append((pred, actual, persistence, g["road"]))
    pred = np.concatenate([r[0] for r in results])
    actual = np.concatenate([r[1] for r in results])
    persistence = np.concatenate([r[2] for r in results])
    return {
        "GRU": metrics(pred, actual),
        "Persistence": metrics(persistence, actual)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/bengaluru_traffic_5min_synthetic.csv")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.csv, parse_dates=["Timestamp"])
    df = df.sort_values(["Road/Intersection Name", "Timestamp"]).reset_index(drop=True)
    groups, feature_names = build_groups(df)

    x_mean, x_std, y_mean, y_std = fit_scalers(groups, 0.70)
    train_ds = WindowDataset(groups, x_mean, x_std, y_mean, y_std, 0.0, 0.70)
    val_ds = WindowDataset(groups, x_mean, x_std, y_mean, y_std, 0.70, 0.85)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TrafficGRU(len(feature_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_val = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())
        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch:02d} | train={np.mean(train_losses):.5f} | val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    report = evaluate(model, groups, x_mean, x_std, y_mean, y_std, device)
    os.makedirs("models", exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "input_size": len(feature_names),
        "hidden_size": 96,
        "layers": 2,
        "dropout": 0.15,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "feature_names": feature_names,
        "numeric_features": FEATURE_NUMERIC,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "roads": sorted(df["Road/Intersection Name"].astype(str).unique()),
        "weather_categories": sorted(df["Weather Conditions"].fillna("Unknown").unique()),
        "data_type": "SYNTHETIC"
    }
    torch.save(checkpoint, "models/traffic_gru_best.pt")
    with open("models/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump({"device": str(device), "best_validation_loss": best_val,
                   "test_metrics": report, "note": "Evaluation is on synthetic data only."}, f, indent=2)
    print(json.dumps(report, indent=2))
    print("Saved models/traffic_gru_best.pt and models/evaluation_report.json")

if __name__ == "__main__":
    main()
