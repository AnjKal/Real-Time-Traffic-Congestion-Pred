import argparse
import os

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "Traffic Volume",
    "Average Speed",
    "Travel Time Index",
    "Congestion Level",
    "Road Capacity Utilization",
    "Incident Reports",
    "Public Transport Usage",
    "Traffic Signal Compliance",
    "Parking Usage",
    "Pedestrian and Cyclist Count",
]

WEATHER_CATEGORIES = ["Clear", "Overcast", "Fog", "Rain", "Windy"]


def prepare_data(csv_path, output_dir, seq_len=12, horizon=12):
    df = pd.read_csv(csv_path)

    required = [
        "Date",
        "Road/Intersection Name",
        "Weather Conditions",
        "Roadwork and Construction Activity",
    ] + NUMERIC_FEATURES

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Road/Intersection Name"]).copy()
    df["Date"] = df["Date"].dt.normalize()
    df["Road/Intersection Name"] = (
        df["Road/Intersection Name"].astype(str).str.strip()
    )

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    nodes = sorted(df["Road/Intersection Name"].unique().tolist())
    dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")

    df = df.groupby(
        ["Date", "Road/Intersection Name"], as_index=False
    ).agg({
        **{col: "mean" for col in NUMERIC_FEATURES},
        "Weather Conditions": (
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        ),
        "Roadwork and Construction Activity": (
            lambda s: "Yes"
            if s.astype(str).str.strip().str.lower().eq("yes").any()
            else "No"
        ),
    })

    index = pd.MultiIndex.from_product(
        [dates, nodes],
        names=["Date", "Road/Intersection Name"],
    )

    grid = (
        df.set_index(["Date", "Road/Intersection Name"])
        .reindex(index)
        .reset_index()
    )

    row_keys = pd.MultiIndex.from_frame(df[["Date", "Road/Intersection Name"]])
    grid_keys = pd.MultiIndex.from_frame(grid[["Date", "Road/Intersection Name"]])
    grid["observed_record"] = grid_keys.isin(row_keys).astype(np.float32)

    raw_speed = pd.to_numeric(grid["Average Speed"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    target_mask = np.isfinite(raw_speed).astype(np.float32)
    target_array = raw_speed.reshape(len(dates), len(nodes), 1)
    target_mask = target_mask.reshape(len(dates), len(nodes), 1)

    train_end = int(len(dates) * 0.70)
    val_end = int(len(dates) * 0.80)
    train_dates = set(dates[:train_end])

    training_source = df[df["Date"].isin(train_dates)]
    numeric_medians = {}
    for col in NUMERIC_FEATURES:
        median = training_source[col].median()
        numeric_medians[col] = float(median) if pd.notna(median) else 0.0

    # Forward-fill only: do not use future observations to fill earlier inputs.
    for node in nodes:
        mask = grid["Road/Intersection Name"] == node

        for col in NUMERIC_FEATURES:
            grid.loc[mask, col] = (
                pd.to_numeric(grid.loc[mask, col], errors="coerce").ffill()
            )

        grid.loc[mask, "Weather Conditions"] = (
            grid.loc[mask, "Weather Conditions"].ffill()
        )
        grid.loc[mask, "Roadwork and Construction Activity"] = (
            grid.loc[mask, "Roadwork and Construction Activity"].ffill()
        )

    for col in NUMERIC_FEATURES:
        grid[col] = grid[col].fillna(numeric_medians[col])

    grid["Weather Conditions"] = (
        grid["Weather Conditions"].fillna("Clear").astype(str).str.strip()
    )
    grid.loc[
        ~grid["Weather Conditions"].isin(WEATHER_CATEGORIES),
        "Weather Conditions",
    ] = "Clear"

    grid["Roadwork and Construction Activity"] = (
        grid["Roadwork and Construction Activity"]
        .fillna("No")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .astype(np.float32)
    )

    weather = pd.get_dummies(grid["Weather Conditions"], prefix="weather")
    for category in WEATHER_CATEGORIES:
        col = f"weather_{category}"
        if col not in weather:
            weather[col] = 0.0
    weather = weather[
        [f"weather_{category}" for category in WEATHER_CATEGORIES]
    ].astype(np.float32)

    day_of_week = grid["Date"].dt.dayofweek.to_numpy()
    grid["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    grid["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    grid = pd.concat(
        [grid.reset_index(drop=True), weather.reset_index(drop=True)],
        axis=1,
    )

    feature_columns = [
        "Average Speed",
        "Traffic Volume",
        "Travel Time Index",
        "Congestion Level",
        "Road Capacity Utilization",
        "Incident Reports",
            "Public Transport Usage",
        "Traffic Signal Compliance",
        "Parking Usage",
        "Pedestrian and Cyclist Count",
    ] + [
        f"weather_{category}" for category in WEATHER_CATEGORIES
    ] + [
        "Roadwork and Construction Activity",
        "dow_sin",
        "dow_cos",
        "observed_record",
    ]

    feature_array = grid[feature_columns].to_numpy(
        dtype=np.float32
    ).reshape(len(dates), len(nodes), len(feature_columns))

    # Fit scaling only on the training period. Index 0 (speed) is left unscaled
    # here because the existing model pipeline expects speed in original units.
    numeric_indices = list(range(1, 10))
    training_features = feature_array[:train_end, :, numeric_indices]
    means = training_features.mean(axis=(0, 1))
    stds = training_features.std(axis=(0, 1))
    stds[stds < 1e-6] = 1.0
    feature_array[:, :, numeric_indices] = (
        feature_array[:, :, numeric_indices] - means
    ) / stds

    if not np.isfinite(feature_array).all():
        raise ValueError("Feature array contains non-finite values after filling.")

    os.makedirs(output_dir, exist_ok=True)

    def create_windows(start, end):
        xs, ys, masks = [], [], []
        for t in range(start + seq_len, end - horizon + 1):
            xs.append(feature_array[t - seq_len:t])
            ys.append(target_array[t:t + horizon])
            masks.append(target_mask[t:t + horizon])

        if not xs:
            raise ValueError(
                f"Not enough dates in split {start}:{end} "
                f"for sequence={seq_len}, horizon={horizon}"
            )
        return np.stack(xs), np.stack(ys), np.stack(masks)

    splits = [
        ("train", 0, train_end),
        ("val", train_end, val_end),
        ("test", val_end, len(dates)),
    ]

    for name, start, end in splits:
        x, y, mask = create_windows(start, end)
        np.savez_compressed(
            os.path.join(output_dir, f"{name}.npz"),
            x=x,
            y=y,
            mask=mask,
        )
        print(
            f"{name}: x={x.shape}, y={y.shape}, "
            f"observed_targets={int(mask.sum())}/{mask.size} "
            f"({100 * mask.mean():.1f}%)"
        )

    (pd.Series(feature_columns).to_csv(
        os.path.join(output_dir, "feature_names.txt"),
        index=False,
        header=False,
    ))
    pd.Series(nodes).to_csv(
        os.path.join(output_dir, "node_ids.txt"),
        index=False,
        header=False,
    )
    np.savez_compressed(
        os.path.join(output_dir, "scaler.npz"),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
        numeric_indices=np.asarray(numeric_indices, dtype=np.int64),
    )

    print(f"Dates: {dates.min().date()} to {dates.max().date()}")
    print(f"Nodes: {len(nodes)}")
    print("Targets remain NaN where Average Speed was not observed.")
    print("Inputs are forward-filled only; no backward interpolation is used.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="../Banglore_traffic_Dataset.csv",
        help="Path to the original traffic CSV",
    )
    parser.add_argument("--output-dir", default="data/BANGALORE")
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    prepare_data(args.csv, args.output_dir, args.seq_len, args.horizon)


if __name__ == "__main__":
    main()
