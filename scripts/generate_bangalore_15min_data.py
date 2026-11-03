
import os
import pickle
import numpy as np
import pandas as pd

RAW_FILE = "data/raw/tomtom_traffic_raw.csv"
SPOTS_FILE = "data/raw/bangalore_traffic_spots.csv"
GRAPH_FILE = "data/sensor_graph/adj_mx_bangalore.pkl"
OUTPUT_DIR = "data/BANGLORE_15MIN"

SEQ_LEN = 12
HORIZON = 12
INTERVAL = "15min"

print("Loading source data...")

traffic = pd.read_csv(RAW_FILE)
spots = pd.read_csv(SPOTS_FILE)

# Load Bangalore graph and preserve its exact node order.
with open(GRAPH_FILE, "rb") as f:
    sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(
        f, encoding="latin1"
    )

sensor_ids = list(sensor_ids)
num_nodes = len(sensor_ids)

assert num_nodes == 100, "Expected exactly 100 junctions."
assert len(sensor_id_to_ind) == 100
assert np.asarray(adj_mx).shape == (100, 100)

print("Graph verified:", num_nodes, "junctions")

# Verify that all graph junctions exist in the traffic dataset.
traffic_ids = set(traffic["junction_id"].astype(str))
graph_ids = set(str(x) for x in sensor_ids)

assert graph_ids.issubset(traffic_ids), (
    "Some graph junction IDs are missing from traffic data."
)

# Convert timestamps to Bangalore local time.
traffic["timestamp"] = pd.to_datetime(
    traffic["timestamp"], utc=True
).dt.tz_convert("Asia/Kolkata")

traffic["junction_id"] = traffic["junction_id"].astype(str)
traffic["currentSpeed"] = pd.to_numeric(
    traffic["currentSpeed"], errors="coerce"
)

# Keep only the 100 graph junctions.
traffic = traffic[
    traffic["junction_id"].isin(graph_ids)
].copy()

# Average readings into 15-minute intervals.
pivot = traffic.pivot_table(
    index="timestamp",
    columns="junction_id",
    values="currentSpeed",
    aggfunc="mean"
)

# Force the exact graph node order.
pivot = pivot.reindex(columns=[str(x) for x in sensor_ids])

pivot = pivot.sort_index().resample(INTERVAL).mean()

print("Aggregated time slots:", len(pivot))
print("Junction columns:", len(pivot.columns))
print("Missing observations before interpolation:",
      int(pivot.isna().sum().sum()))

# Fill only short internal gaps (maximum one 15-minute interval).
pivot = pivot.interpolate(
    method="time",
    limit=1,
    limit_area="inside"
)

print("Missing observations after interpolation:",
      int(pivot.isna().sum().sum()))

# Build features: normalized speed and normalized time of day.
speed = pivot.to_numpy(dtype=np.float32)

# Normalize speed to [0, 1] using a fixed 120 km/h scale.
speed = np.clip(speed, 0, 120) / 120.0

timestamps = pivot.index
minutes = (
    timestamps.hour * 60 + timestamps.minute
).to_numpy(dtype=np.float32)

time_feature = (minutes / 1440.0).reshape(-1, 1)
time_feature = np.repeat(time_feature, num_nodes, axis=1)

features = np.stack(
    [speed, time_feature], axis=-1
).astype(np.float32)

print("Feature matrix:", features.shape)

# Generate 12-step input and 12-step prediction windows.
X, Y = [], []

for i in range(len(features) - SEQ_LEN - HORIZON + 1):
    x_window = features[i:i + SEQ_LEN]
    y_window = features[
        i + SEQ_LEN:i + SEQ_LEN + HORIZON
    ]

    if (
        np.isfinite(x_window).all()
        and np.isfinite(y_window).all()
    ):
        X.append(x_window)
        Y.append(y_window)

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)

print("Valid sequences:", len(X))

if len(X) < 10:
    raise ValueError("Too few valid sequences. Check the source data.")

# Chronological 70/10/20 split.
n = len(X)
n_train = int(n * 0.70)
n_val = int(n * 0.10)

splits = {
    "train": (X[:n_train], Y[:n_train]),
    "val": (
        X[n_train:n_train + n_val],
        Y[n_train:n_train + n_val]
    ),
    "test": (
        X[n_train + n_val:],
        Y[n_train + n_val:]
    )
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

for name, (x_data, y_data) in splits.items():
    output_file = os.path.join(OUTPUT_DIR, name + ".npz")

    np.savez_compressed(
        output_file,
        x=x_data,
        y=y_data
    )

    print(
        name.upper(),
        "X:", x_data.shape,
        "Y:", y_data.shape
    )

print("\n15-MINUTE PREPROCESSING COMPLETED.")
print("All 100 junctions retained.")
print("Output:", OUTPUT_DIR)
