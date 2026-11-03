
import os
import pickle
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

RAW_FILE = "data/raw/tomtom_traffic_raw.csv"
SPOTS_FILE = "data/raw/bangalore_traffic_spots.csv"
GRAPH_FILE = "data/sensor_graph/adj_mx_bangalore.pkl"

OUTPUT_DIR = "data/BANGLORE_REGENERATED"

FREQUENCY = "3min"
NUM_NODES = 100
MAX_INTERPOLATION_GAP = 2

INPUT_STEPS = 12
OUTPUT_STEPS = 12

TRAIN_RATIO = 0.70
TEST_RATIO = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    print("\n[1] Loading source datasets...")

    raw = pd.read_csv(RAW_FILE)
    spots = pd.read_csv(SPOTS_FILE)

    raw["timestamp"] = pd.to_datetime(
        raw["timestamp"],
        utc=True
    ).dt.tz_convert("Asia/Kolkata")

    raw["junction_id"] = raw["junction_id"].astype(str).str.strip()

    raw["currentSpeed"] = pd.to_numeric(
        raw["currentSpeed"],
        errors="coerce"
    )

    print("Raw traffic records:", len(raw))
    print("Junction records:", len(spots))

    return raw, spots


# ============================================================
# 3. VERIFY GRAPH AND JUNCTION ORDER
# ============================================================

def verify_graph(spots, raw):

    print("\n[2] Verifying graph and junction ordering...")

    with open(GRAPH_FILE, "rb") as f:
        graph_ids, sensor_id_to_ind, adj_mx = pickle.load(
            f, encoding="latin1"
        )

    graph_ids = [str(x) for x in graph_ids]

    expected_ids = [
        "J{:03d}".format(i)
        for i in range(1, NUM_NODES + 1)
    ]

    if graph_ids != expected_ids:
        raise ValueError(
            "Graph IDs do not match J001-J100 ordering."
        )

    if len(spots) != NUM_NODES:
        raise ValueError(
            "Expected exactly 100 junctions."
        )

    if not np.array_equal(
        np.asarray(adj_mx).shape,
        (NUM_NODES, NUM_NODES)
    ):
        raise ValueError(
            "Adjacency matrix must be 100 x 100."
        )

    if not np.isfinite(adj_mx).all():
        raise ValueError(
            "Adjacency matrix contains invalid values."
        )

    if spots["name"].duplicated().any():
        raise ValueError("Duplicate junction names found.")

    if raw["junction_id"].nunique() != NUM_NODES:
        raise ValueError(
            "Raw data does not contain exactly 100 junction IDs."
        )

    # Verify every ID-to-index mapping.
    for i, junction_id in enumerate(expected_ids):

        if sensor_id_to_ind.get(junction_id) != i:
            raise ValueError(
                "Incorrect graph mapping for {}".format(junction_id)
            )

        spot_name = str(spots.iloc[i]["name"]).strip()

        names = raw.loc[
            raw["junction_id"] == junction_id,
            "junction_name"
        ].dropna().astype(str).str.strip().unique()

        if len(names) != 1 or names[0] != spot_name:
            raise ValueError(
                "Junction mismatch: {}. Expected {}, found {}".format(
                    junction_id, spot_name, names
                )
            )

    print("Graph contains 100 nodes.")
    print("Junction IDs verified.")
    print("Junction ordering verified.")

    return expected_ids


# ============================================================
# 4. CREATE TRAFFIC SPEED MATRIX
# ============================================================

def create_speed_matrix(raw, expected_ids):

    print("\n[3] Creating three-minute traffic matrix...")

    start_time = raw["timestamp"].min().floor(FREQUENCY)
    end_time = raw["timestamp"].max().floor(FREQUENCY)

    time_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq=FREQUENCY
    )

    speed_matrix = raw.pivot_table(
        index="timestamp",
        columns="junction_id",
        values="currentSpeed",
        aggfunc="mean"
    )

    speed_matrix = speed_matrix.reindex(
        index=time_index,
        columns=expected_ids
    )

    # Count missing observations before interpolation.
    missing_before = int(
        speed_matrix.isna().sum().sum()
    )

    missing_timestamps = int(
        speed_matrix.isna().all(axis=1).sum()
    )

    print("Total time slots:", len(time_index))
    print("Missing observations:", missing_before)
    print("Completely missing timestamps:", missing_timestamps)

    return speed_matrix, time_index


# ============================================================
# 5. HANDLE MISSING OBSERVATIONS
# ============================================================

def handle_missing_values(speed_matrix):

    print("\n[4] Handling missing observations...")

    # Interpolate only gaps of up to two consecutive intervals.
    # Do not extrapolate at the beginning or end.
    speed_matrix = speed_matrix.interpolate(
        method="time",
        limit=MAX_INTERPOLATION_GAP,
        limit_area="inside"
    )

    remaining_missing = int(
        speed_matrix.isna().sum().sum()
    )

    print(
        "Remaining missing observations:",
        remaining_missing
    )

    return speed_matrix


# ============================================================
# 6. CREATE TWO INPUT FEATURES
# ============================================================

def create_features(speed_matrix, time_index):

    print("\n[5] Creating model features...")

    # Feature 1: Traffic speed.
    speed_data = speed_matrix.to_numpy(
        dtype=np.float32
    )

    # Feature 2: Time of day.
    seconds_in_day = (
        time_index.hour * 3600
        + time_index.minute * 60
        + time_index.second
    )

    time_of_day = (
        seconds_in_day.to_numpy(dtype=np.float32)
        / 86400.0
    )

    time_feature = np.tile(
        time_of_day[:, None, None],
        (1, NUM_NODES, 1)
    )

    # Shape: (time, nodes, features)
    data = np.concatenate(
        [
            speed_data[:, :, None],
            time_feature
        ],
        axis=-1
    ).astype(np.float32)

    print("Feature matrix shape:", data.shape)

    return data


# ============================================================
# 7. GENERATE INPUT AND OUTPUT SEQUENCES
# ============================================================

def generate_sequences(data):

    print("\n[6] Generating DCRNN sequences...")

    x_offsets = np.arange(
        -(INPUT_STEPS - 1), 1
    )

    y_offsets = np.arange(
        1, OUTPUT_STEPS + 1
    )

    x_list = []
    y_list = []

    for t in range(
        INPUT_STEPS - 1,
        len(data) - OUTPUT_STEPS
    ):

        x_t = data[t + x_offsets]
        y_t = data[t + y_offsets]

        x_list.append(x_t)
        y_list.append(y_t)

    x = np.stack(x_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)

    print("X shape before filtering:", x.shape)
    print("Y shape before filtering:", y.shape)

    return x, y, x_offsets, y_offsets


# ============================================================
# 8. FILTER INVALID SEQUENCES
# ============================================================

def filter_invalid_sequences(x, y):

    print("\n[7] Filtering sequences with unresolved gaps...")

    valid_mask = (
        np.isfinite(x).all(axis=(1, 2, 3))
        &
        np.isfinite(y).all(axis=(1, 2, 3))
    )

    invalid_count = int(
        np.sum(~valid_mask)
    )

    x = x[valid_mask]
    y = y[valid_mask]

    print("Invalid sequences removed:", invalid_count)
    print("Valid sequences remaining:", len(x))

    if len(x) == 0:
        raise ValueError(
            "No valid sequences remain after filtering."
        )

    return x, y


# ============================================================
# 9. CHRONOLOGICAL DATA SPLIT
# ============================================================

def split_dataset(x, y):

    print("\n[8] Splitting dataset...")

    num_samples = len(x)

    num_train = round(
        num_samples * TRAIN_RATIO
    )

    num_test = round(
        num_samples * TEST_RATIO
    )

    num_val = (
        num_samples - num_train - num_test
    )

    splits = {
        "train": (
            x[:num_train],
            y[:num_train]
        ),

        "val": (
            x[num_train:num_train + num_val],
            y[num_train:num_train + num_val]
        ),

        "test": (
            x[num_train + num_val:],
            y[num_train + num_val:]
        )
    }

    print("Training samples:", num_train)
    print("Validation samples:", num_val)
    print("Testing samples:", num_test)

    return splits


# ============================================================
# 10. SAVE NPZ FILES
# ============================================================

def save_dataset(splits, x_offsets, y_offsets):

    print("\n[9] Saving NPZ files...")

    for split, (x, y) in splits.items():

        output_path = os.path.join(
            OUTPUT_DIR,
            "{}.npz".format(split)
        )

        np.savez_compressed(
            output_path,
            x=x,
            y=y,
            x_offsets=x_offsets.reshape(-1, 1),
            y_offsets=y_offsets.reshape(-1, 1)
        )

        print("\n", split.upper())
        print("X:", x.shape)
        print("Y:", y.shape)
        print("Saved:", output_path)


# ============================================================
# 11. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("BANGALORE DCRNN DATA PREPROCESSING")
    print("=" * 60)

    raw, spots = load_data()

    expected_ids = verify_graph(
        spots,
        raw
    )

    speed_matrix, time_index = create_speed_matrix(
        raw,
        expected_ids
    )

    speed_matrix = handle_missing_values(
        speed_matrix
    )

    data = create_features(
        speed_matrix,
        time_index
    )

    x, y, x_offsets, y_offsets = generate_sequences(
        data
    )

    x, y = filter_invalid_sequences(
        x,
        y
    )

    splits = split_dataset(
        x,
        y
    )

    save_dataset(
        splits,
        x_offsets,
        y_offsets
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("Output directory:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
