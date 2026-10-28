"""Shared settings for the Bangalore -> Kafka -> DCRNN -> dashboard demo.

All relative paths are resolved from the DCRNN repo root, so always run the
scripts from there (e.g. `python kafka_demo/producer.py`).
"""
import os

# ---- Kafka -----------------------------------------------------------------
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_RAW = "traffic.raw"              # one message = one Bangalore row (16 sensors)
TOPIC_PRED = "traffic.predictions"     # one message = one 12-step forecast for all sensors
GROUP_ID = "dcrnn-inference"

# ---- Model / data shape (must match the pretrained config.yaml) --------------
NUM_NODES = 16
INPUT_DIM = 19
SEQ_LEN = 12
HORIZON = 12
STEP_MIN = 24 * 60    # Bangalore dataset is sampled daily

# ---- Files (relative to the DCRNN repo root) --------------------------------
H5_PATH = "bangalore_prep/data/BANGALORE/bangalore_features.h5"
DATASET_DIR = "bangalore_prep/data/BANGALORE"
FEATURE_STATS = "bangalore_prep/data/BANGALORE/feature_stats.json"
SENSOR_IDS_TXT = "bangalore_prep/data/BANGALORE/graph_sensor_ids.txt"
SENSOR_LOC_CSV = "bangalore_prep/data/BANGALORE/nodes.csv"
EDGE_GEOMETRY_FILE = "bangalore_prep/data/BANGALORE/edge_geometry.geojson"
ROAD_ROUTER_URL = os.environ.get("ROAD_ROUTER_URL", "https://router.project-osrm.org")
MODEL_CONFIG = "bangalore_prep/data/BANGALORE/dcrnn_bangalore_multi.yaml"

# ---- Congestion definition ---------------------------------------------------
# free-flow speed per sensor = 85th percentile of its non-zero training speeds
FREEFLOW_PERCENTILE = 85
FREEFLOW_FLOOR_MPH = 1.0
TRAIN_FRACTION = 0.7   # same split as DCRNN's generate_training_data.py

# ratio = speed / free_flow_speed
#   >= 0.75 free flow (green), >= 0.50 moderate (yellow), below 0.50 congested (red)
LEVEL_THRESHOLDS = (0.75, 0.50)
LEVEL_NAMES = ["Free flow", "Moderate", "Congested"]

# ---- Dashboard ---------------------------------------------------------------
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5000
# CARTO basemap key (free, from https://carto.com/basemaps/apikey/). Prefer the environment variable
# CARTO_API_KEY so the key never gets committed; this constant is only a fallback.
CARTO_API_KEY = "cb1_49fz_1_fad9aea9ffeb19dcbb89d3a2"


def load_sensor_ids():
    with open(SENSOR_IDS_TXT) as f:
        return [s.strip() for s in f.read().strip().split(",")]