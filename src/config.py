from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = DATA_DIR / "graph"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")
TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

# DCRNN hyperparameters (frozen for Phase 1)
NUM_NODES = 20
POLL_CADENCE_MIN = 3
SEQ_LEN = 12          # 36 min input history
HORIZON = 7           # 21 min output; read off idx 1 (~5min), 3 (~10min), 6 (~20min)
INPUT_DIM = 2         # speed + time-of-day
OUTPUT_DIM = 1        # speed only

# Bounding box for the OSM download and node sanity checks
BBOX = {
    "north": 13.05,
    "south": 12.90,
    "west":  77.53,   # was 77.56, widen to include Yeshwanthpur
    "east":  77.76,
}
# Gaussian-kernel adjacency
ADJ_EPSILON = 0.1    # sparsity threshold

# Synthetic data config
SYNTHETIC_DAYS = 90
RANDOM_SEED = 42
