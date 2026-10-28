# Bangalore Traffic Pulse -> DCRNN (road-network graph + multi-feature inputs, speed target)

Run from your DCRNN repo root. Copy this folder in, put the Kaggle CSV at data/bangalore_traffic.csv.

## Environments
Steps 1-3 need a modern Python (3.9+) env, NOT the TF1 one:
    pip install pandas numpy tables networkx pyyaml requests matplotlib osmnx scikit-learn
Step 4 and training run in your TF1 'dcrnn' env.

## Order
    python bangalore_prep/selftest.py                                   # optional: dry run on fake data
    python bangalore_prep/01_inspect_data.py --csv bangalore_prep/data/bangalore_traffic.csv --out bangalore_prep/data/BANGALORE
    python bangalore_prep/02_build_graph.py  --csv bangalore_prep/data/bangalore_traffic.csv --out bangalore_prep/data/BANGALORE
    python bangalore_prep/03_build_dataset.py --csv bangalore_prep/data/bangalore_traffic.csv --graph-dir bangalore_prep/data/BANGALORE
    # in the TF1 env:
    python bangalore_prep/npy_to_pkl.py --graph-dir bangalore_prep/data/BANGALORE
    python dcrnn_train.py --config_filename=bangalore_prep/data/BANGALORE/dcrnn_bangalore_speed.yaml   # baseline
    python dcrnn_train.py --config_filename=bangalore_prep/data/BANGALORE/dcrnn_bangalore_multi.yaml   # all features

Step 2 may stop and ask you to fix coordinates in data/BANGALORE/nodes.csv. That is intended.
Check data/BANGALORE/graph_check.png and edges.csv before trusting the graph.

## Ablations that come for free
- speed-only vs multi-feature inputs (two configs above)
- road graph only vs similarity blended in:  02_build_graph.py --sim-mode reweight --beta 0.3
- input gap handling: 03_build_dataset.py --fill ffill --ffill-limit 3
- drop suspicious columns: 03_build_dataset.py --exclude congestion_level travel_time_index

The all-feature configuration uses 19 input channels: speed, the numeric and binary
features discovered in `bangalore_features.h5`, an observed-speed flag, and two
calendar channels. The model keeps `output_dim: 1`, so it forecasts speed while
using every channel as an input.
