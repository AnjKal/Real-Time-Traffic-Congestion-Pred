import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from traffic_gcn import TrafficGCN


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRAPH_PATH = os.path.join(
    BASE_DIR, "data", "sensor_graph", "bangalore_adj_mx.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "BANGALORE",
    "traffic_5mins", "bengaluru_traffic_5min_synthetic.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "traffic_gcn_best.pt"
)

DEVICE = torch.device("cpu")

with open(GRAPH_PATH, "rb") as f:
    graph = pickle.load(f)

nodes, node_mapping, adjacency = graph

adjacency = torch.tensor(adjacency, dtype=torch.float32)

import pandas as pd

df = pd.read_csv(DATA_PATH)
road_column = "Road/Intersection Name"

df[road_column] = df[road_column].astype(str)

road_speeds = (
    df.groupby(road_column)["Average Speed"]
    .mean()
    .reindex(nodes)
)

if road_speeds.isnull().any():
    raise ValueError("Missing traffic speed data for one or more roads")

speeds = road_speeds.to_numpy(dtype=np.float32)

mean_speed = speeds.mean()
std_speed = speeds.std()

if std_speed < 1e-8:
    std_speed = 1.0

normalized_speeds = (speeds - mean_speed) / std_speed

x = torch.tensor(
    normalized_speeds.reshape(-1, 1),
    dtype=torch.float32
)

model = TrafficGCN(
    input_features=1,
    hidden_features=16,
    output_features=1
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()

model.train()

for epoch in range(200):

    optimizer.zero_grad()

    predictions = model(x, adjacency)

    loss = loss_function(predictions, x)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch + 1}/200 | "
            f"Loss: {loss.item():.6f}"
        )

model.eval()

with torch.no_grad():
    predictions = model(x, adjacency).numpy().flatten()

predicted_speeds = predictions * std_speed + mean_speed

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "mean_speed": float(mean_speed),
        "std_speed": float(std_speed),
        "node_mapping": node_mapping
    },
    MODEL_PATH
)

print("\nGCN training completed")
print("Model saved:", MODEL_PATH)

for road, actual, predicted in zip(
    nodes, speeds, predicted_speeds
):
    print(
        f"{road}: Input={actual:.2f} km/h, "
        f"GCN Output={predicted:.2f} km/h"
    )