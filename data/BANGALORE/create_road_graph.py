import os
import pickle
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))

node_file = os.path.join(base_dir, "node_ids.txt")
output_file = os.path.join(
    base_dir,
    "..",
    "sensor_graph",
    "bangalore_adj_mx.pkl"
)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(node_file, "r", encoding="utf-8") as f:
    nodes = [line.strip() for line in f if line.strip()]

node_to_index = {node: i for i, node in enumerate(nodes)}

adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)

edges = [
    ("Trinity Circle", "Anil Kumble Circle", 0.9),
    ("Trinity Circle", "CMH Road", 0.6),
    ("Anil Kumble Circle", "CMH Road", 0.5),

    ("CMH Road", "100 Feet Road", 0.8),
    ("100 Feet Road", "Sony World Junction", 0.7),
    ("Sony World Junction", "Jayanagar 4th Block", 0.5),

    ("Jayanagar 4th Block", "South End Circle", 0.8),
    ("South End Circle", "Silk Board Junction", 0.5),

    ("Silk Board Junction", "Hosur Road", 0.9),
    ("Silk Board Junction", "Sarjapur Road", 0.8),
    ("Sarjapur Road", "Marathahalli Bridge", 0.7),
    ("Marathahalli Bridge", "ITPL Main Road", 0.7),

    ("Hebbal Flyover", "Ballari Road", 0.9),
    ("Hebbal Flyover", "Yeshwanthpur Circle", 0.5),
    ("Yeshwanthpur Circle", "Tumkur Road", 0.8)
]

for road_a, road_b, weight in edges:
    if road_a not in node_to_index or road_b not in node_to_index:
        raise ValueError(f"Unknown road: {road_a}, {road_b}")

    i = node_to_index[road_a]
    j = node_to_index[road_b]

    adjacency[i, j] = weight
    adjacency[j, i] = weight

for i in range(len(nodes)):
    adjacency[i, i] = 1.0

graph_data = [
    nodes,
    node_to_index,
    adjacency
]

if os.path.exists(output_file):
    backup_file = output_file + ".backup"
    with open(output_file, "rb") as f:
        old_data = pickle.load(f)

    with open(backup_file, "wb") as f:
        pickle.dump(old_data, f)

with open(output_file, "wb") as f:
    pickle.dump(graph_data, f)

print("Road graph created successfully")
print("Number of nodes:", len(nodes))
print("Number of edges:", len(edges))
print("Adjacency matrix shape:", adjacency.shape)
print("Non-zero connections:", np.count_nonzero(adjacency))
print("Saved to:", output_file)