
import os
import pickle
import numpy as np

nodes_path = "data/BANGALORE/node_ids.txt"
output_path = "data/sensor_graph/bangalore_adj_mx.pkl"

with open(nodes_path, encoding="utf-8") as file:
    nodes = [line.strip() for line in file if line.strip()]

adjacency = np.eye(len(nodes), dtype=np.float32)
node_to_index = {node: i for i, node in enumerate(nodes)}

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "wb") as file:
    pickle.dump(
        [nodes, node_to_index, adjacency],
        file,
        protocol=2,
    )

print("Nodes:", len(nodes))
print("Adjacency shape:", adjacency.shape)
print("Saved:", output_path)
print("WARNING: This is a temporary identity graph, not the real road network.")
