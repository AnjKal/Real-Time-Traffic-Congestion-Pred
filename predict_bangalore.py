
from pathlib import Path
import numpy as np
import torch
from pytorch_dcrnn import BangaloreDCRNN

data_dir = Path("data/BANGALORE")
checkpoint_path = data_dir / "pytorch_model" / "best_model.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

model = BangaloreDCRNN(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    horizon=checkpoint["horizon"],
    supports=checkpoint["supports"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_data = np.load(data_dir / "test.npz")
x = torch.tensor(test_data["x"][:1], dtype=torch.float32)

with torch.no_grad():
    predictions = model(x).numpy()[0, :, :, 0]

node_ids = checkpoint["node_ids"]

print("Predicted average speed (km/h)")
print("First test sample, next 12 days")

for node_index, node in enumerate(node_ids):
    values = np.round(predictions[:, node_index], 2)
    print(f"{node}: {values.tolist()}")
