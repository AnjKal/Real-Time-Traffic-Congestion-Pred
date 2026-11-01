
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_dcrnn import BangaloreDCRNN

torch.manual_seed(42)

data_dir = "data/BANGALORE"
model_dir = f"{data_dir}/pytorch_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test = np.load(f"{data_dir}/test.npz")
x = test["x"]
y = test["y"]
mask = test["mask"] > 0

checkpoint = torch.load(
    f"{model_dir}/best_model.pt",
    map_location=device,
    weights_only=False
)

supports = checkpoint["supports"].to(device)

model = BangaloreDCRNN(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    horizon=checkpoint["horizon"],
    supports=supports
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

loader = DataLoader(
    TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(mask)),
    batch_size=32,
    shuffle=False
)

predictions = []

with torch.no_grad():
    for batch_x, _, _ in loader:
        predictions.append(model(batch_x.to(device)).cpu().numpy())

pred = np.concatenate(predictions, axis=0)
valid = mask & np.isfinite(y) & np.isfinite(pred)

feature_names = [
    line.strip()
    for line in open(f"{data_dir}/feature_names.txt", encoding="utf-8")
]
speed_index = feature_names.index("Average Speed")

baseline_speed = x[:, -1, :, speed_index]
baseline = np.repeat(
    baseline_speed[:, None, :, None],
    y.shape[1],
    axis=1
)

print("\nHorizon-wise test performance")
print("Horizon | Valid | DCRNN MAE | Baseline MAE | DCRNN RMSE | Baseline RMSE")

for h in range(y.shape[1]):
    valid_h = valid[:, h, :, :]
    actual = y[:, h, :, :][valid_h]
    dcrnn_pred = pred[:, h, :, :][valid_h]
    baseline_pred = baseline[:, h, :, :][valid_h]

    if actual.size == 0:
        print(f"{h + 1:7d} | No valid targets")
        continue

    dcrnn_mae = np.mean(np.abs(dcrnn_pred - actual))
    baseline_mae = np.mean(np.abs(baseline_pred - actual))
    dcrnn_rmse = np.sqrt(np.mean((dcrnn_pred - actual) ** 2))
    baseline_rmse = np.sqrt(np.mean((baseline_pred - actual) ** 2))

    print(
        f"{h + 1:7d} | {actual.size:5d} | "
        f"{dcrnn_mae:10.3f} | {baseline_mae:12.3f} | "
        f"{dcrnn_rmse:10.3f} | {baseline_rmse:13.3f}"
    )
