
import numpy as np
import torch
from pytorch_dcrnn import BangaloreDCRNN

data_dir = "data/BANGALORE"
model_dir = f"{data_dir}/pytorch_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(f"{data_dir}/test.npz")
x = data["x"]
y = data["y"]
mask = data["mask"] > 0

checkpoint = torch.load(
    f"{model_dir}/best_model.pt",
    map_location=device,
    weights_only=False
)

model = BangaloreDCRNN(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    horizon=checkpoint["horizon"],
    supports=checkpoint["supports"].to(device)
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    pred = model(torch.tensor(x, dtype=torch.float32, device=device))
    pred = pred.cpu().numpy()

valid = mask & np.isfinite(y) & np.isfinite(pred)
actual = y[valid]
predicted = pred[valid]

print("Observed target mean:", round(float(actual.mean()), 3))
print("Prediction mean:", round(float(predicted.mean()), 3))
print("Prediction standard deviation:", round(float(predicted.std()), 3))
print("Actual standard deviation:", round(float(actual.std()), 3))
print("Mean prediction error:", round(float((predicted - actual).mean()), 3))
print("Prediction minimum:", round(float(predicted.min()), 3))
print("Prediction maximum:", round(float(predicted.max()), 3))
