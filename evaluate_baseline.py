
import numpy as np

data = np.load("data/BANGALORE/test.npz")
x = data["x"]
y = data["y"]
mask = data["mask"] > 0

feature_names = [
    line.strip()
    for line in open("data/BANGALORE/feature_names.txt", encoding="utf-8")
]

speed_index = feature_names.index("Average Speed")

last_speed = x[:, -1, :, speed_index]
pred = np.repeat(last_speed[:, None, :, None], y.shape[1], axis=1)

valid = mask & np.isfinite(y) & np.isfinite(pred)

actual = y[valid]
predicted = pred[valid]

mae = np.mean(np.abs(predicted - actual))
rmse = np.sqrt(np.mean((predicted - actual) ** 2))

print("Persistence baseline")
print("Valid targets:", len(actual))
print(f"MAE: {mae:.3f} km/h")
print(f"RMSE: {rmse:.3f} km/h")
