
import numpy as np

data = np.load("data/BANGALORE/test.npz")
x = data["x"]
y = data["y"]
mask = data["mask"] > 0

features = [
    line.strip()
    for line in open("data/BANGALORE/feature_names.txt", encoding="utf-8")
]

speed_idx = features.index("Average Speed")
last_input_speed = x[:, -1, :, speed_idx]

print("Last input speed:")
print("Min:", np.nanmin(last_input_speed))
print("Max:", np.nanmax(last_input_speed))
print("Mean:", np.nanmean(last_input_speed))

print("\nObserved test targets:")
print("Min:", np.nanmin(y))
print("Max:", np.nanmax(y))
print("Mean:", np.nanmean(y))

pred = np.repeat(last_input_speed[:, None, :, None], y.shape[1], axis=1)
valid = mask & np.isfinite(y) & np.isfinite(pred)

print("\nBaseline evaluated on", int(valid.sum()), "targets")
print("Baseline MAE:", np.abs(pred[valid] - y[valid]).mean())
print("Baseline RMSE:", np.sqrt(np.square(pred[valid] - y[valid]).mean()))
