
import numpy as np

base = "data/BANGALORE"

train = np.load(f"{base}/train.npz")
val = np.load(f"{base}/val.npz")
test = np.load(f"{base}/test.npz")

def describe(name, data):
    y = data["y"]
    mask = (data["mask"] > 0) & np.isfinite(y)
    values = y[mask]

    print(f"\n{name}")
    print("Valid targets:", values.size)
    print("Mean:", round(float(values.mean()), 3))
    print("Std:", round(float(values.std()), 3))
    print("Min:", round(float(values.min()), 3))
    print("25th percentile:", round(float(np.percentile(values, 25)), 3))
    print("Median:", round(float(np.median(values)), 3))
    print("75th percentile:", round(float(np.percentile(values, 75)), 3))
    print("Max:", round(float(values.max()), 3))

describe("TRAIN", train)
describe("VALIDATION", val)
describe("TEST", test)

train_y = train["y"]
train_mask = (train["mask"] > 0) & np.isfinite(train_y)
train_mean = float(train_y[train_mask].mean())

test_y = test["y"]
test_mask = (test["mask"] > 0) & np.isfinite(test_y)
valid_y = test_y[test_mask]

mean_pred = np.full_like(valid_y, train_mean)

print("\nTraining-mean baseline")
print("Training mean:", round(train_mean, 3))
print("Test MAE:", round(float(np.mean(np.abs(mean_pred - valid_y))), 3))
print("Test RMSE:", round(float(np.sqrt(np.mean((mean_pred - valid_y) ** 2))), 3))
