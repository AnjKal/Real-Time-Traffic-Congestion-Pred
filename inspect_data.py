
import numpy as np

for split in ["train", "val", "test"]:
    data = np.load(f"data/BANGALORE/{split}.npz")
    print(f"\n{split.upper()}")

    for key in data.files:
        arr = data[key]
        print(key, arr.shape, arr.dtype)

        if np.issubdtype(arr.dtype, np.number):
            print("  NaN values:", np.isnan(arr).sum())
