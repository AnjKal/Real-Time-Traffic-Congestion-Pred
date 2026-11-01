
import numpy as np

for split in ["train", "val", "test"]:
    data = np.load(f"data/BANGALORE/{split}.npz")
    y = data["y"]
    mask = data["mask"]

    valid = np.isfinite(y)
    mask_valid = mask > 0

    print(f"\n{split.upper()}")
    print("Total targets:", y.size)
    print("Finite targets:", valid.sum())
    print("Missing targets:", (~valid).sum())
    print("Mask valid targets:", mask_valid.sum())
    print("Mask and target disagree:", np.logical_xor(valid, mask_valid).sum())
    print("Mask contains NaN:", np.isnan(mask).sum())
