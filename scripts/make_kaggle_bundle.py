"""Pack the minimum needed to train Step 4 on Kaggle/Colab into kaggle/bangalore_dcrnn_bundle.zip."""
import zipfile

from src.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "kaggle" / "bangalore_dcrnn_bundle.zip"
FILES = (["src/__init__.py", "src/config.py", "src/model/__init__.py",
          "scripts/step4_train.py", "scripts/step4_smoke_test.py",
          "data/graph/adj_mx.pkl"]
         + [f"data/processed/{n}" for n in ("train.npz", "val.npz", "test.npz", "scaler.pkl")]
         + [f"src/model/dcrnn/{p.name}" for p in (PROJECT_ROOT / "src/model/dcrnn").iterdir()
            if p.suffix in (".py", ".yaml")])


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in FILES:
            z.write(PROJECT_ROOT / rel, f"bangalore-dcrnn/{rel}")
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.0f} MB, {len(FILES)} files)")


if __name__ == "__main__":
    main()
