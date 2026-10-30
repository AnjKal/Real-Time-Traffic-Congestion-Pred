"""Step 3 orchestrator: generate -> validate -> DCRNN tensors, then print a summary."""
import sys

from src.config import PROCESSED_DIR, SYNTHETIC_DIR
from src.synthetic import generator, to_dcrnn_tensors, validate
from src.synthetic.scenarios import SCENARIOS


def dir_mb(path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    print("== Generating ==")
    generator.run()
    print("\n== Validating ==")
    report, ok = validate.run()

    g1 = report["gate1_scenario_proportions"]
    print("\nScenario proportions (actual vs target):")
    for s in SCENARIOS:
        print(f"  {s.name:<26} {g1['actual'][s.id]:6.3f}  vs  {s.target_prop:5.3f}")

    if not ok:
        print("\nSTEP 3 FAIL: validation gates failed; tensors not written")
        return 1

    print("\n== Writing DCRNN tensors ==")
    counts = to_dcrnn_tensors.run()
    print("\nWindows: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    print(f"Disk: data/synthetic {dir_mb(SYNTHETIC_DIR):.1f} MB, "
          f"data/processed {dir_mb(PROCESSED_DIR):.1f} MB")
    print("STEP 3 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
