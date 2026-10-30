"""Step 7: score saved forecasts against the real TomTom observations that followed."""
import sys
from datetime import datetime

from src.inference.forecast_io import (ACCURACY_LOG, FORECAST_DIR, build_accuracy_rows,
                                       rolling_mae, update_accuracy_log)


def main() -> int:
    new_rows = build_accuracy_rows()
    if new_rows.empty:
        print(f"No forecast has a matching observation yet (forecasts in {FORECAST_DIR}).")
        print("Targets are 6/12/21 min after each forecast's last poll; the poller must still "
              "have been running then, at a 3-min cadence.")
        return 0
    log = update_accuracy_log(new_rows)
    print(f"{len(new_rows)} rows scored; log now has {len(log)} rows -> {FORECAST_DIR / ACCURACY_LOG}")
    mae = rolling_mae(log, datetime.now(), hours=24)
    print("\nRolling MAE over the last 24 h (km/h):")
    for horizon_min, value in mae.items():
        print(f"  +{horizon_min:>2} min: {value:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
