"""Step 7: keep predicting as the poller writes fresh data (run in a second terminal)."""
import argparse
import sys
import time

from src.inference.forecast_io import FORECAST_DIR, save_forecast
from src.inference.loader import load_model
from src.inference.predict import format_table, predict_once
from src.tomtom.buffer import RollingBuffer
from src.tomtom.poller import BUFFER_PATH, load_reference_free_flow


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cadence-min", type=float, default=3,
                   help="expected poll spacing; forecasts older than 2x this are flagged")
    p.add_argument("--save-dir", default=str(FORECAST_DIR))
    p.add_argument("--check-s", type=float, default=15, help="how often to look for a new poll")
    args = p.parse_args(argv)

    model, _ = load_model()
    free_flows = load_reference_free_flow()
    last_seen = None
    print(f"watching {BUFFER_PATH} (Ctrl-C to stop)")
    try:
        while True:
            buffer = None
            if BUFFER_PATH.exists():
                try:
                    buffer = RollingBuffer.load(BUFFER_PATH)
                except Exception as e:                  # file caught mid-write or corrupt
                    print(f"could not read buffer ({type(e).__name__}), retrying")
            if buffer is None or not buffer.is_ready():
                print("waiting for buffer...", flush=True)
            else:
                newest = buffer.snapshot()[1][-1]
                if newest != last_seen:                  # predict once per fresh poll
                    last_seen = newest
                    result = predict_once(model, buffer, free_flows)
                    print("\n" + format_table(result))
                    print(f"saved {save_forecast(result, args.save_dir)}", flush=True)
            time.sleep(args.check_s)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
