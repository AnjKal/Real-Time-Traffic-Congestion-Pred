"""Step 6 CLI: run the TomTom poller (Ctrl-C saves the buffer and exits 0)."""
import argparse
import logging
import sys

from src.config import LOG_DIR, TOMTOM_API_KEY
from src.nodes import NODES
from src.tomtom.buffer import RollingBuffer
from src.tomtom.client import TomTomClient
from src.tomtom.poller import BUFFER_PATH, Poller


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cadence-min", type=int, default=7)
    p.add_argument("--window-start", default="08:00", help="HH:MM local time")
    p.add_argument("--window-end", default="22:00", help="HH:MM local time")
    p.add_argument("--daily-quota", type=int, default=2400)
    p.add_argument("--dry-run", action="store_true", help="poll once and exit")
    p.add_argument("--no-save-buffer", action="store_true")
    p.add_argument("--warmup-only", action="store_true",
                   help="ignore the window; poll every cadence until the buffer holds 12 valid "
                        "timesteps, save it, and exit (~36 min at 3-min cadence)")
    return p.parse_args(argv)


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = [logging.FileHandler(LOG_DIR / "poller.log"), logging.StreamHandler(sys.stdout)]
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not TOMTOM_API_KEY or TOMTOM_API_KEY == "your_key_here":
        print("TOMTOM_API_KEY is not set in .env")
        return 1
    if not args.warmup_only and args.window_start >= args.window_end:
        print("--window-start must be earlier than --window-end (same-day windows only)")
        return 1
    setup_logging()

    if args.warmup_only:
        # fresh buffer: a stale one from an earlier run would look "ready" with old data
        poller = Poller(TomTomClient(TOMTOM_API_KEY), RollingBuffer(), NODES, args.cadence_min,
                        None, None, daily_quota=args.daily_quota, buffer_path=BUFFER_PATH)
        return 0 if poller.run_warmup() else 1

    buffer = RollingBuffer.load(BUFFER_PATH) if BUFFER_PATH.exists() else RollingBuffer()
    poller = Poller(TomTomClient(TOMTOM_API_KEY), buffer, NODES, args.cadence_min,
                    args.window_start, args.window_end, daily_quota=args.daily_quota,
                    buffer_path=None if args.no_save_buffer else BUFFER_PATH)
    if args.dry_run:
        print(poller.poll_once())
        return 0
    poller.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
