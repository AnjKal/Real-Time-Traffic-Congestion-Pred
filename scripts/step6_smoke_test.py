"""Step 6 one-shot smoke test: exactly one poll_once() = 20 real TomTom requests."""
import sys

from src.config import TOMTOM_API_KEY
from src.nodes import NODES
from src.tomtom.buffer import RollingBuffer
from src.tomtom.client import TomTomClient
from src.tomtom.poller import Poller


def main() -> int:
    if not TOMTOM_API_KEY or TOMTOM_API_KEY == "your_key_here":
        print("SMOKE FAIL: TOMTOM_API_KEY not set in .env")
        return 1
    # max_retries=0 keeps this to exactly 20 requests even if some nodes fail
    poller = Poller(TomTomClient(TOMTOM_API_KEY), RollingBuffer(), NODES, cadence_min=3,
                    daily_window_start_hhmm=None, daily_window_end_hhmm=None,
                    max_retries=0, buffer_path=None)
    summary = poller.poll_once()

    print(f"{'id':>2} | {'name':<22} | {'speed':>6} | {'free_flow':>9} | {'conf':>5} | frc")
    for node, row in zip(NODES, poller.last_observations):
        print(f"{node.node_id:>2} | {node.name:<22} | {row['speed_kmh']:6.1f} | "
              f"{row['free_flow_kmh']:9.1f} | {row['confidence']:5.2f} | {row['frc']}")
    print(f"\n{summary}")
    if summary["ok"] < 18:
        print(f"SMOKE FAIL: only {summary['ok']}/20 nodes returned data")
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
