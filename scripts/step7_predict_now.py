"""Step 7: one-shot live prediction from the warm buffer."""
import sys

from src.inference.forecast_io import save_forecast
from src.inference.loader import load_model
from src.inference.predict import format_table, predict_once
from src.tomtom.buffer import RollingBuffer
from src.tomtom.poller import BUFFER_PATH, load_reference_free_flow


def main() -> int:
    if not BUFFER_PATH.exists():
        print("BUFFER NOT READY, run step6 warmup first")
        return 1
    buffer = RollingBuffer.load(BUFFER_PATH)
    if not buffer.is_ready():
        print("BUFFER NOT READY, run step6 warmup first")
        return 1

    model, _ = load_model()
    result = predict_once(model, buffer, load_reference_free_flow())
    print(format_table(result))
    print(f"\nSaved forecast: {save_forecast(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
