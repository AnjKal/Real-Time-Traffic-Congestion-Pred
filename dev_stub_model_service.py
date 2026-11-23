"""A stand-in for the model service, for trying the journey agent without the live pipeline.

This is a development aid, not part of the live system. It serves the same `/health`, `/forecast`
and `/nodes` the real `model_service.py` serves, over the real 100 junctions in
`bangalore_traffic_spots.csv`, but with speeds made up instead of predicted by the DCRNN. That is
enough for the web app to draw routes and for the journey agent to decide, so you can watch a
reroute happen without a TomTom key, Kafka, TensorFlow or Python 3.7.

It runs in the same modern Python as the web app:

    python dev_stub_model_service.py                       # port 8001, inner ORR jams at +4 min
    python dev_stub_model_service.py --jam J017,J018       # jam different junctions
    python dev_stub_model_service.py --jam-from 8          # ... later in the forecast
    python dev_stub_model_service.py --free-kmh 45 --jam-kmh 5

Then, in another terminal:

    python run_webapp.py

Junction ids are J001..J100, one per row of `bangalore_traffic_spots.csv` in order. Jamming
junctions that only ONE of the candidate routes passes through is what makes an alternative look
better; junctions shared by every route slow them all down equally (see the limits noted in
README.md).

For real forecasts, use the pipeline: `docker compose up -d`.
"""
import argparse
from datetime import datetime, timedelta

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from webapp.osm_graph import SPOTS_PATH

# Twelve horizons, two minutes apart: the same shape the pipeline produces in test cadence.
HORIZONS = [2 * (step + 1) for step in range(12)]
# The inner ORR stretch, which the Marathahalli -> Silk Board preset runs along.
DEFAULT_JAM = "J005,J006,J007"

SETTINGS = {"jammed": set(), "jam_from": 4, "free_kmh": 40.0, "jam_kmh": 6.0}
SPOTS = pd.read_csv(SPOTS_PATH)
# One fixed input slot, so every forecast belongs to the same poll and the journey agent treats
# the horizons as one book it can read ahead in.
SLOT = datetime.now().replace(second=0, microsecond=0)

app = FastAPI(title="stub model service (development only)")


def _nodes(horizon):
    rows = []
    for index, row in SPOTS.iterrows():
        junction_id = "J%03d" % (index + 1)
        jammed = horizon >= SETTINGS["jam_from"] and junction_id in SETTINGS["jammed"]
        speed = SETTINGS["jam_kmh"] if jammed else SETTINGS["free_kmh"]
        rows.append({"junction_id": junction_id, "name": str(row["name"]),
                     "area": str(row["area"]), "lat": float(row.lat), "lon": float(row.lon),
                     "predicted_kmh": speed, "current_kmh": speed, "reference_kmh": 40.0,
                     "free_flow_kmh": 48.0,
                     "band": "heavy" if speed < 15 else ("moderate" if speed < 25 else "free"),
                     "color": ("#e74c3c" if speed < 15 else
                               ("#f1c40f" if speed < 25 else "#2ecc71"))})
    return rows


@app.get("/health")
def health():
    return {"ready": True, "mode": "pipeline_test", "horizons_min": HORIZONS,
            "step_minutes": 2, "input_window_steps": 12, "warmup_steps": 0,
            "input_last_slot": SLOT.isoformat(timespec="seconds"),
            "sensor_coverage_pct": 100.0,
            "warning": "Stub model service: speeds are made up, not predicted."}


@app.get("/forecast")
def forecast(horizon_min: int = None):
    horizon = int(horizon_min or HORIZONS[0])
    if horizon not in HORIZONS:
        raise HTTPException(400, "horizon_min must be one of %s" % HORIZONS)
    return {"horizon_min": horizon, "horizons_min": HORIZONS, "step_minutes": 2,
            "mode": "pipeline_test", "input_window_steps": 12, "warmup_steps": 0,
            "made_at": datetime.now().isoformat(timespec="seconds"),
            "input_last_slot": SLOT.isoformat(timespec="seconds"),
            "forecast_for": (SLOT + timedelta(minutes=horizon)).isoformat(timespec="seconds"),
            "sensor_coverage_pct": 100.0, "data_age_min": 0.5,
            "warning": "Stub model service: speeds are made up, not predicted.",
            "nodes": _nodes(horizon)}


@app.get("/nodes")
def nodes():
    return {"nodes": _nodes(HORIZONS[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--jam", default=DEFAULT_JAM,
                        help="comma-separated junction ids that slow down (default %s)"
                             % DEFAULT_JAM)
    parser.add_argument("--jam-from", type=int, default=4,
                        help="first forecast horizon, in minutes, at which they slow down")
    parser.add_argument("--free-kmh", type=float, default=40.0)
    parser.add_argument("--jam-kmh", type=float, default=6.0)
    args = parser.parse_args()
    SETTINGS.update({
        "jammed": {part.strip().upper() for part in args.jam.split(",") if part.strip()},
        "jam_from": args.jam_from, "free_kmh": args.free_kmh, "jam_kmh": args.jam_kmh})
    print("Stub model service on http://%s:%d  (development only, speeds are made up)"
          % (args.host, args.port))
    print("  %d junctions at %.0f km/h; %s drop to %.0f km/h from +%d min onward"
          % (len(SPOTS), args.free_kmh, ", ".join(sorted(SETTINGS["jammed"])) or "none",
             args.jam_kmh, args.jam_from))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
