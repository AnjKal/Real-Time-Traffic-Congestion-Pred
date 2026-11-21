"""Latest TomTom free-flow speeds per junction, read from the raw readings archive.

The live pipeline archives every consumed poll to ``data/live/readings.csv`` (written by the model
service after a poll is consumed from ``traffic.raw``). TomTom ``freeFlowSpeed`` is a near-static
property of a road, so the web app associates the LATEST valid free-flow reading with each
junction for the route hover card. This is read-only: it does not touch Kafka, the collector, or
the model, and ``reference_kmh`` (the model's historical reference speed) is never used here.
"""
import os
import threading

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READINGS_PATH = os.path.join(PROJECT_ROOT, "data", "live", "readings.csv")

_lock = threading.Lock()
_cache = {"key": None, "free_flow": {}, "latest_ts": None}


def latest_free_flow_by_junction(path=READINGS_PATH):
    """Return ({junction_id: free_flow_kmh}, latest_reading_timestamp_iso).

    Uses the newest valid (> 0) ``freeFlowSpeed`` per junction. Returns ({}, None) when the archive
    is missing, empty, or unreadable, so a route request never fails because of it. Cached on the
    file's (mtime, size) so repeated requests do not re-read the CSV.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return {}, None
    key = (stat.st_mtime, stat.st_size)
    with _lock:
        if _cache["key"] == key:
            return dict(_cache["free_flow"]), _cache["latest_ts"]
    try:
        free_flow, latest_ts = _read(path)
    except Exception:
        # A malformed/partial CSV must not break routing; fall back to no free-flow.
        return {}, None
    with _lock:
        _cache.update({"key": key, "free_flow": free_flow, "latest_ts": latest_ts})
    return dict(free_flow), latest_ts


def _read(path):
    # Only the three columns we need, so the large coordinates column is never parsed.
    df = pd.read_csv(path, usecols=["timestamp", "junction_id", "freeFlowSpeed"])
    df["junction_id"] = df["junction_id"].astype(str)
    df["freeFlowSpeed"] = pd.to_numeric(df["freeFlowSpeed"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    stamps = df["timestamp"].dropna()
    latest_ts = stamps.max().isoformat() if not stamps.empty else None

    valid = df.dropna(subset=["freeFlowSpeed", "timestamp"])
    valid = valid[valid["freeFlowSpeed"] > 0]
    free_flow = {}
    if not valid.empty:
        # Append-only archive, oldest first: take the last valid reading per junction.
        last = valid.sort_values("timestamp").groupby("junction_id")["freeFlowSpeed"].last()
        free_flow = {str(jid): float(value) for jid, value in last.items()}
    return free_flow, latest_ts
