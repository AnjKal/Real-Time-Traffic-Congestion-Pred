"""Append-only parquet log of TomTom observations, partitioned by date."""
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

TOMTOM_DIR = RAW_DIR / "tomtom"
COLUMNS = ["ts", "node_id", "lat", "lon", "speed_kmh", "free_flow_kmh", "confidence",
           "road_closure", "frc", "raw_payload"]
MAX_PAYLOAD_CHARS = 2048
_lock = threading.Lock()


def _partition_file(store_dir: Path, day) -> Path:
    return Path(store_dir) / f"date={day:%Y-%m-%d}" / "observations.parquet"


def make_row(timestamp, node_id, lat, lon, speed_kmh, free_flow_kmh, confidence,
             road_closure, frc, raw_payload_json) -> dict:
    return {"ts": pd.Timestamp(timestamp), "node_id": int(node_id), "lat": float(lat),
            "lon": float(lon), "speed_kmh": float(speed_kmh),
            "free_flow_kmh": float(free_flow_kmh), "confidence": float(confidence),
            "road_closure": road_closure, "frc": frc,
            "raw_payload": (raw_payload_json or "")[:MAX_PAYLOAD_CHARS]}


def append_rows(rows: list[dict], store_dir: Path | None = None) -> None:
    """Append many rows (dicts with COLUMNS keys) in one write per date partition."""
    store_dir = Path(store_dir or TOMTOM_DIR)
    df = pd.DataFrame(rows, columns=COLUMNS)
    with _lock:
        for day, part in df.groupby(df["ts"].dt.date):
            path = _partition_file(store_dir, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
            tmp = path.with_suffix(".parquet.tmp")
            part.to_parquet(tmp, index=False)
            os.replace(tmp, path)                      # atomic: Ctrl-C cannot corrupt the log


def append_observation(timestamp, node_id, lat, lon, speed_kmh, free_flow_kmh, confidence,
                       road_closure, frc, raw_payload_json, store_dir: Path | None = None) -> None:
    append_rows([make_row(timestamp, node_id, lat, lon, speed_kmh, free_flow_kmh, confidence,
                          road_closure, frc, raw_payload_json)], store_dir)


def read_last_hours(hours: int, store_dir: Path | None = None, now: datetime | None = None
                    ) -> pd.DataFrame:
    store_dir = Path(store_dir or TOMTOM_DIR)
    now = now or datetime.now()
    cutoff = now - timedelta(hours=hours)
    frames = []
    day = cutoff.date()
    while day <= now.date():
        path = _partition_file(store_dir, day)
        if path.exists():
            frames.append(pd.read_parquet(path))
        day += timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    return df[df["ts"] >= cutoff].sort_values(["ts", "node_id"]).reset_index(drop=True)
