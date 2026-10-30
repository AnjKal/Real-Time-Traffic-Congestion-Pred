"""Save forecasts to disk and score them against later TomTom observations."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR
from src.inference.predict import HORIZON_MINUTES
from src.tomtom import storage

FORECAST_DIR = RAW_DIR / "forecasts"
ACCURACY_LOG = "accuracy_log.csv"
MATCH_TOLERANCE_S = 90
NOMINAL_MIN = {"plus_5min": 5, "plus_10min": 10, "plus_20min": 20}
COLUMNS = ["forecast_ts", "horizon_min", "node_id", "name", "predicted_kmh", "actual_kmh",
           "abs_error_kmh", "target_ts"]


def save_forecast(result: dict, save_dir: Path | None = None) -> Path:
    save_dir = Path(save_dir or FORECAST_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(result["forecast_made_at"]).strftime("%Y%m%dT%H%M%S")
    path = save_dir / f"forecast_{stamp}.json"
    path.write_text(json.dumps(result, indent=2))
    return path.resolve()


def build_accuracy_rows(forecast_dir: Path | None = None, store_dir: Path | None = None,
                        now: datetime | None = None) -> pd.DataFrame:
    """Match every saved forecast to the real observation at its target time.

    A forecast predicts relative to its LAST INPUT POLL (not the time it was printed): index k
    of the model output is (k+1)*3 minutes after that poll, i.e. +6 / +12 / +21 min for the
    "+5/+10/+20" labels. Targets that have no observation within 90 s are skipped.
    """
    forecast_dir = Path(forecast_dir or FORECAST_DIR)
    now = now or datetime.now()
    forecasts = [json.loads(p.read_text()) for p in sorted(forecast_dir.glob("forecast_*.json"))]
    if not forecasts:
        return pd.DataFrame(columns=COLUMNS)
    earliest = min(datetime.fromisoformat(f["input_last_poll_at"]) for f in forecasts)
    obs = storage.read_last_hours(int((now - earliest).total_seconds() // 3600) + 2,
                                  store_dir=store_dir, now=now)
    obs = obs[obs["speed_kmh"].notna()]
    poll_times = pd.DatetimeIndex(sorted(obs["ts"].unique()))
    by_ts = {t: g.set_index("node_id")["speed_kmh"] for t, g in obs.groupby("ts")}

    rows = []
    for f in forecasts:
        base = datetime.fromisoformat(f["input_last_poll_at"])
        minutes = f.get("horizon_minutes", HORIZON_MINUTES)
        for label, offset in minutes.items():
            target = pd.Timestamp(base + timedelta(minutes=offset))
            if len(poll_times) == 0:
                continue
            j = poll_times.get_indexer([target], method="nearest")[0]
            if abs((poll_times[j] - target).total_seconds()) > MATCH_TOLERANCE_S:
                continue
            actual = by_ts[poll_times[j]]
            for n in f["nodes"]:
                if n["node_id"] not in actual.index:
                    continue
                pred, act = n[f"pred_{label}_kmh"], float(actual[n["node_id"]])
                rows.append({"forecast_ts": f["forecast_made_at"],
                             "horizon_min": NOMINAL_MIN[label], "node_id": n["node_id"],
                             "name": n["name"], "predicted_kmh": pred, "actual_kmh": act,
                             "abs_error_kmh": round(abs(pred - act), 3),
                             "target_ts": poll_times[j].isoformat()})
    return pd.DataFrame(rows, columns=COLUMNS)


def update_accuracy_log(new_rows: pd.DataFrame, forecast_dir: Path | None = None) -> pd.DataFrame:
    """Append to accuracy_log.csv, de-duplicating so reruns do not double-count."""
    path = Path(forecast_dir or FORECAST_DIR) / ACCURACY_LOG
    log = new_rows if not path.exists() else pd.concat([pd.read_csv(path), new_rows],
                                                        ignore_index=True)
    log = log.drop_duplicates(["forecast_ts", "horizon_min", "node_id"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log


def rolling_mae(log: pd.DataFrame, now: datetime | None = None, hours: int = 24) -> pd.Series:
    now = now or datetime.now()
    recent = log[pd.to_datetime(log["target_ts"]) >= now - timedelta(hours=hours)]
    return recent.groupby("horizon_min")["abs_error_kmh"].mean()
