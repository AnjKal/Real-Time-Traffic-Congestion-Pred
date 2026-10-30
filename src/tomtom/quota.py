"""Daily TomTom request counter persisted to data/raw/tomtom/.quota.json (local calendar day)."""
import json
import os
import threading
from datetime import date
from pathlib import Path

from src.tomtom.storage import TOMTOM_DIR

QUOTA_PATH = TOMTOM_DIR / ".quota.json"
_lock = threading.Lock()


def _read(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _write(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


def _reset(path: Path, today: str) -> dict:
    state = _read(path)
    if state.get("date") != today:
        state = {"date": today, "calls_used": 0}
        _write(path, state)
    return state


def reset_if_new_day(path: Path | None = None, today: date | None = None) -> None:
    with _lock:
        _reset(path or QUOTA_PATH, (today or date.today()).isoformat())


def get_calls_used_today(path: Path | None = None, today: date | None = None) -> int:
    with _lock:
        return int(_reset(path or QUOTA_PATH, (today or date.today()).isoformat())["calls_used"])


def increment(n_calls: int, path: Path | None = None, today: date | None = None) -> None:
    path = path or QUOTA_PATH
    with _lock:
        state = _reset(path, (today or date.today()).isoformat())
        state["calls_used"] = int(state["calls_used"]) + int(n_calls)
        _write(path, state)
