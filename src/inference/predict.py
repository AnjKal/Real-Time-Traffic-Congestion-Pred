"""One-shot live prediction from a warm RollingBuffer."""
from datetime import datetime

import numpy as np
import torch

from src.model.dcrnn.dcrnn_model import forward_batch
from src.nodes import NODES
from src.tomtom.buffer import RollingBuffer

# horizon index in the 7-step output -> label used in the JSON. Index k is (k+1)*3 minutes
# ahead, so these are really +6 / +12 / +21 min; the labels keep the project's 5/10/20 naming.
HORIZONS = {"plus_5min": 1, "plus_10min": 3, "plus_20min": 6}
HORIZON_MINUTES = {label: 3 * (idx + 1) for label, idx in HORIZONS.items()}
CADENCE_MIN = 3
STALE_AFTER_MIN = 10


def congestion_band(speed_over_free_flow: float) -> str:
    if speed_over_free_flow > 0.75:
        return "free"
    if speed_over_free_flow >= 0.40:
        return "moderate"
    return "heavy"


def predict_once(model, buffer: RollingBuffer, free_flows: np.ndarray, device=None,
                 scaler: tuple[float, float] | None = None, now: datetime | None = None) -> dict:
    if not buffer.is_ready():
        raise ValueError("buffer is not ready (needs a full window of valid polls)")
    free_flows = np.asarray(free_flows, dtype=np.float64)
    mean, std = scaler or getattr(model, "speed_scaler", (0.0, 1.0))

    x_np, timestamps = buffer.snapshot()                          # (12, 20, 2)
    x = x_np.copy()
    x[..., 0] = (x[..., 0] - mean) / std                          # same scaling as training
    with torch.no_grad():
        y = forward_batch(model, torch.from_numpy(x).unsqueeze(0).float())   # (1, 7, 20, 1)
    y_norm = y.squeeze(0).squeeze(-1).cpu().numpy() * std + mean   # (7, 20) speed / free-flow
    y_kmh = np.clip(y_norm * free_flows[None, :], 2.0, 1.2 * free_flows[None, :])
    current = x_np[-1, :, 0].astype(np.float64) * free_flows

    now = now or datetime.now()
    last_poll = timestamps[-1]
    warnings = []
    age_min = (now - last_poll).total_seconds() / 60
    if age_min > STALE_AFTER_MIN:
        warnings.append(f"newest poll is {age_min:.0f} min old; forecast is stale")
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(timestamps, timestamps[1:])]
    if max(gaps) > 1.5 * CADENCE_MIN or min(gaps) < 0.5 * CADENCE_MIN:
        warnings.append(f"buffer polls are not {CADENCE_MIN} min apart (gaps {min(gaps):.1f}-"
                        f"{max(gaps):.1f} min); the model was trained on {CADENCE_MIN}-min steps")
    if not np.isfinite(y_kmh).all():
        raise ValueError("model produced non-finite predictions")

    nodes = []
    for i, node in enumerate(NODES):
        ff = float(free_flows[i])
        entry = {"node_id": node.node_id, "name": node.name, "lat": node.lat, "lon": node.lon,
                 "current_kmh": round(float(current[i]), 2), "free_flow_kmh": round(ff, 2),
                 "congestion_now": round(1 - float(current[i]) / ff, 4)}
        for label, idx in HORIZONS.items():
            pred = float(y_kmh[idx, i])
            entry[f"pred_{label}_kmh"] = round(pred, 2)
            entry[f"congestion_{label}"] = round(1 - pred / ff, 4)
            entry[f"band_{label}"] = congestion_band(pred / ff)
        nodes.append(entry)

    return {"forecast_made_at": now.isoformat(timespec="seconds"),
            "input_last_poll_at": last_poll.isoformat(),
            "horizon_minutes": HORIZON_MINUTES, "warnings": warnings, "nodes": nodes}


def format_table(result: dict) -> str:
    short = {"free": "free", "moderate": "mod", "heavy": "heavy"}
    lines = [f"Forecast made at {result['forecast_made_at'].replace('T', ' ')} "
             f"(last poll {result['input_last_poll_at'][11:19]})"]
    lines += [f"WARNING: {w}" for w in result["warnings"]]
    lines += ["", f"{'Node':<24}| Now | +5 min    | +10 min   | +20 min",
              f"{'-' * 24}|-----|-----------|-----------|-----------"]
    for n in result["nodes"]:
        cells = [f"{n[f'pred_{lab}_kmh']:>3.0f} {short[n[f'band_{lab}']]:<5}" for lab in HORIZONS]
        lines.append(f"{n['node_id']:>2} {n['name']:<20} | {n['current_kmh']:>3.0f} | "
                     + " | ".join(cells))
    return "\n".join(lines)
