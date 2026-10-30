"""The 9 traffic scenarios as pure-function speed generators.

Every scenario function has the signature
    fn(timestamps, free_flow, W, hotspot, rng, severity=None) -> (T, N) float32 km/h
and renders the whole timeline; the generator then picks, per timestep, the row from the
scenario that the timestep is labelled with. Scenarios share one demand curve so speeds stay
continuous across label boundaries, and differ in how they scale / shape it.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from src.config import RANDOM_SEED

AR_PHI = 0.85
NOISE_STD_KMH = 2.0
LAGS = (1, 2, 3)                 # neighbor influence lags, in 3-min steps
LAG_WEIGHTS = (0.5, 0.3, 0.2)
PROPAGATION_GAIN = 0.25
BACKGROUND_SLOPE = 0.15          # city-wide slowdown that scales with demand, any node
INCIDENT_DEPTH = 0.75            # max speed-ratio loss at an incident node

# Speed ratio reached at full demand (lower = worse). Unlisted nodes use DEFAULT_HOTSPOT.
HOTSPOT_BY_NODE_ID = {
    10: 0.4, 13: 0.4,            # Silk Board, Marathahalli
    3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5,   # CBD: Majestic, MG Road, Richmond, Shivajinagar
    0: 0.9, 2: 0.9,              # fringe: Hebbal, Yeshwanthpur
}
DEFAULT_HOTSPOT = 0.7

# Minute-of-day segments (start, end, scenario_id). Hours were fitted so that, with the
# 5% incident/recovery overlay, the scenario shares land on the targets below.
WEEKDAY_SCHEDULE = [(0, 150, 1), (150, 420, 2), (420, 630, 3), (630, 805, 4),
                    (805, 1075, 5), (1075, 1285, 6), (1285, 1440, 1)]
WEEKEND_SCHEDULE = [(0, 190, 1), (190, 312, 2), (312, 672, 4), (672, 1198, 7),
                    (1198, 1320, 5), (1320, 1440, 1)]


def hotspot_multipliers(n_nodes: int) -> np.ndarray:
    return np.array([HOTSPOT_BY_NODE_ID.get(i, DEFAULT_HOTSPOT) for i in range(n_nodes)])


def _minute_map(schedule) -> np.ndarray:
    m = np.zeros(1440, dtype=np.int8)
    for start, end, sid in schedule:
        m[start:end] = sid
    assert (m > 0).all()
    return m


def base_labels(ts: pd.DatetimeIndex) -> np.ndarray:
    """Time-of-day / weekday scenario label (1-7) for every timestep."""
    minute = ts.hour.to_numpy() * 60 + ts.minute.to_numpy()
    weekend = ts.dayofweek.to_numpy() >= 5
    return np.where(weekend, _minute_map(WEEKEND_SCHEDULE)[minute],
                    _minute_map(WEEKDAY_SCHEDULE)[minute]).astype(np.int8)


def demand(ts: pd.DatetimeIndex, weekend_profile: bool | None = None) -> np.ndarray:
    """Congestion demand in [0, 1.2]: bimodal weekday rush, flat low weekend, per-day jitter."""
    h = ts.hour.to_numpy() + ts.minute.to_numpy() / 60

    def g(c, s):
        return np.exp(-0.5 * ((h - c) / s) ** 2)

    weekday = np.maximum(g(8.8, 1.5), g(19.2, 1.7)) + 0.35 * g(13.5, 3.0)
    weekend = 0.45 * g(13.5, 3.5) + 0.35 * g(18.5, 2.5)
    is_we = (ts.dayofweek.to_numpy() >= 5) if weekend_profile is None \
        else np.full(len(ts), weekend_profile)
    base = np.where(is_we, weekend, weekday)
    day_idx = np.asarray((ts.normalize() - ts[0].normalize()).days)
    day_factor = np.random.default_rng(RANDOM_SEED).uniform(0.85, 1.15, day_idx.max() + 1)
    return np.clip(base * day_factor[day_idx], 0, 1.2)


def _neighbor_influence(deficit: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Lagged (1-3 step) row-normalised neighbor deficit; a neighbor's downturn drags a node."""
    rowsum = W.sum(axis=1, keepdims=True)
    Wn = np.divide(W, rowsum, out=np.zeros_like(W, dtype=float), where=rowsum > 0)
    influence = np.zeros_like(deficit)
    for lag, w in zip(LAGS, LAG_WEIGHTS):
        shifted = np.zeros_like(deficit)
        shifted[lag:] = deficit[:-lag]
        influence += w * (shifted @ Wn.T)
    return influence


def _ar1_noise(shape, rng) -> np.ndarray:
    eps = rng.normal(0.0, NOISE_STD_KMH, shape)
    return lfilter([1.0], [1.0, -AR_PHI], eps, axis=0)


def _render(d, free_flow, W, hotspot, rng, severity=None) -> np.ndarray:
    d = d[:, None]
    deficit = BACKGROUND_SLOPE * d + (1.0 - hotspot)[None, :] * d
    if severity is not None:
        deficit = deficit + INCIDENT_DEPTH * severity
    deficit = deficit + PROPAGATION_GAIN * _neighbor_influence(deficit, W)
    ratio = np.clip(1.0 - deficit, 0.05, None)
    ff = free_flow[None, :]
    speed = ratio * ff + _ar1_noise(ratio.shape, rng)
    return np.clip(speed, 2.0, 1.2 * ff).astype(np.float32)


def s1_free_flow_night(ts, free_flow, W, hotspot, rng, severity=None):
    return _render(0.1 * demand(ts), free_flow, W, hotspot, rng)


def s2_morning_rush_buildup(ts, free_flow, W, hotspot, rng, severity=None):
    return _render(demand(ts), free_flow, W, hotspot, rng)


def s3_morning_rush_peak(ts, free_flow, W, hotspot, rng, severity=None):
    return _render(1.1 * demand(ts), free_flow, W, hotspot, rng)


def s4_midday_steady(ts, free_flow, W, hotspot, rng, severity=None):
    return _render(0.9 * demand(ts), free_flow, W, hotspot, rng)


def s5_evening_rush_buildup(ts, free_flow, W, hotspot, rng, severity=None):
    return _render(demand(ts), free_flow, W, hotspot, rng)


def s6_evening_rush_peak(ts, free_flow, W, hotspot, rng, severity=None):
    return _render(1.1 * demand(ts), free_flow, W, hotspot, rng)


def s7_weekend_pattern(ts, free_flow, W, hotspot, rng, severity=None):
    return _render(demand(ts, weekend_profile=True), free_flow, W, hotspot, rng)


def s8_incident_spike(ts, free_flow, W, hotspot, rng, severity=None):
    """severity: (T, N) in [0, 1], the incident intensity at each node."""
    return _render(demand(ts), free_flow, W, hotspot, rng, severity)


def s9_partial_recovery(ts, free_flow, W, hotspot, rng, severity=None):
    """severity: (T, N) decaying from the preceding incident back toward 0."""
    return _render(demand(ts), free_flow, W, hotspot, rng, severity)


@dataclass(frozen=True)
class Scenario:
    id: int
    name: str
    target_prop: float
    fn: Callable


SCENARIOS = [
    Scenario(1, "s1_free_flow_night", 0.20, s1_free_flow_night),
    Scenario(2, "s2_morning_rush_buildup", 0.15, s2_morning_rush_buildup),
    Scenario(3, "s3_morning_rush_peak", 0.10, s3_morning_rush_peak),
    Scenario(4, "s4_midday_steady", 0.15, s4_midday_steady),
    Scenario(5, "s5_evening_rush_buildup", 0.15, s5_evening_rush_buildup),
    Scenario(6, "s6_evening_rush_peak", 0.10, s6_evening_rush_peak),
    Scenario(7, "s7_weekend_pattern", 0.10, s7_weekend_pattern),
    Scenario(8, "s8_incident_spike", 0.03, s8_incident_spike),
    Scenario(9, "s9_partial_recovery", 0.02, s9_partial_recovery),
]
TARGET_PROPS = {s.id: s.target_prop for s in SCENARIOS}
assert abs(sum(TARGET_PROPS.values()) - 1.0) < 1e-9
