"""Rolling buffer of the last SEQ_LEN observations, shaped like one DCRNN input window."""
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import INPUT_DIM, NUM_NODES, SEQ_LEN


class RollingBuffer:
    def __init__(self, num_nodes: int = NUM_NODES, seq_len: int = SEQ_LEN,
                 input_dim: int = INPUT_DIM):
        self.num_nodes, self.seq_len, self.input_dim = num_nodes, seq_len, input_dim
        self._data = np.full((seq_len, num_nodes, input_dim), np.nan, dtype=np.float32)
        self._timestamps: list[datetime | None] = [None] * seq_len

    def push(self, timestamp: datetime, speeds_kmh: np.ndarray, free_flows_kmh: np.ndarray) -> None:
        speeds = np.asarray(speeds_kmh, dtype=np.float64)
        ff = np.asarray(free_flows_kmh, dtype=np.float64)
        if speeds.shape != (self.num_nodes,) or ff.shape != (self.num_nodes,):
            raise ValueError(f"expected ({self.num_nodes},) arrays, got {speeds.shape}, {ff.shape}")
        with np.errstate(invalid="ignore", divide="ignore"):
            speed_norm = np.clip(speeds / ff, 0.0, 1.5)       # NaN stays NaN
        tod = (timestamp.hour + timestamp.minute / 60) / 24
        self._data[:-1] = self._data[1:]
        self._data[-1, :, 0] = speed_norm
        self._data[-1, :, 1] = tod
        self._timestamps = self._timestamps[1:] + [timestamp]

    def snapshot(self) -> tuple[np.ndarray, list]:
        return self._data.copy(), list(self._timestamps)

    def is_ready(self) -> bool:
        return not np.isnan(self._data).any()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"num_nodes": self.num_nodes, "seq_len": self.seq_len,
                 "input_dim": self.input_dim, "data": self._data,
                 "timestamps": self._timestamps}
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp, path)                                 # never leave a half-written file

    @classmethod
    def load(cls, path: Path) -> "RollingBuffer":
        with open(path, "rb") as f:
            state = pickle.load(f)                            # our own file, written by save()
        buf = cls(state["num_nodes"], state["seq_len"], state["input_dim"])
        buf._data = state["data"]
        buf._timestamps = state["timestamps"]
        return buf
