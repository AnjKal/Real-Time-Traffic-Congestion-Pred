"""Scheduled TomTom poller: parquet log + rolling buffer + daily quota guard."""
import json
import logging
import math
import pickle
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np

from src.config import PROCESSED_DIR
from src.nodes import Node
from src.tomtom import quota, storage
from src.tomtom.buffer import RollingBuffer
from src.tomtom.client import ClientError, RateLimitError, TomTomClient, TransientError

log = logging.getLogger("poller")
BUFFER_PATH = storage.TOMTOM_DIR / ".buffer.pkl"


def _parse_hhmm(value: str) -> dtime:
    hour, minute = value.split(":")
    return dtime(int(hour), int(minute))


def load_reference_free_flow(path: Path = PROCESSED_DIR / "scaler.pkl") -> np.ndarray:
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f)["free_flow_kmh"], dtype=np.float64)


class Poller:
    def __init__(self, client: TomTomClient, buffer: RollingBuffer, nodes: list[Node],
                 cadence_min: int, daily_window_start_hhmm: str | None,
                 daily_window_end_hhmm: str | None, daily_quota: int = 2400,
                 max_retries: int = 2, backoff_s: float = 5.0, *,
                 free_flow_ref: np.ndarray | None = None, buffer_path: Path | None = BUFFER_PATH,
                 store_dir: Path | None = None, quota_path: Path | None = None,
                 sleep_fn=time.sleep, now_fn=datetime.now):
        self.client, self.buffer, self.nodes = client, buffer, nodes
        self.cadence_s = cadence_min * 60
        self.window_start = _parse_hhmm(daily_window_start_hhmm) if daily_window_start_hhmm else None
        self.window_end = _parse_hhmm(daily_window_end_hhmm) if daily_window_end_hhmm else None
        self.daily_quota, self.max_retries, self.backoff_s = daily_quota, max_retries, backoff_s
        # The model was trained on speed / these free-flow speeds, so the buffer must normalise
        # with the SAME fixed values, not whatever freeFlowSpeed TomTom returns on a given poll.
        self.free_flow_ref = (np.asarray(free_flow_ref, dtype=np.float64)
                              if free_flow_ref is not None else load_reference_free_flow())
        self.buffer_path, self.store_dir, self.quota_path = buffer_path, store_dir, quota_path
        self._sleep, self._now = sleep_fn, now_fn
        self.last_observations: list[dict] = []

    # ------------------------------------------------------------------ one poll
    def _fetch_with_retries(self, node: Node, calls_so_far: int, used_before: int):
        """Returns (data | None, calls_made). Every HTTP attempt counts against the quota."""
        calls = 0
        for attempt in range(self.max_retries + 1):
            if used_before + calls_so_far + calls >= self.daily_quota:
                log.error("quota exhausted mid-poll; skipping node %d", node.node_id)
                return None, calls
            calls += 1
            try:
                return self.client.fetch_point(node.lat, node.lon), calls
            except ClientError as e:
                log.error("node %d permanent error: %s", node.node_id, e)
                return None, calls
            except (RateLimitError, TransientError) as e:
                log.warning("node %d attempt %d/%d failed: %s", node.node_id, attempt + 1,
                            self.max_retries + 1, e)
                if attempt < self.max_retries:
                    self._sleep(self.backoff_s * 2 ** attempt)
        log.error("node %d failed after %d attempts", node.node_id, self.max_retries + 1)
        return None, calls

    def poll_once(self, push: str = "always") -> dict:
        """push="always" pushes every poll to the buffer (NaN for failed nodes);
        push="all_valid" only pushes a poll in which every node returned data."""
        ts = self._now().replace(microsecond=0)
        used_before = quota.get_calls_used_today(self.quota_path)
        n = len(self.nodes)
        speeds = np.full(n, np.nan)
        rows, ok, calls = [], 0, 0
        try:
            for i, node in enumerate(self.nodes):
                data, made = self._fetch_with_retries(node, calls, used_before)
                calls += made
                speed = ff = conf = np.nan
                closure, frc, payload = None, None, ""
                if data is not None:
                    try:
                        speed = float(data["currentSpeed"])
                        ff = float(data.get("freeFlowSpeed", np.nan))
                        conf = float(data.get("confidence", np.nan))
                        closure, frc = data.get("roadClosure"), data.get("frc")
                        payload = json.dumps(data)
                        speeds[i] = speed
                        ok += 1
                    except (KeyError, TypeError, ValueError):
                        log.error("node %d: unusable payload", node.node_id)
                rows.append(storage.make_row(ts, node.node_id, node.lat, node.lon, speed, ff,
                                             conf, closure, frc, payload))
        finally:
            quota.increment(calls, self.quota_path)

        storage.append_rows(rows, self.store_dir)
        pushed = push == "always" or ok == n
        if pushed:
            self.buffer.push(ts, speeds, self.free_flow_ref)
        self.last_observations = rows
        summary = {"ok": ok, "failed": n - ok, "ts": ts.isoformat(), "calls_used": calls,
                   "pushed": pushed}
        log.info("poll %s", summary)
        return summary

    # ------------------------------------------------------------------ scheduling
    def _in_window(self, now: datetime) -> bool:
        if self.window_start is None or self.window_end is None:
            return True
        return self.window_start <= now.time() <= self.window_end

    def _seconds_until_window_start(self, now: datetime) -> float:
        start = datetime.combine(now.date(), self.window_start)
        if now >= start:
            start += timedelta(days=1)
        return (start - now).total_seconds()

    def _seconds_until_next_boundary(self, now: datetime) -> float:
        """Align polls to cadence multiples from the window start (or midnight)."""
        anchor = datetime.combine(now.date(), self.window_start or dtime(0, 0))
        k = math.floor((now - anchor).total_seconds() / self.cadence_s) + 1
        return max(1.0, (anchor + timedelta(seconds=k * self.cadence_s) - now).total_seconds())

    def _seconds_until_midnight(self, now: datetime) -> float:
        return (datetime.combine(now.date() + timedelta(days=1), dtime(0, 0)) - now).total_seconds() + 1

    def _save_buffer(self) -> None:
        if self.buffer_path is not None:
            self.buffer.save(self.buffer_path)

    def run_forever(self) -> None:
        log.info("poller started: cadence %ds, window %s-%s, quota %d/day", self.cadence_s,
                 self.window_start, self.window_end, self.daily_quota)
        try:
            while True:
                now = self._now()
                quota.reset_if_new_day(self.quota_path)
                if not self._in_window(now):
                    self._sleep(self._seconds_until_window_start(now))
                    continue
                used = quota.get_calls_used_today(self.quota_path)
                if used + len(self.nodes) > self.daily_quota:
                    log.error("daily quota reached (%d/%d); sleeping until midnight", used,
                              self.daily_quota)
                    self._sleep(self._seconds_until_midnight(now))
                    continue
                self.poll_once()
                self._save_buffer()
                self._sleep(self._seconds_until_next_boundary(self._now()))
        except KeyboardInterrupt:
            self._save_buffer()
            log.info("shutting down, buffer saved")

    # ------------------------------------------------------------------ warmup
    def _filled_steps(self) -> int:
        data, _ = self.buffer.snapshot()
        return int((~np.isnan(data).any(axis=(1, 2))).sum())

    def run_warmup(self, status_fn=print, max_consecutive_bad: int = 5) -> bool:
        """Poll every cadence until the buffer holds seq_len fully valid timesteps, then save it.

        A poll counts only if every node returned data (otherwise a NaN would sit in the buffer
        and is_ready() could never become True). Bad polls are still logged and quota-counted,
        but are not pushed. Gives up after `max_consecutive_bad` bad polls in a row.
        Returns True when the buffer is ready and saved.
        """
        total_calls, bad_in_a_row, k = 0, 0, 0
        t0 = self._now()
        try:
            while not self.buffer.is_ready():
                if quota.get_calls_used_today(self.quota_path) + len(self.nodes) > self.daily_quota:
                    log.error("daily quota reached during warmup")
                    status_fn("[warmup] daily quota reached - stopping")
                    return False
                s = self.poll_once(push="all_valid")
                total_calls += s["calls_used"]
                bad_in_a_row = 0 if s["pushed"] else bad_in_a_row + 1
                at = datetime.fromisoformat(s["ts"])
                status_fn(f"[warmup] poll {self._filled_steps()}/{self.buffer.seq_len} at "
                          f"{at:%H:%M:%S}  ok={s['ok']} failed={s['failed']}  "
                          f"calls_used={total_calls}" + ("" if s["pushed"] else "  (not counted)"))
                if bad_in_a_row >= max_consecutive_bad:
                    log.error("%d bad polls in a row, giving up", bad_in_a_row)
                    status_fn(f"[warmup] {bad_in_a_row} failed polls in a row - check network/key")
                    return False
                if self.buffer.is_ready():
                    break
                k += 1
                self._sleep(max(1.0, (t0 + timedelta(seconds=k * self.cadence_s)
                                      - self._now()).total_seconds()))
        except KeyboardInterrupt:
            status_fn("[warmup] interrupted - buffer not ready, not saved")
            return False

        _, stamps = self.buffer.snapshot()
        gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        if max(gaps) > 1.5 * self.cadence_s:
            status_fn("[warmup] WARNING: a failed poll left a time gap in the buffer; "
                      "predictions from it will be less accurate")
        if self.buffer_path is not None:
            self.buffer.save(self.buffer_path)
        status_fn(f"WARMUP COMPLETE: {self.buffer.seq_len} timesteps buffered, ready for Step 7")
        return True
