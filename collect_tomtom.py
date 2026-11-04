"""TomTom collector: polls all 100 junctions on a fixed schedule and appends to a CSV.

    python collect_tomtom.py --once                       # one poll now (uses 100 TomTom calls)
    python collect_tomtom.py --interval-min 2 --max-polls 24     # PIPELINE TEST, ~48 minutes
    python collect_tomtom.py --interval-min 15            # real cadence, runs until Ctrl+C

100 calls per poll; the free tier allows 2,500 per day, so a daily guard (default 2,400, kept
in data/live/quota.json) stops the collector before the limit. 24 polls = 2,400 calls: 12 to
fill the model's input and 12 more to check its forecasts against reality. The guard only
counts calls made by this tool, and retries also count, so avoid other TomTom use that day.
Polls are aligned to the clock (e.g. :00, :02, :04 ...) and every row of a poll carries the
slot time, so the model service can line them up. Output columns match the layout of
data/raw/tomtom_traffic_raw.csv (timestamp, junction_id, currentSpeed) plus extra detail.
Works on Python 3.7 and newer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LIVE_DIR = os.path.join(PROJECT_ROOT, "data", "live")
READINGS_PATH = os.path.join(LIVE_DIR, "readings.csv")
QUOTA_PATH = os.path.join(LIVE_DIR, "quota.json")
GRAPH_PATH = os.path.join(PROJECT_ROOT, "data", "sensor_graph", "adj_mx_bangalore.pkl")
SPOTS_PATH = os.path.join(PROJECT_ROOT, "bangalore_traffic_spots.csv")
FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
# Same columns, same order, as the table the model was trained on (data/raw/tomtom_traffic_raw.csv).
# The model itself only reads timestamp, junction_id and currentSpeed; the rest is kept as a record.
COLUMNS = ["timestamp", "junction_id", "junction_name", "frc", "currentSpeed", "freeFlowSpeed",
           "currentTravelTime", "freeFlowTravelTime", "confidence", "roadClosure", "coordinates"]


class TransientError(Exception):
    """Worth retrying: timeout, connection problem, HTTP 429 or 5xx."""


class FatalError(Exception):
    """Retrying is pointless and every later call would fail too (e.g. bad API key)."""


class ClientError(Exception):
    """This one junction failed for a permanent reason (e.g. HTTP 400/404)."""


# --------------------------------------------------------------------------- TomTom access
def make_tomtom_fetch(api_key, timeout_s=10.0):
    """Returns fetch(lat, lon) -> flowSegmentData dict. Error messages never contain the URL
    (the URL holds the API key)."""
    import requests
    session = requests.Session()

    def fetch(lat, lon):
        params = {"point": "%s,%s" % (lat, lon), "unit": "KMPH", "openLr": "false",
                  "key": api_key}
        try:
            r = session.get(FLOW_URL, params=params, timeout=timeout_s)
        except (requests.Timeout, requests.ConnectionError) as e:
            raise TransientError("network error: %s" % type(e).__name__)
        except requests.RequestException as e:
            raise ClientError("request failed: %s" % type(e).__name__)
        if r.status_code == 200:
            try:
                return r.json()["flowSegmentData"]
            except (ValueError, KeyError):
                raise ClientError("HTTP 200 without flowSegmentData")
        if r.status_code in (401, 403):
            raise FatalError("HTTP %d: the TomTom API key was rejected" % r.status_code)
        if r.status_code == 429 or 500 <= r.status_code < 600:
            raise TransientError("HTTP %d" % r.status_code)
        raise ClientError("HTTP %d" % r.status_code)

    return fetch


def load_api_key():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    key = (os.getenv("TOMTOM_API_KEY") or "").strip().strip('"').strip("'").strip()
    return key or None


def load_junctions():
    """[(junction_id, junction_name, lat, lon)] in the model's node order (J001..J100).
    ASSUMPTION: row i of the spots file is graph node i."""
    import pandas as pd
    with open(GRAPH_PATH, "rb") as f:
        sensor_ids = pickle.load(f, encoding="latin1")[0]
    spots = pd.read_csv(SPOTS_PATH)
    if len(spots) != len(sensor_ids):
        raise ValueError("spots file has %d rows but the graph has %d junctions"
                         % (len(spots), len(sensor_ids)))
    return [(str(sid), str(spots.iloc[i]["name"]), float(spots.iloc[i]["lat"]),
             float(spots.iloc[i]["lon"])) for i, sid in enumerate(sensor_ids)]


# --------------------------------------------------------------------------- daily quota
class QuotaTracker(object):
    def __init__(self, path=QUOTA_PATH, today_fn=None):
        self.path = path
        self._today = today_fn or (lambda: datetime.now().strftime("%Y-%m-%d"))

    def used_today(self):
        try:
            with open(self.path) as f:
                state = json.load(f)
        except (IOError, OSError, ValueError):
            return 0
        return int(state.get("calls", 0)) if state.get("date") == self._today() else 0

    def add(self, n):
        used = self.used_today() + int(n)
        folder = os.path.dirname(self.path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"date": self._today(), "calls": used}, f)
        os.replace(tmp, self.path)
        return used


# --------------------------------------------------------------------------- the collector
class Collector(object):
    def __init__(self, fetch, junctions, out_path=READINGS_PATH, interval_min=15,
                 daily_quota=2400, quota=None, workers=5, max_retries=2, backoff_s=1.0,
                 max_qps=8.0, sleep_fn=time.sleep, now_fn=time.time):
        self.fetch, self.junctions, self.out_path = fetch, junctions, out_path
        self.interval_s = int(interval_min) * 60
        self.daily_quota = daily_quota
        self.quota = quota or QuotaTracker()
        self.workers, self.max_retries, self.backoff_s = workers, max_retries, backoff_s
        self._min_gap = 1.0 / max_qps if max_qps else 0.0
        self._sleep, self._now = sleep_fn, now_fn
        self._gate = threading.Lock()
        self._next_call_at = 0.0

    # -- one HTTP call at a time is paced so we stay well below TomTom's per-second limit
    def _pace(self):
        if not self._min_gap:
            return
        with self._gate:
            now = self._now()
            wait = self._next_call_at - now
            self._next_call_at = max(now, self._next_call_at) + self._min_gap
        if wait > 0:
            self._sleep(wait)

    def _one_junction(self, jid, name, lat, lon, budget):
        """-> (row dict | None, http_calls_made). budget: shared dict {'left': n} of allowed calls."""
        calls = 0
        for attempt in range(self.max_retries + 1):
            with self._gate:
                if budget["left"] <= 0:
                    return None, calls
                budget["left"] -= 1
            self._pace()
            calls += 1
            try:
                data = self.fetch(lat, lon)
                return {"junction_id": jid, "junction_name": name,
                        "frc": data.get("frc"), "currentSpeed": float(data["currentSpeed"]),
                        "freeFlowSpeed": data.get("freeFlowSpeed"),
                        "currentTravelTime": data.get("currentTravelTime"),
                        "freeFlowTravelTime": data.get("freeFlowTravelTime"),
                        "confidence": data.get("confidence"),
                        "roadClosure": data.get("roadClosure"),
                        "coordinates": json.dumps(data.get("coordinates"))}, calls
            except TransientError:
                if attempt < self.max_retries:
                    self._sleep(self.backoff_s * (2 ** attempt))
            except (ClientError, KeyError, TypeError, ValueError):
                return None, calls
        return None, calls

    def poll_once(self, slot_epoch):
        """Poll every junction once; append the batch (stamped with the slot time) to the CSV.
        Returns {'ok', 'failed', 'calls', 'slot', 'quota_used'}."""
        used_before = self.quota.used_today()
        budget = {"left": max(0, self.daily_quota - used_before)}
        results, calls_total = [], 0
        fatal = None
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(self._one_junction, jid, name, lat, lon, budget)
                           for jid, name, lat, lon in self.junctions]
                for fut in futures:
                    try:
                        row, calls = fut.result()
                    except FatalError as e:            # bad key: stop everything
                        fatal = e
                        row, calls = None, 1
                    calls_total += calls
                    results.append(row)
        finally:
            used = self.quota.add(calls_total)
        if fatal is not None:
            raise fatal

        stamp = datetime.fromtimestamp(slot_epoch, tz=timezone.utc).isoformat()
        rows = [r for r in results if r is not None]
        if rows:
            new_file = not os.path.exists(self.out_path)
            folder = os.path.dirname(self.out_path)
            if folder and not os.path.isdir(folder):
                os.makedirs(folder)
            with open(self.out_path, "a") as f:             # one write per batch
                writer = csv.writer(f, lineterminator="\n")
                if new_file:
                    writer.writerow(COLUMNS)
                for r in rows:
                    writer.writerow([stamp] + [r[c] for c in COLUMNS[1:]])
        return {"ok": len(rows), "failed": len(self.junctions) - len(rows),
                "calls": calls_total, "slot": stamp, "quota_used": used}

    def _slot_of(self, t):
        return int(t // self.interval_s) * self.interval_s

    def run(self, max_polls=None, start_now=False, once=False, log=print, on_poll=None,
            should_stop=None):
        """Poll on clock boundaries until max_polls, quota, Ctrl+C or should_stop() is true.
        on_poll(summary) is called after every poll. Returns the polls made."""
        stopped = should_stop or (lambda: False)
        n = len(self.junctions)
        polls = 0
        last_slot = None
        try:
            if once:
                summary = self.poll_once(self._slot_of(self._now()))
                log(self._line(1, summary))
                return 1
            if not start_now:                              # wait for the next boundary
                log("[collector] first poll at the next %d-minute mark" % (self.interval_s // 60))
                self._sleep(self._slot_of(self._now()) + self.interval_s - self._now())
            while (max_polls is None or polls < max_polls) and not stopped():
                slot = self._slot_of(self._now())
                if slot == last_slot:                       # woke early: wait for the boundary
                    self._sleep(max(0.5, slot + self.interval_s - self._now()))
                    continue
                if self.quota.used_today() + n > self.daily_quota:
                    log("Daily quota reached (%d/%d calls). Stopping." % (
                        self.quota.used_today(), self.daily_quota))
                    break
                summary = self.poll_once(slot)
                last_slot = slot
                polls += 1
                log(self._line(polls, summary, max_polls))
                if on_poll is not None:
                    on_poll(summary)
                if max_polls is not None and polls >= max_polls:
                    break
                wait = slot + self.interval_s - self._now()
                if wait <= 0:
                    log("WARNING: the poll took longer than the %d-minute interval; skipping "
                        "to the next slot" % (self.interval_s // 60))
                self._sleep(max(0.5, wait))
        except KeyboardInterrupt:
            log("Stopped by user after %d polls." % polls)
        return polls

    @staticmethod
    def _line(i, s, total=None):
        local = datetime.fromisoformat(s["slot"]).astimezone().strftime("%H:%M:%S")
        return "[collector] poll %d%s slot %s  ok=%d failed=%d  calls=%d  quota_today=%d" % (
            i, "/%d" % total if total else "", local, s["ok"], s["failed"], s["calls"],
            s["quota_used"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval-min", type=int, default=15)
    parser.add_argument("--max-polls", type=int, default=None)
    parser.add_argument("--once", action="store_true", help="one poll now, then exit")
    parser.add_argument("--start-now", action="store_true",
                        help="first poll immediately instead of at the next clock boundary")
    parser.add_argument("--daily-quota", type=int, default=2400)
    parser.add_argument("--out", default=READINGS_PATH)
    args = parser.parse_args()

    key = load_api_key()
    if not key or key == "your_key_here":
        print("TOMTOM_API_KEY is not set in .env")
        return 1
    junctions = load_junctions()
    collector = Collector(make_tomtom_fetch(key), junctions, out_path=args.out,
                          interval_min=args.interval_min, daily_quota=args.daily_quota)
    per_poll = len(junctions)
    if args.max_polls:
        print("Plan: %d polls x %d junctions = %d TomTom calls (today so far: %d of %d)"
              % (args.max_polls, per_poll, args.max_polls * per_poll,
                 collector.quota.used_today(), args.daily_quota))
    try:
        collector.run(max_polls=args.max_polls, start_now=args.start_now, once=args.once)
    except FatalError as e:
        print("STOPPED: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
