"""TomTom collector: polls all configured junctions on a fixed schedule.

    python collect_tomtom.py --once                       # one poll now (uses 100 TomTom calls)
    python collect_tomtom.py --interval-min 2 --max-polls 24

Polls are aligned to the clock (e.g. :00, :02, :04 ...). In the live Kafka pipeline, one raw
message is published per junction, followed by a poll-complete marker. The configured provider
rate pacing and retry handling apply; there is no fixed local daily-call quota.
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
TOMTOM_ARCHIVE_PATH = os.path.join(LIVE_DIR, "tomtom_readings_archive.csv")
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


# --------------------------------------------------------------------------- the collector
class Collector(object):
    def __init__(self, fetch, junctions, out_path=TOMTOM_ARCHIVE_PATH, interval_min=2,
                 workers=5, max_retries=2, backoff_s=1.0,
                 max_qps=8.0, sleep_fn=time.sleep, now_fn=time.time, write_csv=True,
                 on_rows=None):
        self.fetch, self.junctions, self.out_path = fetch, junctions, out_path
        self.interval_s = int(interval_min) * 60
        self.workers, self.max_retries, self.backoff_s = workers, max_retries, backoff_s
        self._min_gap = 1.0 / max_qps if max_qps else 0.0
        self._sleep, self._now = sleep_fn, now_fn
        self._gate = threading.Lock()
        self._next_call_at = 0.0
        self.write_csv = write_csv
        self.on_rows = on_rows

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

    def _one_junction(self, jid, name, lat, lon):
        """-> (row dict | None, http_calls_made)."""
        calls = 0
        for attempt in range(self.max_retries + 1):
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
        """Poll every junction once and return its stamped batch summary."""
        results, calls_total = [], 0
        fatal = None
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._one_junction, jid, name, lat, lon)
                       for jid, name, lat, lon in self.junctions]
            for fut in futures:
                try:
                    row, calls = fut.result()
                except FatalError as e:                # bad key: stop everything
                    fatal = e
                    row, calls = None, 1
                calls_total += calls
                results.append(row)
        if fatal is not None:
            raise fatal

        stamp = datetime.fromtimestamp(slot_epoch, tz=timezone.utc).isoformat()
        rows = [r for r in results if r is not None]
        if self.write_csv:
            folder = os.path.dirname(self.out_path)
            if folder and not os.path.isdir(folder):
                os.makedirs(folder)
            new_file = not os.path.exists(self.out_path) or os.path.getsize(self.out_path) == 0
            with open(self.out_path, "a") as f:             # one append per batch
                writer = csv.writer(f, lineterminator="\n")
                if new_file:
                    writer.writerow(COLUMNS)
                for r in rows:
                    writer.writerow([stamp] + [r.get(c) for c in COLUMNS[1:]])
        if self.on_rows is not None:
            self.on_rows(stamp, rows)
        return {"ok": len(rows), "failed": len(self.junctions) - len(rows),
                "calls": calls_total, "slot": stamp}

    def _slot_of(self, t):
        return int(t // self.interval_s) * self.interval_s

    def run(self, max_polls=None, start_now=False, once=False, log=print, on_poll=None,
            should_stop=None):
        """Poll on clock boundaries until max_polls, Ctrl+C or should_stop() is true.
        on_poll(summary) is called after every poll. Returns the polls made."""
        stopped = should_stop or (lambda: False)
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
        return "[collector] poll %d%s slot %s  ok=%d failed=%d  calls=%d" % (
            i, "/%d" % total if total else "", local, s["ok"], s["failed"], s["calls"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval-min", type=int, default=2)
    parser.add_argument("--max-polls", type=int, default=None)
    parser.add_argument("--once", action="store_true", help="one poll now, then exit")
    parser.add_argument("--start-now", action="store_true",
                        help="first poll immediately instead of at the next clock boundary")
    parser.add_argument("--out", default=TOMTOM_ARCHIVE_PATH,
                        help="append raw TomTom readings here (separate from the model archive)")
    args = parser.parse_args()

    key = load_api_key()
    if not key or key == "your_key_here":
        print("TOMTOM_API_KEY is not set in .env")
        return 1
    junctions = load_junctions()
    collector = Collector(make_tomtom_fetch(key), junctions, out_path=args.out,
                          interval_min=args.interval_min)
    per_poll = len(junctions)
    if args.max_polls:
        print("Plan: up to %d polls x %d junctions = %d TomTom calls"
              % (args.max_polls, per_poll, args.max_polls * per_poll))
    try:
        collector.run(max_polls=args.max_polls, start_now=args.start_now, once=args.once)
    except FatalError as e:
        print("STOPPED: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
