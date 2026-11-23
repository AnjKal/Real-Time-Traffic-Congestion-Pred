"""Memory for the journey intelligence agent.

The agent stores only what it needs to answer one question on every forecast: is the route this
traveller chose still the right one for them? That means four things per journey -- the roads the
route runs on, the shape of the route, the arrival time promised when they chose it, and where
they are now -- plus two small bookkeeping fields so one noisy forecast cannot swing a car, and a
separate ledger of switches.

Two files, both under data/live:

  journey_state.json     the journeys being watched, their last few decisions and the traveller's
                         feedback. Rewritten atomically whenever anything changes, so restarting
                         the web app picks the journeys back up.
  journey_switches.json  an append-only record of adopted switches: where the traveller was, what
                         was promised, what the switch claimed to save and -- once the journey
                         finishes -- what it actually saved.

In-process callers (the agent and the simulator) read the live dictionaries under the lock with
``view``; ``recall`` hands out a deep copy and is what the HTTP layer uses.
"""
from __future__ import absolute_import

import copy
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone

LOG = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DIR = os.path.join(PROJECT_ROOT, "data", "live")
STATE_PATH = os.path.join(LIVE_DIR, "journey_state.json")
SWITCH_LOG_PATH = os.path.join(LIVE_DIR, "journey_switches.json")

MAX_DECISIONS = 20       # kept per journey: the page shows five, the rules look back one
MAX_SWITCH_ROWS = 500    # the ledger stays small, but bound it anyway
JOURNEY_TTL_MIN = 180    # forget a journey this long after its last update


def utc_now():
    return datetime.now(timezone.utc)


def iso(moment):
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(text):
    """Parse an ISO timestamp, assuming UTC when it carries no zone. None when unparsable."""
    if not text:
        return None
    try:
        value = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def write_json_atomic(path, value):
    """Write via a temporary file and rename, so a reader never sees a half-written file."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, separators=(",", ":"), allow_nan=False)
    os.replace(temporary, path)


def slim_decision(decision):
    """A decision without the alternative's road geometry, which is far too big to keep."""
    if not decision:
        return decision
    light = {key: value for key, value in decision.items() if key != "alternative"}
    alternative = decision.get("alternative")
    if alternative:
        light["alternative"] = {key: value for key, value in alternative.items()
                                if key != "segments"}
    return light


class JourneyMemory(object):
    """Everything the agent remembers, on disk and under one lock."""

    def __init__(self, state_path=STATE_PATH, switch_log_path=SWITCH_LOG_PATH):
        self.state_path = state_path
        self.switch_log_path = switch_log_path
        self._lock = threading.RLock()
        self._journeys = {}
        self._decisions = {}
        self._switches = []
        self._load()

    # ------------------------------------------------------------------ loading and saving
    def _load(self):
        try:
            with open(self.state_path, "r") as handle:
                state = json.load(handle)
        except (IOError, ValueError):
            state = {}
        self._journeys = state.get("journeys", {}) or {}
        self._decisions = state.get("decisions", {}) or {}
        try:
            with open(self.switch_log_path, "r") as handle:
                ledger = json.load(handle)
        except (IOError, ValueError):
            ledger = {}
        self._switches = ledger.get("switches", []) or []
        if self._journeys:
            LOG.info("Journey memory restored: %d journey(s)", len(self._journeys))

    def _save_state(self):
        """Called with the lock held."""
        try:
            write_json_atomic(self.state_path, {"updated_at": iso(utc_now()),
                                                "journeys": self._journeys,
                                                "decisions": self._decisions})
        except (IOError, OSError, ValueError):
            LOG.exception("Could not write %s; the agent keeps working from memory",
                          self.state_path)

    def _save_switches(self):
        """Called with the lock held."""
        try:
            write_json_atomic(self.switch_log_path, {"updated_at": iso(utc_now()),
                                                     "switch_count": len(self._switches),
                                                     "switches": self._switches})
        except (IOError, OSError, ValueError):
            LOG.exception("Could not write the switch ledger %s", self.switch_log_path)

    @property
    def lock(self):
        """The agent takes this around a decision so a journey cannot change underneath it."""
        return self._lock

    # ------------------------------------------------------------------ the journeys themselves
    def remember(self, journey):
        with self._lock:
            journey["updated_at"] = iso(utc_now())
            self._journeys[journey["journey_id"]] = journey
            self._save_state()
            return journey

    def view(self, journey_id):
        """The live journey dictionary, for in-process callers holding the lock. May be None."""
        return self._journeys.get(journey_id)

    def recall(self, journey_id):
        """A deep copy of one journey, safe to hand outside. None when not watched."""
        with self._lock:
            journey = self._journeys.get(journey_id)
            return copy.deepcopy(journey) if journey else None

    def journey_ids(self):
        with self._lock:
            return list(self._journeys.keys())

    def all_journeys(self):
        with self._lock:
            return copy.deepcopy(list(self._journeys.values()))

    def update_journey(self, journey_id, **fields):
        """Set fields on a watched journey and persist. Returns a copy, or None if unknown."""
        with self._lock:
            journey = self._journeys.get(journey_id)
            if journey is None:
                return None
            journey.update(fields)
            if "position_cum_m" in fields and "position_source" not in fields:
                journey["position_source"] = "position_feed"
            journey["updated_at"] = iso(utc_now())
            self._save_state()
            return copy.deepcopy(journey)

    def touch(self):
        """Persist whatever in-process callers changed on the live dictionaries."""
        with self._lock:
            self._save_state()

    def forget(self, journey_id):
        with self._lock:
            existed = self._journeys.pop(journey_id, None) is not None
            self._decisions.pop(journey_id, None)
            if existed:
                self._save_state()
            return existed

    def prune(self, ttl_minutes=JOURNEY_TTL_MIN):
        """Drop journeys untouched for longer than the TTL. Returns the ids dropped."""
        cutoff = utc_now() - timedelta(minutes=ttl_minutes)
        dropped = []
        with self._lock:
            for journey_id, journey in list(self._journeys.items()):
                seen = parse_iso(journey.get("updated_at"))
                if seen is not None and seen < cutoff:
                    self._journeys.pop(journey_id, None)
                    self._decisions.pop(journey_id, None)
                    dropped.append(journey_id)
            if dropped:
                self._save_state()
        return dropped

    # ------------------------------------------------------------------ decisions
    def record_decision(self, journey_id, decision):
        """Append a decision (without road geometry) to this journey's short history."""
        with self._lock:
            history = self._decisions.setdefault(journey_id, [])
            history.append(slim_decision(decision))
            del history[:-MAX_DECISIONS]
            self._save_state()

    def decisions(self, journey_id, limit=5):
        """The most recent decisions, newest first."""
        with self._lock:
            history = self._decisions.get(journey_id, [])
            rows = history[-int(limit):] if limit else list(history)
            return copy.deepcopy(list(reversed(rows)))

    def last_decision(self, journey_id):
        with self._lock:
            history = self._decisions.get(journey_id, [])
            return copy.deepcopy(history[-1]) if history else None

    # ------------------------------------------------------------------ the traveller's feedback
    # Two separate interruption budgets. Telling someone they are running late must not use up
    # the budget for telling them about a faster road: they are different messages.
    PREFERENCE_DEFAULTS = {"accepted": 0, "dismissed": 0, "switches_taken": 0,
                           "last_switch_tick": None, "last_delay_tick": None}

    def _preference_block(self, journey):
        block = journey.setdefault("preferences", dict(self.PREFERENCE_DEFAULTS))
        for key, value in self.PREFERENCE_DEFAULTS.items():
            block.setdefault(key, value)
        return block

    def preferences(self, journey_id):
        with self._lock:
            journey = self._journeys.get(journey_id)
            if journey is None:
                return None
            return copy.deepcopy(self._preference_block(journey))

    def record_feedback(self, journey_id, response):
        """'accepted' or 'dismissed'. A dismissal is what makes the agent back off."""
        with self._lock:
            journey = self._journeys.get(journey_id)
            if journey is None:
                return None
            preferences = self._preference_block(journey)
            key = "accepted" if response == "accepted" else "dismissed"
            preferences[key] = int(preferences.get(key, 0)) + 1
            preferences["last_switch_tick"] = journey.get("tick", 0)
            if key == "dismissed":
                # Forget the suggestion, so its streak has to build again from scratch.
                journey["suggested"] = None
                journey["pending_alternative"] = None
            journey["updated_at"] = iso(utc_now())
            self._save_state()
            return copy.deepcopy(preferences)

    # ------------------------------------------------------------------ the switch ledger
    def record_switch(self, entry):
        """Append an adopted switch. This outlives the journey, which is the point of it."""
        with self._lock:
            self._switches.append(entry)
            del self._switches[:-MAX_SWITCH_ROWS]
            self._save_switches()
            return copy.deepcopy(entry)

    def close_switches(self, journey_id, achieved_total_min):
        """On arrival, write down what each switch on this journey actually saved.

        ``eta_before_min`` on the row is the whole-journey time the agent was promising just
        before the switch. Comparing it with the time the journey actually took is the honest
        measure of whether the advice was worth taking.
        """
        if achieved_total_min is None:
            return []
        closed = []
        with self._lock:
            for row in self._switches:
                if (row.get("journey_id") != journey_id
                        or row.get("achieved_total_min") is not None):
                    continue
                row["achieved_total_min"] = round(float(achieved_total_min), 1)
                promised = row.get("eta_before_min")
                if promised is not None:
                    row["gain_realised_min"] = round(float(promised) - float(achieved_total_min), 1)
                row["closed_at"] = iso(utc_now())
                closed.append(copy.deepcopy(row))
            if closed:
                self._save_switches()
        return closed

    def switches(self, journey_id=None):
        """Switch rows, newest first."""
        with self._lock:
            rows = [row for row in self._switches
                    if journey_id is None or row.get("journey_id") == journey_id]
            return copy.deepcopy(list(reversed(rows)))
