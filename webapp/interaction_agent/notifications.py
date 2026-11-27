"""Proactive traffic notifications for the User Interaction Agent (Phase 3).

Phase 1 answered questions. Phase 2 remembered the conversation. Phase 3 is the part that speaks
first: it turns the congestion events the incident agent publishes and the decisions the journey
agent publishes into short, relevant messages -- but only about the journeys a traveller has
explicitly asked to be told about.

Where the data comes from (nothing here recomputes any of it)
------------------------------------------------------------
The engine is a *consumer*, exactly like the rest of the interaction agent. It reads:

  * ``data/live/incidents.json``     the incident monitoring agent's snapshot
                                     (``active_events`` + ``recent_resolved``)
  * ``data/live/journey_state.json`` the journey intelligence agent's watched journeys and their
                                     latest decision (via ``LiveDataSources``, which reads journeys
                                     through the existing ``JourneyMemory``)

It never re-derives congestion, never re-routes and never calls a model. It only re-words what the
other two agents already decided, so the notification text can never contradict the page.

Relevance
---------
A notification is only produced for a ``(session, journey)`` pair the traveller subscribed to.
Nothing subscribes on the traveller's behalf: a journey that merely appeared in a tool result is
never monitored. An incident is matched to that journey using evidence the upstream agents already
produced -- either the incident's affected junctions lie on the journey's stored route, or the
journey's own decision already listed the incident id. If neither holds, no personalized
notification is generated.

Notification types
------------------
  incident            a congestion event is open on a monitored journey
  incident_resolved   such an event has cleared
  journey_delay       the journey agent decided to say "you are running late"
  route_alternative   the journey agent is recommending a faster route
  journey_completed   the journey arrived

Deduplication and freshness
---------------------------
The same event processed repeatedly must not notify twice. Every notification carries a dedup key
(session + journey + type + source event) and a material ``signature``; a repeat with the same
signature is dropped, and the decision persists in SQLite so it survives a restart. A cooldown
silences repeats of the same key even when the signature changed -- except when the severity
escalates, which always breaks through. Nothing is ever generated from data older than
``stale_after_min``, so a current alert is never built from a stale snapshot.

Independence
------------
The engine imports no LLM provider and makes no AWS call: messages are rendered from fixed
templates. ``process_notifications`` is a plain function, so a background runtime or a Kafka
consumer can call it later; importing this module starts nothing and no thread is ever spawned.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from webapp.journey_memory import iso, parse_iso, utc_now

LOG = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Same file Phase 2 uses. Its schema is additive and namespaced, so it coexists with the
# conversation tables and the LangGraph checkpoint tables already in that file.
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "live", "interaction_memory.sqlite")

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 5000

# -- tables (all namespaced, all created with IF NOT EXISTS, nothing dropped) --
NOTIFICATION_TABLE = "interaction_notifications"
NOTIFICATION_STATE_TABLE = "interaction_notification_state"
SUBSCRIPTION_TABLE = "interaction_subscriptions"
NOTIFICATION_META_TABLE = "interaction_notification_meta"

# -- notification types --
NOTIF_INCIDENT = "incident"
NOTIF_INCIDENT_RESOLVED = "incident_resolved"
NOTIF_JOURNEY_DELAY = "journey_delay"
NOTIF_ROUTE_ALTERNATIVE = "route_alternative"
NOTIF_JOURNEY_COMPLETED = "journey_completed"
NOTIFICATION_TYPES = (NOTIF_INCIDENT, NOTIF_INCIDENT_RESOLVED, NOTIF_JOURNEY_DELAY,
                      NOTIF_ROUTE_ALTERNATIVE, NOTIF_JOURNEY_COMPLETED)

# -- severities, least to most severe. "info" is for terminal, non-urgent notices (arrival). --
SEVERITY_INFO = "info"
_SEVERITY_RANK = {"info": 0, "low": 1, "moderate": 2, "high": 3}

# -- journey decision action values, copied from journey_agent.py rather than imported so this
#    module never drags the road-graph stack into a unit test. --
ACTION_NOTIFY_DELAY = "notify_delay"
ACTION_RECOMMEND_SWITCH = "recommend_switch"
ACTION_ARRIVED = "arrived"

# Incident statuses that are "open" (the snapshot's active_events side).
OPEN_INCIDENT_STATUSES = ("watch", "active", "recovering")

# Journey statuses after which the journey is over: no *new* congestion alert should be raised on
# one (the arrival notification itself is still emitted -- see journey_candidates).
TERMINAL_JOURNEY_STATUSES = ("arrived", "expired", "completed", "finished")

# Severity the engine assigns to a delay / an alternative, from the figures the journey agent
# already computed. These are presentation thresholds, not a second opinion on the decision.
DELAY_MODERATE_MIN = 8.0
DELAY_HIGH_MIN = 15.0
ALTERNATIVE_MODERATE_GAIN_MIN = 5.0
ALTERNATIVE_HIGH_GAIN_MIN = 12.0
DELAY_SIGNATURE_BUCKET_MIN = 5.0     # a delay must move this far to count as a new signature

_EVENT_TYPE_WORDS = {
    "observed_slowdown": "slowdown",
    "forecast_congestion": "forecast congestion",
    "congestion_build_up": "build-up congestion",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS {meta} (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS {subs} (
    session_id TEXT NOT NULL,
    journey_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, journey_id)
);
CREATE TABLE IF NOT EXISTS {notifications} (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    notification_id TEXT NOT NULL UNIQUE,
    session_id      TEXT NOT NULL,
    journey_id      TEXT,
    source_event_id TEXT,
    notification_type TEXT NOT NULL,
    severity        TEXT NOT NULL,
    title           TEXT NOT NULL,
    message         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    read            INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_{notifications}_session ON {notifications}(session_id, id);
CREATE INDEX IF NOT EXISTS idx_{notifications}_unread ON {notifications}(session_id, read);
CREATE TABLE IF NOT EXISTS {state} (
    dedup_key        TEXT PRIMARY KEY,
    signature        TEXT,
    severity         TEXT,
    last_notified_at TEXT,
    notification_id  TEXT
);
""".format(meta=NOTIFICATION_META_TABLE, subs=SUBSCRIPTION_TABLE,
           notifications=NOTIFICATION_TABLE, state=NOTIFICATION_STATE_TABLE)


# --------------------------------------------------------------------------- helpers

def _num(value, digits=1):
    """Format a number for a message; None stays N/A rather than becoming a guess."""
    if value is None:
        return "N/A"
    try:
        return ("%%.%df" % digits) % float(value)
    except (TypeError, ValueError):
        return "N/A"


def _age_minutes(stamp, now=None):
    moment = parse_iso(stamp)
    if moment is None:
        return None
    reference = now or utc_now()
    return round(max(0.0, (reference - moment).total_seconds() / 60.0), 1)


def _event_age_minutes(event, now=None):
    """How old a single incident event is, from its own timestamps (not the snapshot's)."""
    stamp = event.get("updated_at") or event.get("last_seen") or event.get("first_seen")
    return _age_minutes(stamp, now)


def event_is_fresh(event, now, stale_after_min) -> bool:
    """False when this individual event is older than the freshness window.

    The snapshot can be fresh while one event inside it is hours old (a resolved event lingers in
    ``recent_resolved``); a current alert must never be built from that one. An event carrying no
    timestamp is treated as fresh, since it cannot be shown to be stale.
    """
    age = _event_age_minutes(event, now)
    return age is None or age <= float(stale_after_min)


def severity_rank(severity) -> int:
    """Order severities so an escalation can be told apart from a de-escalation."""
    return _SEVERITY_RANK.get(str(severity or "").lower(), 0)


def _delay_severity(delta_min):
    delta = abs(float(delta_min or 0.0))
    if delta >= DELAY_HIGH_MIN:
        return "high"
    if delta >= DELAY_MODERATE_MIN:
        return "moderate"
    return "low"


def _alternative_severity(gain_min):
    gain = float(gain_min or 0.0)
    if gain >= ALTERNATIVE_HIGH_GAIN_MIN:
        return "high"
    if gain >= ALTERNATIVE_MODERATE_GAIN_MIN:
        return "moderate"
    return "low"


def new_notification_id() -> str:
    """A unique, URL-safe notification id."""
    return "nt-" + uuid.uuid4().hex[:12]


def _dedup_key(session_id, journey_id, notification_type, source_event_id):
    return "%s|%s|%s|%s" % (session_id, journey_id or "", notification_type, source_event_id or "")


def evaluate_dedup(existing, signature, severity, now=None, cooldown_min=0.0):
    """Decide whether a candidate notification should be emitted.

    ``existing`` is the stored dedup state for the same key (or None). Returns
    ``(should_notify, reason)`` where reason is one of:

      new                 nothing has been notified for this key yet
      duplicate           the same material signature was already notified
      severity_increased  severity escalated: always notify, cooldown notwithstanding
      cooldown            a change, but too soon after the last notice for this key
      changed             a material change, past the cooldown
    """
    if not existing:
        return True, "new"
    if existing.get("signature") == signature:
        return False, "duplicate"
    if severity_rank(severity) > severity_rank(existing.get("severity")):
        return True, "severity_increased"
    reference = now or utc_now()
    last = parse_iso(existing.get("last_notified_at"))
    cooldown = max(0.0, float(cooldown_min or 0.0))
    if last is not None and cooldown > 0.0:
        elapsed_min = (reference - last).total_seconds() / 60.0
        if elapsed_min < cooldown:
            return False, "cooldown"
    return True, "changed"


@dataclass(frozen=True)
class Notification:
    """One proactive message, addressed to a session about a journey."""

    session_id: str
    notification_type: str
    title: str
    message: str
    journey_id: Optional[str] = None
    source_event_id: Optional[str] = None
    severity: str = SEVERITY_INFO
    created_at: Optional[str] = None
    notification_id: Optional[str] = None
    read: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {"notification_id": self.notification_id, "session_id": self.session_id,
                "journey_id": self.journey_id, "source_event_id": self.source_event_id,
                "notification_type": self.notification_type, "severity": self.severity,
                "title": self.title, "message": self.message, "created_at": self.created_at,
                "read": bool(self.read)}


# --------------------------------------------------------------------------- the store

class NotificationStore:
    """SQLite-backed subscriptions, notifications and dedup state.

    Follows the same rules as the Phase 2 conversation memory: opening is lazy, the schema is
    additive (``IF NOT EXISTS`` only), every method degrades instead of raising, and one lock plus
    WAL makes concurrent readers and writers safe. ``enabled=False`` makes every method a no-op so
    the call sites stay identical when notifications are switched off.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, enabled: bool = True):
        self.db_path = db_path
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    @classmethod
    def from_config(cls, config) -> "NotificationStore":
        return cls(db_path=config.memory_db_path, enabled=config.notifications_enabled)

    @property
    def available(self) -> bool:
        """True once the database is open. Reading it opens nothing."""
        return self._conn is not None

    def connect(self) -> bool:
        """Open the database if it is not already open. Returns whether the store is usable."""
        return self._ready()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None

    def _ready(self) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if self._conn is not None:
                return True
            connection = None
            try:
                if self.db_path != ":memory:":
                    parent = os.path.dirname(os.path.abspath(self.db_path))
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                connection = sqlite3.connect(self.db_path, check_same_thread=False,
                                             timeout=BUSY_TIMEOUT_MS / 1000.0)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA busy_timeout=%d" % BUSY_TIMEOUT_MS)
                self._create_schema(connection)
            except (sqlite3.Error, OSError) as exc:
                LOG.warning("notification store unavailable at %s: %s", self.db_path, exc)
                if connection is not None:
                    try:
                        connection.close()
                    except sqlite3.Error:
                        pass
                self._conn = None
                return False
            self._conn = connection
            return True

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        """Additive only: create what is missing and leave every existing table alone."""
        connection.executescript(SCHEMA_SQL)
        connection.execute("INSERT OR IGNORE INTO %s(key, value) VALUES('schema_version', ?)"
                           % NOTIFICATION_META_TABLE, (str(SCHEMA_VERSION),))
        connection.commit()

    def _query(self, sql, params=()) -> List[sqlite3.Row]:
        if not self._ready():
            return []
        try:
            with self._lock:
                return self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            LOG.warning("notification store read failed: %s", exc)
            return []

    def _write(self, operations) -> int:
        """Run ``operations`` (callables taking the connection) in one transaction."""
        if not self._ready():
            return 0
        try:
            with self._lock:
                changes = 0
                for operation in operations:
                    cursor = operation(self._conn)
                    if cursor is not None:
                        changes += cursor.rowcount
                self._conn.commit()
                return changes
        except sqlite3.Error as exc:
            LOG.warning("notification store write failed: %s", exc)
            return 0

    # -- subscriptions ---------------------------------------------------------
    def subscribe(self, session_id, journey_id) -> bool:
        """Explicitly monitor a journey for a session. Idempotent."""
        sid, jid = str(session_id or "").strip(), str(journey_id or "").strip()
        if not sid or not jid:
            return False

        def op(connection):
            return connection.execute(
                "INSERT OR IGNORE INTO %s(session_id, journey_id, created_at) VALUES(?,?,?)"
                % SUBSCRIPTION_TABLE, (sid, jid, iso(utc_now())))
        return self._write([op]) > 0

    def unsubscribe(self, session_id, journey_id) -> bool:
        """Stop monitoring a journey. Returns whether a subscription was removed."""
        sid, jid = str(session_id or "").strip(), str(journey_id or "").strip()

        def op(connection):
            return connection.execute(
                "DELETE FROM %s WHERE session_id=? AND journey_id=?" % SUBSCRIPTION_TABLE,
                (sid, jid))
        return self._write([op]) > 0

    def subscriptions(self, session_id=None) -> List[Dict[str, Any]]:
        """Every ``(session, journey)`` subscription, or just one session's when asked."""
        if session_id is None:
            rows = self._query("SELECT session_id, journey_id, created_at FROM %s "
                               "ORDER BY created_at, session_id, journey_id" % SUBSCRIPTION_TABLE)
        else:
            rows = self._query("SELECT session_id, journey_id, created_at FROM %s "
                               "WHERE session_id=? ORDER BY created_at, journey_id"
                               % SUBSCRIPTION_TABLE, (str(session_id),))
        return [{"session_id": row["session_id"], "journey_id": row["journey_id"],
                 "created_at": row["created_at"]} for row in rows]

    def subscribed_journeys(self, session_id) -> List[str]:
        rows = self._query("SELECT journey_id FROM %s WHERE session_id=? ORDER BY created_at"
                           % SUBSCRIPTION_TABLE, (str(session_id),))
        return [row["journey_id"] for row in rows]

    def is_subscribed(self, session_id, journey_id) -> bool:
        rows = self._query("SELECT 1 FROM %s WHERE session_id=? AND journey_id=? LIMIT 1"
                           % SUBSCRIPTION_TABLE, (str(session_id), str(journey_id)))
        return bool(rows)

    # -- notifications ---------------------------------------------------------
    def add(self, notification: Notification, dedup_key=None, signature=None) -> Optional[str]:
        """Persist one notification. Returns its id, or None if nothing was written."""
        nid = notification.notification_id or new_notification_id()
        created = notification.created_at or iso(utc_now())
        row = (nid, notification.session_id, notification.journey_id, notification.source_event_id,
               notification.notification_type, notification.severity, notification.title,
               notification.message, created, 1 if notification.read else 0)

        def insert(connection):
            return connection.execute(
                "INSERT INTO %s(notification_id, session_id, journey_id, source_event_id, "
                "notification_type, severity, title, message, created_at, read) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)" % NOTIFICATION_TABLE, row)

        operations = [insert]
        if dedup_key:
            operations.append(self._state_upsert(dedup_key, signature, notification.severity,
                                                  created, nid))
        if self._write(operations) <= 0:
            return None
        return nid

    def add_guarded(self, notification: Notification, dedup_key=None, signature=None, now=None,
                    cooldown_min=0.0) -> Tuple[Optional[str], str]:
        """Dedup-check and persist atomically; returns ``(notification_id, reason)``.

        ``add`` alone cannot be used for deduplication safely: reading the state, deciding, and
        inserting are three steps a second thread can interleave between, which is how two
        overlapping notification cycles both emit the same alert. Here the read of the dedup state,
        the decision and the two writes happen under the store's lock in one transaction, so only
        one of two racing callers can win. ``reason`` is the same vocabulary ``evaluate_dedup``
        returns, plus ``unavailable``/``skipped``/``error`` when nothing could be written.
        """
        if not dedup_key:
            nid = self.add(notification)
            return (nid, "new") if nid else (None, "skipped")
        if not self._ready():
            return None, "unavailable"
        created = notification.created_at or iso(now or utc_now())
        nid = notification.notification_id or new_notification_id()
        row = (nid, notification.session_id, notification.journey_id, notification.source_event_id,
               notification.notification_type, notification.severity, notification.title,
               notification.message, created, 1 if notification.read else 0)
        with self._lock:
            try:
                current = self._conn.execute(
                    "SELECT signature, severity, last_notified_at, notification_id FROM %s "
                    "WHERE dedup_key=?" % NOTIFICATION_STATE_TABLE, (dedup_key,)).fetchone()
                existing = ({"signature": current["signature"], "severity": current["severity"],
                             "last_notified_at": current["last_notified_at"],
                             "notification_id": current["notification_id"]}
                            if current is not None else None)
                should, reason = evaluate_dedup(existing, signature, notification.severity, now=now,
                                                cooldown_min=cooldown_min)
                if not should:
                    return None, reason
                self._conn.execute(
                    "INSERT INTO %s(notification_id, session_id, journey_id, source_event_id, "
                    "notification_type, severity, title, message, created_at, read) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)" % NOTIFICATION_TABLE, row)
                self._conn.execute(
                    "INSERT OR REPLACE INTO %s(dedup_key, signature, severity, last_notified_at, "
                    "notification_id) VALUES(?,?,?,?,?)" % NOTIFICATION_STATE_TABLE,
                    (dedup_key, signature, notification.severity, created, nid))
                self._conn.commit()
                return nid, reason
            except sqlite3.Error as exc:
                LOG.warning("notification store guarded write failed: %s", exc)
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass
                return None, "error"

    @staticmethod
    def _state_upsert(dedup_key, signature, severity, at, notification_id):
        def op(connection):
            return connection.execute(
                "INSERT OR REPLACE INTO %s(dedup_key, signature, severity, last_notified_at, "
                "notification_id) VALUES(?,?,?,?,?)" % NOTIFICATION_STATE_TABLE,
                (dedup_key, signature, severity, at, notification_id))
        return op

    def list_notifications(self, session_id, unread_only=False, limit=None) -> List[Dict[str, Any]]:
        """This session's notifications, newest first."""
        sql = ("SELECT notification_id, session_id, journey_id, source_event_id, "
               "notification_type, severity, title, message, created_at, read FROM %s "
               "WHERE session_id=?" % NOTIFICATION_TABLE)
        params: List[Any] = [str(session_id)]
        if unread_only:
            sql += " AND read=0"
        sql += " ORDER BY id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [self._row_to_dict(row) for row in self._query(sql, tuple(params))]

    def get(self, session_id, notification_id) -> Optional[Dict[str, Any]]:
        """One notification by id, scoped to a session. None when it is not this session's."""
        rows = self._query("SELECT notification_id, session_id, journey_id, source_event_id, "
                           "notification_type, severity, title, message, created_at, read FROM %s "
                           "WHERE session_id=? AND notification_id=? LIMIT 1" % NOTIFICATION_TABLE,
                           (str(session_id), str(notification_id)))
        return self._row_to_dict(rows[0]) if rows else None

    def mark_read(self, session_id, notification_id=None) -> int:
        """Mark one notification read, or every unread one when no id is given."""
        if notification_id is not None:
            def op(connection):
                return connection.execute(
                    "UPDATE %s SET read=1 WHERE session_id=? AND notification_id=?"
                    % NOTIFICATION_TABLE, (str(session_id), str(notification_id)))
        else:
            def op(connection):
                return connection.execute(
                    "UPDATE %s SET read=1 WHERE session_id=? AND read=0" % NOTIFICATION_TABLE,
                    (str(session_id),))
        return max(0, self._write([op]))

    def unread_count(self, session_id) -> int:
        rows = self._query("SELECT COUNT(*) AS n FROM %s WHERE session_id=? AND read=0"
                           % NOTIFICATION_TABLE, (str(session_id),))
        return int(rows[0]["n"]) if rows else 0

    def notification_count(self, session_id) -> int:
        rows = self._query("SELECT COUNT(*) AS n FROM %s WHERE session_id=?"
                           % NOTIFICATION_TABLE, (str(session_id),))
        return int(rows[0]["n"]) if rows else 0

    # -- dedup state -----------------------------------------------------------
    def dedup_state(self, dedup_key) -> Optional[Dict[str, Any]]:
        rows = self._query("SELECT dedup_key, signature, severity, last_notified_at, "
                           "notification_id FROM %s WHERE dedup_key=?"
                           % NOTIFICATION_STATE_TABLE, (dedup_key,))
        if not rows:
            return None
        row = rows[0]
        return {"dedup_key": row["dedup_key"], "signature": row["signature"],
                "severity": row["severity"], "last_notified_at": row["last_notified_at"],
                "notification_id": row["notification_id"]}

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        return {"notification_id": row["notification_id"], "session_id": row["session_id"],
                "journey_id": row["journey_id"], "source_event_id": row["source_event_id"],
                "notification_type": row["notification_type"], "severity": row["severity"],
                "title": row["title"], "message": row["message"],
                "created_at": row["created_at"], "read": bool(row["read"])}


# --------------------------------------------------------------------------- relevance

def _journey_junctions(journey) -> set:
    """The junctions a monitored journey's stored route runs through."""
    junctions = set()
    for entry in journey.get("spine") or []:
        if isinstance(entry, dict) and entry.get("junction_id"):
            junctions.add(str(entry["junction_id"]))
    for node in journey.get("route_nodes") or []:
        junctions.add(str(node))
    return junctions


def _journey_position_m(journey) -> float:
    try:
        return float(journey.get("position_cum_m") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _journey_remaining_junctions(journey) -> set:
    """The junctions still AHEAD of the traveller's current position.

    A congestion alert must never point at a junction the car has already driven past. The journey
    agent stores its route as a ``spine`` of ``{junction_id, cum_m}`` and the car's distance along
    it as ``position_cum_m`` (both already maintained upstream), so the remaining route is the
    spine entries at or beyond that distance -- no routing is re-derived here. When there is no
    spine to measure against, the whole stored route is used, exactly as before this correction.
    """
    spine = journey.get("spine") or []
    if not spine:
        return _journey_junctions(journey)
    position = _journey_position_m(journey)
    ahead = set()
    for entry in spine:
        if not isinstance(entry, dict) or not entry.get("junction_id"):
            continue
        try:
            cum = float(entry.get("cum_m", 0.0) or 0.0)
        except (TypeError, ValueError):
            cum = 0.0
        if cum >= position:
            ahead.add(str(entry["junction_id"]))
    return ahead


def journey_in_progress(journey, decision) -> bool:
    """False once the journey has finished, so no *new* congestion alert is raised for it."""
    if str(journey.get("status") or "").strip().lower() in TERMINAL_JOURNEY_STATUSES:
        return False
    return str((decision or {}).get("action") or "") != ACTION_ARRIVED


def _incident_junctions(event) -> set:
    """The junctions an incident affects, from either shape the incident agent emits."""
    junctions = {str(junction) for junction in (event.get("affected_junction_ids") or [])}
    for affected in event.get("affected_junctions") or []:
        if isinstance(affected, dict) and affected.get("junction_id"):
            junctions.add(str(affected["junction_id"]))
    return junctions


def _decision_incident_ids(decision) -> set:
    return {str(event_id) for event_id in (decision.get("incident_ids") or [])}


def incident_relevant_to_journey(event, journey_junctions, decision_incident_ids) -> bool:
    """True only when the incident is actually on (or already associated with) this journey.

    Two independent pieces of evidence, both produced upstream: the incident's junctions overlap
    the journey's stored route, or the journey agent's own decision already listed the event id.
    With neither, the incident is someone else's traffic and is not personalized to this user.
    """
    event_id = event.get("event_id")
    if event_id and str(event_id) in (decision_incident_ids or set()):
        return True
    junctions = _incident_junctions(event)
    return bool(junctions and journey_junctions and junctions & journey_junctions)


# --------------------------------------------------------------------------- message builders

def _label(journey):
    return str(journey.get("label") or journey.get("journey_id") or "your journey")


def _event_word(event):
    return _EVENT_TYPE_WORDS.get(str(event.get("event_type") or ""), "congestion event")


def _incident_candidate(session_id, journey_id, journey, event) -> Tuple[Notification, str, str]:
    severity = str(event.get("severity") or "low")
    area = event.get("area") or event.get("anchor_name") or "your route"
    word = _event_word(event)
    parts = ["A %s %s is reported on %s, on your journey '%s'."
             % (severity, word, area, _label(journey))]
    current = event.get("avg_current_speed_kmh")
    free_flow = event.get("avg_free_flow_speed_kmh")
    if current is not None and free_flow is not None:
        parts.append("Traffic is around %s km/h against %s km/h free-flow."
                     % (_num(current), _num(free_flow)))
    count = event.get("affected_junction_count")
    if count:
        parts.append("%d junction(s) affected." % int(count))
    notification = Notification(
        session_id=session_id, notification_type=NOTIF_INCIDENT,
        title="Congestion on your route near %s" % area, message=" ".join(parts),
        journey_id=journey_id, source_event_id=event.get("event_id"), severity=severity)
    signature = "%s|%s|%s" % (event.get("event_type") or "", event.get("status") or "", severity)
    key = _dedup_key(session_id, journey_id, NOTIF_INCIDENT, event.get("event_id"))
    return notification, key, signature


def _resolved_candidate(session_id, journey_id, journey, event) -> Tuple[Notification, str, str]:
    area = event.get("area") or event.get("anchor_name") or "your route"
    word = _event_word(event)
    notification = Notification(
        session_id=session_id, notification_type=NOTIF_INCIDENT_RESOLVED,
        title="Congestion cleared near %s" % area,
        message="The %s near %s that affected your journey '%s' has cleared."
                % (word, area, _label(journey)),
        journey_id=journey_id, source_event_id=event.get("event_id"), severity=SEVERITY_INFO)
    signature = "resolved|%s" % (event.get("event_id") or "")
    key = _dedup_key(session_id, journey_id, NOTIF_INCIDENT_RESOLVED, event.get("event_id"))
    return notification, key, signature


def _delay_candidate(session_id, journey_id, journey, decision) -> Tuple[Notification, str, str]:
    delta = decision.get("eta_delta_min")
    eta = decision.get("eta_min")
    severity = _delay_severity(delta)
    notification = Notification(
        session_id=session_id, notification_type=NOTIF_JOURNEY_DELAY,
        title="Your journey is running late",
        message="Journey '%s' is now about %s min later than promised (ETA %s min)."
                % (_label(journey), _num(delta), _num(eta)),
        journey_id=journey_id, source_event_id=journey_id, severity=severity)
    bucket = int(round(float(delta or 0.0) / DELAY_SIGNATURE_BUCKET_MIN))
    signature = "delay|%s|%d" % (severity, bucket)
    key = _dedup_key(session_id, journey_id, NOTIF_JOURNEY_DELAY, journey_id)
    return notification, key, signature


def _alternative_candidate(session_id, journey_id, journey, decision) -> Tuple[Notification, str, str]:
    alternative = decision.get("alternative") or {}
    gain = alternative.get("gain_min")
    severity = _alternative_severity(gain)
    notification = Notification(
        session_id=session_id, notification_type=NOTIF_ROUTE_ALTERNATIVE,
        title="A faster route is available",
        message=("For journey '%s', an alternative route could save about %s min "
                 "(ETA %s min), turning off after %s km."
                 % (_label(journey), _num(gain), _num(alternative.get("eta_min")),
                    _num(alternative.get("turn_off_after_km"), 2))),
        journey_id=journey_id, source_event_id=journey_id, severity=severity)
    signature = "switch|%s|%s|%s" % (alternative.get("rank"), alternative.get("eta_min"), gain)
    key = _dedup_key(session_id, journey_id, NOTIF_ROUTE_ALTERNATIVE, journey_id)
    return notification, key, signature


def _completed_candidate(session_id, journey_id, journey, decision) -> Tuple[Notification, str, str]:
    achieved = decision.get("achieved_total_min")
    promised = decision.get("baseline_eta_min") or journey.get("baseline_eta_min")
    notification = Notification(
        session_id=session_id, notification_type=NOTIF_JOURNEY_COMPLETED,
        title="You have arrived",
        message="Journey '%s' is complete: about %s min against the %s min promised."
                % (_label(journey), _num(achieved), _num(promised)),
        journey_id=journey_id, source_event_id=journey_id, severity=SEVERITY_INFO)
    signature = "arrived|%s" % (achieved,)
    key = _dedup_key(session_id, journey_id, NOTIF_JOURNEY_COMPLETED, journey_id)
    return notification, key, signature


def journey_candidates(session_id, journey_id, journey, decision):
    """Turn one journey's latest decision into at most one notification candidate.

    The journey agent emits a single action per check, and that action is exactly the traveller
    facing signal -- so we translate the decision, never re-derive it.
    """
    action = (decision or {}).get("action")
    if action == ACTION_ARRIVED:
        return [_completed_candidate(session_id, journey_id, journey, decision)]
    if action == ACTION_RECOMMEND_SWITCH and decision.get("alternative"):
        return [_alternative_candidate(session_id, journey_id, journey, decision)]
    if action == ACTION_NOTIFY_DELAY:
        return [_delay_candidate(session_id, journey_id, journey, decision)]
    return []


# --------------------------------------------------------------------------- the engine

def _emit(store, notification, dedup_key, signature, now, cooldown_min, result):
    """Apply dedup + cooldown, then persist. Records what happened on ``result``.

    The check and the write are one atomic step inside the store (``add_guarded``), so two
    overlapping cycles cannot both emit the same alert. A store that predates ``add_guarded`` falls
    back to the older read-then-write behaviour, which is still correct for a single caller.
    """
    stamped = replace(notification, created_at=notification.created_at or iso(now))
    guarded = getattr(store, "add_guarded", None)
    if guarded is not None:
        notification_id, reason = guarded(stamped, dedup_key=dedup_key, signature=signature,
                                          now=now, cooldown_min=cooldown_min)
        result["reasons"][reason] = result["reasons"].get(reason, 0) + 1
        if notification_id is None:
            if reason in ("duplicate", "cooldown"):
                result["suppressed"] += 1
            else:
                result["skipped"] += 1
            return None
    else:
        existing = store.dedup_state(dedup_key)
        should, reason = evaluate_dedup(existing, signature, notification.severity, now=now,
                                        cooldown_min=cooldown_min)
        result["reasons"][reason] = result["reasons"].get(reason, 0) + 1
        if not should:
            result["suppressed"] += 1
            return None
        notification_id = store.add(stamped, dedup_key=dedup_key, signature=signature)
        if not notification_id:
            result["skipped"] += 1
            return None
    stored = replace(stamped, notification_id=notification_id)
    result["generated"] += 1
    result["notifications"].append(stored.as_dict())
    return notification_id


def _latest_incident_time(snapshot):
    return snapshot.get("made_at") or snapshot.get("timestamp")


def _journey_decision_time(journey, decision):
    return (decision or {}).get("updated_at") or journey.get("updated_at")


def process_notifications(sources, store, config=None, now=None) -> Dict[str, Any]:
    """One notification pass: read upstream data, match it to monitored journeys, persist.

    Independently callable -- a background loop or a future Kafka consumer can call this on a
    timer. It never starts a thread and never imports an LLM. ``sources`` is any object with
    ``incidents()`` and ``journeys()`` (``LiveDataSources`` in production, a fake in tests).

    The result summarises what happened: how many notifications were generated, how many were
    suppressed as duplicates/cooldown, and why.
    """
    from webapp.interaction_agent.config import InteractionConfig

    config = config or InteractionConfig.from_env()
    now = now or utc_now()
    result: Dict[str, Any] = {"enabled": bool(config.notifications_enabled), "generated": 0,
                              "suppressed": 0, "skipped": 0, "scanned_journeys": 0,
                              "notifications": [], "reasons": {},
                              "incidents_fresh": False, "journeys_fresh": 0}
    if not config.notifications_enabled:
        return result
    if store is None or not store.enabled or not store.connect():
        result["enabled"] = False
        return result

    subscriptions = store.subscriptions()
    stale_after = float(getattr(sources, "stale_after_min", config.stale_after_min))

    # -- incidents: fresh snapshot only; a stale one must not become a current alert --
    incident_events: List[Dict[str, Any]] = []
    resolved_events: List[Dict[str, Any]] = []
    snapshot = sources.incidents()
    if snapshot is not None:
        age = _age_minutes(_latest_incident_time(snapshot), now)
        if age is not None and age <= stale_after:
            result["incidents_fresh"] = True
            incident_events = [event for event in (snapshot.get("active_events") or [])
                               if isinstance(event, dict)
                               and str(event.get("status") or "") != "resolved"]
            resolved_events = [event for event in (snapshot.get("recent_resolved") or [])
                               if isinstance(event, dict)]

    # Freshness describes the published incident snapshot, not whether anyone is subscribed.
    # Keep it observable in runtime status even when there is no notification recipient.
    if not subscriptions:
        return result

    # -- journeys, keyed by id so a subscription can be resolved to its current state --
    journeys = {}
    for journey in (sources.journeys() or []):
        if isinstance(journey, dict) and journey.get("journey_id"):
            journeys[journey["journey_id"]] = journey

    cooldown = float(config.notification_cooldown_min)

    for subscription in subscriptions:
        session_id = subscription["session_id"]
        journey_id = subscription["journey_id"]
        journey = journeys.get(journey_id)
        if journey is None:
            result["skipped"] += 1                       # not being watched any more
            continue
        result["scanned_journeys"] += 1

        decision = journey.get("decision") or {}
        decision_ids = _decision_incident_ids(decision)

        # Congestion alerts only concern a journey still being driven, and only the junctions ahead
        # of the traveller -- never one already passed, never a journey already finished.
        if result["incidents_fresh"] and journey_in_progress(journey, decision):
            remaining_junctions = _journey_remaining_junctions(journey)
            for event in incident_events:
                if not event_is_fresh(event, now, stale_after):
                    continue
                if not incident_relevant_to_journey(event, remaining_junctions, decision_ids):
                    continue
                notification, key, signature = _incident_candidate(session_id, journey_id,
                                                                   journey, event)
                _emit(store, notification, key, signature, now, cooldown, result)
            for event in resolved_events:
                if not event_is_fresh(event, now, stale_after):
                    continue
                if not incident_relevant_to_journey(event, remaining_junctions, decision_ids):
                    continue
                notification, key, signature = _resolved_candidate(session_id, journey_id,
                                                                   journey, event)
                _emit(store, notification, key, signature, now, cooldown, result)

        decision_age = _age_minutes(_journey_decision_time(journey, decision), now)
        if decision_age is not None and decision_age <= stale_after:
            result["journeys_fresh"] += 1
            for notification, key, signature in journey_candidates(session_id, journey_id,
                                                                   journey, decision):
                _emit(store, notification, key, signature, now, cooldown, result)

    return result


def build_notification_store(config=None) -> NotificationStore:
    """Build the store from configuration (or the environment when no config is given)."""
    if config is None:
        from webapp.interaction_agent.config import InteractionConfig
        config = InteractionConfig.from_env()
    return NotificationStore.from_config(config)
