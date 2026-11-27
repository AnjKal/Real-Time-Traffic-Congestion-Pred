"""Persistent conversation memory for the User Interaction Agent.

The agent forgets everything between turns unless something remembers it. This module is that
something: a small SQLite store of conversation sessions and their messages, plus the little bit of
context that makes a follow-up question resolvable (which journey, which place, which topic was
last discussed).

Design choices worth knowing:

  * Only ``user`` and ``assistant`` TEXT is stored. Tool results are deliberately NOT persisted --
    a tool result is evidence about the traffic at the moment it ran, and replaying an old one as
    if it were current would be a fabrication. Every new turn runs its tools again.
  * Tables are namespaced with the ``interaction_`` prefix. The configured file already holds an
    unrelated schema (a LangGraph checkpoint store: ``checkpoints`` / ``writes``), so the schema is
    added, never replaced: ``CREATE TABLE IF NOT EXISTS`` only, no drops, no alters of existing
    tables. That is the whole migration.
  * Opening is lazy. Constructing the object touches nothing, so an agent built for tests or for a
    stateless turn never creates a database file.
  * Every operation degrades instead of raising. If SQLite is missing, locked or the path is bad,
    the memory silently reports "nothing remembered" and the agent answers from live tool data
    alone, exactly as it did before this module existed.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from typing import Any, Dict, List, Optional

from webapp.journey_memory import iso, utc_now

LOG = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "live", "interaction_memory.sqlite")

DEFAULT_MAX_MESSAGES = 20
MAX_CONTEXT_ITEMS = 5          # journey ids / places kept per session; enough for a follow-up
SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 5000         # wait for a lock rather than failing a concurrent writer

ROLES = ("user", "assistant")

SESSION_TABLE = "interaction_sessions"
MESSAGE_TABLE = "interaction_messages"
META_TABLE = "interaction_meta"

# Keys whose values are small ordered lists; update_context merges these instead of replacing.
LIST_CONTEXT_KEYS = ("journey_ids", "places")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS {meta} (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS {sessions} (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    context    TEXT NOT NULL DEFAULT '{{}}'
);
CREATE TABLE IF NOT EXISTS {messages} (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES {sessions}(session_id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    tool_calls TEXT
);
CREATE INDEX IF NOT EXISTS idx_{messages}_session ON {messages}(session_id, id);
""".format(meta=META_TABLE, sessions=SESSION_TABLE, messages=MESSAGE_TABLE)


class ConversationMemory:
    """SQLite-backed sessions, messages and context for one agent.

    ``enabled=False`` turns every method into a no-op that keeps the call sites unchanged, which is
    how the agent runs when ``INTERACTION_MEMORY_ENABLED`` is switched off.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, max_messages: int = DEFAULT_MAX_MESSAGES,
                 enabled: bool = True):
        self.db_path = db_path
        self.max_messages = max(1, int(max_messages))
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    # -- lifecycle -------------------------------------------------------------
    @classmethod
    def from_config(cls, config) -> "ConversationMemory":
        return cls(db_path=config.memory_db_path, max_messages=config.memory_max_messages,
                   enabled=config.memory_enabled)

    @property
    def available(self) -> bool:
        """True once the database is open. Reading it does not open anything."""
        return self._conn is not None

    def connect(self) -> bool:
        """Open the database if it is not already open. Returns whether memory is usable."""
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
        """Lazily open the database. False means "behave as if nothing is remembered"."""
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
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA busy_timeout=%d" % BUSY_TIMEOUT_MS)
                self._create_schema(connection)
            except (sqlite3.Error, OSError) as exc:
                LOG.warning("interaction memory unavailable at %s: %s", self.db_path, exc)
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
                           % META_TABLE, (str(SCHEMA_VERSION),))
        connection.commit()

    @staticmethod
    def _write(connection, sql, params=()) -> int:
        cursor = connection.execute(sql, params)
        connection.commit()
        return cursor.rowcount

    def _execute(self, sql, params=()) -> int:
        if not self._ready():
            return 0
        try:
            with self._lock:
                return self._write(self._conn, sql, params)
        except sqlite3.Error as exc:
            LOG.warning("interaction memory write failed: %s", exc)
            return 0

    def _query(self, sql, params=()) -> List[sqlite3.Row]:
        if not self._ready():
            return []
        try:
            with self._lock:
                return self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            LOG.warning("interaction memory read failed: %s", exc)
            return []

    # -- sessions --------------------------------------------------------------
    @staticmethod
    def new_session_id() -> str:
        """A unique, URL-safe conversation id."""
        return "cs-" + uuid.uuid4().hex[:12]

    def create_session(self, session_id: Optional[str] = None, context=None) -> str:
        """Create a session if it does not exist; return its id. Existing sessions are untouched."""
        sid = str(session_id).strip() if session_id else self.new_session_id()
        if not sid:
            sid = self.new_session_id()
        if self._ready():
            now = iso(utc_now())
            self._execute(
                "INSERT OR IGNORE INTO %s(session_id, created_at, updated_at, context) "
                "VALUES(?,?,?,?)" % SESSION_TABLE,
                (sid, now, now, json.dumps(self._clean_context(context), sort_keys=True)))
        return sid

    def ensure_session(self, session_id: Optional[str] = None, context=None) -> str:
        """Return an existing session id, or create the session if it is new."""
        if session_id and self.get_session(session_id) is not None:
            return session_id
        return self.create_session(session_id, context=context)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        rows = self._query("SELECT session_id, created_at, updated_at, context FROM %s "
                           "WHERE session_id=?" % SESSION_TABLE, (session_id,))
        if not rows:
            return None
        row = rows[0]
        return {"session_id": row["session_id"], "created_at": row["created_at"],
                "updated_at": row["updated_at"], "context": self._load_context(row["context"])}

    def session_ids(self) -> List[str]:
        rows = self._query("SELECT session_id FROM %s ORDER BY created_at, session_id"
                           % SESSION_TABLE)
        return [row["session_id"] for row in rows]

    def clear_session(self, session_id: str) -> int:
        """Forget one conversation: delete its messages and reset its context. Returns the count."""
        deleted = self._execute("DELETE FROM %s WHERE session_id=?" % MESSAGE_TABLE, (session_id,))
        self._execute("UPDATE %s SET context='{}', updated_at=? WHERE session_id=?"
                      % SESSION_TABLE, (iso(utc_now()), session_id))
        return max(0, deleted)

    # -- messages --------------------------------------------------------------
    def add_message(self, session_id: str, role: str, content: str, tool_calls=None) -> bool:
        """Append one message. Returns whether it was stored."""
        if role not in ROLES:
            raise ValueError("role must be one of %s, got %r" % (list(ROLES), role))
        if not self._ready():
            return False
        self.create_session(session_id)
        now = iso(utc_now())
        payload = json.dumps(tool_calls) if tool_calls else None
        written = self._execute(
            "INSERT INTO %s(session_id, role, content, created_at, tool_calls) VALUES(?,?,?,?,?)"
            % MESSAGE_TABLE, (session_id, role, str(content or ""), now, payload))
        self._execute("UPDATE %s SET updated_at=? WHERE session_id=?" % SESSION_TABLE,
                      (now, session_id))
        return written > 0

    def recent_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """The newest ``limit`` messages, returned oldest-first so they read chronologically."""
        count = self.max_messages if limit is None else int(limit)
        if count <= 0:
            return []
        rows = self._query(
            "SELECT id, role, content, created_at, tool_calls FROM %s WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?" % MESSAGE_TABLE, (session_id, count))
        return [{"id": row["id"], "role": row["role"], "content": row["content"],
                 "created_at": row["created_at"],
                 "tool_calls": self._load_json(row["tool_calls"])}
                for row in reversed(rows)]

    def history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Just ``role``/``content`` pairs, ready to hand to an LLM provider."""
        return [{"role": row["role"], "content": row["content"]}
                for row in self.recent_messages(session_id, limit)]

    def message_count(self, session_id: str) -> int:
        rows = self._query("SELECT COUNT(*) AS n FROM %s WHERE session_id=?" % MESSAGE_TABLE,
                           (session_id,))
        return int(rows[0]["n"]) if rows else 0

    # -- context ---------------------------------------------------------------
    def get_context(self, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        return dict(session["context"]) if session else {}

    def update_context(self, session_id: str, **changes) -> Dict[str, Any]:
        """Merge changes into a session's context and return the result.

        List-valued keys (journey ids, places) are merged newest-first, de-duplicated and capped, so
        remembering a new journey never forgets the one still being discussed.
        """
        current = self.get_context(session_id)
        for key, value in changes.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                merged: List[Any] = []
                existing = current.get(key)
                existing = existing if isinstance(existing, list) else []
                for item in list(value) + existing:
                    if item not in merged:
                        merged.append(item)
                current[key] = merged[:MAX_CONTEXT_ITEMS]
            else:
                current[key] = value
        if self._ready():
            self.create_session(session_id)
            self._execute("UPDATE %s SET context=?, updated_at=? WHERE session_id=?"
                          % SESSION_TABLE,
                          (json.dumps(current, sort_keys=True), iso(utc_now()), session_id))
        return current

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _clean_context(context) -> Dict[str, Any]:
        return dict(context) if isinstance(context, dict) else {}

    @staticmethod
    def _load_context(raw) -> Dict[str, Any]:
        value = ConversationMemory._load_json(raw)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _load_json(raw):
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None


def build_memory(config=None) -> ConversationMemory:
    """Build memory from configuration (or the environment when no config is given)."""
    if config is None:
        from webapp.interaction_agent.config import InteractionConfig
        config = InteractionConfig.from_env()
    return ConversationMemory.from_config(config)


def history_for(memory: ConversationMemory, session_id: Optional[str],
                limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Safe history load: an empty list when memory is off, unknown or broken."""
    if not session_id or memory is None or not memory.enabled:
        return []
    return memory.history(session_id, limit)
