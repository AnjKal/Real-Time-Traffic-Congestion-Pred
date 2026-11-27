"""Local-demo session ownership for the interaction API (Phase 4).

The web app has no authentication and no user accounts (Phase 4 does not add any). That is fine
for a local demo, but it must not mean "anyone who guesses a session id can read that
conversation". So a conversation is owned by an *opaque bearer credential* the server issues:

  * ``POST /api/chat`` with no session starts a conversation and returns a fresh, random token
    (``secrets.token_urlsafe``) exactly once, alongside the session id.
  * Every later request that touches that conversation's history, subscriptions or notifications
    must present the token (``Authorization: Bearer <token>``). The session id alone proves
    nothing and is never accepted as ownership.

Only a SHA-256 hash of each token is stored, never the token itself, so a leak of the SQLite file
does not hand out live credentials. Like the other Phase 2/3 stores this module is lazy (opening
happens on first use), additive (``CREATE TABLE IF NOT EXISTS`` only, in the same shared database),
thread-safe (one lock, WAL) and it degrades instead of raising: if the database cannot be opened
the credentials still work for the lifetime of the process, held in memory.

This is a demo mechanism, not production auth. See the module docstring notes in
``webapp/interaction_api.py`` for the remaining limitations.
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
import threading
from typing import Dict, List, Optional

from webapp.journey_memory import iso, utc_now

LOG = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The same shared database the conversation memory and notification store use. Additive schema only.
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "live", "interaction_memory.sqlite")

BUSY_TIMEOUT_MS = 5000
SCHEMA_VERSION = 1

CREDENTIAL_TABLE = "interaction_session_credentials"
CREDENTIAL_META_TABLE = "interaction_session_meta"

TOKEN_BYTES = 32                     # token_urlsafe(32) -> ~43 url-safe chars, 256 bits of entropy

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS {meta} (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS {creds} (
    token_hash   TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    last_seen_at TEXT,
    revoked      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_{creds}_session ON {creds}(session_id);
""".format(meta=CREDENTIAL_META_TABLE, creds=CREDENTIAL_TABLE)


def hash_token(token: str) -> str:
    """The stored form of a bearer token. Never store or log the token itself."""
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def new_token() -> str:
    """A fresh, high-entropy, URL-safe session credential."""
    return secrets.token_urlsafe(TOKEN_BYTES)


class SessionGuard:
    """Issues and verifies opaque session credentials.

    ``enabled=False`` makes every method a no-op that reports failure, so the call sites stay
    identical if a deployment wants to disable the mechanism (and accept its consequences).
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, enabled: bool = True):
        self.db_path = db_path
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        # Always-authoritative in-process map (token hash -> session id). Mirrored from the
        # database on first use of a token, so resolution is fast and works even with no database.
        self._memory: Dict[str, str] = {}

    # -- lifecycle -------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._conn is not None

    def connect(self) -> bool:
        """Open the database if it is not already open. Returns whether persistence is usable."""
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
                connection.executescript(SCHEMA_SQL)
                connection.execute("INSERT OR IGNORE INTO %s(key, value) VALUES('schema_version', ?)"
                                   % CREDENTIAL_META_TABLE, (str(SCHEMA_VERSION),))
                connection.commit()
            except (sqlite3.Error, OSError) as exc:
                LOG.warning("session credentials unavailable at %s: %s", self.db_path, exc)
                if connection is not None:
                    try:
                        connection.close()
                    except sqlite3.Error:
                        pass
                self._conn = None
                return False
            self._conn = connection
            return True

    def _query(self, sql, params=()) -> List[sqlite3.Row]:
        if not self._ready():
            return []
        try:
            with self._lock:
                return self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            LOG.warning("session credential read failed: %s", exc)
            return []

    def _execute(self, sql, params=()) -> int:
        if not self._ready():
            return 0
        try:
            with self._lock:
                cursor = self._conn.execute(sql, params)
                self._conn.commit()
                return cursor.rowcount
        except sqlite3.Error as exc:
            LOG.warning("session credential write failed: %s", exc)
            return 0

    # -- issue / verify --------------------------------------------------------
    def issue(self, session_id: str) -> Optional[str]:
        """Mint a credential for ``session_id``. Returns the token (shown to the caller once)."""
        sid = str(session_id or "").strip()
        if not self.enabled or not sid:
            return None
        token = new_token()
        digest = hash_token(token)
        with self._lock:
            self._memory[digest] = sid
        self._execute("INSERT OR REPLACE INTO %s(token_hash, session_id, created_at, last_seen_at, "
                      "revoked) VALUES(?,?,?,?,0)" % CREDENTIAL_TABLE,
                      (digest, sid, iso(utc_now()), iso(utc_now())))
        return token

    def resolve(self, token: str) -> Optional[str]:
        """The session a valid, unrevoked token belongs to, or None."""
        if not self.enabled or not token:
            return None
        digest = hash_token(token)
        with self._lock:
            cached = self._memory.get(digest)
        if cached:
            self._touch(digest)
            return cached
        rows = self._query("SELECT session_id FROM %s WHERE token_hash=? AND revoked=0 LIMIT 1"
                           % CREDENTIAL_TABLE, (digest,))
        if not rows:
            return None
        sid = rows[0]["session_id"]
        with self._lock:
            self._memory[digest] = sid
        self._touch(digest)
        return sid

    def verify(self, token: str, session_id: str) -> bool:
        """True only when the token is valid AND belongs to this exact session."""
        resolved = self.resolve(token)
        return bool(resolved) and resolved == str(session_id or "").strip()

    def revoke(self, token: str) -> bool:
        """Invalidate a credential. Returns whether it existed."""
        if not token:
            return False
        digest = hash_token(token)
        with self._lock:
            existed = self._memory.pop(digest, None) is not None
        removed = self._execute("UPDATE %s SET revoked=1 WHERE token_hash=?" % CREDENTIAL_TABLE,
                                (digest,))
        return existed or removed > 0

    def credential_count(self, session_id: str) -> int:
        """How many live credentials a session has (for tests and diagnostics)."""
        rows = self._query("SELECT COUNT(*) AS n FROM %s WHERE session_id=? AND revoked=0"
                           % CREDENTIAL_TABLE, (str(session_id or ""),))
        return int(rows[0]["n"]) if rows else 0

    # -- internals -------------------------------------------------------------
    def _touch(self, digest: str) -> None:
        self._execute("UPDATE %s SET last_seen_at=? WHERE token_hash=?" % CREDENTIAL_TABLE,
                      (iso(utc_now()), digest))


def build_session_guard(db_path: str = DEFAULT_DB_PATH, enabled: bool = True) -> SessionGuard:
    """Build the guard for the shared interaction database."""
    return SessionGuard(db_path=db_path, enabled=enabled)
