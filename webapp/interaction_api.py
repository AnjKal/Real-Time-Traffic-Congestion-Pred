"""HTTP surface for the User Interaction Agent (Phase 4).

Phase 1 answered questions, Phase 2 remembered them, Phase 3 spoke first. Phase 4 puts all of that
behind the *existing* FastAPI app (``webapp/server.py``) -- no second server, no second framework,
no second conversation implementation. The endpoints here only translate HTTP into calls on the
already-built agent, memory and notification store, and translate their replies back.

What this module owns
---------------------
* :class:`InteractionServices` -- one object holding the config, the agent, the conversation
  memory, the notification store, the session guard and the notification runtime. It is built once
  (from ``webapp/server.py``) and started/stopped by the app's own lifespan.
* :func:`create_interaction_router` -- the ``/api/chat``, history, notification, subscription and
  status routes, returned as an :class:`~fastapi.APIRouter` the app includes.

Session ownership (Section 6)
-----------------------------
The app has no login, so a session is owned by an *opaque bearer credential* issued in the same
response that creates it. Every later request that touches a conversation's history, subscriptions
or notifications must present ``Authorization: Bearer <token>``; the session id alone is never
accepted as proof of ownership. The token is stored only as a SHA-256 hash (``security.py``).

No secrets, no traces
---------------------
Nothing here returns the system prompt, a credential, a raw exception trace or an internal
filesystem path. Provider failure is surfaced as the agent's own message plus an ``error`` string
the agent already produced; the endpoint never echoes a traceback.

Remaining production limitations (documented, not fixed in this phase -- see the report):
  * credentials are bearer tokens with no expiry, rotation or revocation UI;
  * there is no per-user identity, so "ownership" means "holds the token", not "is user X";
  * tokens are stored in the same SQLite file as the conversation, so filesystem access to that
    file is equivalent to access to every conversation;
  * there is no rate limiting, CSRF protection or TLS termination here (a reverse proxy would
    provide those);
  * the status endpoint is unauthenticated by design (it exposes no session data).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from webapp.interaction_agent import (ConversationMemory, InteractionAgent, InteractionConfig,
                                      LiveDataSources, NotificationRuntime, NotificationStore,
                                      SessionGuard, build_default_registry)

LOG = logging.getLogger(__name__)

# A message longer than this is rejected rather than truncated: the agent must never answer a
# question the traveller did not actually finish asking.
MAX_MESSAGE_CHARS = 2000

DEFAULT_HISTORY_LIMIT = 50
MAX_HISTORY_LIMIT = 200
DEFAULT_NOTIFICATION_LIMIT = 50
MAX_NOTIFICATION_LIMIT = 200

# Session ids the memory module creates look like "cs-<hex>"; anything else is a malformed id and
# is rejected before it can be used as a lookup key.
SESSION_ID_RE = re.compile(r"^cs-[A-Za-z0-9_-]{4,64}$")

NOT_AVAILABLE = "The interaction agent is not available."


# --------------------------------------------------------------------------- services container

class InteractionServices:
    """Everything the interaction API needs, built once and owned by the app's lifespan.

    Constructing this opens no database and makes no AWS call: the memory, store and guard all open
    lazily, and the default LLM provider is the deterministic mock. ``start``/``stop`` are what the
    lifespan calls, and neither ever raises -- a failure to open a store or start a worker must not
    take the whole web application down.
    """

    def __init__(self, config: Optional[InteractionConfig] = None, agent=None, sources=None,
                 memory=None, store=None, guard=None, runtime=None):
        self.config = config or InteractionConfig.from_env()
        self.memory = memory if memory is not None else ConversationMemory.from_config(self.config)
        self.store = store if store is not None else NotificationStore.from_config(self.config)
        self.guard = (guard if guard is not None
                      else SessionGuard(db_path=self.config.memory_db_path))
        self.sources = sources if sources is not None else LiveDataSources.from_config(self.config)
        self.agent = agent if agent is not None else InteractionAgent(
            config=self.config, memory=self.memory,
            registry=build_default_registry(self.sources))
        self.runtime = runtime if runtime is not None else NotificationRuntime(
            self.sources, self.store, config=self.config)
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        """Open resources and start the notification runtime. Never raises."""
        for name, resource in (("memory", self.memory), ("store", self.store),
                               ("guard", self.guard)):
            try:
                resource.connect()
            except Exception:                                     # noqa: BLE001 - must not be fatal
                LOG.exception("interaction %s failed to open", name)
        try:
            self.runtime.start()
        except Exception:                                         # noqa: BLE001 - must not be fatal
            LOG.exception("interaction notification runtime failed to start")
        self._started = True

    def stop(self) -> None:
        """Stop the notification runtime and close resources. Never raises."""
        try:
            self.runtime.stop()
        except Exception:                                         # noqa: BLE001 - must not be fatal
            LOG.exception("interaction notification runtime failed to stop")
        for name, resource in (("memory", self.memory), ("store", self.store),
                               ("guard", self.guard)):
            try:
                resource.close()
            except Exception:                                     # noqa: BLE001 - must not be fatal
                LOG.exception("interaction %s failed to close", name)
        self._started = False


def build_interaction_services(config: Optional[InteractionConfig] = None) -> InteractionServices:
    """Build the container from configuration (or the environment when none is given)."""
    return InteractionServices(config=config)


# --------------------------------------------------------------------------- small helpers

def _clamp(value, default, maximum, name="limit") -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise HTTPException(400, "%s must be a whole number" % name)
    if parsed <= 0:
        raise HTTPException(400, "%s must be positive" % name)
    return min(parsed, maximum)


def _availability_summary(reply) -> Dict[str, Any]:
    """A compact, non-sensitive view of which sources answered and how fresh they were."""
    sources: Dict[str, str] = {}
    for call in reply.tool_calls:
        sources[call.name] = call.availability or "unknown"
    return {
        "sources": sources,
        "live": any(call.availability == "live" for call in reply.tool_calls),
        "degraded": any(call.availability in ("missing", "stale", "unavailable")
                        for call in reply.tool_calls),
    }


# --------------------------------------------------------------------------- request models

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ReadRequest(BaseModel):
    session_id: str
    notification_id: Optional[str] = None


class SubscriptionRequest(BaseModel):
    session_id: str
    journey_id: str


# --------------------------------------------------------------------------- the router

def create_interaction_router(services: Optional[InteractionServices]) -> APIRouter:
    """All Phase 4 routes. ``services`` may be None (the container failed to build), in which case
    every data endpoint answers 503 and the status endpoint reports unavailability."""
    router = APIRouter()

    def _require() -> InteractionServices:
        if services is None:
            raise HTTPException(503, NOT_AVAILABLE)
        return services

    def _token(authorization: Optional[str]) -> Optional[str]:
        if not authorization:
            return None
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
        return None

    def _owned(svc: InteractionServices, session_id: Optional[str],
               authorization: Optional[str]) -> str:
        """Return the session id only when the caller can prove ownership, else raise."""
        sid = str(session_id or "").strip()
        if not sid or not SESSION_ID_RE.match(sid):
            raise HTTPException(400, "a valid session_id is required")
        token = _token(authorization)
        if not token:
            raise HTTPException(401, "This conversation requires its session token: send "
                                     "Authorization: Bearer <token>.")
        resolved = svc.guard.resolve(token)
        if not resolved:
            raise HTTPException(401, "The session token is not valid.")
        if resolved != sid:
            raise HTTPException(403, "The session token does not belong to that conversation.")
        return sid

    # --------------------------------------------------------------- session
    @router.post("/api/interaction/session")
    def create_session():
        """Mint a fresh conversation session and its bearer credential.

        ``POST /api/chat`` already issues these for a first message. The notification panel and
        the journey monitor, however, must be usable *before* any question is asked, so the page
        needs a session up front to subscribe and poll. This endpoint is purely additive: it
        starts an empty conversation and returns the same ``{session_id, session_token}`` pair a
        first chat turn would, and it stores no message, so the history stays empty.
        """
        svc = _require()
        session_id = svc.memory.create_session()
        token = svc.guard.issue(session_id)
        return {"session_id": session_id, "session_token": token,
                "memory": {"enabled": bool(svc.memory.enabled)},
                "notifications": {"enabled": bool(svc.store.enabled)}}

    # --------------------------------------------------------------- chat
    @router.post("/api/chat")
    def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
        svc = _require()
        text = (req.message or "").strip()
        if not text:
            raise HTTPException(400, "message must not be empty")
        if len(text) > MAX_MESSAGE_CHARS:
            raise HTTPException(413, "message is too long (max %d characters)" % MAX_MESSAGE_CHARS)

        issued_token = None
        if req.session_id:
            session_id = _owned(svc, req.session_id, authorization)
        else:
            session_id = svc.memory.create_session()
            issued_token = svc.guard.issue(session_id)

        reply = svc.agent.answer(text, session_id=session_id)
        payload = reply.as_dict()
        payload["session_token"] = issued_token
        payload["response"] = reply.text
        payload["tools_used"] = [call.as_dict() for call in reply.tool_calls]
        payload["data"] = _availability_summary(reply)
        return payload

    # --------------------------------------------------------------- conversation history
    @router.get("/api/chat/history")
    def chat_history(session_id: str = Query(...), limit: Optional[int] = Query(None),
                     authorization: Optional[str] = Header(None)):
        svc = _require()
        sid = _owned(svc, session_id, authorization)
        bounded = _clamp(limit, DEFAULT_HISTORY_LIMIT, MAX_HISTORY_LIMIT)
        messages = svc.memory.recent_messages(sid, bounded) if svc.memory.enabled else []
        return {"session_id": sid, "enabled": bool(svc.memory.enabled), "limit": bounded,
                "count": len(messages), "messages": messages}

    @router.delete("/api/chat/history")
    def clear_history(session_id: str = Query(...), authorization: Optional[str] = Header(None)):
        svc = _require()
        sid = _owned(svc, session_id, authorization)
        # Clearing is scoped to this conversation's own messages and context only: it never touches
        # another session, a notification, a subscription or the third-party LangGraph tables.
        cleared = svc.memory.clear_session(sid) if svc.memory.enabled else 0
        return {"session_id": sid, "cleared": cleared}

    # --------------------------------------------------------------- notifications
    @router.get("/api/notifications")
    def list_notifications(session_id: str = Query(...), unread_only: bool = Query(False),
                           limit: Optional[int] = Query(None),
                           authorization: Optional[str] = Header(None)):
        svc = _require()
        sid = _owned(svc, session_id, authorization)
        bounded = _clamp(limit, DEFAULT_NOTIFICATION_LIMIT, MAX_NOTIFICATION_LIMIT)
        rows = svc.store.list_notifications(sid, unread_only=unread_only, limit=bounded)
        return {"session_id": sid, "enabled": bool(svc.store.enabled),
                "unread_only": bool(unread_only), "limit": bounded,
                "count": len(rows), "unread": svc.store.unread_count(sid),
                "notifications": rows}

    @router.get("/api/notifications/unread-count")
    def unread_count(session_id: str = Query(...), authorization: Optional[str] = Header(None)):
        svc = _require()
        sid = _owned(svc, session_id, authorization)
        return {"session_id": sid, "enabled": bool(svc.store.enabled),
                "unread": svc.store.unread_count(sid)}

    @router.post("/api/notifications/read")
    def mark_read(req: ReadRequest, authorization: Optional[str] = Header(None)):
        svc = _require()
        sid = _owned(svc, req.session_id, authorization)
        if not svc.store.enabled:
            return {"session_id": sid, "enabled": False,
                    "notification_id": req.notification_id, "marked": 0}
        if req.notification_id:
            if svc.store.get(sid, req.notification_id) is None:
                raise HTTPException(404, "no notification '%s' for this conversation"
                                         % req.notification_id)
            marked = svc.store.mark_read(sid, req.notification_id)
            return {"session_id": sid, "enabled": True,
                    "notification_id": req.notification_id, "marked": marked}
        marked = svc.store.mark_read(sid)
        return {"session_id": sid, "enabled": True, "notification_id": None, "marked": marked}

    # --------------------------------------------------------------- subscriptions
    @router.get("/api/subscriptions")
    def list_subscriptions(session_id: str = Query(...),
                           authorization: Optional[str] = Header(None)):
        svc = _require()
        sid = _owned(svc, session_id, authorization)
        rows = svc.store.subscriptions(sid)
        return {"session_id": sid, "enabled": bool(svc.store.enabled),
                "count": len(rows), "subscriptions": rows}

    @router.post("/api/subscriptions")
    def subscribe(req: SubscriptionRequest, authorization: Optional[str] = Header(None)):
        svc = _require()
        sid = _owned(svc, req.session_id, authorization)
        journey_id = str(req.journey_id or "").strip()
        if not journey_id:
            raise HTTPException(400, "journey_id is required")
        # Validated against the journeys the journey agent is already watching. This is a read:
        # subscribing never changes that agent's state or its route selection.
        try:
            known = svc.sources.journey(journey_id)
        except Exception:                                         # noqa: BLE001 - treat as unknown
            LOG.warning("could not validate journey %s against the journey agent", journey_id)
            known = None
        if known is None:
            raise HTTPException(404, "no journey '%s' is being watched" % journey_id)
        created = svc.store.subscribe(sid, journey_id)
        return {"session_id": sid, "journey_id": journey_id, "subscribed": True,
                "created": bool(created), "enabled": bool(svc.store.enabled)}

    @router.delete("/api/subscriptions/{journey_id}")
    def unsubscribe(journey_id: str, session_id: str = Query(...),
                    authorization: Optional[str] = Header(None)):
        svc = _require()
        sid = _owned(svc, session_id, authorization)
        removed = svc.store.unsubscribe(sid, str(journey_id or "").strip())
        return {"session_id": sid, "journey_id": journey_id, "removed": bool(removed)}

    # --------------------------------------------------------------- status
    @router.get("/api/interaction/status")
    def interaction_status():
        if services is None:
            return {"available": False, "agent": {"available": False}, "detail": NOT_AVAILABLE}
        svc = services
        config = svc.config
        runtime_state = svc.runtime.status()
        return {
            "available": True,
            "agent": {"available": True, "name": getattr(svc.agent, "name", "interaction"),
                      "provider": config.llm_provider,
                      "tools": svc.agent.tool_names() if svc.agent is not None else []},
            "llm": {"provider": config.llm_provider, "model": config.bedrock_model_id,
                    "uses_bedrock": config.uses_bedrock},
            "memory": {"enabled": bool(svc.memory.enabled),
                       "available": bool(svc.memory.available)},
            "notifications": {"enabled": bool(svc.store.enabled),
                              "available": bool(svc.store.available),
                              "cooldown_min": config.notification_cooldown_min},
            "notification_runtime": runtime_state,
            "last_processing_cycle_at": runtime_state.get("last_cycle_at"),
            "data_sources": _data_sources(svc.sources),
            "stale_after_min": config.stale_after_min,
        }

    return router


def _data_sources(sources) -> Dict[str, Any]:
    """Availability of the published data the agent reads, without leaking filesystem paths."""
    def exists(attr) -> bool:
        path = getattr(sources, attr, None)
        if not path:
            return False
        try:
            return bool(sources.source_available(path))
        except Exception:                                         # noqa: BLE001 - report as absent
            return False

    return {"model_service_url": getattr(sources, "model_service_url", None),
            "incidents_published": exists("incidents_path"),
            "features_published": exists("features_path"),
            "journeys_published": exists("state_path")}
