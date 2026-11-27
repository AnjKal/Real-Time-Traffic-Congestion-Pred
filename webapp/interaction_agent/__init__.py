"""The User Interaction Agent (Phase 1-4).

The third agent in the system: it talks to the traveller. It does not compute congestion, route or
predict anything -- it reads what the incident monitoring agent, the journey intelligence agent and
the feature aggregator have already published, and answers questions about it (Phase 1), remembers
the conversation (Phase 2), speaks first with relevant proactive notifications (Phase 3), and is
served over the existing web app's HTTP API with a controlled notification runtime (Phase 4).

Layers, each replaceable on its own:

    config.py         environment-driven settings; ``INTERACTION_LLM_PROVIDER=mock`` by default
    llm.py            the provider seam: a deterministic mock now, Amazon Bedrock Converse later
    tools.py          read-only tools over the published data, each reporting its own availability
    memory.py         optional SQLite conversation memory (sessions, messages, context)
    notifications.py  proactive notifications: relevance, dedup/cooldown and SQLite storage
    security.py       opaque session credentials, so a session id alone is not ownership
    runtime.py        the background poller that drives ``process_notifications`` on a timer
    prompts.py        the system instructions the provider receives
    agent.py          the tool-calling loop that ties the question-answering layers together

``webapp/interaction_api.py`` wires these onto the FastAPI app (``POST /api/chat`` and friends).

Typical use::

    from webapp.interaction_agent import build_agent

    agent = build_agent()
    reply = agent.answer("Are there any traffic incidents?")
    print(reply.text)

    # conversations (optional), persisted across restarts
    session = agent.new_session()
    agent.answer("What is the status of my journey?", session_id=session)
    agent.answer("What about the delay?", session_id=session)

    # proactive notifications (optional), callable from a background loop
    from webapp.interaction_agent import build_notification_store, process_notifications
    store = build_notification_store()
    store.subscribe(session, journey_id)          # explicit: never auto-subscribed
    process_notifications(LiveDataSources.from_config(), store)

The defaults run with no AWS, no Kafka, no Docker and no TensorFlow.
"""
from webapp.interaction_agent.agent import (AgentReply, InteractionAgent, ToolInvocation,
                                            answer, build_agent, context_update)
from webapp.interaction_agent.config import ConfigError, InteractionConfig
from webapp.interaction_agent.llm import (BedrockConverseProvider, LLMProvider, LLMResponse,
                                          MockLLMProvider, ProviderUnavailable, ToolCall,
                                          build_llm_provider, detect_followup, detect_intent)
from webapp.interaction_agent.memory import ConversationMemory, build_memory
from webapp.interaction_agent.notifications import (NOTIF_INCIDENT, NOTIF_INCIDENT_RESOLVED,
                                                    NOTIF_JOURNEY_COMPLETED, NOTIF_JOURNEY_DELAY,
                                                    NOTIF_ROUTE_ALTERNATIVE, NOTIFICATION_TYPES,
                                                    Notification, NotificationStore,
                                                    build_notification_store, evaluate_dedup,
                                                    event_is_fresh, incident_relevant_to_journey,
                                                    journey_in_progress, process_notifications,
                                                    severity_rank)
from webapp.interaction_agent.prompts import SYSTEM_PROMPT, build_system_prompt
from webapp.interaction_agent.runtime import NotificationRuntime, build_notification_runtime
from webapp.interaction_agent.security import SessionGuard, build_session_guard, hash_token, new_token
from webapp.interaction_agent.tools import LiveDataSources, ToolRegistry, build_default_registry

__all__ = [
    "AgentReply", "InteractionAgent", "ToolInvocation", "answer", "build_agent", "context_update",
    "ConfigError", "InteractionConfig",
    "BedrockConverseProvider", "LLMProvider", "LLMResponse", "MockLLMProvider",
    "ProviderUnavailable", "ToolCall", "build_llm_provider", "detect_followup", "detect_intent",
    "ConversationMemory", "build_memory",
    "NOTIF_INCIDENT", "NOTIF_INCIDENT_RESOLVED", "NOTIF_JOURNEY_DELAY", "NOTIF_ROUTE_ALTERNATIVE",
    "NOTIF_JOURNEY_COMPLETED", "NOTIFICATION_TYPES", "Notification", "NotificationStore",
    "build_notification_store", "process_notifications", "evaluate_dedup", "severity_rank",
    "incident_relevant_to_journey", "event_is_fresh", "journey_in_progress",
    "SessionGuard", "build_session_guard", "hash_token", "new_token",
    "NotificationRuntime", "build_notification_runtime",
    "SYSTEM_PROMPT", "build_system_prompt",
    "LiveDataSources", "ToolRegistry", "build_default_registry",
]
