"""Environment-based configuration for the User Interaction Agent.

The default provider is the deterministic mock, so the agent runs with no AWS account, no Kafka and
no Docker. The Bedrock settings below (region, model id, token cap, temperature, timeouts, retries)
are read now and used only once ``INTERACTION_LLM_PROVIDER`` is switched to ``bedrock`` (Phase 6);
with ``mock`` they are inert.

No credentials or secrets are read or stored here. ``boto3`` resolves those from its own chain
(environment variables, shared config, instance role), so there is nothing to hardcode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

DEFAULT_PROVIDER = "mock"
DEFAULT_REGION = "ap-south-1"                       # Mumbai: nearest region for a Bangalore setup
# Amazon Nova Lite: chosen for a cheap, fast, tool-capable model. The region and the exact inference
# profile will be re-verified once AWS access is restored.
DEFAULT_BEDROCK_MODEL_ID = "amazon.nova-lite-v1:0"
DEFAULT_MODEL_SERVICE_URL = "http://127.0.0.1:8001"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_REQUEST_TIMEOUT_S = 10.0
DEFAULT_STALE_AFTER_MIN = 10.0                      # the model polls every 2 min; 10 min is stale

# Amazon Bedrock Converse (Phase 6). Only read when INTERACTION_LLM_PROVIDER=bedrock. None of these
# is a credential: boto3 resolves AWS credentials from its own chain, so nothing is stored here.
DEFAULT_BEDROCK_TEMPERATURE = 0.2                   # low, so answers stay grounded in tool results
DEFAULT_BEDROCK_CONNECT_TIMEOUT_S = 5.0             # TCP connect to the Bedrock endpoint
DEFAULT_BEDROCK_READ_TIMEOUT_S = 30.0               # a longer read timeout: the model can be slow
DEFAULT_BEDROCK_MAX_RETRIES = 3                     # bounded SDK retries for throttling/5xx

# Conversation memory. The default path is the file that already exists under data/live; the schema
# is added alongside whatever is already in it (see memory.py), never in place of it.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MEMORY_DB_PATH = os.path.join(PROJECT_ROOT, "data", "live", "interaction_memory.sqlite")
DEFAULT_MEMORY_MAX_MESSAGES = 20                    # how many past messages are replayed per turn
DEFAULT_MEMORY_ENABLED = True

# Proactive notifications (Phase 3). The engine is deterministic and calls no LLM, so these are the
# only knobs it needs. Freshness reuses ``stale_after_min`` above: a notification is never built
# from data older than that.
DEFAULT_NOTIFICATIONS_ENABLED = True
DEFAULT_NOTIFICATION_COOLDOWN_MIN = 10.0            # don't re-notify the same journey/topic sooner
DEFAULT_NOTIFICATION_POLL_INTERVAL_S = 120.0        # matches the model's 2-minute poll cadence

SUPPORTED_PROVIDERS = ("mock", "bedrock")

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


class ConfigError(ValueError):
    """An environment value is present but cannot be used."""


def _as_int(value, name, default):
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        raise ConfigError("%s must be a whole number, got %r" % (name, value))
    if parsed <= 0:
        raise ConfigError("%s must be positive, got %r" % (name, value))
    return parsed


def _as_float(value, name, default):
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        raise ConfigError("%s must be a number, got %r" % (name, value))
    if parsed <= 0:
        raise ConfigError("%s must be positive, got %r" % (name, value))
    return parsed


def _as_nonneg_float(value, name, default):
    """Like ``_as_float`` but zero is allowed -- a zero cooldown means "no cooldown"."""
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        raise ConfigError("%s must be a number, got %r" % (name, value))
    if parsed < 0:
        raise ConfigError("%s must not be negative, got %r" % (name, value))
    return parsed


def _as_nonneg_int(value, name, default):
    """Like ``_as_int`` but zero is allowed -- zero retries means "do not retry"."""
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        raise ConfigError("%s must be a whole number, got %r" % (name, value))
    if parsed < 0:
        raise ConfigError("%s must not be negative, got %r" % (name, value))
    return parsed


def _as_bool(value, name, default):
    if value is None or str(value).strip() == "":
        return default
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError("%s must be a boolean (one of %s / %s), got %r"
                      % (name, list(_TRUE), list(_FALSE), value))


@dataclass(frozen=True)
class InteractionConfig:
    """Everything the agent needs to know about its environment, validated once."""

    llm_provider: str = DEFAULT_PROVIDER
    aws_region: str = DEFAULT_REGION
    bedrock_model_id: str = DEFAULT_BEDROCK_MODEL_ID
    bedrock_max_tokens: int = DEFAULT_MAX_TOKENS
    bedrock_temperature: float = DEFAULT_BEDROCK_TEMPERATURE
    bedrock_connect_timeout_s: float = DEFAULT_BEDROCK_CONNECT_TIMEOUT_S
    bedrock_read_timeout_s: float = DEFAULT_BEDROCK_READ_TIMEOUT_S
    bedrock_max_retries: int = DEFAULT_BEDROCK_MAX_RETRIES
    model_service_url: str = DEFAULT_MODEL_SERVICE_URL
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    stale_after_min: float = DEFAULT_STALE_AFTER_MIN
    memory_db_path: str = DEFAULT_MEMORY_DB_PATH
    memory_max_messages: int = DEFAULT_MEMORY_MAX_MESSAGES
    memory_enabled: bool = DEFAULT_MEMORY_ENABLED
    notifications_enabled: bool = DEFAULT_NOTIFICATIONS_ENABLED
    notification_cooldown_min: float = DEFAULT_NOTIFICATION_COOLDOWN_MIN
    notification_poll_interval_s: float = DEFAULT_NOTIFICATION_POLL_INTERVAL_S

    def __post_init__(self):
        if self.llm_provider not in SUPPORTED_PROVIDERS:
            raise ConfigError("INTERACTION_LLM_PROVIDER must be one of %s, got %r"
                              % (list(SUPPORTED_PROVIDERS), self.llm_provider))
        if not str(self.memory_db_path).strip():
            raise ConfigError("INTERACTION_MEMORY_DB_PATH must not be blank")
        if float(self.bedrock_temperature) < 0.0 or float(self.bedrock_temperature) > 1.0:
            raise ConfigError("BEDROCK_TEMPERATURE must be between 0 and 1, got %r"
                              % self.bedrock_temperature)
        if float(self.bedrock_connect_timeout_s) <= 0:
            raise ConfigError("BEDROCK_CONNECT_TIMEOUT must be positive, got %r"
                              % self.bedrock_connect_timeout_s)
        if float(self.bedrock_read_timeout_s) <= 0:
            raise ConfigError("BEDROCK_READ_TIMEOUT must be positive, got %r"
                              % self.bedrock_read_timeout_s)
        if int(self.bedrock_max_retries) < 0:
            raise ConfigError("BEDROCK_MAX_RETRIES must not be negative, got %r"
                              % self.bedrock_max_retries)
        if int(self.memory_max_messages) <= 0:
            raise ConfigError("INTERACTION_MEMORY_MAX_MESSAGES must be positive, got %r"
                              % self.memory_max_messages)
        if float(self.notification_cooldown_min) < 0:
            raise ConfigError("INTERACTION_NOTIFICATION_COOLDOWN_MIN must not be negative, got %r"
                              % self.notification_cooldown_min)
        if float(self.notification_poll_interval_s) <= 0:
            raise ConfigError("INTERACTION_NOTIFICATION_POLL_INTERVAL_S must be positive, got %r"
                              % self.notification_poll_interval_s)

    @property
    def uses_bedrock(self) -> bool:
        return self.llm_provider == "bedrock"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "InteractionConfig":
        """Build from an environment mapping (defaults to ``os.environ``)."""
        source = os.environ if env is None else env
        provider = str(source.get("INTERACTION_LLM_PROVIDER", DEFAULT_PROVIDER)).strip().lower()
        region = str(source.get("AWS_REGION") or source.get("AWS_DEFAULT_REGION")
                     or DEFAULT_REGION).strip()
        model_id = str(source.get("BEDROCK_MODEL_ID") or DEFAULT_BEDROCK_MODEL_ID).strip()
        url = str(source.get("MODEL_SERVICE_URL") or DEFAULT_MODEL_SERVICE_URL).strip()
        # Absent means "use the default"; present-but-blank is a malformed value and is rejected by
        # __post_init__ rather than silently writing the database somewhere unexpected.
        raw_db_path = source.get("INTERACTION_MEMORY_DB_PATH")
        db_path = DEFAULT_MEMORY_DB_PATH if raw_db_path is None else str(raw_db_path).strip()
        return cls(
            llm_provider=provider or DEFAULT_PROVIDER,
            aws_region=region or DEFAULT_REGION,
            bedrock_model_id=model_id,
            bedrock_max_tokens=_as_int(source.get("BEDROCK_MAX_TOKENS"), "BEDROCK_MAX_TOKENS",
                                       DEFAULT_MAX_TOKENS),
            bedrock_temperature=_as_nonneg_float(source.get("BEDROCK_TEMPERATURE"),
                                                 "BEDROCK_TEMPERATURE", DEFAULT_BEDROCK_TEMPERATURE),
            bedrock_connect_timeout_s=_as_float(
                source.get("BEDROCK_CONNECT_TIMEOUT"), "BEDROCK_CONNECT_TIMEOUT",
                DEFAULT_BEDROCK_CONNECT_TIMEOUT_S),
            bedrock_read_timeout_s=_as_float(
                source.get("BEDROCK_READ_TIMEOUT"), "BEDROCK_READ_TIMEOUT",
                DEFAULT_BEDROCK_READ_TIMEOUT_S),
            bedrock_max_retries=_as_nonneg_int(source.get("BEDROCK_MAX_RETRIES"),
                                               "BEDROCK_MAX_RETRIES", DEFAULT_BEDROCK_MAX_RETRIES),
            model_service_url=url.rstrip("/") or DEFAULT_MODEL_SERVICE_URL,
            request_timeout_s=_as_float(source.get("INTERACTION_REQUEST_TIMEOUT"),
                                        "INTERACTION_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT_S),
            stale_after_min=_as_float(source.get("INTERACTION_STALE_AFTER_MIN"),
                                      "INTERACTION_STALE_AFTER_MIN", DEFAULT_STALE_AFTER_MIN),
            memory_db_path=db_path,
            memory_max_messages=_as_int(source.get("INTERACTION_MEMORY_MAX_MESSAGES"),
                                        "INTERACTION_MEMORY_MAX_MESSAGES",
                                        DEFAULT_MEMORY_MAX_MESSAGES),
            memory_enabled=_as_bool(source.get("INTERACTION_MEMORY_ENABLED"),
                                    "INTERACTION_MEMORY_ENABLED", DEFAULT_MEMORY_ENABLED),
            notifications_enabled=_as_bool(source.get("INTERACTION_NOTIFICATIONS_ENABLED"),
                                           "INTERACTION_NOTIFICATIONS_ENABLED",
                                           DEFAULT_NOTIFICATIONS_ENABLED),
            notification_cooldown_min=_as_nonneg_float(
                source.get("INTERACTION_NOTIFICATION_COOLDOWN_MIN"),
                "INTERACTION_NOTIFICATION_COOLDOWN_MIN", DEFAULT_NOTIFICATION_COOLDOWN_MIN),
            notification_poll_interval_s=_as_float(
                source.get("INTERACTION_NOTIFICATION_POLL_INTERVAL_S"),
                "INTERACTION_NOTIFICATION_POLL_INTERVAL_S", DEFAULT_NOTIFICATION_POLL_INTERVAL_S),
        )
