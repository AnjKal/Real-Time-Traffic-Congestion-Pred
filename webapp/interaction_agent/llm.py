"""LLM provider seam for the User Interaction Agent.

The agent talks to a provider through a single method, ``converse``. It takes a conversation, a
list of tool specifications and a system prompt, and returns either text or a set of tool calls to
run. That is exactly the shape of the Amazon Bedrock Converse API:

    converse(messages=..., toolConfig={"tools": [...]}, system=[...])
        -> output.message.content  (a list of ``text`` and/or ``toolUse`` blocks)

so the mock and the real Bedrock provider share one interface, and swapping them is a new subclass
rather than a rewrite.

Two providers:

  * :class:`MockLLMProvider` -- deterministic keyword intent -> tool call -> templated answer. It is
    not a language model. It exists so the orchestration, the tools and the error handling can be
    tested without AWS, and it is what ``INTERACTION_LLM_PROVIDER=mock`` (the default) selects.
  * :class:`BedrockConverseProvider` -- the real Amazon Bedrock Converse client. It is a translation
    layer only: it never invents an answer, it never runs a tool and it never falls back to the
    mock. ``INTERACTION_LLM_PROVIDER=bedrock`` selects it.

The intent rules below identify the source that must be refreshed before a live-data answer. They
do not compose Bedrock responses: Bedrock remains responsible for natural-language answers and may
still request additional tools when the current result is insufficient.
"""
from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class ProviderUnavailable(RuntimeError):
    """The configured provider cannot be used (not enabled, SDK missing, no credentials)."""


@dataclass(frozen=True)
class ToolCall:
    """A request from the provider to run one tool with these arguments."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class LLMResponse:
    """Either a final ``text`` answer, or ``tool_calls`` the agent should run next.

    ``stop_reason`` and ``usage`` are optional metadata a real provider may fill in (Bedrock returns
    them); the neutral orchestration ignores them, so a provider that leaves them unset is fine.
    """

    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(ABC):
    """One method, so a mock and a real model are interchangeable."""

    name = "base"

    @abstractmethod
    def converse(self, messages: Sequence[Dict[str, Any]],
                 tools: Sequence[Dict[str, Any]],
                 system: str = "",
                 context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Given the conversation and available tools, return text or tool calls.

        ``messages`` is a neutral list of dicts:
          * ``{"role": "user", "content": "<text>"}``
          * ``{"role": "assistant", "content": "<text>", "tool_calls": [...]}``
          * ``{"role": "tool", "tool_call_id": ..., "name": ..., "result": {...}}``
        ``tools`` are already ``toolSpec`` blocks, ready for a Bedrock ``toolConfig``.

        ``context`` is optional session context (``journey_ids``, ``places``, ``last_intent``) used
        to resolve a follow-up question such as "what about the delay?". It is context, never
        evidence: a provider must still call a tool to learn the current traffic. Providers that do
        not use it may ignore it.
        """


# --------------------------------------------------------------------------- mock provider

# Intent rules, checked in order: the first match wins. Order matters -- "traffic incidents"
# is an incident question, not a congestion one, so the incident rule is checked first.
INTENT_RULES = (
    ("incidents", ("incident", "accident", "jam", "road closure", "roadblock", "blocked",
                   "slowdown", "congestion event")),
    ("alternatives", ("alternative", "faster route", "another route", "better route", "reroute",
                      "re-route", "other route", "recommend a route", "switch route", "switch",
                      "route change", "save", "saving", "gain")),
    ("system", ("system running", "prediction system", "is the system", "system up",
                "service running", "model running", "system status", "up and running",
                "is it running", "system healthy")),
    ("journey", ("my journey", "journey", "my trip", "my route", "where am i", "how far",
                 "my progress", "delay", "delayed", "eta", "arrival", "arrive", "how long",
                 "remaining")),
    ("congestion", ("congestion", "traffic", "how busy", "how heavy", "overall",
                    "current speed", "traffic speed", "road conditions", "traffic conditions")),
)

INTENT_TOOL = {
    "incidents": "get_incidents",
    "alternatives": "get_route_recommendations",
    "system": "get_system_status",
    "journey": "get_journeys",
    "congestion": "get_congestion_summary",
}

# The reverse map, so the agent can record which topic a turn was about for the next follow-up.
TOOL_INTENT = {
    "get_incidents": "incidents",
    "get_congestion_summary": "congestion",
    "get_journeys": "journey",
    "get_journey": "journey",
    "get_route_recommendations": "alternatives",
    "get_system_status": "system",
}

# A follow-up like "what about the delay?" carries no topic keyword of its own, so these narrow
# phrases recover the topic. Checked in order, only when the main intent rules found nothing.
FOLLOWUP_RULES = (
    ("alternatives", ("save", "saving", "gain", "faster", "quicker", "shorter", "avoid",
                      "alternative", "better route")),
    ("journey", ("delay", "delayed", "late", "eta", "arrival", "arrive", "progress", "how long",
                 "remaining", "left")),
    ("incidents", ("incident", "jam", "blocked", "closure", "they", "those", "them")),
    ("congestion", ("busy", "heavy", "how bad")),
    ("system", ("running", "ready")),
)

# A journey id the page generates, e.g. jr-ff466f71c8. Used only to pick the single-journey tool.
JOURNEY_ID_RE = re.compile(r"\b(jr-[0-9a-fA-F]{4,})\b")

# Words that mark a message as a genuine continuation of the previous topic ("anything else?",
# "what about that?"). Falling back to the last topic is only safe when one of these is present;
# an unrelated request that merely arrives next ("tell me a joke") must not inherit the topic.
FOLLOWUP_CUES = ("what about", "how about", "what else", "anything else", "that", "this", "those",
                 "these", "them", "they", "there", "it", "its", "then", "instead", "more", "again",
                 "too", "also", "else", "now")
_CUE_RE = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(cue) for cue in FOLLOWUP_CUES))
_CONTEXTUAL_FOLLOWUP_RE = re.compile(
    r"^\s*(?:what about|how about|what else|anything else)\b", re.IGNORECASE)


def is_contextual_followup(text: str) -> bool:
    """Whether a question is phrased as a fragment that needs prior conversational context."""
    return bool(_CONTEXTUAL_FOLLOWUP_RE.search(text or ""))

CLARIFY = ("I can help with traffic incidents, the current congestion, your journey, alternative "
           "routes, or prediction-system status. Which one would you like to check?")


def detect_intent(text: str) -> Optional[str]:
    """Map a user message to an intent name, or None when nothing matches."""
    lowered = (text or "").lower()
    for intent, keywords in INTENT_RULES:
        if any(keyword in lowered for keyword in keywords):
            return intent
    return None


def detect_followup(text: str, context: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resolve a follow-up that carries no topic keyword of its own, using session context.

    Returns None when there is nothing reliable to go on, so the caller asks for clarification
    instead of guessing at a topic. A journey mentioned earlier in the conversation is preferred,
    because that is what "the delay" almost always means. With no prior context at all -- or a
    message that is not a continuation of the previous topic -- this returns None rather than
    inventing one.
    """
    context = context or {}
    lowered = (text or "").lower()
    # Nothing has been discussed yet, so there is no earlier topic for a follow-up to attach to.
    if not (context.get("last_intent") or context.get("journey_ids") or context.get("places")):
        return None
    for intent, keywords in FOLLOWUP_RULES:
        if any(keyword in lowered for keyword in keywords):
            return intent
    # Reusing the previous topic is only safe for a real continuation ("anything else?"), never for
    # an unrelated request that merely arrives next ("tell me a joke").
    if not _CUE_RE.search(lowered):
        return None
    last = context.get("last_intent")
    if last in INTENT_TOOL:
        return last
    if context.get("journey_ids"):
        return "journey"
    return None


class MockLLMProvider(LLMProvider):
    """Deterministic stand-in: keyword intent -> tool call -> templated answer.

    Two rules keep it honest across a conversation:

      * Only tool results produced AFTER the most recent user message count as the current turn.
        Older ones are history and are never reused as if they were today's traffic.
      * A follow-up is resolved against session context (last topic, known journeys), and when it
        cannot be, the mock asks for clarification rather than pretending to reason.
    """

    name = "mock"

    def converse(self, messages, tools, system="", context=None):
        results = self._current_turn_tool_results(messages)
        if results:
            text = self._last_user_text(messages)
            intent = detect_intent(text) or detect_followup(text, context)
            preferred_tool = INTENT_TOOL.get(intent)
            if preferred_tool:
                preferred = next(
                    (message.get("result") or {} for message in reversed(results)
                     if message.get("name") == preferred_tool),
                    None,
                )
                if preferred is not None:
                    return LLMResponse(text=self._compose(preferred))
            return LLMResponse(text=self._compose(results[-1].get("result") or {}))
        text = self._last_user_text(messages)
        followup_intent = detect_followup(text, context)
        intent = detect_intent(text) or followup_intent
        journey_ids = (context or {}).get("journey_ids") or []
        if (is_contextual_followup(text) and intent == "journey"
                and not re.search(r"\b(?:journey|trip|route)\b", text, re.IGNORECASE)
                and len(journey_ids) != 1):
            return LLMResponse(text=CLARIFY)
        if intent is None:
            return LLMResponse(text=CLARIFY)
        return LLMResponse(tool_calls=[self._tool_call_for(intent, text, context)])

    @staticmethod
    def _last_user_text(messages):
        for message in reversed(list(messages)):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    @staticmethod
    def _current_turn_tool_results(messages):
        """Tool messages that follow the latest user message -- the current turn, and only it."""
        messages = list(messages)
        last_user = -1
        for index, message in enumerate(messages):
            if message.get("role") == "user":
                last_user = index
        return [message for message in messages[last_user + 1:]
                if message.get("role") == "tool"]

    def _tool_call_for(self, intent, text, context=None):
        name = INTENT_TOOL[intent]
        arguments: Dict[str, Any] = {}
        if intent == "journey":
            match = JOURNEY_ID_RE.search(text or "")
            if match:
                name = "get_journey"
                arguments = {"journey_id": match.group(1)}
            else:
                known = (context or {}).get("journey_ids") or []
                if known:
                    name = "get_journey"
                    arguments = {"journey_id": known[0]}
        return ToolCall(name=name, arguments=arguments, id="call-1")

    def _compose(self, result):
        if not result.get("ok", True):
            return ("I couldn't retrieve that information right now%s."
                    % (": " + str(result["error"]) if result.get("error") else ""))
        lines = []
        if result.get("headline"):
            lines.append(str(result["headline"]))
        for detail in (result.get("details") or [])[:6]:
            lines.append("- " + str(detail))
        note = self._availability_note(result.get("availability"), result.get("tool"))
        if note:
            lines.append(note)
        return "\n".join(lines).strip() or "I don't have data to answer that right now."

    @staticmethod
    def _availability_note(availability, tool):
        if not availability or availability in ("live", "empty"):
            return None
        if availability == "missing":
            return "That data has not been published yet."
        if availability == "stale":
            return "That figure may be out of date."
        if availability == "unavailable":
            return "That source is not reachable right now."
        return None


# --------------------------------------------------------------------------- Bedrock seam

# Amazon Bedrock's Converse API is already the shape of the interface above, so this provider is a
# translation layer: neutral messages -> Converse messages, ``toolSpec`` blocks -> ``toolConfig``,
# and the model's ``content`` blocks -> either final text or tool calls. Two rules it must respect:
#
#   * a ``toolResult`` must be returned in a ``user`` message carrying the SAME ``toolUseId`` the
#     model issued, and every tool result for one assistant turn must come back together;
#   * nothing is invented locally -- whatever the model does not say, the provider does not say.
#
# The client is built lazily on the first ``converse`` call, so importing, constructing and starting
# the web app (and its status endpoint) never touch AWS and never require credentials.

# Safe, user-facing wording for the AWS failures the SDK raises. Deliberately generic: the raw
# ``Error.Message`` can echo request contents, so only the error CODE / category is used here.
_CLIENT_ERROR_MESSAGES = {
    "AccessDeniedException": "access to the model was denied by AWS.",
    "ValidationException": "the request was rejected as invalid.",
    "ThrottlingException": "the request was throttled; please try again shortly.",
    "TooManyRequestsException": "the request was throttled; please try again shortly.",
    "ModelNotReadyException": "the model is not ready yet.",
    "ModelTimeoutException": "the model took too long to respond.",
    "ServiceUnavailableException": "the service is temporarily unavailable.",
    "InternalServerException": "AWS reported an internal error.",
    "ResourceNotFoundException": "the configured model or inference profile was not found.",
    "UnrecognizedClientException": "the AWS credentials were not recognised.",
    "ExpiredTokenException": "the AWS credentials have expired.",
    "InvalidSignatureException": "the request could not be signed with the current credentials.",
}

_BOTOCORE_ERROR_MESSAGES = {
    "NoCredentialsError": "no AWS credentials were found.",
    "PartialCredentialsError": "the AWS credentials are incomplete.",
    "EndpointConnectionError": "the Amazon Bedrock endpoint could not be reached.",
    "ConnectTimeoutError": "connecting to Amazon Bedrock timed out.",
    "ReadTimeoutError": "Amazon Bedrock did not respond in time.",
    "ConnectionClosedError": "the connection to Amazon Bedrock was closed unexpectedly.",
    "SSLError": "a secure connection to Amazon Bedrock could not be established.",
    "ProxyConnectionError": "a proxy connection to Amazon Bedrock failed.",
    "ParamValidationError": "the request was rejected before it was sent (invalid parameters).",
}

_CLIENT_ERROR_DEFAULT = "AWS rejected the request."
_BEDROCK_GENERIC_ERROR = "Amazon Bedrock could not be reached. Please try again."
_BEDROCK_MALFORMED_ERROR = "Amazon Bedrock returned a response that could not be understood."


class BedrockConverseProvider(LLMProvider):
    """Amazon Bedrock Converse provider (Phase 6).

    Talks to ``bedrock-runtime`` through the Converse API, which has the same shape as this module's
    interface. The default model is Amazon Nova Lite; ``BEDROCK_MODEL_ID`` accepts a plain model id,
    an inference-profile id or an inference-profile ARN, so the exact target can be set after the
    account is approved without a code change.

    Testing seam: pass ``client=`` (a ready client, e.g. a ``Stubber``-wrapped one) or
    ``client_factory=`` (called once, lazily, on the first ``converse``). Neither is used in
    production, where ``boto3.client("bedrock-runtime", ...)`` is built on demand.
    """

    name = "bedrock"

    def __init__(self, config, client=None, client_factory=None):
        self.config = config
        self._client = client
        self._client_factory = client_factory
        self._client_lock = threading.Lock()

    # -- client lifecycle ------------------------------------------------------
    def _build_client(self):
        """Build a bedrock-runtime client. Imports boto3 here so a mock-only install needs no AWS."""
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as exc:                        # boto3 is optional until Bedrock is used
            raise ProviderUnavailable(
                "boto3 is not installed, so Amazon Bedrock cannot be used. Install it "
                "(pip install 'boto3>=1.34') or keep INTERACTION_LLM_PROVIDER=mock.") from exc
        sdk_config = BotoConfig(
            region_name=self.config.aws_region,
            connect_timeout=float(self.config.bedrock_connect_timeout_s),
            read_timeout=float(self.config.bedrock_read_timeout_s),
            retries={"max_attempts": int(self.config.bedrock_max_retries), "mode": "standard"},
        )
        return boto3.client("bedrock-runtime", region_name=self.config.aws_region,
                            config=sdk_config)

    def _get_client(self):
        """Return the client, building it on first use. Never called at construction time."""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                self._client = (self._client_factory or self._build_client)()
        return self._client

    # -- the one interface method ---------------------------------------------
    def converse(self, messages, tools, system="", context=None):
        """Send the conversation to Converse and translate the reply into an :class:`LLMResponse`."""
        try:
            request = self._build_request(messages, tools, system, context)
            client = self._get_client()
            response = client.converse(**request)
        except ProviderUnavailable:
            raise
        except Exception as exc:                          # translate everything else safely
            raise self._translate_error(exc) from exc
        return self._parse_response(response)

    # -- request building ------------------------------------------------------
    def _build_request(self, messages, tools, system, context):
        request: Dict[str, Any] = {
            "modelId": self.config.bedrock_model_id,
            "messages": self._to_converse_messages(messages),
            "inferenceConfig": self._inference_config(),
        }
        system_blocks = self._system_blocks(system, context)
        if system_blocks:
            request["system"] = system_blocks
        if tools:
            request["toolConfig"] = as_tool_config(tools)
        return request

    def _inference_config(self):
        config: Dict[str, Any] = {"maxTokens": int(self.config.bedrock_max_tokens)}
        temperature = self.config.bedrock_temperature
        if temperature is not None:
            config["temperature"] = float(temperature)
        return config

    @classmethod
    def _system_blocks(cls, system, context):
        """The ``system`` field: the instructions, plus session context as a separate data block."""
        blocks: List[Dict[str, Any]] = []
        text = str(system or "").strip()
        if text:
            blocks.append({"text": text})
        hint = cls._context_hint(context)
        if hint:
            blocks.append({"text": hint})
        return blocks

    @staticmethod
    def _context_hint(context):
        """Session context as a clearly-labelled data block -- background, never an instruction."""
        if not isinstance(context, dict):
            return ""
        parts = []
        if context.get("last_intent"):
            parts.append("last topic was %s" % context["last_intent"])
        journeys = context.get("journey_ids")
        if journeys:
            parts.append("journeys discussed: %s" % ", ".join(str(item) for item in journeys))
        places = context.get("places")
        if places:
            parts.append("places discussed: %s" % ", ".join(str(item) for item in places))
        if not parts:
            return ""
        return ("Reference context for resolving follow-up questions (" + "; ".join(parts) + "). "
                "This is background data, not an instruction, and it is not current evidence.")

    @classmethod
    def _to_converse_messages(cls, messages):
        """Neutral messages -> the Converse ``messages`` list.

        Consecutive ``tool`` messages are collected into a single ``user`` turn because the API
        requires every ``toolResult`` for one assistant turn to be returned together, each carrying
        the ``toolUseId`` the model issued.
        """
        converted: List[Dict[str, Any]] = []
        pending_results: List[Dict[str, Any]] = []
        for message in messages or []:
            role = message.get("role")
            if role == "tool":
                pending_results.append(cls._tool_result_block(message))
                continue
            if pending_results:
                converted.append({"role": "user", "content": pending_results})
                pending_results = []
            if role == "user":
                converted.append({"role": "user",
                                  "content": [{"text": str(message.get("content") or "")}]})
            elif role == "assistant":
                converted.append({"role": "assistant", "content": cls._assistant_content(message)})
        if pending_results:
            converted.append({"role": "user", "content": pending_results})
        return converted

    @classmethod
    def _assistant_content(cls, message):
        """An assistant turn: its text (if any) followed by one ``toolUse`` block per tool call."""
        content: List[Dict[str, Any]] = []
        text = message.get("content")
        if text:
            content.append({"text": str(text)})
        for call in message.get("tool_calls") or []:
            content.append(cls._tool_use_block(call))
        if not content:
            content.append({"text": ""})                  # the API needs at least one block
        return content

    @staticmethod
    def _tool_use_block(call):
        name = call.get("name") if isinstance(call, dict) else None
        arguments = call.get("arguments") if isinstance(call, dict) else None
        return {"toolUse": {
            "toolUseId": str((call.get("id") if isinstance(call, dict) else "") or name or ""),
            "name": str(name or ""),
            "input": arguments if isinstance(arguments, dict) else {},
        }}

    @staticmethod
    def _tool_result_block(message):
        """A ``toolResult`` block, preserving the ``toolUseId`` the model issued for this call."""
        result = message.get("result")
        payload = result if isinstance(result, dict) else {"result": result}
        return {"toolResult": {
            "toolUseId": str(message.get("tool_call_id") or message.get("name") or ""),
            "content": [{"json": payload}],
            "status": "error" if payload.get("ok") is False else "success",
        }}

    # -- response parsing ------------------------------------------------------
    @staticmethod
    def _parse_response(response):
        """Turn a Converse response into text and/or tool calls, tolerating missing optional bits."""
        if not isinstance(response, dict):
            raise ProviderUnavailable(_BEDROCK_MALFORMED_ERROR)
        output = response.get("output")
        if not isinstance(output, dict):
            raise ProviderUnavailable(_BEDROCK_MALFORMED_ERROR)
        message = output.get("message")
        if not isinstance(message, dict):
            raise ProviderUnavailable(_BEDROCK_MALFORMED_ERROR)
        content = message.get("content")
        if content is None:
            content = []
        if not isinstance(content, list):
            raise ProviderUnavailable(_BEDROCK_MALFORMED_ERROR)

        texts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if "text" in block and block.get("text") is not None:
                texts.append(str(block["text"]))
            elif "toolUse" in block:
                tool_use = block.get("toolUse")
                if not isinstance(tool_use, dict) or not tool_use.get("name"):
                    continue                              # a malformed block is skipped, not fatal
                arguments = tool_use.get("input")
                tool_calls.append(ToolCall(
                    name=str(tool_use["name"]),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    id=str(tool_use.get("toolUseId") or "")))
        text = "\n".join(part for part in texts if part).strip() or None
        usage = response.get("usage")
        return LLMResponse(text=text, tool_calls=tool_calls,
                           stop_reason=response.get("stopReason"),
                           usage=usage if isinstance(usage, dict) else None)

    # -- error translation -----------------------------------------------------
    @staticmethod
    def _translate_error(exc):
        """Map any SDK exception onto a safe :class:`ProviderUnavailable` (no traces, no secrets)."""
        try:
            from botocore import exceptions as boto_exc
        except ImportError:                               # no SDK: nothing to classify
            return ProviderUnavailable(_BEDROCK_GENERIC_ERROR)
        if isinstance(exc, boto_exc.ClientError):
            code = ""
            try:
                code = str(((exc.response or {}).get("Error") or {}).get("Code") or "")
            except Exception:                             # noqa: BLE001 - keep the message safe
                code = ""
            detail = _CLIENT_ERROR_MESSAGES.get(code, _CLIENT_ERROR_DEFAULT)
            return ProviderUnavailable("Amazon Bedrock request failed: %s" % detail)
        for klass in type(exc).__mro__:
            detail = _BOTOCORE_ERROR_MESSAGES.get(klass.__name__)
            if detail:
                return ProviderUnavailable("Amazon Bedrock request failed: %s" % detail)
        return ProviderUnavailable(_BEDROCK_GENERIC_ERROR)


def build_llm_provider(config):
    """Return the provider named by the configuration. Mock unless Bedrock is explicitly asked for."""
    if config.llm_provider == "mock":
        return MockLLMProvider()
    if config.llm_provider == "bedrock":
        return BedrockConverseProvider(config)
    raise ProviderUnavailable("unsupported LLM provider %r" % config.llm_provider)


def as_tool_config(tools):
    """Wrap tool specs in the ``toolConfig`` shape Bedrock's Converse API expects."""
    return {"tools": list(tools)}
