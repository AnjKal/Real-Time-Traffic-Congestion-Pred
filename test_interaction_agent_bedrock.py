"""Isolated, offline tests for the Amazon Bedrock Converse provider (Phase 6).

    python -m unittest test_interaction_agent_bedrock -v

No test here makes a real AWS call, and none needs credentials, Kafka, Docker or the network. The
SDK is exercised two ways:

  * an injected recording client (dependency injection) for behaviour -- scripted responses and
    scripted exceptions;
  * botocore's own ``Stubber``, on a client built with dummy credentials, to prove the request we
    send matches the real ``bedrock-runtime`` service model.

These are SIMULATED responses. Nothing here claims that live Bedrock inference has been verified;
that must be done against a real account (see the Phase 6 report).
"""
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from unittest import mock

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import boto3
from botocore.exceptions import (ClientError, ConnectTimeoutError, ConnectionClosedError,
                                 EndpointConnectionError, NoCredentialsError, ParamValidationError,
                                 PartialCredentialsError, ReadTimeoutError)
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from webapp import server as srv
from webapp.interaction_agent import (BedrockConverseProvider, ConversationMemory, InteractionAgent,
                                      InteractionConfig, LLMResponse, MockLLMProvider,
                                      ProviderUnavailable, ToolCall, build_llm_provider)
from webapp.interaction_agent.agent import MAX_STEPS, PROVIDER_ERROR_REPLY, STEP_LIMIT_REPLY
from webapp.interaction_agent.tools import build_default_registry
from webapp.interaction_api import InteractionServices

# Reuse the API test fixtures so the data shape is identical to the rest of the suite.
from test_interaction_agent_api import (INCIDENT_EVENT, FakeSources, default_sources, fresh,
                                        journey, old)

MODEL_ID = "amazon.nova-lite-v1:0"


# --------------------------------------------------------------------------- fixtures

def text_response(text, stop_reason="end_turn", usage=None):
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": stop_reason,
            "usage": usage or {"inputTokens": 12, "outputTokens": 7, "totalTokens": 19},
            "metrics": {"latencyMs": 42}}


def tool_response(calls, text=None, stop_reason="tool_use"):
    content = []
    if text is not None:
        content.append({"text": text})
    for call in calls:
        content.append({"toolUse": {"toolUseId": call["id"], "name": call["name"],
                                    "input": call.get("input", {})}})
    return {"output": {"message": {"role": "assistant", "content": content}},
            "stopReason": stop_reason,
            "usage": {"inputTokens": 20, "outputTokens": 9, "totalTokens": 29},
            "metrics": {"latencyMs": 51}}


class RecordingClient(object):
    """A stand-in ``bedrock-runtime`` client: records requests, replays scripted replies.

    An entry in ``responses`` may be a dict (returned) or an ``Exception`` (raised).
    """

    def __init__(self, responses=(), error=None):
        self.requests = []
        self._responses = list(responses)
        self._error = error

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        if self._error is not None:
            raise self._error
        if self._responses:
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return text_response("done")


def bedrock_config(**overrides):
    kwargs = {"llm_provider": "bedrock"}
    kwargs.update(overrides)
    return InteractionConfig(**kwargs)


def stubbed_provider(expected_response, expected_params, config=None):
    """A provider whose client is a real bedrock-runtime client behind a ``Stubber``."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          aws_access_key_id="test", aws_secret_access_key="test")
    stubber = Stubber(client)
    stubber.add_response("converse", expected_response, expected_params)
    stubber.activate()
    provider = BedrockConverseProvider(config or bedrock_config(aws_region="us-east-1"),
                                       client=client)
    return provider, stubber


def tool_result_blocks(request):
    """All ``toolResult`` blocks in a converted request, in order."""
    blocks = []
    for message in request["messages"]:
        if message.get("role") != "user":
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict) and "toolResult" in block:
                blocks.append(block["toolResult"])
    return blocks


# --------------------------------------------------------------------------- 1. client lifetime

class TestClientLifetime(unittest.TestCase):
    def test_client_is_not_built_at_construction(self):
        built = []
        provider = BedrockConverseProvider(
            bedrock_config(),
            client_factory=lambda: built.append("built") or RecordingClient())
        self.assertEqual(built, [])                                  # constructing touches nothing
        provider.converse([{"role": "user", "content": "hi"}], tools=[])
        self.assertEqual(built, ["built"])                           # built lazily, on first use

    def test_client_is_built_once_and_cached(self):
        built = []
        client = RecordingClient([text_response("a"), text_response("b")])
        provider = BedrockConverseProvider(
            bedrock_config(), client_factory=lambda: built.append(1) or client)
        provider.converse([{"role": "user", "content": "1"}], tools=[])
        provider.converse([{"role": "user", "content": "2"}], tools=[])
        self.assertEqual(len(built), 1)

    def test_injected_client_is_used_directly(self):
        client = RecordingClient([text_response("hello")])
        provider = BedrockConverseProvider(bedrock_config(), client=client)
        response = provider.converse([{"role": "user", "content": "hi"}], tools=[])
        self.assertEqual(response.text, "hello")
        self.assertEqual(len(client.requests), 1)

    def test_built_client_honours_region_timeouts_and_retries(self):
        config = bedrock_config(aws_region="eu-west-1", bedrock_connect_timeout_s=3.0,
                                bedrock_read_timeout_s=17.0, bedrock_max_retries=5)
        with mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test",
                                          "AWS_SECRET_ACCESS_KEY": "test"}):
            client = BedrockConverseProvider(config)._build_client()
        self.assertEqual(client.meta.region_name, "eu-west-1")
        self.assertEqual(client.meta.config.connect_timeout, 3.0)
        self.assertEqual(client.meta.config.read_timeout, 17.0)
        # botocore normalises our "max_attempts": n into a bounded "total_max_attempts": n + 1.
        self.assertEqual(client.meta.config.retries["mode"], "standard")
        self.assertEqual(client.meta.config.retries["total_max_attempts"], 6)


# --------------------------------------------------------------------------- 2. request structure

class TestConverseRequestStructure(unittest.TestCase):
    """The Stubber validates our request against the real bedrock-runtime service model."""

    def test_plain_text_turn(self):
        provider, stubber = stubbed_provider(
            text_response("ok"),
            {"modelId": MODEL_ID,
             "messages": [{"role": "user", "content": [{"text": "how is traffic?"}]}],
             "system": [{"text": "SYS"}],
             "inferenceConfig": {"maxTokens": 1024, "temperature": 0.2}})
        self.addCleanup(stubber.deactivate)
        response = provider.converse([{"role": "user", "content": "how is traffic?"}],
                                     tools=[], system="SYS")
        self.assertEqual(response.text, "ok")
        stubber.assert_no_pending_responses()

    def test_tools_are_sent_as_a_tool_config(self):
        spec = {"toolSpec": {"name": "get_incidents", "description": "incidents",
                             "inputSchema": {"json": {"type": "object", "properties": {}}}}}
        provider, stubber = stubbed_provider(
            tool_response([{"id": "tu-9", "name": "get_incidents", "input": {}}]),
            {"modelId": MODEL_ID,
             "messages": [{"role": "user", "content": [{"text": "incidents?"}]}],
             "inferenceConfig": {"maxTokens": 1024, "temperature": 0.2},
             "toolConfig": {"tools": [spec]}})
        self.addCleanup(stubber.deactivate)
        response = provider.converse([{"role": "user", "content": "incidents?"}], tools=[spec])
        self.assertEqual([call.name for call in response.tool_calls], ["get_incidents"])
        self.assertEqual(response.tool_calls[0].id, "tu-9")
        stubber.assert_no_pending_responses()

    def test_tool_result_round_trip_preserves_tool_use_id(self):
        result = {"tool": "get_incidents", "ok": True, "availability": "live",
                  "headline": "1 event", "details": [], "data": {}, "error": None}
        messages = [
            {"role": "user", "content": "incidents?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "tu-9", "name": "get_incidents", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "tu-9", "name": "get_incidents", "result": result},
        ]
        provider, stubber = stubbed_provider(
            text_response("One event is open."),
            {"modelId": MODEL_ID,
             "messages": [
                 {"role": "user", "content": [{"text": "incidents?"}]},
                 {"role": "assistant",
                  "content": [{"toolUse": {"toolUseId": "tu-9", "name": "get_incidents",
                                           "input": {}}}]},
                 {"role": "user",
                  "content": [{"toolResult": {"toolUseId": "tu-9", "content": [{"json": result}],
                                              "status": "success"}}]},
             ],
             "inferenceConfig": {"maxTokens": 1024, "temperature": 0.2}})
        self.addCleanup(stubber.deactivate)
        response = provider.converse(messages, tools=[])
        self.assertEqual(response.text, "One event is open.")
        stubber.assert_no_pending_responses()


class TestRequestBuilding(unittest.TestCase):
    """Precise assertions on the request, independent of the service model."""

    def _request(self, **kwargs):
        client = RecordingClient([text_response("ok")])
        provider = BedrockConverseProvider(kwargs.pop("config", bedrock_config()), client=client)
        provider.converse(kwargs.pop("messages"), tools=kwargs.pop("tools", []),
                          system=kwargs.pop("system", ""), context=kwargs.pop("context", None))
        return client.requests[0]

    def test_model_id_max_tokens_and_temperature_are_configurable(self):
        request = self._request(
            config=bedrock_config(bedrock_model_id="arn:aws:bedrock:us-east-1:1:inference-profile/x",
                                  bedrock_max_tokens=256, bedrock_temperature=0.7),
            messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(request["modelId"], "arn:aws:bedrock:us-east-1:1:inference-profile/x")
        self.assertEqual(request["inferenceConfig"], {"maxTokens": 256, "temperature": 0.7})

    def test_system_instructions_and_context_are_separate_blocks(self):
        request = self._request(
            messages=[{"role": "user", "content": "what about the delay?"}], system="SYS",
            context={"last_intent": "journey", "journey_ids": ["jr-1"], "places": ["Silk Board"]})
        self.assertEqual(request["system"][0], {"text": "SYS"})
        self.assertIn("jr-1", request["system"][1]["text"])
        self.assertIn("Silk Board", request["system"][1]["text"])
        self.assertIn("not an instruction", request["system"][1]["text"])

    def test_no_system_and_no_context_omits_the_system_field(self):
        request = self._request(messages=[{"role": "user", "content": "hi"}])
        self.assertNotIn("system", request)

    def test_no_tools_omits_the_tool_config(self):
        request = self._request(messages=[{"role": "user", "content": "hi"}], tools=[])
        self.assertNotIn("toolConfig", request)

    def test_history_becomes_alternating_user_and_assistant_messages(self):
        request = self._request(messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "and now?"}])
        self.assertEqual(request["messages"], [
            {"role": "user", "content": [{"text": "hello"}]},
            {"role": "assistant", "content": [{"text": "hi there"}]},
            {"role": "user", "content": [{"text": "and now?"}]}])

    def test_assistant_text_and_tool_use_share_one_content_list(self):
        request = self._request(messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "let me check",
             "tool_calls": [{"id": "a", "name": "get_incidents", "arguments": {"x": 1}}]},
            {"role": "tool", "tool_call_id": "a", "name": "get_incidents", "result": {"ok": True}}])
        assistant = request["messages"][1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"][0], {"text": "let me check"})
        self.assertEqual(assistant["content"][1]["toolUse"]["toolUseId"], "a")
        self.assertEqual(assistant["content"][1]["toolUse"]["input"], {"x": 1})

    def test_consecutive_tool_results_are_grouped_into_one_user_turn(self):
        request = self._request(messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "a", "name": "get_incidents", "arguments": {}},
                            {"id": "b", "name": "get_system_status", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "a", "name": "get_incidents", "result": {"ok": True}},
            {"role": "tool", "tool_call_id": "b", "name": "get_system_status",
             "result": {"ok": True}}])
        tool_turns = [m for m in request["messages"]
                      if m["role"] == "user" and any("toolResult" in b for b in m["content"])]
        self.assertEqual(len(tool_turns), 1)
        ids = [b["toolResult"]["toolUseId"] for b in tool_turns[0]["content"]]
        self.assertEqual(ids, ["a", "b"])

    def test_an_errored_tool_result_is_marked_as_an_error(self):
        request = self._request(messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "a", "name": "get_journey", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "a", "name": "get_journey",
             "result": {"ok": False, "error": "missing journey_id"}}])
        blocks = tool_result_blocks(request)
        self.assertEqual(blocks[0]["status"], "error")


# --------------------------------------------------------------------------- 3. response parsing

class TestResponseParsing(unittest.TestCase):
    def _parse(self, response):
        return BedrockConverseProvider._parse_response(response)

    def test_plain_text(self):
        parsed = self._parse(text_response("Two incidents are open."))
        self.assertEqual(parsed.text, "Two incidents are open.")
        self.assertFalse(parsed.wants_tools)

    def test_multiple_text_blocks_are_joined(self):
        parsed = self._parse({"output": {"message": {"content": [{"text": "one"},
                                                               {"text": "two"}]}},
                              "stopReason": "end_turn"})
        self.assertEqual(parsed.text, "one\ntwo")

    def test_single_tool_call(self):
        parsed = self._parse(tool_response([{"id": "tu-1", "name": "get_incidents",
                                             "input": {"area": "Silk Board"}}]))
        self.assertEqual(parsed.tool_calls, [ToolCall(name="get_incidents",
                                                      arguments={"area": "Silk Board"},
                                                      id="tu-1")])
        self.assertTrue(parsed.wants_tools)
        self.assertIsNone(parsed.text)

    def test_multiple_tool_calls(self):
        parsed = self._parse(tool_response([
            {"id": "tu-1", "name": "get_incidents", "input": {}},
            {"id": "tu-2", "name": "get_congestion_summary", "input": {}}]))
        self.assertEqual([call.name for call in parsed.tool_calls],
                         ["get_incidents", "get_congestion_summary"])
        self.assertEqual([call.id for call in parsed.tool_calls], ["tu-1", "tu-2"])

    def test_text_and_tool_call_together(self):
        parsed = self._parse(tool_response([{"id": "tu-1", "name": "get_incidents", "input": {}}],
                                           text="Let me check."))
        self.assertEqual(parsed.text, "Let me check.")
        self.assertTrue(parsed.wants_tools)

    def test_stop_reason_and_usage_are_captured(self):
        parsed = self._parse(text_response("hi", stop_reason="max_tokens",
                                           usage={"inputTokens": 5, "outputTokens": 6,
                                                  "totalTokens": 11}))
        self.assertEqual(parsed.stop_reason, "max_tokens")
        self.assertEqual(parsed.usage["totalTokens"], 11)

    def test_malformed_responses_are_rejected_safely(self):
        for bad in ("nope", {}, {"output": None}, {"output": {"message": "x"}},
                    {"output": {"message": {"content": "not-a-list"}}}):
            with self.assertRaises(ProviderUnavailable):
                self._parse(bad)

    def test_a_malformed_tool_use_block_is_skipped_not_fatal(self):
        parsed = self._parse({"output": {"message": {"content": [
            {"toolUse": {"toolUseId": "x"}},                      # no name -> skipped
            "not-a-block",                                        # not a dict -> skipped
            {"toolUse": {"toolUseId": "ok", "name": "get_incidents", "input": {}}}]}},
            "stopReason": "tool_use"})
        self.assertEqual([call.name for call in parsed.tool_calls], ["get_incidents"])
        self.assertEqual(parsed.tool_calls[0].id, "ok")

    def test_absent_usage_is_none(self):
        parsed = self._parse({"output": {"message": {"content": [{"text": "hi"}]}},
                              "stopReason": "end_turn"})
        self.assertIsNone(parsed.usage)
        self.assertEqual(parsed.text, "hi")


# --------------------------------------------------------------------------- 4. error handling

class TestBedrockErrorHandling(unittest.TestCase):
    def _raised_message(self, exc):
        provider = BedrockConverseProvider(bedrock_config(), client=RecordingClient(error=exc))
        with self.assertRaises(ProviderUnavailable) as context:
            provider.converse([{"role": "user", "content": "hi"}], tools=[])
        return str(context.exception)

    @staticmethod
    def _client_error(code, message="SECRET-REQUEST-CONTENTS"):
        return ClientError({"Error": {"Code": code, "Message": message}}, "Converse")

    def test_client_error_codes_map_to_safe_messages(self):
        cases = {
            "AccessDeniedException": "denied",
            "ValidationException": "invalid",
            "ThrottlingException": "throttled",
            "ModelNotReadyException": "not ready",
            "ServiceUnavailableException": "temporarily unavailable",
            "ModelTimeoutException": "too long",
            "ResourceNotFoundException": "not found",
        }
        for code, fragment in cases.items():
            text = self._raised_message(self._client_error(code))
            self.assertIn(fragment, text, code)
            self.assertNotIn("SECRET-REQUEST-CONTENTS", text, code)
            self.assertNotIn("Traceback", text, code)

    def test_unknown_client_error_code_is_still_safe(self):
        text = self._raised_message(self._client_error("SomeBrandNewException"))
        self.assertIn("rejected", text)
        self.assertNotIn("SECRET-REQUEST-CONTENTS", text)

    def test_credential_errors(self):
        for exc, fragment in ((NoCredentialsError(), "credentials"),
                              (PartialCredentialsError(provider="env", cred_var="AWS_SECRET"),
                               "incomplete")):
            text = self._raised_message(exc)
            self.assertIn(fragment, text)
            self.assertNotIn("Traceback", text)

    def test_network_and_timeout_errors(self):
        cases = ((EndpointConnectionError(endpoint_url="https://bedrock-runtime.amazonaws.com"),
                  "could not be reached"),
                 (ConnectTimeoutError(endpoint_url="https://x"), "timed out"),
                 (ReadTimeoutError(endpoint_url="https://x"), "did not respond"),
                 (ConnectionClosedError(endpoint_url="https://x"), "closed"))
        for exc, fragment in cases:
            self.assertIn(fragment, self._raised_message(exc))

    def test_param_validation_error(self):
        text = self._raised_message(ParamValidationError(report="Invalid type for parameter"))
        self.assertIn("invalid parameters", text)

    def test_stubber_client_error_is_translated(self):
        client = boto3.client("bedrock-runtime", region_name="us-east-1",
                              aws_access_key_id="test", aws_secret_access_key="test")
        stubber = Stubber(client)
        stubber.add_client_error("converse", service_error_code="ThrottlingException",
                                 service_message="slow down", http_status_code=429)
        stubber.activate()
        self.addCleanup(stubber.deactivate)
        provider = BedrockConverseProvider(bedrock_config(), client=client)
        with self.assertRaises(ProviderUnavailable) as context:
            provider.converse([{"role": "user", "content": "hi"}], tools=[])
        self.assertIn("throttled", str(context.exception))

    def test_missing_boto3_is_reported_not_crashed(self):
        provider = BedrockConverseProvider(bedrock_config())
        with mock.patch.dict(sys.modules, {"boto3": None}):
            with self.assertRaises(ProviderUnavailable) as context:
                provider.converse([{"role": "user", "content": "hi"}], tools=[])
        self.assertIn("boto3", str(context.exception))


# --------------------------------------------------------------------------- 5. agent loop

class TestAgentLoopWithBedrock(unittest.TestCase):
    def _agent(self, responses, sources=None, **config_kwargs):
        config_kwargs.setdefault("memory_db_path", ":memory:")
        config = bedrock_config(**config_kwargs)
        client = RecordingClient(responses)
        provider = BedrockConverseProvider(config, client=client)
        registry = build_default_registry(sources or default_sources())
        agent = InteractionAgent(config=config, provider=provider, registry=registry,
                                 memory=ConversationMemory(db_path=":memory:", enabled=False))
        return agent, client

    def test_single_tool_call_then_grounded_answer(self):
        agent, client = self._agent([
            tool_response([{"id": "tu-1", "name": "get_incidents", "input": {}}]),
            text_response("One congestion event is open near Silk Board.")])
        reply = agent.answer("Any traffic incidents aahead of me?")
        self.assertTrue(reply.ok)
        self.assertEqual(reply.provider, "bedrock")
        self.assertEqual(reply.text, "One congestion event is open near Silk Board.")
        self.assertEqual([call.name for call in reply.tool_calls],
                         ["get_incidents", "get_journeys"])
        self.assertEqual(reply.tool_calls[0].availability, "live")
        self.assertEqual(tool_result_blocks(client.requests[0]), [])
        # the tool result went back to Bedrock with the same toolUseId, after the assistant turn
        follow_up = client.requests[1]["messages"]
        self.assertTrue(any(
            block.get("toolUse", {}).get("toolUseId") == "tu-1"
            for message in follow_up if message["role"] == "assistant"
            for block in message.get("content", [])))
        blocks = tool_result_blocks(client.requests[1])
        self.assertIn("tu-1", [block["toolUseId"] for block in blocks])
        self.assertIn("related-get_journeys-1", [block["toolUseId"] for block in blocks])

    def test_unsupported_live_answer_retries_with_tools_selected_by_bedrock(self):
        route = journey()
        route["decision"].update({
            "incidents_ahead": 0,
            "incident_ids": [],
            "tool_calls": [{"tool": "check_incidents_on_route", "ok": True,
                            "summary": "0 reported and 0 watched on the road ahead"}],
        })
        sources = FakeSources(
            incidents={"made_at": fresh(), "active_events": [dict(INCIDENT_EVENT)]},
            journeys={route["journey_id"]: route},
        )
        agent, client = self._agent([
            text_response("There are two incidents ahead."),
            tool_response([{"id": "tu-1", "name": "get_incidents", "input": {}}]),
            text_response("Journey Intelligence reports zero active incidents ahead."),
        ], sources=sources)

        reply = agent.answer("Any traffic incidents in front of my car?")

        self.assertEqual(len(client.requests), 3)
        self.assertEqual(tool_result_blocks(client.requests[0]), [])
        retry_instruction = client.requests[1]["messages"][-1]["content"][0]["text"].lower()
        self.assertIn("use the relevant available read-only tool", retry_instruction)
        self.assertEqual([call.name for call in reply.tool_calls],
                         ["get_incidents", "get_journeys"])
        self.assertIn("zero active incidents ahead", reply.text)

    def test_multiple_tool_calls_are_all_executed(self):
        agent, client = self._agent([
            tool_response([{"id": "tu-1", "name": "get_incidents", "input": {}},
                           {"id": "tu-2", "name": "get_system_status", "input": {}}]),
            text_response("Here is what I found.")])
        reply = agent.answer("incidents and system status?")
        self.assertEqual([call.name for call in reply.tool_calls],
                         ["get_incidents", "get_system_status", "get_journeys"])
        blocks = tool_result_blocks(client.requests[1])
        self.assertTrue({"tu-1", "tu-2"}.issubset(
            {block["toolUseId"] for block in blocks}))

    def test_multi_step_tool_execution(self):
        agent, client = self._agent([
            tool_response([{"id": "a", "name": "get_incidents", "input": {}}]),
            tool_response([{"id": "b", "name": "get_congestion_summary", "input": {}}]),
            text_response("Done.")])
        reply = agent.answer("what is going on?")
        self.assertEqual(reply.steps, 3)
        self.assertEqual([call.name for call in reply.tool_calls],
                         ["get_incidents", "get_journeys", "get_congestion_summary"])
        self.assertEqual(reply.text, "Done.")

    def test_step_limit_stops_a_looping_model(self):
        looping = tool_response([{"id": "x", "name": "get_incidents", "input": {}}])
        agent, client = self._agent([looping] * (MAX_STEPS + 3))
        reply = agent.answer("loop forever")
        self.assertEqual(reply.steps, MAX_STEPS)
        self.assertEqual(reply.text, STEP_LIMIT_REPLY)
        self.assertEqual(len(client.requests), MAX_STEPS)             # never called a fifth time

    def test_unknown_tool_is_rejected_safely(self):
        agent, client = self._agent([
            tool_response([{"id": "u", "name": "delete_everything", "input": {}}]),
            text_response("I can't do that.")])
        reply = agent.answer("delete everything")
        self.assertTrue(reply.ok)                                     # the turn still succeeds
        self.assertFalse(reply.tool_calls[0].ok)
        self.assertEqual(reply.tool_calls[0].availability, "unavailable")
        self.assertEqual(reply.text, "I can't do that.")
        self.assertEqual(tool_result_blocks(client.requests[1])[0]["status"], "error")

    def test_invalid_tool_arguments_are_reported_not_crashing(self):
        agent, _ = self._agent([
            tool_response([{"id": "j", "name": "get_journey", "input": {}}]),   # no journey_id
            text_response("Which journey do you mean?")])
        reply = agent.answer("what about that journey?")
        self.assertTrue(reply.ok)
        self.assertTrue(any(not call.ok for call in reply.tool_calls))

    def test_missing_traffic_data_is_reported_not_invented(self):
        sources = FakeSources(incidents=None, features=None, journeys={}, switches=[], health=None)
        agent, _ = self._agent([
            tool_response([{"id": "m", "name": "get_incidents", "input": {}}]),
            text_response("I have no incident data yet.")], sources=sources)
        reply = agent.answer("any incidents?")
        self.assertEqual(reply.tool_calls[0].availability, "missing")

    def test_stale_traffic_data_is_reported_as_stale(self):
        stale = {"made_at": old(120), "active_events": [dict(INCIDENT_EVENT)],
                 "recent_resolved": []}
        sources = FakeSources(incidents=stale, features={"made_at": fresh(), "areas": []},
                              journeys={}, switches=[], health=None)
        agent, _ = self._agent([
            tool_response([{"id": "s", "name": "get_incidents", "input": {}}]),
            text_response("That data may be out of date.")], sources=sources)
        reply = agent.answer("incidents?")
        self.assertEqual(reply.tool_calls[0].availability, "stale")

    def test_provider_failure_does_not_fall_back_to_the_mock(self):
        provider = BedrockConverseProvider(
            bedrock_config(),
            client=RecordingClient(error=ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "Converse")))
        agent = InteractionAgent(config=bedrock_config(), provider=provider,
                                 registry=build_default_registry(default_sources()),
                                 memory=ConversationMemory(db_path=":memory:", enabled=False))
        reply = agent.answer("are there any incidents?")
        self.assertFalse(reply.ok)
        self.assertEqual(reply.provider, "bedrock")                   # the source stays transparent
        self.assertEqual(reply.text, PROVIDER_ERROR_REPLY)
        self.assertNotIn("congestion event", reply.text)               # nothing invented

    def test_conversation_history_is_replayed_to_bedrock(self):
        memory = ConversationMemory(db_path=":memory:", enabled=True)
        self.addCleanup(memory.close)
        client = RecordingClient([text_response("first answer"), text_response("second answer")])
        provider = BedrockConverseProvider(bedrock_config(), client=client)
        agent = InteractionAgent(config=bedrock_config(memory_db_path=":memory:"), provider=provider,
                                 registry=build_default_registry(default_sources()), memory=memory)
        session = agent.new_session()
        agent.answer("hello", session_id=session)
        agent.answer("what about the delay?", session_id=session)
        second = client.requests[1]["messages"]
        self.assertEqual(second[0], {"role": "user", "content": [{"text": "hello"}]})
        self.assertEqual(second[1], {"role": "assistant", "content": [{"text": "first answer"}]})
        self.assertEqual(second[-1], {"role": "user", "content": [{"text": "what about the delay?"}]})

    def test_history_never_replays_old_tool_results_as_evidence(self):
        memory = ConversationMemory(db_path=":memory:", enabled=True)
        self.addCleanup(memory.close)
        client = RecordingClient([
            tool_response([{"id": "t1", "name": "get_incidents", "input": {}}]),
            text_response("One incident is open."),
            tool_response([{"id": "t2", "name": "get_incidents", "input": {}}]),
            text_response("Still one incident.")])
        provider = BedrockConverseProvider(bedrock_config(), client=client)
        agent = InteractionAgent(config=bedrock_config(memory_db_path=":memory:"), provider=provider,
                                 registry=build_default_registry(default_sources()), memory=memory)
        session = agent.new_session()
        agent.answer("incidents?", session_id=session)
        agent.answer("and now?", session_id=session)
        # The next turn starts without replaying tool output from the previous turn.
        first_request_of_turn_two = client.requests[2]
        self.assertEqual(tool_result_blocks(first_request_of_turn_two), [])
        second_request_of_turn_two = client.requests[3]
        blocks = tool_result_blocks(second_request_of_turn_two)
        self.assertEqual([block["toolUseId"] for block in blocks],
                         ["t2", "related-get_journeys-1"])


# --------------------------------------------------------------------------- 6. provider selection

class TestProviderSelection(unittest.TestCase):
    def test_bedrock_is_selected_by_configuration(self):
        provider = build_llm_provider(bedrock_config())
        self.assertIsInstance(provider, BedrockConverseProvider)
        self.assertEqual(provider.name, "bedrock")

    def test_mock_remains_the_default_and_is_unchanged(self):
        provider = build_llm_provider(InteractionConfig.from_env({}))
        self.assertIsInstance(provider, MockLLMProvider)
        response = provider.converse([{"role": "user", "content": "are there any traffic incidents?"}],
                                     tools=[])
        self.assertTrue(response.wants_tools)
        self.assertEqual(response.tool_calls[0].name, "get_incidents")

    def test_bedrock_provider_uses_no_keyword_matching(self):
        # A Bedrock provider is a pure translation layer: no intent rules, no templated answers.
        provider = BedrockConverseProvider(bedrock_config(),
                                           client=RecordingClient([text_response("ok")]))
        self.assertFalse(hasattr(provider, "INTENT_RULES"))
        for attribute in ("detect_intent", "INTENT_TOOL", "CLARIFY"):
            self.assertFalse(hasattr(provider, attribute))


# --------------------------------------------------------------------------- 7. configuration

class TestBedrockConfig(unittest.TestCase):
    def test_defaults(self):
        config = InteractionConfig.from_env({})
        self.assertEqual(config.bedrock_temperature, 0.2)
        self.assertEqual(config.bedrock_connect_timeout_s, 5.0)
        self.assertEqual(config.bedrock_read_timeout_s, 30.0)
        self.assertEqual(config.bedrock_max_retries, 3)

    def test_env_overrides(self):
        config = InteractionConfig.from_env({
            "INTERACTION_LLM_PROVIDER": "bedrock", "BEDROCK_TEMPERATURE": "0.9",
            "BEDROCK_CONNECT_TIMEOUT": "2", "BEDROCK_READ_TIMEOUT": "45",
            "BEDROCK_MAX_RETRIES": "0"})
        self.assertTrue(config.uses_bedrock)
        self.assertEqual(config.bedrock_temperature, 0.9)
        self.assertEqual(config.bedrock_connect_timeout_s, 2.0)
        self.assertEqual(config.bedrock_read_timeout_s, 45.0)
        self.assertEqual(config.bedrock_max_retries, 0)

    def test_invalid_values_are_rejected(self):
        from webapp.interaction_agent import ConfigError
        for env in ({"BEDROCK_TEMPERATURE": "1.5"}, {"BEDROCK_TEMPERATURE": "abc"},
                    {"BEDROCK_CONNECT_TIMEOUT": "0"}, {"BEDROCK_READ_TIMEOUT": "-3"},
                    {"BEDROCK_MAX_RETRIES": "-1"}, {"BEDROCK_MAX_RETRIES": "many"}):
            with self.assertRaises(ConfigError):
                InteractionConfig.from_env(env)

    def test_no_credentials_are_stored(self):
        self.assertFalse(any("KEY" in name or "SECRET" in name
                             for name in InteractionConfig.__dataclass_fields__))


# --------------------------------------------------------------------------- 8. web app wiring

class TestWebAppDoesNotTouchAws(unittest.TestCase):
    """Bedrock mode must not make an AWS call at startup, at status time, or via the chat route."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _services(self, **overrides):
        overrides.setdefault("memory_db_path", os.path.join(self.tmp.name, "i.sqlite"))
        overrides.setdefault("notifications_enabled", False)
        services = InteractionServices(config=bedrock_config(**overrides), sources=default_sources())
        self.addCleanup(services.memory.close)
        self.addCleanup(services.store.close)
        self.addCleanup(services.guard.close)
        return services

    def test_startup_and_status_endpoint_never_call_aws(self):
        with mock.patch("boto3.client", side_effect=AssertionError("an AWS call was attempted")):
            services = self._services()
            app = srv.create_app(model_service_url="http://fake", graph=None, spots=None,
                                 interaction=services)
            with TestClient(app) as client:                            # runs the lifespan
                status = client.get("/api/interaction/status")
                self.assertEqual(status.status_code, 200)
                body = status.json()
                self.assertTrue(body["available"])
                self.assertEqual(body["llm"]["provider"], "bedrock")
                self.assertTrue(body["llm"]["uses_bedrock"])

    def test_status_reports_the_provider_transparently(self):
        services = self._services(notifications_enabled=True, notification_cooldown_min=0.0)
        app = srv.create_app(model_service_url="http://fake", graph=None, spots=None,
                             interaction=services)
        body = TestClient(app).get("/api/interaction/status").json()
        self.assertEqual(body["llm"]["model"], MODEL_ID)
        self.assertEqual(body["agent"]["provider"], "bedrock")

    def test_chat_api_is_compatible_with_a_bedrock_provider(self):
        memory = ConversationMemory(db_path=os.path.join(self.tmp.name, "chat.sqlite"), enabled=True)
        self.addCleanup(memory.close)
        sources = default_sources()
        config = bedrock_config(memory_db_path=memory.db_path, notifications_enabled=True,
                                notification_cooldown_min=0.0)
        client = RecordingClient([
            tool_response([{"id": "t1", "name": "get_incidents", "input": {}}]),
            text_response("One congestion event is open.")])
        provider = BedrockConverseProvider(config, client=client)
        agent = InteractionAgent(config=config, provider=provider,
                                 registry=build_default_registry(sources), memory=memory)
        services = InteractionServices(config=config, sources=sources, memory=memory, agent=agent)
        self.addCleanup(services.memory.close)
        self.addCleanup(services.store.close)
        self.addCleanup(services.guard.close)
        app = srv.create_app(model_service_url="http://fake", graph=None, spots=None,
                             interaction=services)
        body = TestClient(app).post("/api/chat",
                                    json={"message": "are there any incidents?"}).json()
        self.assertTrue(body["ok"], body)
        self.assertEqual(body["provider"], "bedrock")
        self.assertEqual(body["response"], "One congestion event is open.")
        self.assertIn("get_incidents", [call["name"] for call in body["tools_used"]])
        self.assertEqual(body["data"]["sources"]["get_incidents"], "live")


if __name__ == "__main__":
    unittest.main()
