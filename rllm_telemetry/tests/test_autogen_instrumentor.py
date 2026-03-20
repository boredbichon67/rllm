"""Tests for AG2 (AutoGen) instrumentation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from rllm_telemetry.autogen_instrumentor import (
    AutogenInstrumentor,
    _build_agent_info,
    _current_invocation_id,
    _current_session_id,
    _extract_tool_infos,
    _extract_usage,
)
from rllm_telemetry.config import RllmConfig
from rllm_telemetry.exporter import BaseExporter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class CollectingExporter(BaseExporter):
    """In-memory exporter that collects enqueued records for assertions."""

    def __init__(self):
        self.records: list[tuple[str, dict[str, Any]]] = []

    def enqueue(self, span_type, data):
        self.records.append((span_type, data))

    async def start(self):
        pass

    async def close(self):
        pass

    def types(self) -> list[str]:
        return [r[0] for r in self.records]

    def by_type(self, span_type: str) -> list[dict[str, Any]]:
        return [r[1] for r in self.records if r[0] == span_type]


class FakeAgent:
    """Minimal mock of an AG2 ConversableAgent."""

    def __init__(self, name="test_agent", description="A test agent"):
        self.name = name
        self.description = description

    def initiate_chat(self, recipient, *args, message=None, max_turns=None, **kwargs):
        # Simulate calling generate_reply on both agents
        self.generate_reply(messages=[{"role": "user", "content": message}])
        return MagicMock(chat_id=1, chat_history=[], cost={})

    async def a_initiate_chat(self, recipient, *args, message=None, max_turns=None, **kwargs):
        return MagicMock(chat_id=1, chat_history=[], cost={})

    def generate_reply(self, messages=None, sender=None, **kwargs):
        return "Hello!"

    async def a_generate_reply(self, messages=None, sender=None, **kwargs):
        return "Hello!"

    def execute_function(self, func_call, call_id=None, verbose=False):
        return True, {"name": func_call.get("name"), "content": "result"}

    async def a_execute_function(self, func_call, call_id=None, verbose=False):
        return True, {"name": func_call.get("name"), "content": "result"}


@pytest.fixture
def config():
    return RllmConfig(backend="stdout", api_key="test", agent_endpoint="")


@pytest.fixture
def exporter():
    return CollectingExporter()


@pytest.fixture
def instrumentor(config, exporter):
    inst = AutogenInstrumentor(config=config)
    inst._exporter = exporter
    inst._started = True
    return inst


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_build_agent_info_single(self):
        agent = FakeAgent(name="assistant")
        info = _build_agent_info(agent)
        assert info.name == "assistant"
        assert info.type == "FakeAgent"
        assert info.sub_agents == []

    def test_build_agent_info_with_recipient(self):
        agent = FakeAgent(name="user")
        recipient = FakeAgent(name="assistant")
        info = _build_agent_info(agent, recipient)
        assert info.name == "user"
        assert len(info.sub_agents) == 1
        assert info.sub_agents[0].name == "assistant"

    def test_extract_usage_none(self):
        assert _extract_usage(MagicMock(usage=None)) is None

    def test_extract_usage_valid(self):
        response = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        response.usage.total_tokens = 150
        usage = _extract_usage(response)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_extract_tool_infos_none(self):
        assert _extract_tool_infos({}) is None

    def test_extract_tool_infos_valid(self):
        config = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                    },
                }
            ]
        }
        infos = _extract_tool_infos(config)
        assert len(infos) == 1
        assert infos[0].name == "get_weather"


# ---------------------------------------------------------------------------
# Instrumentation tests
# ---------------------------------------------------------------------------


class TestInstrumentAgent:
    def test_patches_applied(self, instrumentor):
        agent = FakeAgent()
        instrumentor.instrument_agent(agent)

        assert hasattr(agent.initiate_chat, "__rllm_wrapped__")
        assert hasattr(agent.generate_reply, "__rllm_wrapped__")
        assert hasattr(agent.execute_function, "__rllm_wrapped__")
        assert hasattr(agent.a_initiate_chat, "__rllm_wrapped__")
        assert hasattr(agent.a_generate_reply, "__rllm_wrapped__")
        assert hasattr(agent.a_execute_function, "__rllm_wrapped__")

    def test_idempotent(self, instrumentor):
        agent = FakeAgent()
        instrumentor.instrument_agent(agent)
        first_initiate = agent.initiate_chat
        instrumentor.instrument_agent(agent)
        assert agent.initiate_chat is first_initiate


class TestInitiateChat:
    def test_emits_session_and_invocation(self, instrumentor, exporter):
        agent = FakeAgent(name="user_proxy")
        recipient = FakeAgent(name="assistant")
        instrumentor.instrument_agent(agent)

        agent.initiate_chat(recipient, message="Hello!")

        types = exporter.types()
        assert "session" in types
        assert "invocation.start" in types
        assert "invocation.end" in types

        # Session record
        session = exporter.by_type("session")[0]
        assert session["app_name"] == "user_proxy"

        # Invocation records
        inv_start = exporter.by_type("invocation.start")[0]
        inv_end = exporter.by_type("invocation.end")[0]
        assert inv_start["invocation_id"] == inv_end["invocation_id"]
        assert inv_start["session_id"] == session["session_id"]
        assert inv_end["duration_ms"] is not None
        assert inv_end["duration_ms"] >= 0

    def test_user_message_captured(self, instrumentor, exporter):
        agent = FakeAgent()
        recipient = FakeAgent()
        instrumentor.instrument_agent(agent)

        agent.initiate_chat(recipient, message="What is 2+2?")

        inv = exporter.by_type("invocation.start")[0]
        assert inv["user_message"] == {"role": "user", "content": "What is 2+2?"}


class TestGenerateReply:
    def test_emits_agent_span(self, instrumentor, exporter):
        agent = FakeAgent(name="assistant")
        instrumentor.instrument_agent(agent)

        # Set context (normally done by initiate_chat)
        token_inv = _current_invocation_id.set("inv-123")
        token_sess = _current_session_id.set("sess-456")
        try:
            result = agent.generate_reply(messages=[{"role": "user", "content": "Hi"}])
        finally:
            _current_invocation_id.reset(token_inv)
            _current_session_id.reset(token_sess)

        assert result == "Hello!"

        types = exporter.types()
        assert "agent.start" in types
        assert "agent.end" in types

        span = exporter.by_type("agent.end")[0]
        assert span["agent_name"] == "assistant"
        assert span["invocation_id"] == "inv-123"
        assert span["session_id"] == "sess-456"
        assert span["duration_ms"] >= 0

    def test_skips_without_invocation_context(self, instrumentor, exporter):
        agent = FakeAgent()
        instrumentor.instrument_agent(agent)

        # No invocation context set
        result = agent.generate_reply()
        assert result == "Hello!"
        assert len(exporter.records) == 0


class TestExecuteFunction:
    def test_emits_tool_span(self, instrumentor, exporter):
        agent = FakeAgent(name="executor")
        instrumentor.instrument_agent(agent)

        token_inv = _current_invocation_id.set("inv-123")
        token_sess = _current_session_id.set("sess-456")
        instrumentor._invocations["inv-123"] = MagicMock(tool_call_count=0, error_count=0)
        try:
            is_success, result = agent.execute_function(
                {"name": "get_weather", "arguments": '{"city": "SF"}'},
                call_id="call-1",
            )
        finally:
            _current_invocation_id.reset(token_inv)
            _current_session_id.reset(token_sess)

        assert is_success is True

        types = exporter.types()
        assert "tool.start" in types
        assert "tool.end" in types

        start = exporter.by_type("tool.start")[0]
        assert start["tool_name"] == "get_weather"
        assert start["args"] == {"city": "SF"}

        end = exporter.by_type("tool.end")[0]
        assert end["duration_ms"] >= 0

    def test_skips_without_invocation_context(self, instrumentor, exporter):
        agent = FakeAgent()
        instrumentor.instrument_agent(agent)

        is_success, result = agent.execute_function({"name": "get_weather", "arguments": "{}"})
        assert is_success is True
        assert len(exporter.records) == 0


class TestErrorSafety:
    def test_telemetry_error_does_not_crash_agent(self, instrumentor, exporter):
        """Telemetry errors should be swallowed — agent continues normally."""
        agent = FakeAgent()
        instrumentor.instrument_agent(agent)

        # Make exporter raise on every enqueue
        def boom(*args, **kwargs):
            raise RuntimeError("exporter is broken")

        exporter.enqueue = boom

        # Set context so telemetry code runs
        token_inv = _current_invocation_id.set("inv-123")
        token_sess = _current_session_id.set("sess-456")
        try:
            # generate_reply wraps telemetry in try/except, but the original
            # call should still succeed even if enqueue raises
            result = agent.generate_reply(messages=[{"role": "user", "content": "Hi"}])
            # The agent should still return its result
            assert result == "Hello!"
        finally:
            _current_invocation_id.reset(token_inv)
            _current_session_id.reset(token_sess)


class TestEndToEnd:
    def test_full_flow(self, instrumentor, exporter):
        """Simulate a full initiate_chat flow and verify all spans emitted."""
        user = FakeAgent(name="user_proxy")
        assistant = FakeAgent(name="assistant")
        instrumentor.instrument_agent(user)
        instrumentor.instrument_agent(assistant)

        user.initiate_chat(assistant, message="Roll a die")

        types = exporter.types()
        # Should have: session, invocation.start, agent.start, agent.end, invocation.end
        assert "session" in types
        assert "invocation.start" in types
        assert "agent.start" in types
        assert "agent.end" in types
        assert "invocation.end" in types


class TestAutoDetect:
    def test_instrument_detects_autogen(self):
        """Verify instrument() auto-detects AG2 agents."""
        from autogen.agentchat.conversable_agent import ConversableAgent

        import rllm_telemetry

        # Use a real ConversableAgent (with no LLM config so it won't try to call)
        agent = ConversableAgent(name="test", llm_config=False)

        result = rllm_telemetry.instrument(agent, backend="stdout", agent_endpoint="")
        assert isinstance(result, AutogenInstrumentor)
        assert hasattr(agent.initiate_chat, "__rllm_wrapped__")

    def test_instrument_autogen_explicit(self):
        """Verify instrument_autogen() works directly."""
        from autogen.agentchat.conversable_agent import ConversableAgent

        import rllm_telemetry

        agent = ConversableAgent(name="test2", llm_config=False)
        result = rllm_telemetry.instrument_autogen(agent, backend="stdout", agent_endpoint="")
        assert isinstance(result, AutogenInstrumentor)
        assert hasattr(agent.generate_reply, "__rllm_wrapped__")
        assert hasattr(agent.execute_function, "__rllm_wrapped__")
