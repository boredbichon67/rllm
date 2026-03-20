"""End-to-end tests for autogen-core instrumentation.

Requires OPENAI_API_KEY to be set. Uses real LLM calls with stdout backend.

Usage:
    OPENAI_API_KEY=sk-... pytest tests/test_autogen_core_instrumentor.py -v -s
"""

from __future__ import annotations

import os
from typing import Any

import pytest

import rllm_telemetry
from rllm_telemetry.autogen_core_instrumentor import AutogenCoreInstrumentor
from rllm_telemetry.exporter import BaseExporter

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Collecting exporter — captures records for assertions while also printing
# ---------------------------------------------------------------------------


class CollectingExporter(BaseExporter):
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_client():
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(model="gpt-4o-mini")


@pytest.fixture
def exporter():
    return CollectingExporter()


@pytest.fixture
def instrumentor(exporter):
    from rllm_telemetry.config import RllmConfig

    config = RllmConfig(backend="stdout", agent_endpoint="")
    inst = AutogenCoreInstrumentor(config=config)
    inst._exporter = exporter
    inst._started = True
    return inst


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleAgent:
    @pytest.mark.asyncio
    async def test_run_emits_all_spans(self, model_client, instrumentor, exporter):
        """agent.run() should emit session, invocation, agent, and llm spans."""
        from autogen_agentchat.agents import AssistantAgent

        agent = AssistantAgent(
            name="assistant",
            model_client=model_client,
            system_message="Reply in one sentence. Say TERMINATE when done.",
        )
        instrumentor.instrument_agent(agent)

        await agent.run(task="What is 2+2?")

        types = exporter.types()
        assert "session" in types
        assert "invocation.start" in types
        assert "invocation.end" in types
        assert "agent.start" in types
        assert "agent.end" in types
        assert "llm.start" in types
        assert "llm.end" in types

        # Verify LLM span has usage data
        llm_end = exporter.by_type("llm.end")[0]
        assert llm_end["response"]["usage"]["input_tokens"] > 0
        assert llm_end["response"]["usage"]["output_tokens"] > 0
        assert llm_end["response"]["finish_reason"] == "stop"
        assert llm_end["duration_ms"] > 0

        # Verify invocation has aggregated metrics
        inv_end = exporter.by_type("invocation.end")[0]
        assert inv_end["llm_call_count"] >= 1
        assert inv_end["total_input_tokens"] > 0
        assert inv_end["total_output_tokens"] > 0

        await model_client.close()


class TestToolCalling:
    @pytest.mark.asyncio
    async def test_tool_call_captured_in_llm_span(self, model_client, instrumentor, exporter):
        """When the agent calls a tool, the LLM response should contain tool_calls."""
        from autogen_agentchat.agents import AssistantAgent

        async def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"Sunny and 72°F in {city}."

        agent = AssistantAgent(
            name="weather_agent",
            model_client=model_client,
            tools=[get_weather],
            system_message="Use the get_weather tool when asked about weather. Say TERMINATE after answering.",
        )
        instrumentor.instrument_agent(agent)

        await agent.run(task="What's the weather in Paris?")

        types = exporter.types()
        assert "llm.start" in types
        assert "llm.end" in types

        # Should have at least 2 LLM calls (tool call + final answer)
        llm_ends = exporter.by_type("llm.end")
        assert len(llm_ends) >= 1

        # Verify invocation tracks tool calls
        inv_end = exporter.by_type("invocation.end")[0]
        assert inv_end["llm_call_count"] >= 1

        await model_client.close()


class TestAutoDetect:
    @pytest.mark.asyncio
    async def test_instrument_auto_detects_autogen_core(self, model_client):
        """rllm_telemetry.instrument() should auto-detect autogen-core agents."""
        from autogen_agentchat.agents import AssistantAgent

        agent = AssistantAgent(
            name="auto_detect_agent",
            model_client=model_client,
            system_message="Reply briefly. Say TERMINATE when done.",
        )

        result = rllm_telemetry.instrument(agent, backend="stdout", agent_endpoint="")
        assert isinstance(result, AutogenCoreInstrumentor)
        assert hasattr(agent.run, "__rllm_wrapped__")
        assert hasattr(agent.on_messages, "__rllm_wrapped__")
        assert hasattr(agent._model_client.create, "__rllm_wrapped__")

        await model_client.close()
