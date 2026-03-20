"""autogen-core (autogen-agentchat) instrumentation for rllm_telemetry.

Monkey-patches autogen-core agent and model client methods to capture the full
execution trace using the same schema hierarchy as the ADK and AG2 plugins:

    Session → Invocation → AgentSpan → LlmSpan

Uses instance-level patching — no import-order dependency.

Usage::

    import rllm_telemetry
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent("assistant", model_client=client)

    rllm_telemetry.instrument(agent, backend="stdout", agent_endpoint="")

    result = await agent.run(task="What is 2+2?")
"""

from __future__ import annotations

import contextvars
import functools
import logging
import time
import uuid
from typing import Any

from .config import RllmConfig
from .exporter import BaseExporter, create_exporter
from .schemas import (
    AgentInfo,
    AgentSpanRecord,
    InvocationRecord,
    LlmRequest,
    LlmResponseData,
    LlmSpanRecord,
    SessionRecord,
    ToolInfo,
    UsageMetadata,
)

logger = logging.getLogger("rllm_telemetry.autogen_core")

# ---------------------------------------------------------------------------
# Context propagation
# ---------------------------------------------------------------------------

_current_invocation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_rllm_ac_invocation_id", default=None)
_current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_rllm_ac_session_id", default=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_agent_info(agent) -> AgentInfo:
    """Build AgentInfo from an autogen-core BaseChatAgent."""
    return AgentInfo(
        name=getattr(agent, "name", "unknown"),
        description=getattr(agent, "description", None) or "",
        type=type(agent).__name__,
    )


def _build_team_agent_info(team) -> AgentInfo:
    """Build AgentInfo tree from a team with participants."""
    participants = getattr(team, "_participants", [])
    sub_agents = [_build_agent_info(p) for p in participants]
    return AgentInfo(
        name=getattr(team, "_team_id", "team"),
        description=getattr(team, "description", None) or "",
        type=type(team).__name__,
        sub_agents=sub_agents,
    )


def _extract_tool_infos(tools) -> list[ToolInfo] | None:
    """Extract tool info from autogen-core tool sequence."""
    if not tools:
        return None
    infos = []
    for tool in tools:
        if hasattr(tool, "name"):
            infos.append(
                ToolInfo(
                    name=tool.name,
                    description=getattr(tool, "description", None),
                )
            )
        elif isinstance(tool, dict):
            func = tool.get("function", tool)
            infos.append(
                ToolInfo(
                    name=func.get("name", "unknown"),
                    description=func.get("description"),
                )
            )
    return infos or None


def _serialize_messages(messages) -> list[dict[str, Any]] | None:
    """Serialize autogen-core LLMMessage sequence to dicts."""
    if not messages:
        return None
    result = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            result.append(msg.model_dump(mode="json", exclude_none=True))
        elif isinstance(msg, dict):
            result.append(msg)
        else:
            result.append({"_raw": str(msg)})
    return result


def _extract_model_name(client) -> str | None:
    """Extract model name from a ChatCompletionClient.

    Supports OpenAIChatCompletionClient with standard models and
    LiteLLM proxy (custom base_url + model names like 'anthropic/claude-3-haiku').
    """
    # autogen-core stores the model in _create_args["model"]
    create_args = getattr(client, "_create_args", None)
    if create_args and isinstance(create_args, dict) and "model" in create_args:
        return create_args["model"]
    # Fallback: _resolved_model (may include version suffix)
    if hasattr(client, "_resolved_model") and client._resolved_model:
        return client._resolved_model
    # Fallback: _raw_config
    raw_config = getattr(client, "_raw_config", None)
    if raw_config and isinstance(raw_config, dict) and "model" in raw_config:
        return raw_config["model"]
    # Last resort: model_info family
    try:
        info = client.model_info
        return info.get("family", None)
    except Exception:
        return None


def _task_to_user_message(task) -> dict[str, Any] | None:
    """Convert autogen-core task argument to a user_message dict."""
    if task is None:
        return None
    if isinstance(task, str):
        return {"role": "user", "content": task}
    if hasattr(task, "model_dump"):
        return task.model_dump(mode="json", exclude_none=True)
    if isinstance(task, list | tuple):
        contents = []
        for m in task:
            if hasattr(m, "model_dump"):
                contents.append(m.model_dump(mode="json", exclude_none=True))
            else:
                contents.append({"_raw": str(m)})
        return {"role": "user", "content": contents}
    return {"role": "user", "content": str(task)}


# ---------------------------------------------------------------------------
# Core instrumentor
# ---------------------------------------------------------------------------


class AutogenCoreInstrumentor:
    """Holds telemetry state and exporter for autogen-core instrumentation."""

    def __init__(self, config: RllmConfig) -> None:
        self._config = config
        self._exporter: BaseExporter = create_exporter(config)
        self._started = False
        self._invocations: dict[str, InvocationRecord] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_sync(self) -> None:
        """Synchronously start the exporter (creates agent session on server)."""
        if self._started:
            return
        import httpx

        from .exporter import AgentSpanExporter

        exporter = self._exporter
        if isinstance(exporter, AgentSpanExporter):
            exporter._sync_mode = True
            exporter._closed = False
            try:
                config = self._config
                key = config.agent_api_key or config.api_key
                headers = {"Authorization": f"Bearer {key}"} if key else {}
                name = config.agent_session_name or f"agent-{uuid.uuid4().hex[:8]}"
                resp = httpx.post(
                    f"{config.agent_endpoint}/api/agent-sessions",
                    json={"name": name},
                    headers=headers,
                    timeout=config.timeout_seconds,
                )
                resp.raise_for_status()
                exporter._agent_session_id = resp.json()["id"]
                logger.info("Agent session created (sync): %s", exporter._agent_session_id)
            except Exception as exc:
                logger.warning("Failed to create agent session (sync): %s", exc)
                print(
                    f"[rllm_telemetry] ERROR: Failed to create agent session — {type(exc).__name__}: {exc}",
                    file=__import__("sys").stderr,
                )
                exporter._agent_session_id = None
            exporter._inner._closed = False
        else:
            if hasattr(exporter, "_closed"):
                exporter._closed = False
        self._started = True

    async def close(self) -> None:
        await self._exporter.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def instrument_agent(self, agent) -> None:
        """Instrument a single autogen-core BaseChatAgent.

        Patches ``run()``, ``run_stream()``, ``on_messages()``,
        ``on_messages_stream()``, and the agent's model client if available.
        """
        self._patch_run(agent)
        self._patch_on_messages(agent)
        # Instrument model client if accessible
        model_client = getattr(agent, "_model_client", None)
        if model_client is not None:
            self.instrument_model_client(model_client)

    def instrument_team(self, team) -> None:
        """Instrument a team (BaseGroupChat) and all its participant agents."""
        self._patch_team_run(team)
        # Instrument all participant agents
        participants = getattr(team, "_participants", [])
        for participant in participants:
            self.instrument_agent(participant)

    def instrument_model_client(self, client) -> None:
        """Patch ``ChatCompletionClient.create()`` on a client instance."""
        self._patch_create(client)

    # ------------------------------------------------------------------
    # Patch: run() / run_stream() → Session + Invocation
    # ------------------------------------------------------------------

    def _patch_run(self, agent) -> None:
        if hasattr(agent, "run") and not hasattr(agent.run, "__rllm_wrapped__"):
            original = agent.run
            instrumentor = self

            @functools.wraps(original)
            async def traced_run(*, task=None, cancellation_token=None, **kwargs):
                if not instrumentor._started:
                    instrumentor.start_sync()

                session_id = str(uuid.uuid4())
                invocation_id = str(uuid.uuid4())

                try:
                    instrumentor._exporter.enqueue(
                        "session",
                        SessionRecord(
                            session_id=session_id,
                            app_name=agent.name,
                            user_id="autogen_core",
                            created_at=time.time(),
                        ).model_dump(exclude_none=True),
                    )
                except Exception:
                    logger.exception("Failed to enqueue session")

                inv_record = InvocationRecord(
                    invocation_id=invocation_id,
                    session_id=session_id,
                    app_name=agent.name,
                    user_id="autogen_core",
                    user_message=_task_to_user_message(task),
                    root_agent=_build_agent_info(agent),
                    started_at=time.time(),
                )
                instrumentor._invocations[invocation_id] = inv_record
                try:
                    instrumentor._exporter.enqueue("invocation.start", inv_record.model_dump(exclude_none=True))
                except Exception:
                    logger.exception("Failed to enqueue invocation.start")

                token_inv = _current_invocation_id.set(invocation_id)
                token_sess = _current_session_id.set(session_id)
                try:
                    result = await original(task=task, cancellation_token=cancellation_token, **kwargs)
                    return result
                except Exception:
                    inv_record.error_count += 1
                    raise
                finally:
                    _current_invocation_id.reset(token_inv)
                    _current_session_id.reset(token_sess)
                    inv_record.ended_at = time.time()
                    inv_record.duration_ms = (inv_record.ended_at - inv_record.started_at) * 1000
                    try:
                        instrumentor._exporter.enqueue(
                            "invocation.end",
                            inv_record.model_dump(exclude_none=True),
                        )
                    except Exception:
                        logger.exception("Failed to enqueue invocation.end")
                    instrumentor._invocations.pop(invocation_id, None)

            traced_run.__rllm_wrapped__ = True
            agent.run = traced_run

        # run_stream — async generator
        if hasattr(agent, "run_stream") and not hasattr(agent.run_stream, "__rllm_wrapped__"):
            original_stream = agent.run_stream
            instrumentor = self

            @functools.wraps(original_stream)
            async def traced_run_stream(*, task=None, cancellation_token=None, **kwargs):
                if not instrumentor._started:
                    instrumentor.start_sync()

                session_id = str(uuid.uuid4())
                invocation_id = str(uuid.uuid4())

                try:
                    instrumentor._exporter.enqueue(
                        "session",
                        SessionRecord(
                            session_id=session_id,
                            app_name=agent.name,
                            user_id="autogen_core",
                            created_at=time.time(),
                        ).model_dump(exclude_none=True),
                    )
                except Exception:
                    logger.exception("Failed to enqueue session")

                inv_record = InvocationRecord(
                    invocation_id=invocation_id,
                    session_id=session_id,
                    app_name=agent.name,
                    user_id="autogen_core",
                    user_message=_task_to_user_message(task),
                    root_agent=_build_agent_info(agent),
                    started_at=time.time(),
                )
                instrumentor._invocations[invocation_id] = inv_record
                try:
                    instrumentor._exporter.enqueue(
                        "invocation.start",
                        inv_record.model_dump(exclude_none=True),
                    )
                except Exception:
                    logger.exception("Failed to enqueue invocation.start")

                token_inv = _current_invocation_id.set(invocation_id)
                token_sess = _current_session_id.set(session_id)
                try:
                    async for item in original_stream(task=task, cancellation_token=cancellation_token, **kwargs):
                        yield item
                except Exception:
                    inv_record.error_count += 1
                    raise
                finally:
                    _current_invocation_id.reset(token_inv)
                    _current_session_id.reset(token_sess)
                    inv_record.ended_at = time.time()
                    inv_record.duration_ms = (inv_record.ended_at - inv_record.started_at) * 1000
                    try:
                        instrumentor._exporter.enqueue(
                            "invocation.end",
                            inv_record.model_dump(exclude_none=True),
                        )
                    except Exception:
                        logger.exception("Failed to enqueue invocation.end")
                    instrumentor._invocations.pop(invocation_id, None)

            traced_run_stream.__rllm_wrapped__ = True
            agent.run_stream = traced_run_stream

    def _patch_team_run(self, team) -> None:
        """Patch run()/run_stream() on a team (BaseGroupChat)."""
        # Teams also have run() and run_stream() with the same TaskRunner interface
        # We use the same pattern but build agent info from team participants
        if hasattr(team, "run") and not hasattr(team.run, "__rllm_wrapped__"):
            original = team.run
            instrumentor = self

            @functools.wraps(original)
            async def traced_run(*, task=None, cancellation_token=None, **kwargs):
                if not instrumentor._started:
                    instrumentor.start_sync()

                session_id = str(uuid.uuid4())
                invocation_id = str(uuid.uuid4())

                team_name = getattr(team, "_team_id", type(team).__name__)

                try:
                    instrumentor._exporter.enqueue(
                        "session",
                        SessionRecord(
                            session_id=session_id,
                            app_name=team_name,
                            user_id="autogen_core",
                            created_at=time.time(),
                        ).model_dump(exclude_none=True),
                    )
                except Exception:
                    logger.exception("Failed to enqueue session")

                inv_record = InvocationRecord(
                    invocation_id=invocation_id,
                    session_id=session_id,
                    app_name=team_name,
                    user_id="autogen_core",
                    user_message=_task_to_user_message(task),
                    root_agent=_build_team_agent_info(team),
                    started_at=time.time(),
                )
                instrumentor._invocations[invocation_id] = inv_record
                try:
                    instrumentor._exporter.enqueue(
                        "invocation.start",
                        inv_record.model_dump(exclude_none=True),
                    )
                except Exception:
                    logger.exception("Failed to enqueue invocation.start")

                token_inv = _current_invocation_id.set(invocation_id)
                token_sess = _current_session_id.set(session_id)
                try:
                    result = await original(task=task, cancellation_token=cancellation_token, **kwargs)
                    return result
                except Exception:
                    inv_record.error_count += 1
                    raise
                finally:
                    _current_invocation_id.reset(token_inv)
                    _current_session_id.reset(token_sess)
                    inv_record.ended_at = time.time()
                    inv_record.duration_ms = (inv_record.ended_at - inv_record.started_at) * 1000
                    try:
                        instrumentor._exporter.enqueue(
                            "invocation.end",
                            inv_record.model_dump(exclude_none=True),
                        )
                    except Exception:
                        logger.exception("Failed to enqueue invocation.end")
                    instrumentor._invocations.pop(invocation_id, None)

            traced_run.__rllm_wrapped__ = True
            team.run = traced_run

        if hasattr(team, "run_stream") and not hasattr(team.run_stream, "__rllm_wrapped__"):
            original_stream = team.run_stream
            instrumentor = self

            @functools.wraps(original_stream)
            async def traced_run_stream(*, task=None, cancellation_token=None, **kwargs):
                if not instrumentor._started:
                    instrumentor.start_sync()

                session_id = str(uuid.uuid4())
                invocation_id = str(uuid.uuid4())
                team_name = getattr(team, "_team_id", type(team).__name__)

                try:
                    instrumentor._exporter.enqueue(
                        "session",
                        SessionRecord(
                            session_id=session_id,
                            app_name=team_name,
                            user_id="autogen_core",
                            created_at=time.time(),
                        ).model_dump(exclude_none=True),
                    )
                except Exception:
                    logger.exception("Failed to enqueue session")

                inv_record = InvocationRecord(
                    invocation_id=invocation_id,
                    session_id=session_id,
                    app_name=team_name,
                    user_id="autogen_core",
                    user_message=_task_to_user_message(task),
                    root_agent=_build_team_agent_info(team),
                    started_at=time.time(),
                )
                instrumentor._invocations[invocation_id] = inv_record
                try:
                    instrumentor._exporter.enqueue(
                        "invocation.start",
                        inv_record.model_dump(exclude_none=True),
                    )
                except Exception:
                    logger.exception("Failed to enqueue invocation.start")

                token_inv = _current_invocation_id.set(invocation_id)
                token_sess = _current_session_id.set(session_id)
                try:
                    async for item in original_stream(task=task, cancellation_token=cancellation_token, **kwargs):
                        yield item
                except Exception:
                    inv_record.error_count += 1
                    raise
                finally:
                    _current_invocation_id.reset(token_inv)
                    _current_session_id.reset(token_sess)
                    inv_record.ended_at = time.time()
                    inv_record.duration_ms = (inv_record.ended_at - inv_record.started_at) * 1000
                    try:
                        instrumentor._exporter.enqueue(
                            "invocation.end",
                            inv_record.model_dump(exclude_none=True),
                        )
                    except Exception:
                        logger.exception("Failed to enqueue invocation.end")
                    instrumentor._invocations.pop(invocation_id, None)

            traced_run_stream.__rllm_wrapped__ = True
            team.run_stream = traced_run_stream

    # ------------------------------------------------------------------
    # Patch: on_messages() / on_messages_stream() → AgentSpan
    # ------------------------------------------------------------------

    def _patch_on_messages(self, agent) -> None:
        if hasattr(agent, "on_messages") and not hasattr(agent.on_messages, "__rllm_wrapped__"):
            original = agent.on_messages
            instrumentor = self

            @functools.wraps(original)
            async def traced_on_messages(messages, cancellation_token=None, **kwargs):
                inv_id = _current_invocation_id.get(None)
                if inv_id is None:
                    return await original(messages, cancellation_token, **kwargs)

                session_id = _current_session_id.get("unknown")
                inv = instrumentor._invocations.get(inv_id)

                record = AgentSpanRecord(
                    span_id=str(uuid.uuid4()),
                    invocation_id=inv_id,
                    session_id=session_id,
                    agent_name=agent.name,
                    agent_description=getattr(agent, "description", None) or "",
                    agent_type=type(agent).__name__,
                    started_at=time.time(),
                )
                try:
                    instrumentor._exporter.enqueue("agent.start", record.model_dump(exclude_none=True))
                except Exception:
                    logger.exception("Failed to enqueue agent.start")

                try:
                    result = await original(messages, cancellation_token, **kwargs)
                    return result
                except Exception as exc:
                    record.error = f"{type(exc).__name__}: {exc}"
                    if inv:
                        inv.error_count += 1
                    raise
                finally:
                    record.ended_at = time.time()
                    record.duration_ms = (record.ended_at - record.started_at) * 1000
                    try:
                        instrumentor._exporter.enqueue("agent.end", record.model_dump(exclude_none=True))
                    except Exception:
                        logger.exception("Failed to enqueue agent.end")

            traced_on_messages.__rllm_wrapped__ = True
            agent.on_messages = traced_on_messages

        if hasattr(agent, "on_messages_stream") and not hasattr(agent.on_messages_stream, "__rllm_wrapped__"):
            original_stream = agent.on_messages_stream
            instrumentor = self

            @functools.wraps(original_stream)
            async def traced_on_messages_stream(messages, cancellation_token=None, **kwargs):
                inv_id = _current_invocation_id.get(None)
                if inv_id is None:
                    async for item in original_stream(messages, cancellation_token, **kwargs):
                        yield item
                    return

                session_id = _current_session_id.get("unknown")
                inv = instrumentor._invocations.get(inv_id)

                record = AgentSpanRecord(
                    span_id=str(uuid.uuid4()),
                    invocation_id=inv_id,
                    session_id=session_id,
                    agent_name=agent.name,
                    agent_description=getattr(agent, "description", None) or "",
                    agent_type=type(agent).__name__,
                    started_at=time.time(),
                )
                try:
                    instrumentor._exporter.enqueue("agent.start", record.model_dump(exclude_none=True))
                except Exception:
                    logger.exception("Failed to enqueue agent.start")

                try:
                    async for item in original_stream(messages, cancellation_token, **kwargs):
                        yield item
                except Exception as exc:
                    record.error = f"{type(exc).__name__}: {exc}"
                    if inv:
                        inv.error_count += 1
                    raise
                finally:
                    record.ended_at = time.time()
                    record.duration_ms = (record.ended_at - record.started_at) * 1000
                    try:
                        instrumentor._exporter.enqueue("agent.end", record.model_dump(exclude_none=True))
                    except Exception:
                        logger.exception("Failed to enqueue agent.end")

            traced_on_messages_stream.__rllm_wrapped__ = True
            agent.on_messages_stream = traced_on_messages_stream

    # ------------------------------------------------------------------
    # Patch: ChatCompletionClient.create() → LlmSpan
    # ------------------------------------------------------------------

    def _patch_create(self, client) -> None:
        if hasattr(client, "create") and not hasattr(client.create, "__rllm_wrapped__"):
            original = client.create
            instrumentor = self

            @functools.wraps(original)
            async def traced_create(messages, *, tools=None, **kwargs):
                if tools is None:
                    tools = []
                inv_id = _current_invocation_id.get(None)
                if inv_id is None:
                    return await original(messages, tools=tools, **kwargs)

                session_id = _current_session_id.get("unknown")
                inv = instrumentor._invocations.get(inv_id)
                model = _extract_model_name(client)

                request_data = LlmRequest(
                    model=model,
                    tools=_extract_tool_infos(tools),
                )
                if instrumentor._config.capture_content:
                    request_data.contents = _serialize_messages(messages)

                record = LlmSpanRecord(
                    span_id=str(uuid.uuid4()),
                    invocation_id=inv_id,
                    session_id=session_id,
                    agent_name="model_client",
                    request=request_data,
                    started_at=time.time(),
                )
                try:
                    instrumentor._exporter.enqueue("llm.start", record.model_dump(exclude_none=True))
                except Exception:
                    logger.exception("Failed to enqueue llm.start")

                try:
                    result = await original(messages, tools=tools, **kwargs)
                except Exception as exc:
                    record.response = LlmResponseData(
                        error_code=type(exc).__name__,
                        error_message=str(exc),
                    )
                    if inv:
                        inv.error_count += 1
                        inv.llm_call_count += 1
                    record.ended_at = time.time()
                    record.duration_ms = (record.ended_at - record.started_at) * 1000
                    try:
                        instrumentor._exporter.enqueue("llm.end", record.model_dump(exclude_none=True))
                    except Exception:
                        logger.exception("Failed to enqueue llm.end")
                    raise

                # Extract response data from CreateResult
                usage = None
                if hasattr(result, "usage") and result.usage:
                    usage = UsageMetadata(
                        input_tokens=getattr(result.usage, "prompt_tokens", None),
                        output_tokens=getattr(result.usage, "completion_tokens", None),
                    )

                content_data = None
                if instrumentor._config.capture_content:
                    if isinstance(result.content, str):
                        content_data = {
                            "role": "assistant",
                            "content": result.content,
                        }
                    elif isinstance(result.content, list):
                        # List[FunctionCall]
                        content_data = {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": fc.id,
                                    "function": {
                                        "name": fc.name,
                                        "arguments": fc.arguments,
                                    },
                                }
                                for fc in result.content
                                if hasattr(fc, "name")
                            ],
                        }

                record.response = LlmResponseData(
                    content=content_data,
                    finish_reason=getattr(result, "finish_reason", None),
                    model_version=model,
                    usage=usage,
                )
                record.ended_at = time.time()
                record.duration_ms = (record.ended_at - record.started_at) * 1000

                if inv:
                    inv.llm_call_count += 1
                    if usage:
                        inv.total_input_tokens += usage.input_tokens or 0
                        inv.total_output_tokens += usage.output_tokens or 0

                try:
                    instrumentor._exporter.enqueue("llm.end", record.model_dump(exclude_none=True))
                except Exception:
                    logger.exception("Failed to enqueue llm.end")

                return result

            traced_create.__rllm_wrapped__ = True
            client.create = traced_create
