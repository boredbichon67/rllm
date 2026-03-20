"""AutoGen (ag2) instrumentation for rllm_telemetry.

Monkey-patches AG2 agent methods to capture the full execution trace
using the same schema hierarchy as the ADK plugin:

    Session → Invocation → AgentSpan → LlmSpan / ToolSpan

Uses the same pattern as AG2's own first-party OpenTelemetry integration
(``autogen.opentelemetry.instrumentators``), with idempotency guards
and error-safe wrappers.

Usage::

    import rllm_telemetry
    from autogen import AssistantAgent, UserProxyAgent

    assistant = AssistantAgent("assistant", llm_config=...)
    user = UserProxyAgent("user", ...)

    rllm_telemetry.instrument([assistant, user], backend="stdout")
    user.initiate_chat(assistant, message="Hello!")
"""

from __future__ import annotations

import contextvars
import functools
import json
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
    ToolSpanRecord,
    UsageMetadata,
)

logger = logging.getLogger("rllm_telemetry.autogen")

# ---------------------------------------------------------------------------
# Context propagation — tracks current invocation across threads
# ---------------------------------------------------------------------------

_current_invocation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_rllm_current_invocation_id", default=None)
_current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_rllm_current_session_id", default=None)


# ---------------------------------------------------------------------------
# Error-safe wrapper
# ---------------------------------------------------------------------------


def _safe_telemetry(func):
    """Swallow exceptions in telemetry code so the agent keeps running."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception(
                "rllm_telemetry autogen error in %s; skipping.",
                func.__name__,
            )
            return None

    return wrapper


def _safe_telemetry_async(func):
    """Async version of _safe_telemetry."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception:
            logger.exception(
                "rllm_telemetry autogen error in %s; skipping.",
                func.__name__,
            )
            return None

    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_agent_info(agent, recipient=None) -> AgentInfo:
    """Build AgentInfo tree from AG2 agents."""
    sub_agents = []
    if recipient and hasattr(recipient, "name"):
        sub_agents.append(
            AgentInfo(
                name=recipient.name,
                description=getattr(recipient, "description", None) or "",
                type=type(recipient).__name__,
            )
        )
    return AgentInfo(
        name=agent.name,
        description=getattr(agent, "description", None) or "",
        type=type(agent).__name__,
        sub_agents=sub_agents,
    )


def _extract_usage(response) -> UsageMetadata | None:
    """Extract token usage from an OpenAI ChatCompletion response."""
    if not hasattr(response, "usage") or not response.usage:
        return None
    usage = response.usage
    return UsageMetadata(
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _extract_tool_infos(config: dict[str, Any]) -> list[ToolInfo] | None:
    """Extract tool schemas from OpenAI-format function definitions."""
    tools = config.get("tools") or config.get("functions")
    if not tools:
        return None
    infos = []
    for tool in tools:
        if isinstance(tool, dict):
            func = tool.get("function", tool)
            infos.append(
                ToolInfo(
                    name=func.get("name", "unknown"),
                    description=func.get("description"),
                )
            )
    return infos or None


def _get_model_from_wrapper(wrapper) -> str | None:
    """Extract model name from OpenAIWrapper's config_list."""
    config_list = getattr(wrapper, "_config_list", None)
    if config_list and isinstance(config_list, list) and config_list:
        return config_list[0].get("model")
    return None


def _safe_dict(obj: Any) -> dict[str, Any] | None:
    """Best-effort conversion to a JSON-safe dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    try:
        return obj.model_dump(mode="json", exclude_none=True)
    except Exception:
        pass
    try:
        return dict(obj)
    except Exception:
        return {"_raw": str(obj)}


def _truncate_strings(obj: Any, max_len: int) -> Any:
    """Recursively truncate string values."""
    if isinstance(obj, str):
        return obj[:max_len] + "...[TRUNCATED]" if len(obj) > max_len else obj
    if isinstance(obj, dict):
        return {k: _truncate_strings(v, max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_strings(v, max_len) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Core instrumentor
# ---------------------------------------------------------------------------


class AutogenInstrumentor:
    """Holds telemetry state and exporter for AG2 instrumentation.

    A single instrumentor instance is shared across all instrumented agents
    in a session.  Mirrors the ADK plugin's state-tracking pattern.
    """

    def __init__(self, config: RllmConfig) -> None:
        self._config = config
        self._exporter: BaseExporter = create_exporter(config)
        self._started = False

        # Open invocations: invocation_id → InvocationRecord
        self._invocations: dict[str, InvocationRecord] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_sync(self) -> None:
        """Synchronously start the exporter using httpx sync client.

        This ensures the agent session is created on the server before
        sync AG2 code (``initiate_chat``) blocks the event loop.
        All span sending will use sync httpx via a thread pool.
        """
        if self._started:
            return
        import httpx

        from .exporter import AgentSpanExporter

        exporter = self._exporter
        # If wrapped in AgentSpanExporter, create the session synchronously
        # and mark it as sync mode (no async client will be created)
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
                logger.info(
                    "Agent session created (sync): %s",
                    exporter._agent_session_id,
                )
            except Exception as exc:
                logger.warning("Failed to create agent session (sync): %s", exc)
                print(
                    f"[rllm_telemetry] ERROR: Failed to create agent session — {type(exc).__name__}: {exc}",
                    file=__import__("sys").stderr,
                )
                exporter._agent_session_id = None
            # Start the inner exporter (stdout is sync no-op)
            exporter._inner._closed = False
        else:
            # Non-AgentSpan exporter (e.g., plain stdout) — just mark as open
            if hasattr(exporter, "_closed"):
                exporter._closed = False
        self._started = True

    def _ensure_started_sync(self) -> None:
        """No-op if already started, otherwise best-effort sync start."""
        if self._started:
            return
        self.start_sync()

    async def start(self) -> None:
        """Async start the exporter."""
        if self._started:
            return
        await self._exporter.start()
        self._started = True

    async def close(self) -> None:
        """Flush pending records and release resources."""
        try:
            await self._exporter.close()
        except Exception:
            logger.exception("Error closing exporter")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def instrument_agent(self, agent) -> None:
        """Instrument a single AG2 ConversableAgent.

        Patches ``initiate_chat``, ``generate_reply``, and
        ``execute_function`` (both sync and async variants).
        """
        self._patch_initiate_chat(agent)
        self._patch_generate_reply(agent)
        self._patch_execute_function(agent)

    def instrument_llm_wrapper(self) -> None:
        """Global patch on ``OpenAIWrapper.create()`` for LlmSpan capture."""
        try:
            from autogen.oai import client as oai_client_module
            from autogen.oai.client import OpenAIWrapper
        except ImportError:
            logger.warning("ag2 not installed; skipping LLM wrapper instrumentation.")
            return

        original_create = OpenAIWrapper.create
        if hasattr(original_create, "__rllm_wrapped__"):
            return

        instrumentor = self

        def traced_create(wrapper_self, **config: Any) -> Any:
            inv_id = _current_invocation_id.get(None)
            if inv_id is None:
                return original_create(wrapper_self, **config)

            session_id = _current_session_id.get("unknown")
            inv = instrumentor._invocations.get(inv_id)

            # Extract agent name
            agent = config.get("agent")
            agent_name = agent.name if agent and hasattr(agent, "name") else "unknown"

            # Extract model
            model = config.get("model") or _get_model_from_wrapper(wrapper_self)

            # Build LlmRequest
            request_data = LlmRequest(
                model=model,
                tools=_extract_tool_infos(config),
            )
            if instrumentor._config.capture_content and "messages" in config:
                messages = config["messages"]
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                request_data.system_instruction = sys_msgs[0]["content"] if sys_msgs else None
                request_data.contents = messages

            record = LlmSpanRecord(
                span_id=str(uuid.uuid4()),
                invocation_id=inv_id,
                session_id=session_id,
                agent_name=agent_name,
                request=request_data,
                started_at=time.time(),
            )
            instrumentor._exporter.enqueue("llm.start", record.model_dump(exclude_none=True))

            try:
                response = original_create(wrapper_self, **config)
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
                instrumentor._exporter.enqueue("llm.end", record.model_dump(exclude_none=True))
                raise

            # Extract response data
            usage = _extract_usage(response)
            content_data = None
            finish_reason = None
            model_version = getattr(response, "model", None)

            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                finish_reason = str(choice.finish_reason) if choice.finish_reason else None
                if instrumentor._config.capture_content and hasattr(choice, "message"):
                    msg = choice.message
                    content_data = {
                        "role": "assistant",
                        "content": getattr(msg, "content", None),
                    }
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        content_data["tool_calls"] = [
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ]

            record.response = LlmResponseData(
                content=content_data,
                finish_reason=finish_reason,
                model_version=model_version,
                usage=usage,
            )
            record.ended_at = time.time()
            record.duration_ms = (record.ended_at - record.started_at) * 1000

            # Update invocation aggregates
            if inv:
                inv.llm_call_count += 1
                if usage:
                    inv.total_input_tokens += usage.input_tokens or 0
                    inv.total_output_tokens += usage.output_tokens or 0

            instrumentor._exporter.enqueue("llm.end", record.model_dump(exclude_none=True))
            return response

        traced_create.__rllm_wrapped__ = True
        OpenAIWrapper.create = traced_create
        oai_client_module.OpenAIWrapper.create = traced_create

    # ------------------------------------------------------------------
    # Patch: initiate_chat → Session + Invocation
    # ------------------------------------------------------------------

    def _patch_initiate_chat(self, agent) -> None:
        # --- Sync ---
        if hasattr(agent, "initiate_chat") and not hasattr(agent.initiate_chat, "__rllm_wrapped__"):
            original = agent.initiate_chat
            instrumentor = self

            @functools.wraps(original)
            def traced_initiate_chat(
                recipient,
                *args,
                message=None,
                max_turns=None,
                **kwargs,
            ):
                instrumentor._ensure_started_sync()

                session_id = str(uuid.uuid4())
                invocation_id = str(uuid.uuid4())

                # Session record
                session_record = SessionRecord(
                    session_id=session_id,
                    app_name=agent.name,
                    user_id="autogen",
                    created_at=time.time(),
                )
                instrumentor._exporter.enqueue("session", session_record.model_dump(exclude_none=True))

                # User message
                user_message = None
                if isinstance(message, str):
                    user_message = {"role": "user", "content": message}
                elif isinstance(message, dict):
                    user_message = message

                # Invocation start
                inv_record = InvocationRecord(
                    invocation_id=invocation_id,
                    session_id=session_id,
                    app_name=agent.name,
                    user_id="autogen",
                    user_message=user_message,
                    root_agent=_build_agent_info(agent, recipient),
                    started_at=time.time(),
                )
                instrumentor._invocations[invocation_id] = inv_record
                instrumentor._exporter.enqueue("invocation.start", inv_record.model_dump(exclude_none=True))

                token_inv = _current_invocation_id.set(invocation_id)
                token_sess = _current_session_id.set(session_id)
                try:
                    result = original(
                        recipient,
                        *args,
                        message=message,
                        max_turns=max_turns,
                        **kwargs,
                    )
                    return result
                except Exception:
                    inv_record.error_count += 1
                    raise
                finally:
                    _current_invocation_id.reset(token_inv)
                    _current_session_id.reset(token_sess)
                    inv_record.ended_at = time.time()
                    inv_record.duration_ms = (inv_record.ended_at - inv_record.started_at) * 1000
                    instrumentor._exporter.enqueue("invocation.end", inv_record.model_dump(exclude_none=True))
                    instrumentor._invocations.pop(invocation_id, None)

            traced_initiate_chat.__rllm_wrapped__ = True
            agent.initiate_chat = traced_initiate_chat

        # --- Async ---
        if hasattr(agent, "a_initiate_chat") and not hasattr(agent.a_initiate_chat, "__rllm_wrapped__"):
            original_async = agent.a_initiate_chat
            instrumentor = self

            @functools.wraps(original_async)
            async def traced_a_initiate_chat(
                recipient,
                *args,
                message=None,
                max_turns=None,
                **kwargs,
            ):
                instrumentor._ensure_started_sync()

                session_id = str(uuid.uuid4())
                invocation_id = str(uuid.uuid4())

                session_record = SessionRecord(
                    session_id=session_id,
                    app_name=agent.name,
                    user_id="autogen",
                    created_at=time.time(),
                )
                instrumentor._exporter.enqueue("session", session_record.model_dump(exclude_none=True))

                user_message = None
                if isinstance(message, str):
                    user_message = {"role": "user", "content": message}
                elif isinstance(message, dict):
                    user_message = message

                inv_record = InvocationRecord(
                    invocation_id=invocation_id,
                    session_id=session_id,
                    app_name=agent.name,
                    user_id="autogen",
                    user_message=user_message,
                    root_agent=_build_agent_info(agent, recipient),
                    started_at=time.time(),
                )
                instrumentor._invocations[invocation_id] = inv_record
                instrumentor._exporter.enqueue("invocation.start", inv_record.model_dump(exclude_none=True))

                token_inv = _current_invocation_id.set(invocation_id)
                token_sess = _current_session_id.set(session_id)
                try:
                    result = await original_async(
                        recipient,
                        *args,
                        message=message,
                        max_turns=max_turns,
                        **kwargs,
                    )
                    return result
                except Exception:
                    inv_record.error_count += 1
                    raise
                finally:
                    _current_invocation_id.reset(token_inv)
                    _current_session_id.reset(token_sess)
                    inv_record.ended_at = time.time()
                    inv_record.duration_ms = (inv_record.ended_at - inv_record.started_at) * 1000
                    instrumentor._exporter.enqueue("invocation.end", inv_record.model_dump(exclude_none=True))
                    instrumentor._invocations.pop(invocation_id, None)

            traced_a_initiate_chat.__rllm_wrapped__ = True
            agent.a_initiate_chat = traced_a_initiate_chat

    # ------------------------------------------------------------------
    # Patch: generate_reply → AgentSpan
    # ------------------------------------------------------------------

    def _patch_generate_reply(self, agent) -> None:
        # --- Sync ---
        if hasattr(agent, "generate_reply") and not hasattr(agent.generate_reply, "__rllm_wrapped__"):
            original = agent.generate_reply
            instrumentor = self

            @functools.wraps(original)
            def traced_generate_reply(messages=None, sender=None, **kwargs):
                inv_id = _current_invocation_id.get(None)
                if inv_id is None:
                    return original(messages, sender, **kwargs)

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
                    result = original(messages, sender, **kwargs)
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

            traced_generate_reply.__rllm_wrapped__ = True
            agent.generate_reply = traced_generate_reply

        # --- Async ---
        if hasattr(agent, "a_generate_reply") and not hasattr(agent.a_generate_reply, "__rllm_wrapped__"):
            original_async = agent.a_generate_reply
            instrumentor = self

            @functools.wraps(original_async)
            async def traced_a_generate_reply(messages=None, sender=None, **kwargs):
                inv_id = _current_invocation_id.get(None)
                if inv_id is None:
                    return await original_async(messages, sender, **kwargs)

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
                    result = await original_async(messages, sender, **kwargs)
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

            traced_a_generate_reply.__rllm_wrapped__ = True
            agent.a_generate_reply = traced_a_generate_reply

    # ------------------------------------------------------------------
    # Patch: execute_function → ToolSpan
    # ------------------------------------------------------------------

    def _patch_execute_function(self, agent) -> None:
        # --- Sync ---
        if hasattr(agent, "execute_function") and not hasattr(agent.execute_function, "__rllm_wrapped__"):
            original = agent.execute_function
            instrumentor = self

            @functools.wraps(original)
            def traced_execute_function(func_call, call_id=None, verbose=False):
                inv_id = _current_invocation_id.get(None)
                if inv_id is None:
                    return original(func_call, call_id, verbose)

                session_id = _current_session_id.get("unknown")
                inv = instrumentor._invocations.get(inv_id)
                func_name = func_call.get("name", "unknown")

                # Parse tool arguments
                args_data = None
                if instrumentor._config.capture_tools:
                    raw_args = func_call.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            args_data = json.loads(raw_args)
                        except (json.JSONDecodeError, TypeError):
                            args_data = {"_raw": raw_args}
                    elif isinstance(raw_args, dict):
                        args_data = raw_args
                    if args_data and instrumentor._config.max_content_length > 0:
                        args_data = _truncate_strings(args_data, instrumentor._config.max_content_length)

                record = ToolSpanRecord(
                    span_id=str(uuid.uuid4()),
                    invocation_id=inv_id,
                    session_id=session_id,
                    agent_name=agent.name,
                    tool_name=func_name,
                    tool_type="function",
                    args=args_data,
                    started_at=time.time(),
                )
                try:
                    instrumentor._exporter.enqueue("tool.start", record.model_dump(exclude_none=True))
                except Exception:
                    logger.exception("Failed to enqueue tool.start")

                try:
                    is_success, result = original(func_call, call_id, verbose)

                    if instrumentor._config.capture_tools and isinstance(result, dict):
                        result_data = _safe_dict(result)
                        if instrumentor._config.max_content_length > 0:
                            result_data = _truncate_strings(result_data, instrumentor._config.max_content_length)
                        record.result = result_data

                    if not is_success:
                        content = result.get("content", "execution failed") if isinstance(result, dict) else str(result)
                        record.error = content
                        if inv:
                            inv.error_count += 1

                    return is_success, result
                except Exception as exc:
                    record.error = f"{type(exc).__name__}: {exc}"
                    if inv:
                        inv.error_count += 1
                    raise
                finally:
                    record.ended_at = time.time()
                    record.duration_ms = (record.ended_at - record.started_at) * 1000
                    if inv:
                        inv.tool_call_count += 1
                    try:
                        instrumentor._exporter.enqueue("tool.end", record.model_dump(exclude_none=True))
                    except Exception:
                        logger.exception("Failed to enqueue tool.end")

            traced_execute_function.__rllm_wrapped__ = True
            agent.execute_function = traced_execute_function

        # --- Async ---
        if hasattr(agent, "a_execute_function") and not hasattr(agent.a_execute_function, "__rllm_wrapped__"):
            original_async = agent.a_execute_function
            instrumentor = self

            @functools.wraps(original_async)
            async def traced_a_execute_function(func_call, call_id=None, verbose=False):
                inv_id = _current_invocation_id.get(None)
                if inv_id is None:
                    return await original_async(func_call, call_id, verbose)

                session_id = _current_session_id.get("unknown")
                inv = instrumentor._invocations.get(inv_id)
                func_name = func_call.get("name", "unknown")

                args_data = None
                if instrumentor._config.capture_tools:
                    raw_args = func_call.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            args_data = json.loads(raw_args)
                        except (json.JSONDecodeError, TypeError):
                            args_data = {"_raw": raw_args}
                    elif isinstance(raw_args, dict):
                        args_data = raw_args
                    if args_data and instrumentor._config.max_content_length > 0:
                        args_data = _truncate_strings(args_data, instrumentor._config.max_content_length)

                record = ToolSpanRecord(
                    span_id=str(uuid.uuid4()),
                    invocation_id=inv_id,
                    session_id=session_id,
                    agent_name=agent.name,
                    tool_name=func_name,
                    tool_type="function",
                    args=args_data,
                    started_at=time.time(),
                )
                try:
                    instrumentor._exporter.enqueue("tool.start", record.model_dump(exclude_none=True))
                except Exception:
                    logger.exception("Failed to enqueue tool.start")

                try:
                    is_success, result = await original_async(func_call, call_id, verbose)

                    if instrumentor._config.capture_tools and isinstance(result, dict):
                        result_data = _safe_dict(result)
                        if instrumentor._config.max_content_length > 0:
                            result_data = _truncate_strings(result_data, instrumentor._config.max_content_length)
                        record.result = result_data

                    if not is_success:
                        content = result.get("content", "execution failed") if isinstance(result, dict) else str(result)
                        record.error = content
                        if inv:
                            inv.error_count += 1

                    return is_success, result
                except Exception as exc:
                    record.error = f"{type(exc).__name__}: {exc}"
                    if inv:
                        inv.error_count += 1
                    raise
                finally:
                    record.ended_at = time.time()
                    record.duration_ms = (record.ended_at - record.started_at) * 1000
                    if inv:
                        inv.tool_call_count += 1
                    try:
                        instrumentor._exporter.enqueue("tool.end", record.model_dump(exclude_none=True))
                    except Exception:
                        logger.exception("Failed to enqueue tool.end")

            traced_a_execute_function.__rllm_wrapped__ = True
            agent.a_execute_function = traced_a_execute_function
