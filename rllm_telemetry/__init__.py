"""Rllm Telemetry — Agent observability for Google ADK, AG2 (AutoGen), and autogen-core."""

from .compare import ComparisonResult, compare
from .config import RllmConfig
from .dataset import load_dataset
from .eval import AsyncEval, Eval, ExperimentResult
from .exporter import AgentSpanExporter, AgentTrajectoryExporter, BaseExporter, BigQueryExporter, BigQueryValidationError, HttpExporter, StdoutExporter, create_exporter, get_bq_table_schema
from .schemas import (
    AgentInfo,
    AgentSpanRecord,
    EventActionsData,
    EventRecord,
    ExperimentCaseRecord,
    ExperimentRecord,
    ExperimentSummary,
    GenerationConfig,
    InvocationRecord,
    LlmRequest,
    LlmResponseData,
    LlmSpanRecord,
    ScoreRecord,
    SessionRecord,
    SessionStartRecord,
    ToolDataRecord,
    ToolInfo,
    ToolSpanRecord,
    TraceEnvelope,
    UsageMetadata,
)
from .scorers import JUDGE_PROMPTS, Contains, ExactMatch, LlmJudge, Score, Scorer
from .trajectory_export import export_trajectories

# ADK plugin — optional dependency
try:
    from .plugin import RllmTelemetryPlugin
except ImportError:
    RllmTelemetryPlugin = None  # type: ignore[assignment,misc]

__all__ = [
    # Plugin (primary API)
    "RllmTelemetryPlugin",
    "RllmConfig",
    # Exporters
    "AgentSpanExporter",
    "AgentTrajectoryExporter",
    "BaseExporter",
    "BigQueryExporter",
    "BigQueryValidationError",
    "HttpExporter",
    "StdoutExporter",
    "create_exporter",
    "get_bq_table_schema",
    # Scorers
    "Score",
    "Scorer",
    "ExactMatch",
    "Contains",
    "LlmJudge",
    "JUDGE_PROMPTS",
    # Experiments
    "Eval",
    "AsyncEval",
    "ExperimentResult",
    # Comparison
    "compare",
    "ComparisonResult",
    # Dataset loading
    "load_dataset",
    # Trajectory export
    "export_trajectories",
    # Convenience
    "instrument",
    "instrument_autogen",
    "instrument_autogen_core",
    # Schemas (for custom backends / testing)
    "AgentInfo",
    "AgentSpanRecord",
    "EventActionsData",
    "EventRecord",
    "GenerationConfig",
    "InvocationRecord",
    "LlmRequest",
    "LlmResponseData",
    "LlmSpanRecord",
    "SessionRecord",
    "SessionStartRecord",
    "ToolDataRecord",
    "ToolInfo",
    "ToolSpanRecord",
    "ExperimentRecord",
    "ExperimentCaseRecord",
    "ExperimentSummary",
    "ScoreRecord",
    "TraceEnvelope",
    "UsageMetadata",
]


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------


def _is_autogen_agent(obj) -> bool:
    """Check if obj is an AG2 ConversableAgent without hard import."""
    try:
        from autogen.agentchat.conversable_agent import ConversableAgent

        return isinstance(obj, ConversableAgent)
    except ImportError:
        return False


def _is_autogen_core_agent(obj) -> bool:
    """Check if obj is an autogen-core BaseChatAgent without hard import."""
    try:
        from autogen_agentchat.agents._base_chat_agent import BaseChatAgent

        return isinstance(obj, BaseChatAgent)
    except ImportError:
        return False


def _is_autogen_core_team(obj) -> bool:
    """Check if obj is an autogen-core BaseGroupChat without hard import."""
    try:
        from autogen_agentchat.teams._group_chat._base_group_chat import BaseGroupChat

        return isinstance(obj, BaseGroupChat)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# AG2 (AutoGen) instrumentation
# ---------------------------------------------------------------------------


def instrument_autogen(
    agents,
    *,
    api_key: str = "",
    endpoint: str = "",
    agent_endpoint: str = "",
    agent_session_name: str = "",
    **kwargs,
):
    """One-liner to attach rllm telemetry to AG2 (AutoGen) agents.

    Example::

        import rllm_telemetry
        from autogen import AssistantAgent, UserProxyAgent

        assistant = AssistantAgent("assistant", llm_config=...)
        user = UserProxyAgent("user")

        rllm_telemetry.instrument_autogen([assistant, user], backend="stdout")
        user.initiate_chat(assistant, message="Hello!")

    Args:
        agents: A single ConversableAgent or a list of agents.
        api_key: Optional API key for the rllm backend.
        endpoint: Optional telemetry endpoint URL.
        agent_endpoint: Optional rllm_ui backend URL for trajectory streaming.
        agent_session_name: Optional human-readable session name.
        **kwargs: Passed to :class:`RllmConfig`.

    Returns:
        The :class:`AutogenInstrumentor` instance (call ``.close()`` when done).
    """
    from .autogen_instrumentor import AutogenInstrumentor

    config_kwargs = {"api_key": api_key, **kwargs}
    if endpoint:
        config_kwargs["endpoint"] = endpoint
    # Allow explicit empty string to disable agent endpoint
    if agent_endpoint is not None:
        config_kwargs["agent_endpoint"] = agent_endpoint
    if agent_session_name:
        config_kwargs["agent_session_name"] = agent_session_name

    config = RllmConfig(**config_kwargs)
    instrumentor = AutogenInstrumentor(config=config)

    # Eagerly start the exporter so the agent session is created on the
    # server before sync AG2 code (initiate_chat) blocks the event loop.
    instrumentor.start_sync()

    # Global LLM wrapper patch
    instrumentor.instrument_llm_wrapper()

    # Instrument agents
    if isinstance(agents, list):
        for agent in agents:
            instrumentor.instrument_agent(agent)
    else:
        instrumentor.instrument_agent(agents)

    return instrumentor


# ---------------------------------------------------------------------------
# autogen-core instrumentation
# ---------------------------------------------------------------------------


def instrument_autogen_core(
    target,
    *,
    api_key: str = "",
    endpoint: str = "",
    agent_endpoint: str = "",
    agent_session_name: str = "",
    **kwargs,
):
    """One-liner to attach rllm telemetry to autogen-core agents or teams.

    Example::

        import rllm_telemetry
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        agent = AssistantAgent("assistant", model_client=client)

        rllm_telemetry.instrument_autogen_core(agent, backend="stdout")
        result = await agent.run(task="Hello!")

    Args:
        target: A BaseChatAgent, BaseGroupChat, or list of agents.
        api_key: Optional API key for the rllm backend.
        endpoint: Optional telemetry endpoint URL.
        agent_endpoint: Optional rllm_ui backend URL for trajectory streaming.
        agent_session_name: Optional human-readable session name.
        **kwargs: Passed to :class:`RllmConfig`.

    Returns:
        The :class:`AutogenCoreInstrumentor` instance (call ``.close()`` when done).
    """
    from .autogen_core_instrumentor import AutogenCoreInstrumentor

    config_kwargs = {"api_key": api_key, **kwargs}
    if endpoint:
        config_kwargs["endpoint"] = endpoint
    if agent_endpoint is not None:
        config_kwargs["agent_endpoint"] = agent_endpoint
    if agent_session_name:
        config_kwargs["agent_session_name"] = agent_session_name

    config = RllmConfig(**config_kwargs)
    instrumentor = AutogenCoreInstrumentor(config=config)

    # Eagerly start the exporter so the agent session is created on the
    # server before any agent code runs.
    instrumentor.start_sync()

    if isinstance(target, list):
        for item in target:
            if _is_autogen_core_team(item):
                instrumentor.instrument_team(item)
            else:
                instrumentor.instrument_agent(item)
    elif _is_autogen_core_team(target):
        instrumentor.instrument_team(target)
    else:
        instrumentor.instrument_agent(target)

    return instrumentor


# ---------------------------------------------------------------------------
# Unified instrument() — auto-detects ADK Runner vs AG2 vs autogen-core
# ---------------------------------------------------------------------------


def instrument(
    target,
    *,
    api_key: str = "",
    endpoint: str = "",
    agent_endpoint: str = "",
    agent_session_name: str = "",
    **kwargs,
):
    """One-liner convenience to attach Rllm telemetry.

    Automatically detects whether ``target`` is an ADK Runner, AG2 agent(s),
    or autogen-core agent/team.

    Example (ADK)::

        rllm_telemetry.instrument(runner, backend="stdout")

    Example (AG2)::

        rllm_telemetry.instrument([assistant, user], backend="stdout")

    Example (autogen-core)::

        rllm_telemetry.instrument(agent, backend="stdout")
    """
    # Auto-detect autogen-core agents/teams (check BEFORE ag2 since both
    # might be installed and autogen-core is the newer framework)
    is_ac = False
    if isinstance(target, list):
        is_ac = any(_is_autogen_core_agent(a) or _is_autogen_core_team(a) for a in target)
    else:
        is_ac = _is_autogen_core_agent(target) or _is_autogen_core_team(target)

    if is_ac:
        return instrument_autogen_core(
            target,
            api_key=api_key,
            endpoint=endpoint,
            agent_endpoint=agent_endpoint,
            agent_session_name=agent_session_name,
            **kwargs,
        )

    # Auto-detect AG2 agents
    is_autogen = False
    if isinstance(target, list):
        is_autogen = any(_is_autogen_agent(a) for a in target)
    else:
        is_autogen = _is_autogen_agent(target)

    if is_autogen:
        return instrument_autogen(
            target,
            api_key=api_key,
            endpoint=endpoint,
            agent_endpoint=agent_endpoint,
            agent_session_name=agent_session_name,
            **kwargs,
        )

    # ADK Runner path
    if RllmTelemetryPlugin is None:
        raise ImportError("google-adk is required for ADK Runner instrumentation. Install it with: pip install rllm-telemetry[adk]")
    config_kwargs = {"api_key": api_key, **kwargs}
    if endpoint:
        config_kwargs["endpoint"] = endpoint
    if agent_endpoint is not None:
        config_kwargs["agent_endpoint"] = agent_endpoint
    if agent_session_name:
        config_kwargs["agent_session_name"] = agent_session_name
    config = RllmConfig(**config_kwargs)
    plugin = RllmTelemetryPlugin(config=config)
    target.plugin_manager.register_plugin(plugin)
    return plugin
