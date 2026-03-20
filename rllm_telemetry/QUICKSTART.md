# rllm_telemetry — Quick Start

Add real-time observability to your agent in **one line of code**. Supports Google ADK, AG2 (AutoGen), and autogen-core.

## 1. Install

`rllm_telemetry` is bundled with the `rllm` package. Install directly from GitHub with only the extras you need:

```bash
# For autogen-core (autogen-agentchat + autogen-ext)
pip install "rllm[telemetry-autogen-core] @ git+https://github.com/boredbichon67/rllm.git@dev-rllm_telemetry"

# For AG2 (autogen community fork)
pip install "rllm[telemetry-autogen] @ git+https://github.com/boredbichon67/rllm.git@dev-rllm_telemetry"

# For Google ADK
pip install "rllm[telemetry-adk] @ git+https://github.com/boredbichon67/rllm.git@dev-rllm_telemetry"
```

After install, `import rllm_telemetry` works out of the box — the framework extras only add the agent SDK dependencies.

## 2. Get your API key

1. Go to **https://rllm-platform.up.railway.app**
2. **Sign up** for an account
3. Navigate to **Settings → API Key → Regenerate**
4. Copy the key and set it as an environment variable:

```bash
export RLLM_API_KEY=rllm_<your-key>
```

> **Local-only testing?** Skip the API key and use `backend="stdout", agent_endpoint=""` to print spans to your terminal — no account needed.

## 3. Add one line to your agent code

### autogen-core

```python
import asyncio
import rllm_telemetry
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent("assistant", model_client=model_client, tools=[...])

    instrumentor = rllm_telemetry.instrument(agent)  # ← this is it

    result = await agent.run(task="Roll a 20-sided die")

    await instrumentor.close()
    await model_client.close()

asyncio.run(main())
```

### AG2 (AutoGen fork)

```python
import asyncio
import rllm_telemetry
from autogen import AssistantAgent, LLMConfig, UserProxyAgent

async def main():
    assistant = AssistantAgent("assistant", llm_config=LLMConfig(...))
    user = UserProxyAgent("user", human_input_mode="NEVER", code_execution_config=False)

    instrumentor = rllm_telemetry.instrument([assistant, user])  # ← this is it

    user.initiate_chat(assistant, message="Roll a 20-sided die")

    await instrumentor.close()

asyncio.run(main())
```

### Google ADK

```python
import asyncio
import rllm_telemetry
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

async def main():
    agent = Agent(model=LiteLlm(model="anthropic/claude-haiku-4-5-20251001"), name="assistant", tools=[...])
    runner = InMemoryRunner(agent=agent, app_name="my_app")

    instrumentor = rllm_telemetry.instrument(runner)  # ← this is it

    session = await runner.session_service.create_session(app_name="my_app", user_id="user1")
    content = types.Content(role="user", parts=[types.Part.from_text(text="Roll a 20-sided die")])
    async for event in runner.run_async(user_id="user1", session_id=session.id, new_message=content):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)

    await instrumentor.close()

asyncio.run(main())
```

`instrument()` auto-detects the framework. You can also call the explicit variants: `instrument_autogen_core()`, `instrument_autogen()`.

## 4. Run the examples

Each example rolls a 20-sided die and checks if the result is prime using tool calls. Spans are printed to stdout and streamed to the rllm cloud dashboard.

### Environment variables

Set these before running any example:

```bash
# Required — rllm cloud dashboard API key (see step 2)
export RLLM_API_KEY=rllm_<your-key>

# Required — at least one LLM provider key
export OPENAI_API_KEY=sk-...        # for autogen-core (OpenAI provider)
export ANTHROPIC_API_KEY=sk-...     # for ADK and AG2 (Anthropic provider)
```

### autogen-core example

Requires `OPENAI_API_KEY`. Uses `gpt-4o-mini` by default.

```bash
cd rllm_telemetry

# Default (gpt-4o-mini)
python examples/autogen_core_example.py

# Use a different model
python examples/autogen_core_example.py --model gpt-4o
```

### AG2 example

Uses Anthropic by default. Pass `--provider openai` for OpenAI.

```bash
cd rllm_telemetry

# Default (Anthropic claude-haiku-4-5)
python examples/ag2_example.py

# Use OpenAI instead
python examples/ag2_example.py --provider openai --model gpt-4o-mini
```

### ADK example

Uses Anthropic via LiteLLM by default. Any LiteLLM model string works.

```bash
cd rllm_telemetry

# Default (Anthropic claude-haiku-4-5 via LiteLLM)
python examples/adk_example.py

# Use OpenAI via LiteLLM
python examples/adk_example.py --model openai/gpt-4o-mini
```

### autogen-core with LiteLLM proxy

Requires a running LiteLLM proxy. See `examples/autogen_core_litellm_example.py` for details.

```bash
# Terminal 1: Start LiteLLM proxy
ANTHROPIC_API_KEY=sk-... litellm --model anthropic/claude-haiku-4-5-20251001 --port 4000

# Terminal 2: Run example
cd rllm_telemetry
python examples/autogen_core_litellm_example.py
```

## What you'll see

- **Live session timeline** — each agent run appears as a session with a full span tree
- **LLM calls** — model, tokens, latency, full request/response content
- **Tool calls** — tool name, args, results, duration
- **Dashboard** — aggregate metrics, top models, top tools, error rates

## Options

```python
# Stream to the rllm observability dashboard (default)
instrumentor = rllm_telemetry.instrument(agent)

# Print spans to terminal only (no server, no API key needed)
instrumentor = rllm_telemetry.instrument(agent, backend="stdout", agent_endpoint="")

# Custom session name in the dashboard
instrumentor = rllm_telemetry.instrument(agent, agent_session_name="my-experiment")
```

## Cleanup

Always close the instrumentor to flush pending spans and mark the session as completed:

```python
instrumentor = rllm_telemetry.instrument(agent)
# ... run your agent ...
await instrumentor.close()
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ValueError: An API key is required` | Set `RLLM_API_KEY` env var, or use `agent_endpoint=""` for local-only |
| `401 Unauthorized` | Check your API key is valid (Settings → API Key) |
| Spans don't appear in dashboard | Check the ClickHouse data source tab in Observability |
| `ImportError: ag2 not installed` | Run `pip install "rllm[telemetry-autogen] @ git+https://github.com/boredbichon67/rllm.git@dev-rllm_telemetry"` |
| `ImportError: autogen_agentchat` | Run `pip install "rllm[telemetry-autogen-core] @ git+https://github.com/boredbichon67/rllm.git@dev-rllm_telemetry"` |
| `ImportError: google.adk` | Run `pip install "rllm[telemetry-adk] @ git+https://github.com/boredbichon67/rllm.git@dev-rllm_telemetry"` |
