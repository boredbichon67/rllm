"""End-to-end autogen-core example using a LiteLLM proxy with Anthropic.

Demonstrates rllm_telemetry instrumentation when the autogen-core agent
uses a LiteLLM proxy to route requests to Anthropic (or any other provider).

Usage:
    1. Start a LiteLLM proxy (in a separate terminal):
           ANTHROPIC_API_KEY=sk-... litellm --model anthropic/claude-haiku-4-5-20251001 --port 4000

    2. Run this example:
           RLLM_API_KEY=rllm_... python examples/autogen_core_litellm_example.py

    Custom proxy URL or model:
        python examples/autogen_core_litellm_example.py --base-url http://my-proxy:8000 --model my-custom-model
"""

import argparse
import asyncio
import random

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

import rllm_telemetry

RLLM_TELEMETRY_ENDPOINT = "https://rllm-platform-api.up.railway.app"

# ----- Tools -----


async def roll_die(sides: int) -> str:
    """Roll a die with the given number of sides.

    Args:
        sides: Number of sides on the die.

    Returns:
        The result of the roll.
    """
    result = random.randint(1, sides)
    return f"Rolled a {result} on a d{sides}."


async def check_prime(n: int) -> str:
    """Check if a number is prime.

    Args:
        n: The number to check.

    Returns:
        Whether the number is prime.
    """
    if n <= 1:
        return f"{n} is not prime."
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return f"{n} is not prime (divisible by {i})."
    return f"{n} is prime!"


# ----- Main -----


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="anthropic/claude-haiku-4-5-20251001",
        help="Model name as configured in the LiteLLM proxy",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:4000",
        help="LiteLLM proxy URL (default: http://localhost:4000)",
    )
    args = parser.parse_args()

    # Create model client pointing at LiteLLM proxy.
    # api_key="no-key" because the proxy handles auth.
    model_client = OpenAIChatCompletionClient(
        model=args.model,
        base_url=args.base_url,
        api_key="no-key",
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )

    # Create agent with tools
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[roll_die, check_prime],
        system_message=("You are a helpful assistant. When asked to roll dice, use the roll_die tool. When asked to check primes, use the check_prime tool. Give concise answers. Reply TERMINATE when the task is fully done."),
    )

    # ---- One-liner instrumentation (stdout + cloud streaming) ----
    instrumentor = rllm_telemetry.instrument(
        agent,
        backend="stdout",
        agent_endpoint=RLLM_TELEMETRY_ENDPOINT,
    )

    print("\n" + "=" * 60)
    print(f"autogen-core + LiteLLM proxy ({args.model})")
    print(f"Proxy: {args.base_url}")
    print("=" * 60 + "\n")

    result = await agent.run(
        task="Roll a 20-sided die, then check if the result is prime.",
        cancellation_token=CancellationToken(),
    )

    print("\n" + "=" * 60)
    print("Task Result:")
    for msg in result.messages:
        print(f"  [{type(msg).__name__}] {msg}")
    print("=" * 60)

    await instrumentor.close()
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
