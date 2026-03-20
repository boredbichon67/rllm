"""End-to-end autogen-core example with rllm_telemetry.

Demonstrates the one-liner instrumentation for autogen-core agents
with tool calling — same agent as the AG2 and ADK examples.
Spans are printed to stdout AND streamed to the rllm cloud endpoint.

Usage:
    OPENAI_API_KEY=sk-... RLLM_API_KEY=rllm_... python examples/autogen_core_example.py
    OPENAI_API_KEY=sk-... RLLM_API_KEY=rllm_... python examples/autogen_core_example.py --model gpt-4o
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
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    args = parser.parse_args()

    # Create model client
    model_client = OpenAIChatCompletionClient(model=args.model)

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
    print(f"autogen-core + rllm_telemetry ({args.model})")
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
