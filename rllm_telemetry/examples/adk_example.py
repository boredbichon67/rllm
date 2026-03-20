"""End-to-end Google ADK example with rllm_telemetry.

Demonstrates the one-liner instrumentation for ADK agents with tool calling
— same agent as the AG2 and autogen-core examples.
Spans are printed to stdout AND streamed to the rllm cloud endpoint.

Usage:
    # With Anthropic (via LiteLLM):
    ANTHROPIC_API_KEY=sk-... RLLM_API_KEY=rllm_... python examples/adk_example.py

    # With OpenAI (via LiteLLM):
    OPENAI_API_KEY=sk-... RLLM_API_KEY=rllm_... python examples/adk_example.py --model openai/gpt-4o-mini
"""

import argparse
import asyncio
import random

from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

import rllm_telemetry

RLLM_TELEMETRY_ENDPOINT = "https://rllm-platform-api.up.railway.app"

# ----- Tools -----


def roll_die(sides: int) -> str:
    """Roll a die with the given number of sides.

    Args:
        sides: The integer number of sides the die has.

    Returns:
        The result of the roll.
    """
    result = random.randint(1, sides)
    return f"Rolled a {result} on a d{sides}."


def check_prime(n: int) -> str:
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


# ----- Agent -----


def make_agent(model: str) -> Agent:
    return Agent(
        model=LiteLlm(model=model),
        name="assistant",
        description="Agent that can roll dice and check prime numbers.",
        instruction=("You roll dice and answer questions about the outcome of the dice rolls. When asked to roll a die, call the roll_die tool. When asked to check primes, call the check_prime tool. Give concise answers."),
        tools=[roll_die, check_prime],
    )


# ----- Main -----


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="anthropic/claude-haiku-4-5-20251001",
        help="LiteLLM model string (e.g. anthropic/claude-haiku-4-5-20251001, openai/gpt-4o-mini)",
    )
    args = parser.parse_args()

    agent = make_agent(args.model)
    app_name = "adk_telemetry_example"
    user_id = "user1"

    runner = InMemoryRunner(agent=agent, app_name=app_name)

    # ---- One-liner instrumentation (stdout + cloud streaming) ----
    instrumentor = rllm_telemetry.instrument(
        runner,
        backend="stdout",
        agent_endpoint=RLLM_TELEMETRY_ENDPOINT,
    )

    session = await runner.session_service.create_session(app_name=app_name, user_id=user_id)

    print("\n" + "=" * 60)
    print(f"ADK + rllm_telemetry ({args.model})")
    print("=" * 60)

    prompts = [
        "Roll a die with 20 sides, then check if the result is prime.",
    ]

    for prompt in prompts:
        print(f"\nUSER: {prompt}\n")

        content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"AGENT ({event.author}): {part.text}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    await instrumentor.close()


if __name__ == "__main__":
    asyncio.run(main())
