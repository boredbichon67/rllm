"""End-to-end AG2 (AutoGen) example with rllm_telemetry.

Demonstrates the one-liner instrumentation for AG2 agents with tool calling.
Spans are printed to stdout AND streamed to the rllm cloud endpoint.

Usage:
    # With Anthropic (default):
    ANTHROPIC_API_KEY=sk-... RLLM_API_KEY=rllm_... python examples/ag2_example.py

    # With OpenAI:
    OPENAI_API_KEY=sk-... RLLM_API_KEY=rllm_... python examples/ag2_example.py --provider openai --model gpt-4o-mini
"""

import argparse
import random

from autogen import AssistantAgent, LLMConfig, UserProxyAgent, register_function

import rllm_telemetry

RLLM_TELEMETRY_ENDPOINT = "https://rllm-platform-api.up.railway.app"

# ----- Tools -----


def roll_die(sides: int) -> str:
    """Roll a die with the given number of sides.

    Args:
        sides: Number of sides on the die.

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


# ----- Main -----


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: claude-haiku-4-5-20251001 for anthropic, gpt-4o-mini for openai)",
    )
    args = parser.parse_args()

    # Build provider-specific LLM config
    model = args.model
    if args.provider == "anthropic":
        from autogen.oai.anthropic import AnthropicLLMConfigEntry

        model = model or "claude-haiku-4-5-20251001"
        llm_config = LLMConfig(AnthropicLLMConfigEntry(model=model, api_type="anthropic"))
    else:
        model = model or "gpt-4o-mini"
        llm_config = LLMConfig({"model": model})

    # Create agents
    assistant = AssistantAgent(
        name="assistant",
        system_message=("You are a helpful assistant. When asked to roll dice, use the roll_die tool. When asked to check primes, use the check_prime tool. Give concise answers. Reply TERMINATE when the task is fully done."),
        llm_config=llm_config,
    )

    user = UserProxyAgent(
        name="user_proxy_ag2",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content") or ""),
        code_execution_config=False,
    )

    # Register tools
    register_function(
        roll_die,
        caller=assistant,
        executor=user,
        name="roll_die",
        description="Roll a die with the given number of sides",
    )
    register_function(
        check_prime,
        caller=assistant,
        executor=user,
        name="check_prime",
        description="Check if a number is prime",
    )

    # ---- One-liner instrumentation (stdout + cloud streaming) ----
    instrumentor = rllm_telemetry.instrument(
        [assistant, user],
        backend="stdout",
        agent_endpoint=RLLM_TELEMETRY_ENDPOINT,
    )

    # Run conversation
    print("\n" + "=" * 60)
    print(f"Starting AG2 conversation ({args.provider}/{model})")
    print("=" * 60 + "\n")

    user.initiate_chat(
        assistant,
        message="Roll a 20-sided die, then check if the result is prime.",
        max_turns=5,
    )

    await instrumentor.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
