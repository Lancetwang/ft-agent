import argparse
import sys
from typing import Annotated, Literal

from ft_agent import Agent
from ft_agent.core import Flow, TraceOptions, make_trace_options
from ft_agent.llm import DeepSeekLLM, ToolAwareLLMNode
from ft_agent.tools import Tool, ToolCallNode, ToolExecutor, tool


SYSTEM_PROMPT = """
You are a small CLI assistant with two demo tools.

Available tools:
- get_weather: weather for Shanghai or Tokyo.
- tell_joke: short demo jokes about a requested topic.

Guidelines:
- If a final answer contains weather information, use get_weather first.
- If a final answer contains a joke, use tell_joke first.
- If one request needs several tools, call the relevant tools before answering.

# Tools

## get_weather
Look up demo weather for Shanghai or Tokyo.
Input: city, normalized to one of: Shanghai, Tokyo.
Use this when the user asks about weather for either city.

## tell_joke
Return a short demo joke for a requested topic.
Input: optional topic, normalized to English.
Use this when the user asks for a joke.

# Response
Answer naturally in the user's language after tool results are available.
""".strip()

@tool(
    description=(
        "Look up demo weather for Shanghai or Tokyo. Use the English city name as "
        "the city argument."
    )
)
def get_weather(
    city: Annotated[
        Literal["Shanghai", "Tokyo"],
        "English city name.",
    ],
) -> dict[str, str]:
    weather = {
        "Shanghai": {"condition": "sunny", "temperature": "24C"},
        "Tokyo": {"condition": "rainy", "temperature": "18C"},
    }
    result = weather.get(city)
    if result is None:
        return {
            "city": city,
            "status": "unsupported_city",
            "supported_cities": ["Shanghai", "Tokyo"],
            "source": "mock",
        }
    return {"city": city, **result, "source": "mock"}


@tool(description="Return a short demo joke for the requested topic.")
def tell_joke(
    topic: Annotated[
        str,
        "Optional joke topic, normalized to English.",
    ] = "weather",
) -> dict[str, str]:
    return {
        "topic": topic,
        "joke": "I asked the cloud for a forecast. It said it was feeling under the weather.",
        "source": "mock",
    }


def build_tools() -> list[Tool]:
    return [get_weather, tell_joke]


def build_messages(payload: dict) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(payload.get("history", []))
    return messages


def build_agent() -> Agent:
    tools = build_tools()
    llm_node = ToolAwareLLMNode(
        llm=DeepSeekLLM(),
        messages=build_messages,
        tools=[tool.to_llm_format() for tool in tools],
        chat_kwargs={"temperature": 0, "tool_choice": "auto"},
    )
    tool_node = ToolCallNode(executor=ToolExecutor(tools))

    llm_node - "tool_call" >> tool_node
    tool_node - "chat" >> llm_node

    return Agent(Flow(llm_node))


def run_turn(
    agent: Agent,
    history: list[dict[str, str]],
    user_input: str,
    *,
    trace: TraceOptions | None = None,
) -> bool:
    if user_input.lower() in {"exit", "quit", "q"}:
        return False

    history.append({"role": "user", "content": user_input})
    result = agent.run({"history": history}, trace=trace)
    safe_print(result.payload["answer"])
    return True


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def run_scripted_demo(trace: TraceOptions | None = None) -> None:
    agent = build_agent()
    history: list[dict[str, str]] = []
    for user_input in [
        "What is the weather in Shanghai?",
        "How about Tokyo?",
        "Tell me a joke about the weather.",
        "Compare Shanghai and Tokyo weather, then tell one joke.",
    ]:
        print(f"> {user_input}")
        run_turn(agent, history, user_input, trace=trace)


def run_interactive(trace: TraceOptions | None = None) -> None:
    agent = build_agent()
    history: list[dict[str, str]] = []
    print("ft-agent tool chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if not run_turn(agent, history, user_input, trace=trace):
            print("bye")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ft-agent tool chatbot example.")
    parser.add_argument("--demo", action="store_true", help="Run scripted demo turns.")
    parser.add_argument("--trace", action="store_true", help="Print trace events while the flow runs.")
    parser.add_argument(
        "--trace-events",
        default="node,tool,flow",
        help="Comma-separated trace categories: node, tool, flow.",
    )
    return parser.parse_args()


def build_trace_options(args: argparse.Namespace) -> TraceOptions:
    categories = [item.strip() for item in args.trace_events.split(",") if item.strip()]
    return make_trace_options(
        enabled=args.trace,
        include=categories,
        print_to_console=args.trace,
        printer=safe_print,
    )


if __name__ == "__main__":
    args = parse_args()
    trace_options = build_trace_options(args)
    if args.demo:
        run_scripted_demo(trace=trace_options)
    else:
        run_interactive(trace=trace_options)
