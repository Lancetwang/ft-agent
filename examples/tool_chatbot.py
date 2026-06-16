import sys

from ft_agent import Agent
from ft_agent.core import Flow
from ft_agent.llm import DeepSeekLLM, ToolAwareLLMNode
from ft_agent.tools import Tool, ToolCallNode, ToolExecutor, ToolResult


SYSTEM_PROMPT = """
You are a small CLI assistant with two demo tools.

# Tool Use
- Use tools when they can ground the answer.
- Weather and jokes are tool-backed demo capabilities in this example.
- For weather questions about Shanghai or Tokyo, use get_weather.
- For joke requests, use tell_joke rather than improvising the joke yourself.
- A final answer that contains weather information needs get_weather results.
- A final answer that contains a joke needs a tell_joke result.
- If one user request needs several facts or abilities, call the needed tools before
  giving the final answer.
- After tool results are available, answer naturally in the user's language.
""".strip()

SHOW_TOOL_TRACE = True


def get_weather(city: str) -> dict[str, str]:
    city_name = normalize_city(city)
    weather = {
        "Shanghai": {"condition": "sunny", "temperature": "24C"},
        "Tokyo": {"condition": "rainy", "temperature": "18C"},
    }
    result = weather.get(city_name)
    if result is None:
        return {
            "city": city,
            "status": "unsupported_city",
            "supported_cities": ["Shanghai", "Tokyo"],
            "source": "mock",
        }
    return {"city": city_name, "input_city": city, **result, "source": "mock"}


def normalize_city(city: str) -> str:
    aliases = {
        "shanghai": "Shanghai",
        "\u4e0a\u6d77": "Shanghai",
        "\u4e0a\u6d77\u5e02": "Shanghai",
        "tokyo": "Tokyo",
        "\u4e1c\u4eac": "Tokyo",
        "\u6771\u4eac": "Tokyo",
    }
    return aliases.get(city.strip().lower(), city.strip())


def tell_joke(topic: str = "weather") -> dict[str, str]:
    return {
        "topic": topic,
        "joke": "I asked the cloud for a forecast. It said it was feeling under the weather.",
        "source": "mock",
    }


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description=(
                "Look up demo weather for Shanghai or Tokyo. The city argument can be "
                "the city name from the user's message, such as Shanghai, Tokyo, "
                "Chinese Shanghai, or Chinese Tokyo."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name from the user request.",
                    }
                },
                "required": ["city"],
            },
            fn=get_weather,
        ),
        Tool(
            name="tell_joke",
            description=(
                "Return a short demo joke for the requested topic. Use this tool "
                "whenever the user asks for a joke in this demo."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional joke topic from the user request.",
                    }
                },
            },
            fn=tell_joke,
        ),
    ]


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


def run_turn(agent: Agent, history: list[dict[str, str]], user_input: str) -> bool:
    if user_input.lower() in {"exit", "quit", "q"}:
        return False

    history.append({"role": "user", "content": user_input})
    result = agent.run({"history": history})
    if SHOW_TOOL_TRACE:
        print_tool_trace(result.payload.get("tool_results", []))
    safe_print(result.payload["answer"])
    return True


def print_tool_trace(results: list[ToolResult]) -> None:
    for result in results:
        status = "error" if result.is_error else "ok"
        safe_print(f"[tool:{status}] {result.content}")


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def run_scripted_demo() -> None:
    agent = build_agent()
    history: list[dict[str, str]] = []
    for user_input in [
        "What is the weather in Shanghai?",
        "How about Tokyo?",
        "Tell me a joke about the weather.",
        "Compare Shanghai and Tokyo weather, then tell one joke.",
    ]:
        print(f"> {user_input}")
        run_turn(agent, history, user_input)


def run_interactive() -> None:
    agent = build_agent()
    history: list[dict[str, str]] = []
    print("ft-agent tool chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if not run_turn(agent, history, user_input):
            print("bye")
            break


if __name__ == "__main__":
    run_interactive()
