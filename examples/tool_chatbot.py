import sys

from ft_agent import Agent
from ft_agent.core import Flow
from ft_agent.llm import DeepSeekLLM, ToolAwareLLMNode
from ft_agent.tools import Tool, ToolCallNode, ToolExecutor


SYSTEM_PROMPT = (
    "You are a concise assistant with tools. Use get_weather for weather questions "
    "about Shanghai or Tokyo. Use tell_joke when the user asks for a joke. If a user "
    "asks for both weather and a joke, you may call both tools before answering. "
    "After tool results are available, answer naturally and briefly."
)


def get_weather(city: str) -> dict[str, str]:
    weather = {
        "shanghai": {"condition": "sunny", "temperature": "24C"},
        "tokyo": {"condition": "rainy", "temperature": "18C"},
    }
    result = weather.get(city.lower(), {"condition": "unknown", "temperature": "unknown"})
    return {"city": city, **result, "source": "mock"}


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
            description="Get mocked weather for Shanghai or Tokyo.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name. Supported examples: Shanghai, Tokyo.",
                    }
                },
                "required": ["city"],
            },
            fn=get_weather,
        ),
        Tool(
            name="tell_joke",
            description="Tell a short mocked joke.",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional joke topic.",
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
    answer = result.payload["answer"]
    safe_print(answer)
    return True


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
    run_scripted_demo()
