from ft_agent import Agent
from ft_agent.core import Flow
from ft_agent.llm import DeepSeekLLM, ToolAwareLLMNode
from ft_agent.tools import Tool, ToolCallNode, ToolExecutor


SYSTEM_PROMPT = """
You are a weather assistant.

# Tool Use
- Use get_weather for weather questions about Shanghai or Tokyo.
- Answer from the tool result once it is available.
- Keep the final answer brief and natural.
""".strip()


def build_messages(payload: dict) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(payload.get("history", []))
    return messages


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


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description=(
                "Look up demo weather for Shanghai or Tokyo. The city argument can be "
                "the city name from the user's message."
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
        )
    ]


def build_weather_agent() -> Agent:
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


if __name__ == "__main__":
    agent = build_weather_agent()
    result = agent.run(
        {
            "history": [
                {
                    "role": "user",
                    "content": "What is the weather in Shanghai? Use the weather tool.",
                }
            ]
        }
    )
    print(result.payload["answer"])
