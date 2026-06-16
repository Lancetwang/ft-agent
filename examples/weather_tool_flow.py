from ft_agent import Agent
from ft_agent.core import Flow
from ft_agent.llm import DeepSeekLLM, ToolAwareLLMNode
from ft_agent.tools import Tool, ToolCallNode, ToolExecutor


SYSTEM_PROMPT = (
    "You are a weather assistant. When the user asks about weather, you must call "
    "the get_weather tool before answering. This includes Chinese city names such "
    "as 上海 and 东京. After receiving tool results, answer briefly using the tool "
    "result. Do not mention that the tool is mocked unless the result says so."
)


def build_messages(payload: dict) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(payload.get("history", []))
    return messages


def get_weather(city: str) -> dict[str, str]:
    normalized_city = normalize_city(city)
    weather = {
        "Shanghai": {"condition": "sunny", "temperature": "24C"},
        "Tokyo": {"condition": "rainy", "temperature": "18C"},
    }
    result = weather.get(normalized_city, {"condition": "unknown", "temperature": "unknown"})
    return {
        "city": normalized_city,
        "input_city": city,
        **result,
        "source": "mock",
    }


def normalize_city(city: str) -> str:
    aliases = {
        "shanghai": "Shanghai",
        "上海": "Shanghai",
        "上海市": "Shanghai",
        "tokyo": "Tokyo",
        "东京": "Tokyo",
        "東京": "Tokyo",
    }
    return aliases.get(city.strip().lower(), city)


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description=(
                "Get mocked weather for Shanghai/上海 or Tokyo/东京. Use this whenever "
                "the user asks about weather for those cities."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name. Supported: Shanghai, 上海, Tokyo, 东京.",
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
