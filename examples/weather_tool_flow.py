from ft_agent import Agent
from ft_agent.core import Flow
from ft_agent.llm import DeepSeekLLM, ToolAwareLLMNode
from ft_agent.tools import Tool, ToolCallNode, ToolExecutor


SYSTEM_PROMPT = """
You are a weather assistant.

# Tools

## get_weather
Look up demo weather for Shanghai or Tokyo.
Input: city, normalized to one of: Shanghai, Tokyo.
Use this when the user asks about weather for either city.

# Response
Answer briefly and naturally after the tool result is available.
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
        "tokyo": "Tokyo",
    }
    return aliases.get(city.strip().lower(), city.strip())


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description=(
                "Look up demo weather for Shanghai or Tokyo. Use the English city "
                "name as the city argument."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "enum": ["Shanghai", "Tokyo"],
                        "description": "English city name.",
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
