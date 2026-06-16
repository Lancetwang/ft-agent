from ft_agent import Agent
from ft_agent.core import Flow
from ft_agent.llm import DeepSeekLLM, ToolAwareLLMNode
from ft_agent.tools import Tool, ToolCallNode, ToolExecutor


SYSTEM_PROMPT = (
    "You are a weather assistant. When the user asks about weather, you must call "
    "the get_weather tool before answering. After receiving tool results, answer "
    "briefly using the tool result. Do not mention that the tool is mocked unless "
    "the result says so."
)


def build_messages(payload: dict) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(payload.get("history", []))
    return messages


def get_weather(city: str) -> dict[str, str]:
    return {
        "city": city,
        "condition": "sunny",
        "temperature": "24C",
        "source": "mock",
    }


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description=(
                "Get the weather for a city. Use this whenever the user asks about "
                "weather. The implementation returns mocked data for this example."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, such as Shanghai or Tokyo.",
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
