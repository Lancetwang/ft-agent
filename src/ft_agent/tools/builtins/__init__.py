from ft_agent.tools.base import Tool
from ft_agent.tools.builtins.weather import get_mock_weather


def get_builtin_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="Get mocked weather for a city. This does not call a real API.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, such as Shanghai or Beijing.",
                    }
                },
                "required": ["city"],
            },
            fn=get_mock_weather,
        )
    ]
