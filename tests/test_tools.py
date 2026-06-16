import unittest

from ft_agent.tools import Tool, ToolCallNode, ToolExecutor


def get_weather(city: str) -> dict[str, str]:
    return {"city": city, "condition": "sunny", "source": "mock"}


def weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Get mocked weather.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        fn=get_weather,
    )


class ToolTests(unittest.TestCase):
    def test_tool_executes_function(self) -> None:
        tool = weather_tool()

        result = tool.execute(city="Shanghai")

        self.assertEqual(tool.name, "get_weather")
        self.assertEqual(result["city"], "Shanghai")
        self.assertEqual(result["source"], "mock")

    def test_executor_runs_openai_style_tool_call(self) -> None:
        executor = ToolExecutor([weather_tool()])
        assistant_message = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Shanghai"}',
                    },
                }
            ]
        }

        tool_calls = executor.parse_tool_calls(assistant_message)
        results = executor.execute_all(tool_calls)

        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertIn('"city": "Shanghai"', results[0].content)
        self.assertFalse(results[0].is_error)

    def test_executor_handles_unknown_tool(self) -> None:
        executor = ToolExecutor()
        tool_call = executor.parse_tool_calls(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "missing", "arguments": "{}"},
                    }
                ]
            }
        )[0]

        result = executor.execute(tool_call)

        self.assertTrue(result.is_error)
        self.assertIn("not found", result.content)

    def test_tool_call_node_appends_tool_messages(self) -> None:
        node = ToolCallNode(executor=ToolExecutor([weather_tool()]), next_action="chat")
        payload = {
            "assistant_message": {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Shanghai"}',
                        },
                    }
                ]
            },
            "history": [],
        }

        action, state = node.exec(payload)

        self.assertEqual(action, "chat")
        self.assertEqual(state["history"][0]["role"], "tool")
        self.assertEqual(state["history"][0]["tool_call_id"], "call_1")


if __name__ == "__main__":
    unittest.main()
