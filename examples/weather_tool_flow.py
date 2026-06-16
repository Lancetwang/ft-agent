import json

from ft_agent import Agent
from ft_agent.core import CallableNode, Flow
from ft_agent.tools import ToolCallNode


def mock_assistant_tool_call(payload: dict) -> tuple[str, dict]:
    payload["assistant_message"] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_weather_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": payload["city"]}),
                },
            }
        ],
    }
    return "tool_call", payload


def answer_from_tool_result(payload: dict) -> dict:
    result = payload["tool_results"][0]
    payload["answer"] = f"Weather result: {result.content}"
    return payload


request_node = CallableNode(mock_assistant_tool_call)
tool_node = ToolCallNode(next_action="answer")
answer_node = CallableNode(answer_from_tool_result)

request_node - "tool_call" >> tool_node
tool_node - "answer" >> answer_node

result = Agent(Flow(request_node)).run({"city": "Shanghai", "history": []})
print(result.payload["answer"])
