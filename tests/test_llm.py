import unittest
from types import SimpleNamespace

from ft_agent.config import DeepSeekConfig
from ft_agent.llm.deepseek import DeepSeekLLM


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self.response


class FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(response),
        )


def chunk(content: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
            )
        ]
    )


class DeepSeekLLMTests(unittest.TestCase):
    def test_non_stream_chat(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                )
            ]
        )
        client = FakeClient(response)
        llm = DeepSeekLLM(config=DeepSeekConfig(api_key="test"), client=client)

        content = llm.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(content, "ok")
        self.assertFalse(client.chat.completions.last_request["stream"])
        self.assertEqual(
            client.chat.completions.last_request["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_stream_chat_collects_deltas(self) -> None:
        client = FakeClient([chunk("o"), chunk("k")])
        llm = DeepSeekLLM(config=DeepSeekConfig(api_key="test"), client=client)
        seen: list[str] = []

        content = llm.chat(
            [{"role": "user", "content": "hello"}],
            stream=True,
            on_delta=seen.append,
        )

        self.assertEqual(content, "ok")
        self.assertEqual(seen, ["o", "k"])
        self.assertTrue(client.chat.completions.last_request["stream"])

    def test_chat_message_keeps_tool_calls(self) -> None:
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Shanghai"}',
            },
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=[tool_call]),
                )
            ],
            usage=None,
        )
        client = FakeClient(response)
        llm = DeepSeekLLM(config=DeepSeekConfig(api_key="test"), client=client)

        message = llm.chat_message(
            [{"role": "user", "content": "weather"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["tool_calls"], [tool_call])
        self.assertEqual(client.chat.completions.last_request["tools"][0]["type"], "function")


if __name__ == "__main__":
    unittest.main()
