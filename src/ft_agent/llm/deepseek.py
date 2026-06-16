from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openai import OpenAI

from ft_agent.config import DeepSeekConfig, load_deepseek_config

Message = Mapping[str, str]
DeltaHandler = Callable[[str], None]


def create_deepseek_client(config: DeepSeekConfig | None = None) -> OpenAI:
    config = config or load_deepseek_config()
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


class DeepSeekLLM:
    def __init__(
        self,
        config: DeepSeekConfig | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.config = config or load_deepseek_config()
        self.client = client or create_deepseek_client(self.config)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        thinking: bool = False,
        stream: bool = False,
        on_delta: DeltaHandler | None = None,
        **kwargs: Any,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
            "stream": stream,
            **kwargs,
        }
        if tools is not None:
            request["tools"] = list(tools)
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**request)
        if stream:
            return self._collect_stream(response, on_delta)
        return response.choices[0].message.content or ""

    def chat_message(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        thinking: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
            **kwargs,
        }
        if tools is not None:
            request["tools"] = list(tools)
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        result: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            result["tool_calls"] = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in tool_calls
            ]

        if response.usage:
            result["usage"] = response.usage.model_dump()

        return result

    @staticmethod
    def _collect_stream(response: Any, on_delta: DeltaHandler | None = None) -> str:
        chunks: list[str] = []
        for event in response:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content or ""
            if not delta:
                continue
            chunks.append(delta)
            if on_delta is not None:
                on_delta(delta)
        return "".join(chunks)
