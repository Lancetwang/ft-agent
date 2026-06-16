from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI

from ft_agent.config import DeepSeekConfig, load_deepseek_config

Message = Mapping[str, str]


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
        **kwargs: Any,
    ) -> str:
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
        return response.choices[0].message.content or ""
