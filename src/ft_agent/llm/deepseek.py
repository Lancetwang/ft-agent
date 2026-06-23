from collections.abc import Callable, Mapping, Sequence
import time
from typing import Any

from openai import OpenAI

from ft_agent.config import DeepSeekConfig, load_deepseek_config
from agent_core.core.trace import get_trace_recorder

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
        started = time.perf_counter()
        recorder = get_trace_recorder(None)
        usage: dict[str, Any] | None = None
        success = False
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

        try:
            response = self.client.chat.completions.create(**request)
            if stream:
                content, usage = self._collect_stream(response, on_delta)
                success = True
                return content
            usage = _usage_dict(getattr(response, "usage", None))
            success = True
            return response.choices[0].message.content or ""
        finally:
            _emit_llm_trace(
                recorder,
                event="llm.chat",
                model=self.config.model,
                stream=stream,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                usage=usage,
                success=success,
            )

    def chat_message(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        thinking: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        recorder = get_trace_recorder(None)
        usage: dict[str, Any] | None = None
        success = False
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

        try:
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

            usage = _usage_dict(getattr(response, "usage", None))
            if usage:
                result["usage"] = usage

            success = True
            return result
        finally:
            _emit_llm_trace(
                recorder,
                event="llm.chat_message",
                model=self.config.model,
                stream=False,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                usage=usage,
                success=success,
            )

    @staticmethod
    def _collect_stream(
        response: Any,
        on_delta: DeltaHandler | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        chunks: list[str] = []
        usage: dict[str, Any] | None = None
        for event in response:
            event_usage = _usage_dict(getattr(event, "usage", None))
            if event_usage:
                usage = event_usage
            if not event.choices:
                continue
            delta = event.choices[0].delta.content or ""
            if not delta:
                continue
            chunks.append(delta)
            if on_delta is not None:
                on_delta(delta)
        return "".join(chunks), usage


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        value = usage.model_dump()
    elif isinstance(usage, Mapping):
        value = dict(usage)
    else:
        value = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    return {str(key): item for key, item in value.items() if item is not None}


def _emit_llm_trace(
    recorder: Any,
    *,
    event: str,
    model: str,
    stream: bool,
    elapsed_ms: float,
    usage: dict[str, Any] | None,
    success: bool,
) -> None:
    if recorder is None:
        return
    recorder.emit(
        event,
        category="llm",
        data={
            "model": model,
            "stream": stream,
            "elapsed_ms": round(elapsed_ms, 2),
            "usage": usage or {},
            "success": success,
        },
    )
