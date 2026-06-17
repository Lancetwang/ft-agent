from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ft_agent.core import ExecResult, Node, Payload
from ft_agent.llm import DeepSeekLLM
from ft_agent.llm.deepseek import Message


ROUTER_SYSTEM_PROMPT = """
You are the router for an agent pipeline focused on Fischer-Tropsch catalysts.

# Task
Classify the user's question and decide the next step.
If clarification is needed, ask one concise clarification question that captures
the minimum missing information needed before planning.

# Domain
Relevant questions are strongly related to Fischer-Tropsch catalysts, including
catalyst composition, active metals, supports, promoters, preparation, reaction
conditions, selectivity, deactivation, regeneration, characterization, mechanisms,
literature, and experiment design.

# Output
Return only one JSON object:
{
  "is_relevant": true,
  "needs_clarification": false,
  "clarification_question": null,
  "deliverable_question": "The question to hand to the next agent."
}
""".strip()


class RouterParseError(ValueError):
    pass


@dataclass(frozen=True)
class RouterDecision:
    is_relevant: bool
    needs_clarification: bool
    clarification_question: str | None = None
    deliverable_question: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, fallback_question: str) -> RouterDecision:
        is_relevant = _bool_value(data.get("is_relevant", False))
        clarification_question = _optional_text(data.get("clarification_question"))
        needs_clarification = _bool_value(data.get("needs_clarification", False))
        deliverable_question = _optional_text(data.get("deliverable_question")) or fallback_question

        if not is_relevant:
            needs_clarification = False
            clarification_question = None

        return cls(
            is_relevant=is_relevant,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            deliverable_question=deliverable_question,
            raw=dict(data),
        )

    @property
    def action(self) -> str:
        if not self.is_relevant:
            return "irrelevant"
        if self.needs_clarification:
            return "clarify"
        return "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_relevant": self.is_relevant,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "deliverable_question": self.deliverable_question,
        }


class RouterNode(Node):
    def __init__(
        self,
        *,
        llm: DeepSeekLLM,
        input_key: str = "question",
        output_key: str = "router_decision",
        raw_output_key: str = "router_raw_output",
        system_prompt: str = ROUTER_SYSTEM_PROMPT,
        chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.input_key = input_key
        self.output_key = output_key
        self.raw_output_key = raw_output_key
        self.system_prompt = system_prompt
        self.chat_kwargs = dict(chat_kwargs or {})

    def exec(self, payload: Payload) -> ExecResult:
        state = dict(payload or {})
        question = str(state.get(self.input_key, "")).strip()
        chat_kwargs = {"temperature": 0, **self.chat_kwargs}
        content = self.llm.chat(
            self._messages(question),
            **chat_kwargs,
        )
        decision = RouterDecision.from_dict(
            _loads_json_object(content),
            fallback_question=question,
        )
        state[self.input_key] = question
        state[self.output_key] = decision
        state[self.raw_output_key] = content
        return decision.action, state

    def _messages(self, question: str) -> list[Message]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]


def _loads_json_object(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise RouterParseError("Router response did not contain a JSON object.") from None
        try:
            value = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RouterParseError("Router response contained invalid JSON.") from exc

    if not isinstance(value, dict):
        raise RouterParseError("Router response must be a JSON object.")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)
