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
If prior clarification turns are provided, use them to update the deliverable
question and decide whether another clarification is still necessary.

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
class RouterClarificationTurn:
    question: str
    answer: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RouterClarificationTurn:
        return cls(
            question=str(data.get("question", "")).strip(),
            answer=str(data.get("answer", "")).strip(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "question": self.question,
            "answer": self.answer,
        }


@dataclass(frozen=True)
class RouterContext:
    original_question: str
    clarification_turns: list[RouterClarificationTurn] = field(default_factory=list)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        question: str,
        context_key: str,
    ) -> RouterContext:
        raw_context = payload.get(context_key)
        if not isinstance(raw_context, Mapping):
            return cls(original_question=question)

        original_question = _optional_text(raw_context.get("original_question")) or question
        raw_turns = raw_context.get("clarification_turns", [])
        turns = [
            RouterClarificationTurn.from_dict(item)
            for item in raw_turns
            if isinstance(item, Mapping)
        ]
        return cls(original_question=original_question, clarification_turns=turns)

    def with_turn(self, question: str, answer: str) -> RouterContext:
        return RouterContext(
            original_question=self.original_question,
            clarification_turns=[
                *self.clarification_turns,
                RouterClarificationTurn(question=question, answer=answer),
            ],
        )

    @property
    def clarification_rounds(self) -> int:
        return len(self.clarification_turns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "clarification_turns": [
                turn.to_dict()
                for turn in self.clarification_turns
            ],
        }


@dataclass(frozen=True)
class RouterDecision:
    is_relevant: bool
    needs_clarification: bool
    clarification_question: str | None = None
    deliverable_question: str = ""
    clarification_rounds: int = 0
    max_clarification_rounds: int = 3
    max_clarification_reached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        fallback_question: str,
        clarification_rounds: int = 0,
        max_clarification_rounds: int = 3,
    ) -> RouterDecision:
        is_relevant = _bool_value(data.get("is_relevant", False))
        clarification_question = _optional_text(data.get("clarification_question"))
        needs_clarification = _bool_value(data.get("needs_clarification", False))
        deliverable_question = _optional_text(data.get("deliverable_question")) or fallback_question
        max_clarification_reached = clarification_rounds >= max_clarification_rounds

        if not is_relevant:
            needs_clarification = False
            clarification_question = None
        elif max_clarification_reached and needs_clarification:
            needs_clarification = False
            clarification_question = None

        return cls(
            is_relevant=is_relevant,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            deliverable_question=deliverable_question,
            clarification_rounds=clarification_rounds,
            max_clarification_rounds=max_clarification_rounds,
            max_clarification_reached=max_clarification_reached,
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
            "clarification_rounds": self.clarification_rounds,
            "max_clarification_rounds": self.max_clarification_rounds,
            "max_clarification_reached": self.max_clarification_reached,
        }


class RouterNode(Node):
    def __init__(
        self,
        *,
        llm: DeepSeekLLM,
        input_key: str = "question",
        clarification_response_key: str = "clarification_response",
        context_key: str = "router_context",
        output_key: str = "router_decision",
        raw_output_key: str = "router_raw_output",
        chat_kwargs_key: str = "router_chat_kwargs",
        max_clarification_rounds: int = 3,
        system_prompt: str = ROUTER_SYSTEM_PROMPT,
        chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if max_clarification_rounds < 0:
            raise ValueError("max_clarification_rounds must be non-negative.")
        self.llm = llm
        self.input_key = input_key
        self.clarification_response_key = clarification_response_key
        self.context_key = context_key
        self.output_key = output_key
        self.raw_output_key = raw_output_key
        self.chat_kwargs_key = chat_kwargs_key
        self.max_clarification_rounds = max_clarification_rounds
        self.system_prompt = system_prompt
        self.chat_kwargs = dict(chat_kwargs or {})

    def exec(self, payload: Payload) -> ExecResult:
        state = dict(payload or {})
        question = str(state.get(self.input_key, "")).strip()
        context = RouterContext.from_payload(
            state,
            question=question,
            context_key=self.context_key,
        )
        context = self._append_clarification_response(state, context)
        chat_kwargs = self._chat_kwargs(state)
        content = self.llm.chat(
            self._messages(context),
            **chat_kwargs,
        )
        decision = RouterDecision.from_dict(
            _loads_json_object(content),
            fallback_question=_fallback_question(context),
            clarification_rounds=context.clarification_rounds,
            max_clarification_rounds=self.max_clarification_rounds,
        )
        state[self.input_key] = context.original_question
        state[self.context_key] = context.to_dict()
        state[self.output_key] = decision
        state[self.raw_output_key] = content
        state.pop(self.clarification_response_key, None)
        return decision.action, state

    def _append_clarification_response(
        self,
        state: Mapping[str, Any],
        context: RouterContext,
    ) -> RouterContext:
        response = _optional_text(state.get(self.clarification_response_key))
        if response is None:
            return context

        previous_decision = state.get(self.output_key)
        if isinstance(previous_decision, RouterDecision):
            clarification_question = previous_decision.clarification_question
        elif isinstance(previous_decision, Mapping):
            clarification_question = _optional_text(previous_decision.get("clarification_question"))
        else:
            clarification_question = None

        return context.with_turn(
            question=clarification_question or "Clarification requested by the router.",
            answer=response,
        )

    def _chat_kwargs(self, state: Mapping[str, Any]) -> dict[str, Any]:
        chat_kwargs = {"temperature": 0, **self.chat_kwargs}
        common_kwargs = state.get("chat_kwargs", {})
        if isinstance(common_kwargs, Mapping):
            chat_kwargs.update(common_kwargs)
        node_kwargs = state.get(self.chat_kwargs_key, {})
        if isinstance(node_kwargs, Mapping):
            chat_kwargs.update(node_kwargs)
        return chat_kwargs

    def _messages(self, context: RouterContext) -> list[Message]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": _router_user_content(
                    context,
                    remaining_clarifications=max(
                        self.max_clarification_rounds - context.clarification_rounds,
                        0,
                    ),
                ),
            },
        ]


def _router_user_content(
    context: RouterContext,
    *,
    remaining_clarifications: int,
) -> str:
    lines = [
        f"Original question: {context.original_question}",
        f"Clarification rounds used: {context.clarification_rounds}",
        f"Remaining clarification rounds: {remaining_clarifications}",
    ]
    if context.clarification_turns:
        lines.append("Clarification history:")
        for index, turn in enumerate(context.clarification_turns, start=1):
            lines.append(f"{index}. Router asked: {turn.question}")
            lines.append(f"   User answered: {turn.answer}")
    if remaining_clarifications <= 0:
        lines.append(
            "No clarification rounds remain. Do not ask another clarification question; "
            "produce the best deliverable_question from the available information."
        )
    return "\n".join(lines)


def _fallback_question(context: RouterContext) -> str:
    if not context.clarification_turns:
        return context.original_question
    answers = " ".join(turn.answer for turn in context.clarification_turns)
    return f"{context.original_question} {answers}".strip()


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
