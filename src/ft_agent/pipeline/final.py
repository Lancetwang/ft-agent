from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_core import ExecResult, Node, Payload
from ft_agent.llm import DeepSeekLLM
from ft_agent.llm.deepseek import Message
from ft_agent.pipeline.router import RouterDecision


FINAL_ANSWER_SYSTEM_PROMPT = """
You are the final response node for an agent pipeline focused on Fischer-Tropsch catalysts.

# Role
Turn the current pipeline state into a user-facing answer.

# Rules
- If the router marked the question as irrelevant, answer the user's question directly as a normal assistant.
- Do not invent tool calls or internal steps.
""".strip()


class FinalAnswerNode(Node):
    def __init__(
        self,
        *,
        llm: DeepSeekLLM,
        input_key: str = "question",
        router_decision_key: str = "router_decision",
        report_key: str = "writer_report",
        report_path_key: str = "writer_report_path",
        review_key: str = "supervisor_review",
        output_key: str = "answer",
        chat_kwargs_key: str = "final_chat_kwargs",
        action: str = "answered",
        system_prompt: str = FINAL_ANSWER_SYSTEM_PROMPT,
        chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.input_key = input_key
        self.router_decision_key = router_decision_key
        self.report_key = report_key
        self.report_path_key = report_path_key
        self.review_key = review_key
        self.output_key = output_key
        self.chat_kwargs_key = chat_kwargs_key
        self.action = action
        self.system_prompt = system_prompt
        self.chat_kwargs = dict(chat_kwargs or {})

    def exec(self, payload: Payload) -> ExecResult:
        state = dict(payload or {})
        clarification = self._clarification_answer(state)
        if clarification is not None:
            state[self.output_key] = clarification
            return self.action, state
        report_answer = self._report_answer(state)
        if report_answer is not None:
            state[self.output_key] = report_answer
            return self.action, state

        content = self.llm.chat(
            self._messages(state),
            **self._chat_kwargs(state),
        )
        state[self.output_key] = content
        return self.action, state

    def _clarification_answer(self, state: Mapping[str, Any]) -> str | None:
        decision = state.get(self.router_decision_key)
        if isinstance(decision, RouterDecision):
            if decision.action == "clarify":
                return decision.clarification_question or (
                    "Please add the missing catalyst, reaction, or target-performance context."
                )
        if isinstance(decision, Mapping):
            needs_clarification = bool(decision.get("needs_clarification"))
            question = _optional_text(decision.get("clarification_question"))
            if needs_clarification:
                return question or "Please add the missing catalyst, reaction, or target-performance context."
        return None

    def _report_answer(self, state: Mapping[str, Any]) -> str | None:
        report = _optional_text(state.get(self.report_key))
        if report is None:
            return None

        lines: list[str] = []
        review_summary = _review_summary(state.get(self.review_key))
        if review_summary:
            lines.append(review_summary)
        report_path = _optional_text(state.get(self.report_path_key))
        if report_path:
            lines.append(f"Report path: {report_path}")
        if lines:
            lines.append("")
        lines.append(report)
        return "\n".join(lines)

    def _messages(self, state: Mapping[str, Any]) -> list[Message]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": _final_user_content(
                    state,
                    input_key=self.input_key,
                    router_decision_key=self.router_decision_key,
                    report_key=self.report_key,
                    report_path_key=self.report_path_key,
                    review_key=self.review_key,
                ),
            },
        ]

    def _chat_kwargs(self, state: Mapping[str, Any]) -> dict[str, Any]:
        chat_kwargs = {"temperature": 0, **self.chat_kwargs}
        common_kwargs = state.get("chat_kwargs", {})
        if isinstance(common_kwargs, Mapping):
            chat_kwargs.update(common_kwargs)
        node_kwargs = state.get(self.chat_kwargs_key, {})
        if isinstance(node_kwargs, Mapping):
            chat_kwargs.update(node_kwargs)
        return chat_kwargs


def _final_user_content(
    state: Mapping[str, Any],
    *,
    input_key: str,
    router_decision_key: str,
    report_key: str,
    report_path_key: str,
    review_key: str,
) -> str:
    lines = [
        f"Original user question: {state.get(input_key, '')}",
    ]

    decision = state.get(router_decision_key)
    if isinstance(decision, RouterDecision):
        lines.append(f"Router action: {decision.action}")
        lines.append(f"Deliverable question: {decision.deliverable_question}")
    elif isinstance(decision, Mapping):
        lines.append(f"Router decision: {dict(decision)}")

    report = _optional_text(state.get(report_key))
    if report is None:
        lines.append(
            "No specialist report was produced. If the router action is irrelevant, answer directly."
        )
        return "\n".join(lines)

    lines.append(f"Report path: {state.get(report_path_key, '')}")
    review = state.get(review_key)
    if review is not None:
        lines.append(f"Supervisor review: {review}")
    lines.append("Final report:")
    lines.append(report)
    return "\n".join(lines)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _review_summary(review: Any) -> str | None:
    if review is None:
        return None
    approved = getattr(review, "approved", None)
    summary = _optional_text(getattr(review, "summary", None))
    max_reached = getattr(review, "max_revision_reached", False)
    if isinstance(review, Mapping):
        approved = review.get("approved", approved)
        summary = _optional_text(review.get("summary")) or summary
        max_reached = bool(review.get("max_revision_reached", max_reached))
    if approved is None and summary is None:
        return None
    status = "approved" if approved else "not approved"
    if max_reached and not approved:
        status = "max revisions reached"
    if summary:
        return f"Supervisor {status}: {summary}"
    return f"Supervisor {status}."
