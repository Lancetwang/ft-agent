from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ft_agent.core import ExecResult, Node, Payload
from ft_agent.core.trace import get_trace_recorder
from ft_agent.llm import DeepSeekLLM
from ft_agent.llm.deepseek import Message
from ft_agent.tools import Tool, ToolCall, ToolExecutor, build_read_file_tool


SUPERVISOR_SYSTEM_PROMPT = """
You are the supervisor for an agent pipeline focused on Fischer-Tropsch catalysts.

# Role
Review the writer's experiment report against the provided criteria.

# Rules
- Read the report content provided to you.
- If the report is good enough, approve it.
- If it is not good enough, give concrete issues that the writer can fix.
- Prefer line-specific feedback when possible.

# Output
Return only one JSON object:
{
  "approved": false,
  "summary": "Short review summary.",
  "issues": [
    {
      "line_start": 10,
      "line_end": 15,
      "severity": "major",
      "message": "What is weak or missing.",
      "suggestion": "How the writer should fix it."
    }
  ]
}
""".strip()


class SupervisorParseError(ValueError):
    pass


@dataclass(frozen=True)
class SupervisorCriterion:
    name: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
        }


@dataclass(frozen=True)
class SupervisorIssue:
    line_start: int | None
    line_end: int | None
    severity: str
    message: str
    suggestion: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SupervisorIssue:
        return cls(
            line_start=_optional_int(data.get("line_start")),
            line_end=_optional_int(data.get("line_end")),
            severity=_optional_text(data.get("severity")) or "minor",
            message=_required_text(data, "message"),
            suggestion=_required_text(data, "suggestion"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_start": self.line_start,
            "line_end": self.line_end,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class SupervisorReview:
    approved: bool
    summary: str
    issues: list[SupervisorIssue] = field(default_factory=list)
    revision_rounds: int = 0
    max_revision_rounds: int = 3
    max_revision_reached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        revision_rounds: int,
        max_revision_rounds: int,
    ) -> SupervisorReview:
        issues_data = data.get("issues", [])
        issues = [
            SupervisorIssue.from_dict(item)
            for item in issues_data
            if isinstance(item, Mapping)
        ]
        approved = _bool_value(data.get("approved", False))
        max_reached = revision_rounds >= max_revision_rounds
        return cls(
            approved=approved,
            summary=_optional_text(data.get("summary")) or "",
            issues=issues,
            revision_rounds=revision_rounds,
            max_revision_rounds=max_revision_rounds,
            max_revision_reached=max_reached,
            raw=dict(data),
        )

    @property
    def action(self) -> str:
        if self.approved or self.max_revision_reached:
            return "approved"
        return "revise"

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "summary": self.summary,
            "issues": [issue.to_dict() for issue in self.issues],
            "revision_rounds": self.revision_rounds,
            "max_revision_rounds": self.max_revision_rounds,
            "max_revision_reached": self.max_revision_reached,
        }


class SupervisorNode(Node):
    def __init__(
        self,
        *,
        llm: DeepSeekLLM,
        criteria: Sequence[SupervisorCriterion] | None = None,
        report_workspace: str | Path = "artifacts",
        report_path_key: str = "writer_report_path",
        review_key: str = "supervisor_review",
        raw_output_key: str = "supervisor_raw_output",
        file_excerpt_key: str = "supervisor_file_excerpt",
        revision_rounds_key: str = "supervisor_revision_rounds",
        chat_kwargs_key: str = "supervisor_chat_kwargs",
        max_revision_rounds: int = 3,
        max_read_chars: int = 20000,
        system_prompt: str = SUPERVISOR_SYSTEM_PROMPT,
        chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if max_revision_rounds < 0:
            raise ValueError("max_revision_rounds must be non-negative.")
        self.llm = llm
        self.criteria = list(criteria or default_supervisor_criteria())
        self.read_tool = build_read_file_tool(report_workspace)
        self.executor = ToolExecutor([self.read_tool])
        self.report_path_key = report_path_key
        self.review_key = review_key
        self.raw_output_key = raw_output_key
        self.file_excerpt_key = file_excerpt_key
        self.revision_rounds_key = revision_rounds_key
        self.chat_kwargs_key = chat_kwargs_key
        self.max_revision_rounds = max_revision_rounds
        self.max_read_chars = max_read_chars
        self.system_prompt = system_prompt
        self.chat_kwargs = dict(chat_kwargs or {})

    def exec(self, payload: Payload) -> ExecResult:
        state = dict(payload or {})
        report_path = str(state.get(self.report_path_key) or "reports/latest.md")
        revision_rounds = int(state.get(self.revision_rounds_key, 0))
        read_result = self._read_report(report_path, state)
        state[self.file_excerpt_key] = read_result.content
        content = self.llm.chat(
            self._messages(report_path, read_result.content),
            **self._chat_kwargs(state),
        )
        review = SupervisorReview.from_dict(
            _loads_json_object(content),
            revision_rounds=revision_rounds,
            max_revision_rounds=self.max_revision_rounds,
        )
        state[self.review_key] = review
        state[self.raw_output_key] = content
        if review.action == "revise":
            state[self.revision_rounds_key] = revision_rounds + 1
        return review.action, state

    def _read_report(self, report_path: str, state: Mapping[str, Any]) -> Any:
        tool_call = ToolCall(
            id="supervisor_read_report",
            name="read_file",
            arguments={
                "path": report_path,
                "start_line": 1,
                "max_chars": self.max_read_chars,
            },
        )
        recorder = get_trace_recorder(state)
        if recorder is not None:
            recorder.emit(
                "tool.call",
                category="tool",
                data={
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                },
            )
        result = self.executor.execute(tool_call)
        if recorder is not None:
            trace_content = _trace_preview(result.content)
            recorder.emit(
                "tool.result",
                category="tool",
                data={
                    "tool_call_id": result.tool_call_id,
                    "content": trace_content,
                    "is_error": result.is_error,
                    "truncated": trace_content != result.content,
                },
            )
        return result

    def _messages(self, report_path: str, report_excerpt: str) -> list[Message]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Report path: {report_path}\n\n"
                    f"Criteria:\n{json.dumps([criterion.to_dict() for criterion in self.criteria], ensure_ascii=False, indent=2)}\n\n"
                    f"Report content with line numbers:\n{report_excerpt}"
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


def default_supervisor_criteria() -> list[SupervisorCriterion]:
    return [
        SupervisorCriterion(
            name="report_structure",
            description="The report should include objective, background, method, characterization, testing, expected results, risks, and evidence notes.",
        ),
        SupervisorCriterion(
            name="domain_specificity",
            description="The report should be specific to Fischer-Tropsch catalysts and avoid generic catalyst-writing.",
        ),
        SupervisorCriterion(
            name="experimental_actionability",
            description="The report should contain concrete experimental conditions, variables, measurements, and evaluation criteria.",
        ),
        SupervisorCriterion(
            name="evidence_grounding",
            description="The report should use retrieved scientific context and identify evidence or citation notes.",
        ),
    ]


def _loads_json_object(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise SupervisorParseError("Supervisor response did not contain a JSON object.") from None
        try:
            value = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SupervisorParseError("Supervisor response contained invalid JSON.") from exc

    if not isinstance(value, dict):
        raise SupervisorParseError("Supervisor response must be a JSON object.")
    return value


def _required_text(data: Mapping[str, Any], key: str) -> str:
    text = _optional_text(data.get(key))
    if text is None:
        raise SupervisorParseError(f"Missing required review field: {key}.")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _trace_preview(content: str, limit: int = 2000) -> str:
    if len(content) <= limit:
        return content
    omitted = len(content) - limit
    return f"{content[:limit]}\n... <truncated {omitted} chars>"
