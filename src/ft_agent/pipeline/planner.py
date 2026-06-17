from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ft_agent.core import ExecResult, Node, Payload
from ft_agent.llm import DeepSeekLLM
from ft_agent.llm.deepseek import Message
from ft_agent.pipeline.router import RouterDecision


PLANNER_SYSTEM_PROMPT = """
You are the planner for an agent pipeline focused on Fischer-Tropsch catalysts.

# Task
Create a concrete, traceable plan for the writer.

# Rules
- Use only the provided writer capabilities.
- Do not call tools or invent tool arguments.
- Each step must be executable by the writer.
- Each step must name exactly one capability.
- Use dependencies when a step needs earlier outputs.

# Output
Return only one JSON object:
{
  "deliverable_question": "The question being planned for.",
  "summary": "Short plan summary.",
  "steps": [
    {
      "id": "s1",
      "capability": "capability_name",
      "instruction": "Concrete instruction for the writer.",
      "expected_output": "What this step should produce.",
      "depends_on": []
    }
  ]
}
""".strip()


class PlannerParseError(ValueError):
    pass


@dataclass(frozen=True)
class WriterCapability:
    name: str
    description: str
    kind: str = "intrinsic"
    inputs: list[str] = field(default_factory=list)
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "inputs": self.inputs,
            "output": self.output,
        }


@dataclass(frozen=True)
class PlanStep:
    id: str
    capability: str
    instruction: str
    expected_output: str
    depends_on: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlanStep:
        return cls(
            id=_required_text(data, "id"),
            capability=_required_text(data, "capability"),
            instruction=_required_text(data, "instruction"),
            expected_output=_required_text(data, "expected_output"),
            depends_on=[
                str(item).strip()
                for item in data.get("depends_on", [])
                if str(item).strip()
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "capability": self.capability,
            "instruction": self.instruction,
            "expected_output": self.expected_output,
            "depends_on": self.depends_on,
        }


@dataclass(frozen=True)
class PlannerPlan:
    deliverable_question: str
    summary: str
    steps: list[PlanStep]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        fallback_question: str,
        capabilities: Sequence[WriterCapability],
    ) -> PlannerPlan:
        steps_data = data.get("steps")
        if not isinstance(steps_data, list) or not steps_data:
            raise PlannerParseError("Planner response must contain a non-empty steps list.")

        plan = cls(
            deliverable_question=_optional_text(data.get("deliverable_question"))
            or fallback_question,
            summary=_optional_text(data.get("summary")) or "",
            steps=[PlanStep.from_dict(item) for item in steps_data if isinstance(item, Mapping)],
            raw=dict(data),
        )
        if not plan.steps:
            raise PlannerParseError("Planner response must contain valid plan steps.")
        _validate_plan(plan, capabilities)
        return plan

    def to_dict(self) -> dict[str, Any]:
        return {
            "deliverable_question": self.deliverable_question,
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
        }


class PlannerNode(Node):
    def __init__(
        self,
        *,
        llm: DeepSeekLLM,
        capabilities: Sequence[WriterCapability] | None = None,
        router_decision_key: str = "router_decision",
        question_key: str = "question",
        output_key: str = "planner_plan",
        raw_output_key: str = "planner_raw_output",
        chat_kwargs_key: str = "planner_chat_kwargs",
        action: str = "planned",
        system_prompt: str = PLANNER_SYSTEM_PROMPT,
        chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.capabilities = list(capabilities or default_writer_capabilities())
        self.router_decision_key = router_decision_key
        self.question_key = question_key
        self.output_key = output_key
        self.raw_output_key = raw_output_key
        self.chat_kwargs_key = chat_kwargs_key
        self.action = action
        self.system_prompt = system_prompt
        self.chat_kwargs = dict(chat_kwargs or {})

    def exec(self, payload: Payload) -> ExecResult:
        state = dict(payload or {})
        deliverable_question = self._deliverable_question(state)
        chat_kwargs = self._chat_kwargs(state)
        content = self.llm.chat(
            self._messages(deliverable_question),
            **chat_kwargs,
        )
        plan = PlannerPlan.from_dict(
            _loads_json_object(content),
            fallback_question=deliverable_question,
            capabilities=self.capabilities,
        )
        state[self.output_key] = plan
        state[self.raw_output_key] = content
        return self.action, state

    def _deliverable_question(self, state: Mapping[str, Any]) -> str:
        decision = state.get(self.router_decision_key)
        if isinstance(decision, RouterDecision):
            return decision.deliverable_question
        if isinstance(decision, Mapping):
            question = _optional_text(decision.get("deliverable_question"))
            if question:
                return question
        return str(state.get(self.question_key, "")).strip()

    def _chat_kwargs(self, state: Mapping[str, Any]) -> dict[str, Any]:
        chat_kwargs = {"temperature": 0, **self.chat_kwargs}
        common_kwargs = state.get("chat_kwargs", {})
        if isinstance(common_kwargs, Mapping):
            chat_kwargs.update(common_kwargs)
        node_kwargs = state.get(self.chat_kwargs_key, {})
        if isinstance(node_kwargs, Mapping):
            chat_kwargs.update(node_kwargs)
        return chat_kwargs

    def _messages(self, deliverable_question: str) -> list[Message]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": _planner_user_content(
                    deliverable_question,
                    self.capabilities,
                ),
            },
        ]


def default_writer_capabilities() -> list[WriterCapability]:
    return [
        WriterCapability(
            name="search_science_knowledge_base",
            description=(
                "Tool-backed retrieval from the scientific knowledge base. The writer "
                "may use it repeatedly with its own queries until enough context is gathered."
            ),
            kind="tool",
            inputs=["query", "top_k"],
            output="Relevant scientific evidence, findings, and citations.",
        ),
        WriterCapability(
            name="search_template_knowledge_base",
            description=(
                "Tool-backed retrieval from the template knowledge base. It returns "
                "the most similar report template for the writer to follow."
            ),
            kind="tool",
            inputs=["query"],
            output="The most similar experiment-report template.",
        ),
        WriterCapability(
            name="write_experimental_report",
            description=(
                "Intrinsic writer ability. Compose the final experiment report from "
                "the plan, retrieved scientific context, and retrieved template."
            ),
            kind="intrinsic",
            inputs=["plan", "scientific_context", "template"],
            output="Final experiment report.",
        ),
    ]


def _planner_user_content(
    deliverable_question: str,
    capabilities: Sequence[WriterCapability],
) -> str:
    capability_json = json.dumps(
        [capability.to_dict() for capability in capabilities],
        ensure_ascii=False,
        indent=2,
    )
    return (
        f"Deliverable question:\n{deliverable_question}\n\n"
        f"Writer capabilities:\n{capability_json}"
    )


def _validate_plan(plan: PlannerPlan, capabilities: Sequence[WriterCapability]) -> None:
    capability_names = {capability.name for capability in capabilities}
    seen_ids: set[str] = set()
    for step in plan.steps:
        if step.id in seen_ids:
            raise PlannerParseError(f"Duplicate plan step id: {step.id}.")
        if step.capability not in capability_names:
            raise PlannerParseError(f"Unknown writer capability: {step.capability}.")
        missing_dependencies = [
            dependency
            for dependency in step.depends_on
            if dependency not in seen_ids
        ]
        if missing_dependencies:
            raise PlannerParseError(
                f"Unknown or future plan dependencies: "
                f"{', '.join(sorted(set(missing_dependencies)))}."
            )
        seen_ids.add(step.id)


def _loads_json_object(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise PlannerParseError("Planner response did not contain a JSON object.") from None
        try:
            value = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PlannerParseError("Planner response contained invalid JSON.") from exc

    if not isinstance(value, dict):
        raise PlannerParseError("Planner response must be a JSON object.")
    return value


def _required_text(data: Mapping[str, Any], key: str) -> str:
    text = _optional_text(data.get(key))
    if text is None:
        raise PlannerParseError(f"Missing required plan field: {key}.")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
