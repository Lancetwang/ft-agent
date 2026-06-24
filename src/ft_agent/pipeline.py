from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Annotated, Any

from agent_core import Agent, Flow, LLM, Node, Payload, tool

LLMFactory = Callable[[], LLM]


ROUTER_PROMPT = """
You are the router for a Fischer-Tropsch catalyst research agent.

Decide whether the user's question is strongly related to Fischer-Tropsch
catalysts. Ask one concise clarification question only when planning would be
unsafe without more details.

Return only JSON:
{
  "is_relevant": true,
  "needs_clarification": false,
  "clarification_question": null,
  "deliverable_question": "question for the planner"
}
""".strip()


PLANNER_PROMPT = """
You are the planner for a Fischer-Tropsch catalyst research agent.

Create a short, executable plan for the writer. The writer can retrieve
scientific context, retrieve one report template, and write a final experiment
report.

Return only JSON:
{
  "summary": "short summary",
  "steps": [
    {"id": "s1", "instruction": "specific writer instruction"}
  ]
}
""".strip()


FINAL_PROMPT = """
You are the final answer agent. If the question is outside Fischer-Tropsch
catalysts, answer normally. Otherwise, deliver the available report or
clarification question without exposing internal implementation details.
""".strip()


WRITER_PROMPT = """
You are the writer for a Fischer-Tropsch catalyst research agent.

Use the available tools when useful, then write a complete Markdown experiment
report. The report should be specific, actionable, and grounded in retrieved
scientific context or template structure when available.
""".strip()


SUPERVISOR_PROMPT = """
You are the supervisor for a Fischer-Tropsch catalyst report.

Review the report against structure, domain specificity, actionability, and
evidence grounding. Return only JSON:
{
  "approved": true,
  "summary": "short review summary",
  "issues": []
}
""".strip()


@dataclass(frozen=True)
class RouterDecision:
    is_relevant: bool
    needs_clarification: bool
    clarification_question: str | None
    deliverable_question: str

    @property
    def action(self) -> str:
        if not self.is_relevant:
            return "irrelevant"
        if self.needs_clarification:
            return "clarify"
        return "ready"


@dataclass(frozen=True)
class PlanStep:
    id: str
    instruction: str


@dataclass(frozen=True)
class PlannerPlan:
    summary: str
    steps: list[PlanStep]


@dataclass(frozen=True)
class SupervisorReview:
    approved: bool
    summary: str
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def action(self) -> str:
        return "approved" if self.approved else "revise"


class RouterNode(Node):
    def __init__(self, *, llm: LLM | None = None) -> None:
        super().__init__()
        self.llm = llm or LLM()

    def exec(self, payload: Payload):
        state = dict(payload or {})
        question = _router_question(state)
        content = _chat_json(
            self.llm,
            [
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        decision = RouterDecision(
            is_relevant=bool(content.get("is_relevant", False)),
            needs_clarification=bool(content.get("needs_clarification", False)),
            clarification_question=_optional_text(content.get("clarification_question")),
            deliverable_question=_optional_text(content.get("deliverable_question")) or question,
        )
        state["question"] = question
        state["router_decision"] = decision
        return decision.action, state


class PlannerNode(Node):
    def __init__(self, *, llm: LLM | None = None) -> None:
        super().__init__()
        self.llm = llm or LLM()

    def exec(self, payload: Payload):
        state = dict(payload or {})
        question = _deliverable_question(state)
        content = _chat_json(
            self.llm,
            [
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        steps = [
            PlanStep(
                id=str(item.get("id") or f"s{index}"),
                instruction=str(item.get("instruction") or item),
            )
            for index, item in enumerate(content.get("steps", []), start=1)
            if isinstance(item, Mapping)
        ]
        if not steps:
            steps = [PlanStep(id="s1", instruction=f"Write a report for: {question}")]
        state["planner_plan"] = PlannerPlan(
            summary=str(content.get("summary") or "Prepare a Fischer-Tropsch report."),
            steps=steps,
        )
        return "planned", state


class WriterNode(Node):
    def __init__(
        self,
        *,
        llm: LLM | None = None,
        report_workspace: str | Path = "artifacts",
    ) -> None:
        super().__init__()
        self.agent = Agent(
            model=llm,
            instructions=WRITER_PROMPT,
            tools=[search_science_knowledge_base, search_template_knowledge_base],
            chat_kwargs={"temperature": 0.2, "max_tokens": 1800, "tool_choice": "auto"},
        )
        self.report_workspace = Path(report_workspace)

    def exec(self, payload: Payload):
        state = dict(payload or {})
        context = self.agent.new_context()
        report = self.agent.chat(_writer_request(state), context=context, max_steps=10)
        report_path = Path("reports/latest.md")
        full_path = self.report_workspace / report_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(report, encoding="utf-8")
        state["writer_report"] = report
        state["writer_report_path"] = str(report_path).replace("\\", "/")
        return "written", state


class SupervisorNode(Node):
    def __init__(self, *, llm: LLM | None = None, max_revisions: int = 2) -> None:
        super().__init__()
        self.llm = llm or LLM()
        self.max_revisions = max_revisions

    def exec(self, payload: Payload):
        state = dict(payload or {})
        rounds = int(state.get("supervisor_revision_rounds", 0))
        content = _chat_json(
            self.llm,
            [
                {"role": "system", "content": SUPERVISOR_PROMPT},
                {"role": "user", "content": str(state.get("writer_report", ""))},
            ],
        )
        review = SupervisorReview(
            approved=bool(content.get("approved", False)) or rounds >= self.max_revisions,
            summary=str(content.get("summary") or ""),
            issues=list(content.get("issues", [])) if isinstance(content.get("issues"), list) else [],
        )
        state["supervisor_review"] = review
        if review.action == "revise":
            state["supervisor_revision_rounds"] = rounds + 1
        return review.action, state


class FinalAnswerNode(Node):
    def __init__(self, *, llm: LLM | None = None) -> None:
        super().__init__()
        self.llm = llm or LLM()

    def exec(self, payload: Payload):
        state = dict(payload or {})
        decision = state.get("router_decision")
        if isinstance(decision, RouterDecision) and decision.action == "clarify":
            state["answer"] = decision.clarification_question or "Please provide more detail."
            return "answered", state
        if state.get("writer_report"):
            review = state.get("supervisor_review")
            prefix = _review_summary(review)
            path = state.get("writer_report_path")
            state["answer"] = "\n\n".join(
                part for part in [prefix, f"Report path: {path}" if path else "", state["writer_report"]] if part
            )
            return "answered", state

        content = self.llm.chat_message(
            [
                {"role": "system", "content": FINAL_PROMPT},
                {"role": "user", "content": str(state.get("question", ""))},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        state["answer"] = str(content.get("content", ""))
        return "answered", state


class RouterAgent(Agent):
    pass


class PlannerAgent(Agent):
    pass


class WriterAgent(Agent):
    pass


class SupervisorAgent(Agent):
    pass


class FinalAgent(Agent):
    pass


def build_ft_pipeline_flow(
    *,
    llm_factory: LLMFactory = LLM,
    report_workspace: str | Path = "artifacts",
) -> Flow:
    router = RouterAgent(Flow(RouterNode(llm=llm_factory())), action=None)
    planner = PlannerAgent(Flow(PlannerNode(llm=llm_factory())), action=None)
    writer = WriterAgent(
        Flow(WriterNode(llm=llm_factory(), report_workspace=report_workspace)),
        action=None,
    )
    supervisor = SupervisorAgent(Flow(SupervisorNode(llm=llm_factory())), action=None)
    final = FinalAgent(Flow(FinalAnswerNode(llm=llm_factory())), action=None)

    router - "irrelevant" >> final
    router - "clarify" >> final
    router - "ready" >> planner
    planner - "planned" >> writer
    writer - "written" >> supervisor
    supervisor - "revise" >> writer
    supervisor - "approved" >> final
    return Flow(router)


def build_ft_agent(
    *,
    llm_factory: LLMFactory = LLM,
    report_workspace: str | Path = "artifacts",
) -> Agent:
    return Agent(
        build_ft_pipeline_flow(
            llm_factory=llm_factory,
            report_workspace=report_workspace,
        )
    )


@tool(description="Retrieve mock Fischer-Tropsch scientific context.")
def search_science_knowledge_base(
    query: Annotated[str, "Scientific query."],
    top_k: Annotated[int, "Number of snippets."] = 3,
) -> dict[str, Any]:
    snippets = [
        "Cobalt FT catalysts often favor long-chain hydrocarbons and need high dispersion.",
        "Supports such as silica, alumina, and titania affect reducibility and stability.",
        "Common deactivation modes include sintering, oxidation, carbon deposition, and sulfur poisoning.",
    ]
    return {"query": query, "results": snippets[: max(1, min(top_k, len(snippets)))]}


@tool(description="Retrieve one mock experiment report template.")
def search_template_knowledge_base(query: Annotated[str, "Template query."]) -> dict[str, Any]:
    return {
        "query": query,
        "sections": [
            "Objective",
            "Background",
            "Catalyst design",
            "Experimental method",
            "Characterization",
            "Reaction testing",
            "Expected results",
            "Risks",
        ],
    }


def _chat_json(llm: LLM, messages: list[dict[str, str]]) -> dict[str, Any]:
    content = llm.chat_message(messages, temperature=0, max_tokens=900)["content"]
    return _loads_json_object(str(content))


def _loads_json_object(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return {}
        try:
            value = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _router_question(state: Mapping[str, Any]) -> str:
    if state.get("clarification_response"):
        return f"{state.get('question', '')}\nClarification: {state['clarification_response']}"
    return str(state.get("question", "")).strip()


def _deliverable_question(state: Mapping[str, Any]) -> str:
    decision = state.get("router_decision")
    if isinstance(decision, RouterDecision):
        return decision.deliverable_question
    return str(state.get("question", "")).strip()


def _writer_request(state: Mapping[str, Any]) -> str:
    plan = state.get("planner_plan")
    if isinstance(plan, PlannerPlan):
        plan_text = json.dumps(asdict(plan), ensure_ascii=False, indent=2)
    else:
        plan_text = str(plan or state.get("question", ""))
    review = state.get("supervisor_review")
    return f"Write or revise the experiment report from this plan:\n{plan_text}\n\nSupervisor review:\n{review or 'none'}"


def _review_summary(review: Any) -> str | None:
    if isinstance(review, SupervisorReview):
        status = "approved" if review.approved else "not approved"
        return f"Supervisor {status}: {review.summary}".strip()
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
