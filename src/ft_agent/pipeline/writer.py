from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from ft_agent.core import ExecResult, Node, Payload
from ft_agent.llm import DeepSeekLLM
from ft_agent.llm.deepseek import Message
from ft_agent.pipeline.planner import PlannerPlan
from ft_agent.tools import Tool, ToolExecutor, ToolResult, tool


WRITER_SYSTEM_PROMPT = """
You are the writer for an agent pipeline focused on Fischer-Tropsch catalysts.

# Role
Produce the final experiment report.

# Tools
You may retrieve scientific context and one report template. Use these tools when
they help you produce a better report. Stop using tools when you have enough
context to write.

# Output
The final deliverable is an experiment report.
""".strip()


@tool(description="Retrieve mock scientific context from the scientific knowledge base.")
def search_science_knowledge_base(
    query: Annotated[str, "Scientific retrieval query chosen by the writer."],
    top_k: Annotated[int, "Number of mock context snippets to retrieve."] = 3,
) -> dict[str, Any]:
    snippets = [
        {
            "title": "Cobalt FT catalyst deactivation overview",
            "content": (
                "Common deactivation routes include sintering, carbon deposition, "
                "oxidation of cobalt sites, water effects, and sulfur poisoning."
            ),
            "source": "mock-science-kb",
        },
        {
            "title": "Support and promoter effects",
            "content": (
                "Supports such as alumina, silica, and titania influence dispersion, "
                "reducibility, metal-support interaction, and long-term stability."
            ),
            "source": "mock-science-kb",
        },
        {
            "title": "Experimental reporting for FT catalysts",
            "content": (
                "Useful reports include catalyst composition, preparation method, "
                "reduction conditions, reactor setup, H2/CO ratio, conversion, "
                "selectivity, time-on-stream, and characterization before and after reaction."
            ),
            "source": "mock-science-kb",
        },
    ]
    return {
        "query": query,
        "top_k": top_k,
        "results": snippets[: max(1, min(top_k, len(snippets)))],
    }


@tool(description="Retrieve the most similar mock experiment-report template.")
def search_template_knowledge_base(
    query: Annotated[str, "Template retrieval query chosen by the writer."],
) -> dict[str, Any]:
    return {
        "query": query,
        "template": {
            "name": "Fischer-Tropsch catalyst experiment report",
            "sections": [
                "Title",
                "Objective",
                "Background and rationale",
                "Catalyst and material design",
                "Experimental method",
                "Characterization plan",
                "Reaction testing plan",
                "Expected results and evaluation criteria",
                "Risks and troubleshooting",
                "References or evidence notes",
            ],
            "source": "mock-template-kb",
        },
    }


class WriterNode(Node):
    def __init__(
        self,
        *,
        llm: DeepSeekLLM,
        tools: Sequence[Tool] | None = None,
        plan_key: str = "planner_plan",
        output_key: str = "writer_report",
        messages_key: str = "writer_messages",
        tool_results_key: str = "writer_tool_results",
        chat_kwargs_key: str = "writer_chat_kwargs",
        action: str = "written",
        max_tool_rounds: int = 6,
        system_prompt: str = WRITER_SYSTEM_PROMPT,
        chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be non-negative.")
        self.llm = llm
        self.tools = list(tools or default_writer_tools())
        self.executor = ToolExecutor(self.tools)
        self.plan_key = plan_key
        self.output_key = output_key
        self.messages_key = messages_key
        self.tool_results_key = tool_results_key
        self.chat_kwargs_key = chat_kwargs_key
        self.action = action
        self.max_tool_rounds = max_tool_rounds
        self.system_prompt = system_prompt
        self.chat_kwargs = dict(chat_kwargs or {})

    def exec(self, payload: Payload) -> ExecResult:
        state = dict(payload or {})
        messages = self._initial_messages(state)
        tool_results: list[ToolResult] = []
        decision_kwargs = self._chat_kwargs(state, allow_stream=False)

        for _ in range(self.max_tool_rounds):
            assistant_message = self.llm.chat_message(
                messages,
                tools=[tool.to_llm_format() for tool in self.tools],
                tool_choice="auto",
                **decision_kwargs,
            )
            messages.append(assistant_message)
            tool_calls = self.executor.parse_tool_calls(assistant_message)
            if not tool_calls:
                break
            for tool_call in tool_calls:
                result = self.executor.execute(tool_call)
                tool_results.append(result)
                messages.append(result.to_message())

        final_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "Write the final experiment report now. Use the plan and any "
                    "retrieved scientific context or template already available."
                ),
            },
        ]
        report = self.llm.chat(
            final_messages,
            **self._chat_kwargs(state, allow_stream=True),
        )
        state[self.output_key] = report
        state[self.messages_key] = final_messages + [{"role": "assistant", "content": report}]
        state[self.tool_results_key] = tool_results
        return self.action, state

    def _initial_messages(self, state: Mapping[str, Any]) -> list[Message]:
        plan = state.get(self.plan_key)
        if isinstance(plan, PlannerPlan):
            plan_payload = plan.to_dict()
        elif isinstance(plan, Mapping):
            plan_payload = dict(plan)
        else:
            plan_payload = {"deliverable_question": str(state.get("question", "")).strip()}

        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Plan to execute:\n"
                    f"{plan_payload}\n\n"
                    "Decide what context or template retrieval is useful before writing."
                ),
            },
        ]

    def _chat_kwargs(self, state: Mapping[str, Any], *, allow_stream: bool) -> dict[str, Any]:
        chat_kwargs = {"temperature": 0, **self.chat_kwargs}
        common_kwargs = state.get("chat_kwargs", {})
        if isinstance(common_kwargs, Mapping):
            chat_kwargs.update(common_kwargs)
        node_kwargs = state.get(self.chat_kwargs_key, {})
        if isinstance(node_kwargs, Mapping):
            chat_kwargs.update(node_kwargs)
        if not allow_stream:
            chat_kwargs.pop("stream", None)
            chat_kwargs.pop("on_delta", None)
        return chat_kwargs


def default_writer_tools() -> list[Tool]:
    return [
        search_science_knowledge_base,
        search_template_knowledge_base,
    ]
