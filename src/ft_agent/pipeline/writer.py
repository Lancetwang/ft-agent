from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

from agent_core import ExecResult, Node, Payload
from agent_core.core.trace import get_trace_recorder
from ft_agent.llm import DeepSeekLLM
from ft_agent.llm.deepseek import Message
from ft_agent.pipeline.planner import PlannerPlan
from agent_core import Tool, ToolCall, ToolExecutor, ToolResult, build_file_tools, tool


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
        report_workspace: str | Path = "artifacts",
        default_report_path: str = "reports/latest.md",
        plan_key: str = "planner_plan",
        review_key: str = "supervisor_review",
        output_key: str = "writer_report",
        report_path_key: str = "writer_report_path",
        messages_key: str = "writer_messages",
        tool_results_key: str = "writer_tool_results",
        chat_kwargs_key: str = "writer_chat_kwargs",
        action: str = "written",
        max_tool_rounds: int = 6,
        max_report_attempts: int = 2,
        system_prompt: str = WRITER_SYSTEM_PROMPT,
        chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be non-negative.")
        if max_report_attempts < 1:
            raise ValueError("max_report_attempts must be at least 1.")
        self.llm = llm
        self.report_workspace = Path(report_workspace)
        self.default_report_path = default_report_path
        if tools is None:
            self.tools = default_writer_tools(self.report_workspace)
        else:
            self.tools = _with_file_tools(tools, self.report_workspace)
        self.executor = ToolExecutor(self.tools)
        self.plan_key = plan_key
        self.review_key = review_key
        self.output_key = output_key
        self.report_path_key = report_path_key
        self.messages_key = messages_key
        self.tool_results_key = tool_results_key
        self.chat_kwargs_key = chat_kwargs_key
        self.action = action
        self.max_tool_rounds = max_tool_rounds
        self.max_report_attempts = max_report_attempts
        self.system_prompt = system_prompt
        self.chat_kwargs = dict(chat_kwargs or {})

    def exec(self, payload: Payload) -> ExecResult:
        state = dict(payload or {})
        messages = self._initial_messages(state)
        tool_results: list[ToolResult] = []
        decision_kwargs = self._chat_kwargs(state, allow_stream=False)
        recorder = get_trace_recorder(state)

        for round_index in range(1, self.max_tool_rounds + 1):
            assistant_message = self.llm.chat_message(
                messages,
                tools=[tool.to_llm_format() for tool in self.tools],
                tool_choice="auto",
                **decision_kwargs,
            )
            messages.append(assistant_message)
            tool_calls = self.executor.parse_tool_calls(assistant_message)
            if recorder is not None:
                recorder.emit(
                    "tool.round",
                    category="tool",
                    data={
                        "round": round_index,
                        "tool_call_count": len(tool_calls),
                    },
                )
            if not tool_calls:
                break
            for tool_call in tool_calls:
                if recorder is not None:
                    recorder.emit(
                        "tool.call",
                        category="tool",
                        data={
                            "round": round_index,
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    )
                result = self.executor.execute(tool_call)
                tool_results.append(result)
                messages.append(result.to_message())
                if recorder is not None:
                    recorder.emit(
                        "tool.result",
                        category="tool",
                        data={
                            "round": round_index,
                            "tool_call_id": result.tool_call_id,
                            "content": result.content,
                            "is_error": result.is_error,
                        },
                    )

        final_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "Write the final experiment report now as complete Markdown. "
                    "Use the plan and any retrieved scientific context or template "
                    "already available. Do not call tools, do not emit tool-call "
                    "markup, and do not merely say the report was already written."
                ),
            },
        ]
        report = self._generate_report(final_messages, state)
        report_path = str(state.get(self.report_path_key) or self.default_report_path)
        write_result = self._execute_tool(
            ToolCall(
                id="writer_write_report",
                name="write_file",
                arguments={"path": report_path, "content": report},
            ),
            round_index="artifact",
            recorder=recorder,
        )
        tool_results.append(write_result)
        state[self.output_key] = report
        state[self.report_path_key] = report_path
        state[self.messages_key] = final_messages + [{"role": "assistant", "content": report}]
        state[self.tool_results_key] = tool_results
        return self.action, state

    def _generate_report(self, messages: list[Message], state: Mapping[str, Any]) -> str:
        report_messages = list(messages)
        report = ""
        for attempt in range(1, self.max_report_attempts + 1):
            report = self.llm.chat(
                report_messages,
                **self._chat_kwargs(state, allow_stream=False),
            )
            if not _looks_like_tool_markup(report):
                self._emit_report_delta(report, state)
                return report
            if attempt < self.max_report_attempts:
                report_messages.extend(
                    [
                        {"role": "assistant", "content": report},
                        {
                            "role": "user",
                            "content": (
                                "The previous response was a tool call or tool-call markup, "
                                "not the experiment report. Write the complete Markdown "
                                "experiment report content now. Do not call read_file, "
                                "write_file, or any other tool. Do not include DSML, XML, "
                                "or tool-call tags."
                            ),
                        },
                    ]
                )

        raise ValueError("Writer produced tool-call markup instead of a report.")

    def _emit_report_delta(self, report: str, state: Mapping[str, Any]) -> None:
        chat_kwargs = self._chat_kwargs(state, allow_stream=True)
        if not chat_kwargs.get("stream"):
            return
        on_delta = chat_kwargs.get("on_delta")
        if not callable(on_delta):
            return
        for chunk in _chunk_text(report):
            on_delta(chunk)

    def _execute_tool(
        self,
        tool_call: ToolCall,
        *,
        round_index: int | str,
        recorder: Any,
    ) -> ToolResult:
        if recorder is not None:
            recorder.emit(
                "tool.call",
                category="tool",
                data={
                    "round": round_index,
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                },
            )
        result = self.executor.execute(tool_call)
        if recorder is not None:
            recorder.emit(
                "tool.result",
                category="tool",
                data={
                    "round": round_index,
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                },
            )
        return result

    def _initial_messages(self, state: Mapping[str, Any]) -> list[Message]:
        plan = state.get(self.plan_key)
        if isinstance(plan, PlannerPlan):
            plan_payload = plan.to_dict()
        elif isinstance(plan, Mapping):
            plan_payload = dict(plan)
        else:
            plan_payload = {"deliverable_question": str(state.get("question", "")).strip()}
        review = state.get(self.review_key)
        revision_context = ""
        if review is not None:
            revision_context = (
                f"\nExisting report path: {state.get(self.report_path_key, self.default_report_path)}\n\n"
                f"Supervisor review:\n{review}\n"
            )

        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Plan to execute:\n"
                    f"{plan_payload}\n\n"
                    f"{revision_context}\n"
                    "Decide what context, template retrieval, or file edits are useful before writing."
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


def default_writer_tools(report_workspace: str | Path = "artifacts") -> list[Tool]:
    return [
        search_science_knowledge_base,
        search_template_knowledge_base,
        *build_file_tools(report_workspace),
    ]


def _with_file_tools(tools: Sequence[Tool], report_workspace: str | Path) -> list[Tool]:
    merged = list(tools)
    names = {tool.name for tool in merged}
    for file_tool in build_file_tools(report_workspace):
        if file_tool.name not in names:
            merged.append(file_tool)
            names.add(file_tool.name)
    return merged


def _looks_like_tool_markup(content: str) -> bool:
    lowered = content.lower()
    markers = (
        "tool_calls",
        "invoke name=",
        "read_file",
        "write_file",
        "dsml",
    )
    return any(marker in lowered for marker in markers)


def _chunk_text(text: str, size: int = 32) -> list[str]:
    chunks: list[str] = []
    current = ""
    for char in text:
        current += char
        if len(current) >= size or char in "\n。！？.!?":
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks
