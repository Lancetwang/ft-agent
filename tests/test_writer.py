import unittest
from tempfile import TemporaryDirectory

from ft_agent.core import Flow
from ft_agent.pipeline import PlanStep, PlannerPlan, WriterNode


class FakeLLM:
    def __init__(self) -> None:
        self.chat_message_calls = 0
        self.chat_calls = 0
        self.last_chat_kwargs = None

    def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs):
        self.chat_message_calls += 1
        if self.chat_message_calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_science_knowledge_base",
                            "arguments": '{"query": "cobalt FT catalyst stability", "top_k": 2}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "search_template_knowledge_base",
                            "arguments": '{"query": "FT catalyst experiment report"}',
                        },
                    },
                ],
            }
        return {"role": "assistant", "content": "READY_TO_WRITE"}

    def chat(self, messages, **kwargs):
        self.chat_calls += 1
        self.last_chat_kwargs = kwargs
        on_delta = kwargs.get("on_delta")
        if on_delta is not None:
            on_delta("Final ")
            on_delta("report")
        return "Final report"


class ToolMarkupThenReportLLM(FakeLLM):
    def chat(self, messages, **kwargs):
        self.chat_calls += 1
        self.last_chat_kwargs = kwargs
        if self.chat_calls == 1:
            return '<DSML tool_calls><invoke name="read_file"></invoke>'
        return "# Final report\n\nActual report content."


def sample_plan() -> PlannerPlan:
    return PlannerPlan(
        deliverable_question="Design an experiment report for cobalt FT catalyst stability.",
        summary="Retrieve context and template, then write.",
        steps=[
            PlanStep(
                id="s1",
                capability="search_science_knowledge_base",
                instruction="Retrieve scientific context.",
                expected_output="Evidence.",
            ),
            PlanStep(
                id="s2",
                capability="search_template_knowledge_base",
                instruction="Retrieve a template.",
                expected_output="Template.",
            ),
            PlanStep(
                id="s3",
                capability="write_experimental_report",
                instruction="Write the report.",
                expected_output="Experiment report.",
                depends_on=["s1", "s2"],
            ),
        ],
    )


class WriterTests(unittest.TestCase):
    def test_writer_executes_tools_and_writes_report(self) -> None:
        llm = FakeLLM()
        with TemporaryDirectory() as temp_dir:
            node = WriterNode(llm=llm, report_workspace=temp_dir)

            action, state = node.exec({"planner_plan": sample_plan()})

        self.assertEqual(action, "written")
        self.assertEqual(state["writer_report"], "Final report")
        self.assertEqual(state["writer_report_path"], "reports/latest.md")
        self.assertEqual(len(state["writer_tool_results"]), 3)
        self.assertIn("mock-science-kb", state["writer_tool_results"][0].content)
        self.assertIn("mock-template-kb", state["writer_tool_results"][1].content)
        self.assertIn("latest.md", state["writer_tool_results"][2].content)

    def test_writer_streams_final_report(self) -> None:
        llm = FakeLLM()
        with TemporaryDirectory() as temp_dir:
            node = WriterNode(llm=llm, report_workspace=temp_dir)
            seen: list[str] = []

            node.exec(
                {
                    "planner_plan": sample_plan(),
                    "writer_chat_kwargs": {
                        "stream": True,
                        "on_delta": seen.append,
                    },
                }
            )

        self.assertEqual(seen, ["Final report"])
        self.assertNotIn("stream", llm.last_chat_kwargs)

    def test_writer_retries_tool_markup_report(self) -> None:
        llm = ToolMarkupThenReportLLM()
        with TemporaryDirectory() as temp_dir:
            node = WriterNode(llm=llm, report_workspace=temp_dir)

            _, state = node.exec({"planner_plan": sample_plan()})

        self.assertEqual(llm.chat_calls, 2)
        self.assertEqual(state["writer_report"], "# Final report\n\nActual report content.")
        self.assertNotIn("tool_calls", state["writer_report"])

    def test_writer_emits_tool_trace_events(self) -> None:
        llm = FakeLLM()
        with TemporaryDirectory() as temp_dir:
            node = WriterNode(llm=llm, report_workspace=temp_dir)

            result = Flow(node).run({"planner_plan": sample_plan()}, trace=True)
        tool_events = [event for event in result.trace if event.category == "tool"]
        tool_calls = [event for event in tool_events if event.event == "tool.call"]
        tool_results = [event for event in tool_events if event.event == "tool.result"]

        self.assertEqual(tool_events[0].event, "tool.round")
        self.assertEqual(tool_events[0].data["tool_call_count"], 2)
        self.assertEqual(
            [event.data["name"] for event in tool_calls[:2]],
            ["search_science_knowledge_base", "search_template_knowledge_base"],
        )
        self.assertEqual(tool_calls[0].data["arguments"]["top_k"], 2)
        self.assertEqual(tool_calls[-1].data["name"], "write_file")
        self.assertEqual(len(tool_results), 3)
        self.assertFalse(tool_results[0].data["is_error"])


if __name__ == "__main__":
    unittest.main()
