import json
import tempfile
import unittest

from ft_agent.pipeline import RouterDecision, build_ft_pipeline_flow


class FakeLLM:
    def __init__(self, responses: list[dict | str]) -> None:
        self.responses = list(responses)

    def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs):
        response = self.responses.pop(0)
        if isinstance(response, dict):
            response = json.dumps(response)
        return {"role": "assistant", "content": response}


class PipelineTests(unittest.TestCase):
    def test_irrelevant_question_routes_to_final(self) -> None:
        llms = iter(
            [
                FakeLLM(
                    [
                        {
                            "is_relevant": False,
                            "needs_clarification": False,
                            "clarification_question": None,
                            "deliverable_question": "hello",
                        }
                    ]
                ),
                FakeLLM([]),
                FakeLLM([]),
                FakeLLM([]),
                FakeLLM(["direct answer"]),
            ]
        )

        result = build_ft_pipeline_flow(llm_factory=lambda: next(llms)).run({"question": "hello"})

        self.assertEqual(result.payload["answer"], "direct answer")
        self.assertEqual(result.path, ["RouterAgent", "FinalAgent"])
        self.assertIsInstance(result.payload["router_decision"], RouterDecision)

    def test_relevant_question_runs_report_path(self) -> None:
        llms = iter(
            [
                FakeLLM(
                    [
                        {
                            "is_relevant": True,
                            "needs_clarification": False,
                            "clarification_question": None,
                            "deliverable_question": "cobalt FT stability report",
                        }
                    ]
                ),
                FakeLLM(
                    [
                        {
                            "summary": "write report",
                            "steps": [{"id": "s1", "instruction": "draft report"}],
                        }
                    ]
                ),
                FakeLLM(["# Report\ncontent"]),
                FakeLLM([{"approved": True, "summary": "good", "issues": []}]),
                FakeLLM([]),
            ]
        )

        with tempfile.TemporaryDirectory() as workspace:
            result = build_ft_pipeline_flow(
                llm_factory=lambda: next(llms),
                report_workspace=workspace,
            ).run(
                {"question": "cobalt FT stability"},
                max_steps=10,
            )

        self.assertIn("# Report", result.payload["answer"])
        self.assertEqual(result.payload["writer_report_path"], "reports/latest.md")
        self.assertEqual(
            result.path,
            ["RouterAgent", "PlannerAgent", "WriterAgent", "SupervisorAgent", "FinalAgent"],
        )


if __name__ == "__main__":
    unittest.main()
