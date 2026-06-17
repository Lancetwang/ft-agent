import unittest
from tempfile import TemporaryDirectory

from ft_agent.core import Flow
from ft_agent.pipeline import SupervisorNode, SupervisorReview
from ft_agent.tools import build_write_file_tool


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_messages = None
        self.last_kwargs = None

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return self.content


class SupervisorTests(unittest.TestCase):
    def test_supervisor_approves_report(self) -> None:
        with TemporaryDirectory() as temp_dir:
            build_write_file_tool(temp_dir).execute(
                path="reports/latest.md",
                content="# Report\n\nObjective\n\nMethod\n\nExpected results\n",
            )
            node = SupervisorNode(
                llm=FakeLLM(
                    """
                    {
                      "approved": true,
                      "summary": "Good enough.",
                      "issues": []
                    }
                    """
                ),
                report_workspace=temp_dir,
            )

            action, state = node.exec({"writer_report_path": "reports/latest.md"})

        self.assertEqual(action, "approved")
        self.assertIsInstance(state["supervisor_review"], SupervisorReview)
        self.assertTrue(state["supervisor_review"].approved)
        self.assertIn("1: # Report", state["supervisor_file_excerpt"])

    def test_supervisor_requests_revision(self) -> None:
        with TemporaryDirectory() as temp_dir:
            build_write_file_tool(temp_dir).execute(
                path="reports/latest.md",
                content="# Report\n\nToo vague.\n",
            )
            node = SupervisorNode(
                llm=FakeLLM(
                    """
                    {
                      "approved": false,
                      "summary": "Needs more experimental detail.",
                      "issues": [
                        {
                          "line_start": 3,
                          "line_end": 3,
                          "severity": "major",
                          "message": "The method is too vague.",
                          "suggestion": "Add concrete conditions and measurements."
                        }
                      ]
                    }
                    """
                ),
                report_workspace=temp_dir,
            )

            action, state = node.exec({"writer_report_path": "reports/latest.md"})

        self.assertEqual(action, "revise")
        self.assertEqual(state["supervisor_revision_rounds"], 1)
        self.assertFalse(state["supervisor_review"].approved)
        self.assertEqual(state["supervisor_review"].issues[0].line_start, 3)

    def test_supervisor_emits_read_file_trace(self) -> None:
        with TemporaryDirectory() as temp_dir:
            build_write_file_tool(temp_dir).execute(path="reports/latest.md", content="# Report\n")
            node = SupervisorNode(
                llm=FakeLLM('{"approved": true, "summary": "ok", "issues": []}'),
                report_workspace=temp_dir,
            )

            result = Flow(node).run({"writer_report_path": "reports/latest.md"}, trace=True)

        tool_calls = [event for event in result.trace if event.event == "tool.call"]
        self.assertEqual(tool_calls[0].data["name"], "read_file")


if __name__ == "__main__":
    unittest.main()
