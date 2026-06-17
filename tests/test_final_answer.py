import unittest

from ft_agent.pipeline import FinalAnswerNode, RouterDecision, SupervisorReview


class FakeLLM:
    def __init__(self, content: str = "final answer") -> None:
        self.content = content
        self.calls = 0
        self.last_messages = None
        self.last_kwargs = None

    def chat(self, messages, **kwargs):
        self.calls += 1
        self.last_messages = messages
        self.last_kwargs = kwargs
        return self.content


class FinalAnswerTests(unittest.TestCase):
    def test_irrelevant_question_uses_llm_answer(self) -> None:
        llm = FakeLLM("Direct answer.")
        node = FinalAnswerNode(llm=llm)

        action, state = node.exec(
            {
                "question": "What is Python?",
                "router_decision": RouterDecision(
                    is_relevant=False,
                    needs_clarification=False,
                    deliverable_question="What is Python?",
                ),
                "final_chat_kwargs": {"stream": False},
            }
        )

        self.assertEqual(action, "answered")
        self.assertEqual(state["answer"], "Direct answer.")
        self.assertEqual(llm.calls, 1)
        self.assertIn("Router action: irrelevant", llm.last_messages[1]["content"])
        self.assertFalse(llm.last_kwargs["stream"])

    def test_clarification_returns_question_without_llm_call(self) -> None:
        llm = FakeLLM()
        node = FinalAnswerNode(llm=llm)

        _, state = node.exec(
            {
                "router_decision": RouterDecision(
                    is_relevant=True,
                    needs_clarification=True,
                    clarification_question="Which catalyst system should I focus on?",
                    deliverable_question="Improve FT catalyst stability.",
                )
            }
        )

        self.assertEqual(state["answer"], "Which catalyst system should I focus on?")
        self.assertEqual(llm.calls, 0)

    def test_report_answer_is_formatted_without_llm_call(self) -> None:
        llm = FakeLLM("Delivered report.")
        node = FinalAnswerNode(llm=llm)

        _, state = node.exec(
            {
                "question": "Write the report.",
                "router_decision": RouterDecision(
                    is_relevant=True,
                    needs_clarification=False,
                    deliverable_question="Write cobalt FT report.",
                ),
                "writer_report": "# Report\n\nContent",
                "writer_report_path": "reports/latest.md",
                "supervisor_review": SupervisorReview(
                    approved=True,
                    summary="Good enough.",
                ),
            }
        )

        self.assertIn("Supervisor approved: Good enough.", state["answer"])
        self.assertIn("Report path: reports/latest.md", state["answer"])
        self.assertIn("# Report", state["answer"])
        self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
