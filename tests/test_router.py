import unittest

from ft_agent.core import CallableNode, Flow
from ft_agent.pipeline import RouterDecision, RouterNode, RouterParseError


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_messages = None
        self.last_kwargs = None

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return self.content


class RouterTests(unittest.TestCase):
    def test_router_routes_ready_question(self) -> None:
        llm = FakeLLM(
            """
            {
              "is_relevant": true,
              "needs_clarification": false,
              "clarification_question": null,
              "deliverable_question": "How does cobalt particle size affect FT selectivity?"
            }
            """
        )
        node = RouterNode(llm=llm)

        action, state = node.exec({"question": "cobalt FT catalyst particle size"})

        self.assertEqual(action, "ready")
        self.assertIsInstance(state["router_decision"], RouterDecision)
        self.assertTrue(state["router_decision"].is_relevant)
        self.assertEqual(
            state["router_decision"].deliverable_question,
            "How does cobalt particle size affect FT selectivity?",
        )
        self.assertEqual(llm.last_kwargs["temperature"], 0)

    def test_router_routes_irrelevant_question(self) -> None:
        node = RouterNode(
            llm=FakeLLM(
                """
                {
                  "is_relevant": "false",
                  "needs_clarification": true,
                  "clarification_question": "Which catalyst?",
                  "deliverable_question": "What is the weather?"
                }
                """
            )
        )

        action, state = node.exec({"question": "What is the weather?"})
        decision = state["router_decision"]

        self.assertEqual(action, "irrelevant")
        self.assertFalse(decision.is_relevant)
        self.assertFalse(decision.needs_clarification)
        self.assertIsNone(decision.clarification_question)

    def test_router_routes_clarification_question(self) -> None:
        node = RouterNode(
            llm=FakeLLM(
                """
                {
                  "is_relevant": true,
                  "needs_clarification": true,
                  "clarification_question": "Are you asking about cobalt or iron catalysts?",
                  "deliverable_question": "How can I improve FT catalyst performance?"
                }
                """
            )
        )

        action, state = node.exec({"question": "How can I improve it?"})

        self.assertEqual(action, "clarify")
        self.assertEqual(
            state["router_decision"].clarification_question,
            "Are you asking about cobalt or iron catalysts?",
        )

    def test_router_works_in_flow(self) -> None:
        router = RouterNode(
            llm=FakeLLM(
                """
                {
                  "is_relevant": true,
                  "needs_clarification": false,
                  "clarification_question": null,
                  "deliverable_question": "Summarize cobalt FT catalyst deactivation."
                }
                """
            )
        )
        final = CallableNode(lambda payload: {"answer": payload["router_decision"].deliverable_question})
        router - "ready" >> final

        result = Flow(router).run({"question": "cobalt deactivation"})

        self.assertEqual(result.payload["answer"], "Summarize cobalt FT catalyst deactivation.")
        self.assertEqual(result.path, ["RouterNode", "CallableNode"])

    def test_router_raises_on_invalid_json(self) -> None:
        node = RouterNode(llm=FakeLLM("not json"))

        with self.assertRaises(RouterParseError):
            node.exec({"question": "hello"})


if __name__ == "__main__":
    unittest.main()
