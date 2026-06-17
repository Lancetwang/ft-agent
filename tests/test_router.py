import unittest

from ft_agent.core import CallableNode, Flow
from ft_agent.pipeline import RouterDecision, RouterNode, RouterParseError


class FakeLLM:
    def __init__(self, content: str | list[str]) -> None:
        self.contents = content if isinstance(content, list) else [content]
        self.calls = 0
        self.last_messages = None
        self.last_kwargs = None

    def chat(self, messages, **kwargs):
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        self.last_messages = messages
        self.last_kwargs = kwargs
        return content


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

    def test_router_accepts_payload_chat_kwargs(self) -> None:
        llm = FakeLLM(
            """
            {
              "is_relevant": false,
              "needs_clarification": false,
              "clarification_question": null,
              "deliverable_question": "Weather"
            }
            """
        )
        node = RouterNode(llm=llm)
        seen: list[str] = []
        on_delta = seen.append

        node.exec(
            {
                "question": "weather",
                "router_chat_kwargs": {
                    "stream": True,
                    "on_delta": on_delta,
                },
            }
        )

        self.assertTrue(llm.last_kwargs["stream"])
        self.assertIs(llm.last_kwargs["on_delta"], on_delta)

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

    def test_router_uses_clarification_response_for_next_decision(self) -> None:
        llm = FakeLLM(
            [
                """
                {
                  "is_relevant": true,
                  "needs_clarification": true,
                  "clarification_question": "Are you asking about cobalt or iron catalysts?",
                  "deliverable_question": "How can I improve FT catalyst performance?"
                }
                """,
                """
                {
                  "is_relevant": true,
                  "needs_clarification": false,
                  "clarification_question": null,
                  "deliverable_question": "How can cobalt FT catalyst stability be improved?"
                }
                """,
            ]
        )
        node = RouterNode(llm=llm)

        first_action, first_state = node.exec({"question": "How can I improve it?"})
        second_state = {
            **first_state,
            "clarification_response": "I mean cobalt catalyst stability.",
        }
        second_action, second_state = node.exec(second_state)

        self.assertEqual(first_action, "clarify")
        self.assertEqual(second_action, "ready")
        self.assertEqual(second_state["router_decision"].clarification_rounds, 1)
        self.assertIn(
            "User answered: I mean cobalt catalyst stability.",
            llm.last_messages[1]["content"],
        )
        self.assertEqual(
            second_state["router_decision"].deliverable_question,
            "How can cobalt FT catalyst stability be improved?",
        )
        self.assertNotIn("clarification_response", second_state)

    def test_router_stops_clarifying_after_max_rounds(self) -> None:
        node = RouterNode(
            llm=FakeLLM(
                """
                {
                  "is_relevant": true,
                  "needs_clarification": true,
                  "clarification_question": "Which catalyst property?",
                  "deliverable_question": "Improve FT catalyst performance."
                }
                """
            ),
            max_clarification_rounds=1,
        )
        state = {
            "question": "How can I improve it?",
            "router_context": {
                "original_question": "How can I improve it?",
                "clarification_turns": [
                    {
                        "question": "Which catalyst?",
                        "answer": "Cobalt FT catalyst.",
                    }
                ],
            },
        }

        action, state = node.exec(state)
        decision = state["router_decision"]

        self.assertEqual(action, "ready")
        self.assertTrue(decision.max_clarification_reached)
        self.assertFalse(decision.needs_clarification)
        self.assertIsNone(decision.clarification_question)

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
