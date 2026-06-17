import unittest

from ft_agent.core import Flow
from ft_agent.pipeline import (
    PlannerNode,
    PlannerParseError,
    PlannerPlan,
    RouterDecision,
    WriterCapability,
)


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_messages = None
        self.last_kwargs = None

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return self.content


class PlannerTests(unittest.TestCase):
    def test_planner_builds_plan_from_router_decision(self) -> None:
        llm = FakeLLM(
            """
            {
              "deliverable_question": "How does cobalt particle size affect methane selectivity?",
              "summary": "Retrieve evidence, draft, find a template, then write.",
              "steps": [
                {
                  "id": "s1",
                  "capability": "search_science_knowledge_base",
                  "instruction": "Search literature on cobalt particle size and methane selectivity.",
                  "expected_output": "Relevant findings and citations.",
                  "depends_on": []
                },
                {
                  "id": "s2",
                  "capability": "draft_from_evidence",
                  "instruction": "Draft the mechanism explanation from the evidence.",
                  "expected_output": "Evidence-grounded draft.",
                  "depends_on": ["s1"]
                }
              ]
            }
            """
        )
        node = PlannerNode(llm=llm)
        decision = RouterDecision(
            is_relevant=True,
            needs_clarification=False,
            deliverable_question="How does cobalt particle size affect methane selectivity?",
        )

        action, state = node.exec({"router_decision": decision})
        plan = state["planner_plan"]

        self.assertEqual(action, "planned")
        self.assertIsInstance(plan, PlannerPlan)
        self.assertEqual(plan.steps[0].capability, "search_science_knowledge_base")
        self.assertEqual(plan.steps[1].depends_on, ["s1"])
        self.assertEqual(llm.last_kwargs["temperature"], 0)
        self.assertIn("Writer capabilities", llm.last_messages[1]["content"])

    def test_planner_rejects_unknown_capability(self) -> None:
        node = PlannerNode(
            llm=FakeLLM(
                """
                {
                  "deliverable_question": "Question",
                  "summary": "Bad plan.",
                  "steps": [
                    {
                      "id": "s1",
                      "capability": "call_hidden_tool",
                      "instruction": "Do something unavailable.",
                      "expected_output": "Something.",
                      "depends_on": []
                    }
                  ]
                }
                """
            )
        )

        with self.assertRaises(PlannerParseError):
            node.exec({"question": "Question"})

    def test_planner_rejects_unknown_dependency(self) -> None:
        node = PlannerNode(
            llm=FakeLLM(
                """
                {
                  "deliverable_question": "Question",
                  "summary": "Bad dependency.",
                  "steps": [
                    {
                      "id": "s1",
                      "capability": "search_science_knowledge_base",
                      "instruction": "Search.",
                      "expected_output": "Evidence.",
                      "depends_on": ["missing"]
                    }
                  ]
                }
                """
            )
        )

        with self.assertRaises(PlannerParseError):
            node.exec({"question": "Question"})

    def test_planner_rejects_future_dependency(self) -> None:
        node = PlannerNode(
            llm=FakeLLM(
                """
                {
                  "deliverable_question": "Question",
                  "summary": "Future dependency.",
                  "steps": [
                    {
                      "id": "s1",
                      "capability": "draft_from_evidence",
                      "instruction": "Draft from evidence.",
                      "expected_output": "Draft.",
                      "depends_on": ["s2"]
                    },
                    {
                      "id": "s2",
                      "capability": "search_science_knowledge_base",
                      "instruction": "Search evidence.",
                      "expected_output": "Evidence.",
                      "depends_on": []
                    }
                  ]
                }
                """
            )
        )

        with self.assertRaises(PlannerParseError):
            node.exec({"question": "Question"})

    def test_planner_uses_custom_capabilities(self) -> None:
        capability = WriterCapability(
            name="search_science_knowledge_base",
            description="Search science.",
        )
        node = PlannerNode(
            llm=FakeLLM(
                """
                {
                  "deliverable_question": "Question",
                  "summary": "Search only.",
                  "steps": [
                    {
                      "id": "s1",
                      "capability": "search_science_knowledge_base",
                      "instruction": "Search relevant evidence.",
                      "expected_output": "Evidence.",
                      "depends_on": []
                    }
                  ]
                }
                """
            ),
            capabilities=[capability],
        )

        result = Flow(node).run({"question": "Question"})

        self.assertEqual(result.action, "planned")
        self.assertEqual(result.payload["planner_plan"].steps[0].capability, capability.name)


if __name__ == "__main__":
    unittest.main()
