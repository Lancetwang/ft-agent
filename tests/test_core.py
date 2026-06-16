import unittest

from ft_agent.core import CallableNode, Flow, FlowError, Node


class CoreFlowTests(unittest.TestCase):
    def test_action_routes_to_one_successor(self) -> None:
        def classify(payload: dict) -> tuple[str, dict]:
            return "question", payload

        def answer(payload: dict) -> dict:
            payload["reply"] = "ok"
            return payload

        start = CallableNode(classify)
        answer_node = CallableNode(answer)
        start - "question" >> answer_node

        result = Flow(start).run({"input": "hello?"})

        self.assertEqual(result.action, "default")
        self.assertEqual(result.payload["reply"], "ok")
        self.assertEqual(result.path, ["CallableNode", "CallableNode"])

    def test_retry(self) -> None:
        calls = {"count": 0}

        class FlakyNode(Node):
            def exec(self, payload):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise ValueError("try again")
                return "default", payload

        Flow(FlakyNode(max_retries=2)).run({})

        self.assertEqual(calls["count"], 2)

    def test_max_steps_guard(self) -> None:
        node = CallableNode(lambda payload: payload)
        node >> node

        with self.assertRaises(FlowError):
            Flow(node).run({}, max_steps=2)


if __name__ == "__main__":
    unittest.main()
