import unittest

from ft_agent.core import CallableNode, Flow, FlowError, Node, make_trace_options
from ft_agent.core.trace import TRACE_KEY


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

    def test_flow_collects_trace_events(self) -> None:
        node = CallableNode(lambda payload: payload)

        result = Flow(node).run({}, trace=True)

        self.assertEqual(result.path, ["CallableNode"])
        self.assertNotIn(TRACE_KEY, result.payload)
        self.assertEqual(
            [event.event for event in result.trace],
            ["node.start", "node.end", "flow.end"],
        )
        self.assertEqual(result.trace[0].step, 1)
        self.assertEqual(result.trace[1].action, "default")

    def test_flow_trace_can_filter_categories(self) -> None:
        node = CallableNode(lambda payload: payload)
        trace = make_trace_options(include=["flow"])

        result = Flow(node).run({}, trace=trace)

        self.assertEqual([event.category for event in result.trace], ["flow"])

    def test_flow_trace_does_not_leak_into_original_payload(self) -> None:
        def copy_payload(payload: dict) -> dict:
            return {"ok": payload.get("ok")}

        payload = {"ok": True}

        Flow(CallableNode(copy_payload)).run(payload, trace=True)

        self.assertNotIn(TRACE_KEY, payload)


if __name__ == "__main__":
    unittest.main()
