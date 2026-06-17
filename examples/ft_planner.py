import argparse
import sys

from ft_agent import Agent
from ft_agent.core import CallableNode, Flow, TraceOptions, make_trace_options
from ft_agent.llm import DeepSeekLLM
from ft_agent.pipeline import FinalAnswerNode, PlannerNode, PlannerPlan, RouterDecision, RouterNode


def final_planned(payload: dict) -> dict:
    plan: PlannerPlan = payload["planner_plan"]
    payload["answer"] = f"Planner produced {len(plan.steps)} executable steps."
    return payload


def build_flow() -> Flow:
    router = RouterNode(llm=DeepSeekLLM())
    planner = PlannerNode(llm=DeepSeekLLM())
    final_node = FinalAnswerNode(llm=DeepSeekLLM())
    planned_node = CallableNode(final_planned)

    router - "irrelevant" >> final_node
    router - "clarify" >> final_node
    router - "ready" >> planner
    planner - "planned" >> planned_node

    return Flow(router)


def run_pipeline(payload: dict, agent: Agent, trace: TraceOptions | None = None, *, stream: bool = False) -> dict:
    result = agent.run(payload, trace=trace)
    if stream:
        print()
    print_result(result.payload)
    return result.payload


def print_result(payload: dict) -> None:
    decision: RouterDecision = payload["router_decision"]
    print(f"route: {decision.action}")
    print(f"deliverable_question: {decision.deliverable_question}")
    if decision.clarification_question:
        print(f"clarification: {decision.clarification_question}")
    if "planner_plan" in payload:
        print_plan(payload["planner_plan"])
    print(f"answer: {payload['answer']}")


def print_plan(plan: PlannerPlan) -> None:
    print(f"plan_summary: {plan.summary}")
    for step in plan.steps:
        depends_on = ", ".join(step.depends_on) if step.depends_on else "-"
        print(f"{step.id}. {step.capability} depends_on={depends_on}")
        print(f"   instruction: {step.instruction}")
        print(f"   expected_output: {step.expected_output}")


def run_conversation(
    agent: Agent,
    question: str,
    trace: TraceOptions | None = None,
    *,
    stream: bool = False,
) -> None:
    payload = {"question": question, **stream_payload(stream)}
    while True:
        payload = run_pipeline(payload, agent, trace=trace, stream=stream)
        decision: RouterDecision = payload["router_decision"]
        if decision.action != "clarify":
            return

        try:
            clarification_response = input("clarification> ").strip()
        except EOFError:
            print("no clarification response available")
            return
        if clarification_response.lower() in {"exit", "quit", "q"}:
            print("bye")
            return
        if not clarification_response:
            continue
        payload["clarification_response"] = clarification_response
        payload.update(stream_payload(stream))


def stream_payload(enabled: bool) -> dict:
    if not enabled:
        return {}
    return {
        "router_chat_kwargs": {
            "stream": True,
            "on_delta": make_stream_printer("router"),
        },
        "planner_chat_kwargs": {
            "stream": True,
            "on_delta": make_stream_printer("planner"),
        },
        "final_chat_kwargs": {
            "stream": True,
            "on_delta": make_stream_printer("final"),
        },
    }


def make_stream_printer(stage: str):
    started = {"value": False}

    def on_delta(delta: str) -> None:
        if not started["value"]:
            print(f"\n[{stage}:stream] ", end="", flush=True)
            started["value"] = True
        print(safe_text(delta), end="", flush=True)

    return on_delta


def safe_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def print_safe(text: str) -> None:
    print(safe_text(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ft-agent planner example.")
    parser.add_argument("question", nargs="*", help="Question to route and plan.")
    parser.add_argument("--trace", action="store_true", help="Print trace events.")
    parser.add_argument("--stream", action="store_true", help="Print raw LLM deltas for router and planner.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = make_trace_options(print_to_console=True, printer=print_safe) if args.trace else None
    agent = Agent(build_flow())

    if args.question:
        run_conversation(agent, " ".join(args.question), trace=trace, stream=args.stream)
        return

    print("ft-agent planner. Type 'exit' to quit.")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("bye")
            break
        if question:
            run_conversation(agent, question, trace=trace, stream=args.stream)


if __name__ == "__main__":
    main()
