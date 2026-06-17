import argparse

from ft_agent import Agent
from ft_agent.core import CallableNode, Flow, TraceOptions, make_trace_options
from ft_agent.llm import DeepSeekLLM
from ft_agent.pipeline import RouterDecision, RouterNode


def final_irrelevant(payload: dict) -> dict:
    payload["answer"] = (
        "This question is not strongly related to Fischer-Tropsch catalysts, "
        "so I will not route it into the FT catalyst pipeline."
    )
    return payload


def final_clarify(payload: dict) -> dict:
    decision: RouterDecision = payload["router_decision"]
    payload["answer"] = decision.clarification_question or (
        "Please add the missing catalyst, reaction, or target-performance context."
    )
    return payload


def final_ready(payload: dict) -> dict:
    decision: RouterDecision = payload["router_decision"]
    payload["answer"] = f"Ready for planner: {decision.deliverable_question}"
    return payload


def build_flow() -> Flow:
    router = RouterNode(llm=DeepSeekLLM())
    irrelevant_node = CallableNode(final_irrelevant)
    clarify_node = CallableNode(final_clarify)
    ready_node = CallableNode(final_ready)

    router - "irrelevant" >> irrelevant_node
    router - "clarify" >> clarify_node
    router - "ready" >> ready_node

    return Flow(router)


def run_once(agent: Agent, question: str, trace: TraceOptions | None = None) -> None:
    result = agent.run({"question": question}, trace=trace)
    decision: RouterDecision = result.payload["router_decision"]
    print(f"route: {decision.action}")
    print(f"relevant: {decision.is_relevant}")
    print(f"needs_clarification: {decision.needs_clarification}")
    if decision.clarification_question:
        print(f"clarification: {decision.clarification_question}")
    print(f"deliverable_question: {decision.deliverable_question}")
    print(f"answer: {result.payload['answer']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ft-agent router example.")
    parser.add_argument("question", nargs="*", help="Question to route.")
    parser.add_argument("--trace", action="store_true", help="Print trace events.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = make_trace_options(print_to_console=True) if args.trace else None
    agent = Agent(build_flow())

    if args.question:
        run_once(agent, " ".join(args.question), trace=trace)
        return

    print("ft-agent router. Type 'exit' to quit.")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("bye")
            break
        if question:
            run_once(agent, question, trace=trace)


if __name__ == "__main__":
    main()
