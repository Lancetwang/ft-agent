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


def run_router(payload: dict, agent: Agent, trace: TraceOptions | None = None) -> dict:
    result = agent.run(payload, trace=trace)
    print_result(result.payload)
    return result.payload


def print_result(payload: dict) -> None:
    decision: RouterDecision = payload["router_decision"]
    print(f"route: {decision.action}")
    print(f"relevant: {decision.is_relevant}")
    print(f"needs_clarification: {decision.needs_clarification}")
    print(f"clarification_rounds: {decision.clarification_rounds}/{decision.max_clarification_rounds}")
    if decision.max_clarification_reached:
        print("max_clarification_reached: True")
    if decision.clarification_question:
        print(f"clarification: {decision.clarification_question}")
    print(f"deliverable_question: {decision.deliverable_question}")
    print(f"answer: {payload['answer']}")


def run_conversation(agent: Agent, question: str, trace: TraceOptions | None = None) -> None:
    payload = {"question": question}
    while True:
        payload = run_router(payload, agent, trace=trace)
        decision: RouterDecision = payload["router_decision"]
        if decision.action != "clarify":
            return

        clarification_response = input("clarification> ").strip()
        if clarification_response.lower() in {"exit", "quit", "q"}:
            print("bye")
            return
        if not clarification_response:
            continue
        payload["clarification_response"] = clarification_response


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
        run_conversation(agent, " ".join(args.question), trace=trace)
        return

    print("ft-agent router. Type 'exit' to quit.")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("bye")
            break
        if question:
            run_conversation(agent, question, trace=trace)


if __name__ == "__main__":
    main()
