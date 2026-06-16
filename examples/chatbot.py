import sys

from ft_agent import Agent
from ft_agent.core import CallableNode, Flow
from ft_agent.llm import DeepSeekLLM, LLMNode


SYSTEM_PROMPT = "You are a concise, helpful chatbot."


def route_input(payload: dict) -> tuple[str, dict]:
    user_input = payload["input"].strip()
    payload["input"] = user_input
    if user_input.lower() in {"exit", "quit", "q"}:
        return "exit", payload
    return "chat", payload


def build_messages(payload: dict) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(payload.get("history", []))
    messages.append({"role": "user", "content": payload["input"]})
    return messages


def remember_turn(payload: dict) -> dict:
    history = payload.setdefault("history", [])
    history.append({"role": "user", "content": payload["input"]})
    history.append({"role": "assistant", "content": payload["answer"]})
    return payload


def build_chat_flow() -> Flow:
    route_node = CallableNode(route_input)
    llm_node = LLMNode(
        llm=DeepSeekLLM(),
        messages=build_messages,
        chat_kwargs={"stream": True, "temperature": 0},
    )
    remember_node = CallableNode(remember_turn)

    route_node - "chat" >> llm_node >> remember_node
    return Flow(route_node)


def run_once(agent: Agent, history: list[dict[str, str]], user_input: str) -> bool:
    result = agent.run(
        {
            "input": user_input,
            "history": history,
            "chat_kwargs": {"on_delta": lambda delta: print(delta, end="", flush=True)},
        }
    )
    payload = result.payload
    if result.action == "exit":
        return False
    print()
    return True


def main() -> None:
    agent = Agent(build_chat_flow())
    history: list[dict[str, str]] = []

    if len(sys.argv) > 1:
        run_once(agent, history, " ".join(sys.argv[1:]))
        return

    print("ft-agent chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("> ")
        if not run_once(agent, history, user_input):
            print("bye")
            break


if __name__ == "__main__":
    main()
