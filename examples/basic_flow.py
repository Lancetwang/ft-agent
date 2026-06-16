from ft_agent import Agent
from ft_agent.core import CallableNode, Flow, NodeResult


def classify(context: dict) -> NodeResult:
    text = context["input"]
    route = "question" if text.endswith("?") else "statement"
    return NodeResult(route=route, updates={"kind": route})


def answer_question(context: dict) -> dict:
    return {"reply": f"Question received: {context['input']}"}


def answer_statement(context: dict) -> dict:
    return {"reply": f"Statement received: {context['input']}"}


flow = Flow(
    nodes=[
        CallableNode("classify", classify),
        CallableNode("answer_question", answer_question),
        CallableNode("answer_statement", answer_statement),
    ],
    start="classify",
    transitions={
        "classify": {
            "question": "answer_question",
            "statement": "answer_statement",
        },
        "answer_question": {"default": None},
        "answer_statement": {"default": None},
    },
)

result = Agent(flow).run({"input": "Hello?"})
print(result.context["reply"])
print(result.path)
