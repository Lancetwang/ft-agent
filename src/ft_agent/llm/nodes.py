from collections.abc import Callable, Sequence

from ft_agent.core.node import Context, Node, NodeResult
from ft_agent.llm.deepseek import DeepSeekLLM, Message


class LLMNode(Node):
    def __init__(
        self,
        name: str,
        *,
        llm: DeepSeekLLM,
        messages: Callable[[Context], Sequence[Message]],
        output_key: str = "answer",
        route: str = "default",
    ) -> None:
        super().__init__(name)
        self.llm = llm
        self.messages = messages
        self.output_key = output_key
        self.route = route

    def run(self, context: Context) -> NodeResult:
        content = self.llm.chat(self.messages(context), temperature=0)
        return NodeResult(route=self.route, output=content, updates={self.output_key: content})
