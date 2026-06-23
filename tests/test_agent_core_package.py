import unittest

from agent_core import Agent, CallableNode, Flow, RunContext, Tool, get_current_context, tool
from ft_agent import Agent as FtAgent
from ft_agent.core import Flow as FtFlow
from ft_agent.core import RunContext as FtRunContext
from ft_agent.core import get_current_context as ft_get_current_context
from ft_agent.tools import Tool as FtTool


class AgentCorePackageTests(unittest.TestCase):
    def test_agent_core_can_be_used_directly(self) -> None:
        node = CallableNode(lambda payload: {"message": payload["message"].upper()})
        result = Agent(Flow(node)).run({"message": "ok"})

        self.assertEqual(result.payload["message"], "OK")

    def test_ft_agent_exports_runtime_compatibility(self) -> None:
        self.assertIs(FtAgent, Agent)
        self.assertIs(FtFlow, Flow)
        self.assertIs(FtTool, Tool)
        self.assertIs(FtRunContext, RunContext)
        self.assertIs(ft_get_current_context, get_current_context)

    def test_agent_core_exports_tool_decorator(self) -> None:
        @tool(description="Echo text.")
        def echo(text: str) -> str:
            return text

        self.assertEqual(echo.execute(text="hello"), "hello")
        self.assertEqual(echo.to_llm_format()["function"]["name"], "echo")


if __name__ == "__main__":
    unittest.main()
