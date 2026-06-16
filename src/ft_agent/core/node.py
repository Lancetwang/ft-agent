from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


Context = dict[str, Any]


@dataclass(frozen=True)
class NodeResult:
    route: str = "default"
    output: Any = None
    updates: Mapping[str, Any] = field(default_factory=dict)


class Node(ABC):
    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("Node name cannot be empty.")
        self.name = name

    @abstractmethod
    def run(self, context: Context) -> NodeResult:
        raise NotImplementedError


class CallableNode(Node):
    def __init__(
        self,
        name: str,
        fn: Callable[[Context], NodeResult | Mapping[str, Any] | Any],
    ) -> None:
        super().__init__(name)
        self._fn = fn

    def run(self, context: Context) -> NodeResult:
        result = self._fn(context)
        if isinstance(result, NodeResult):
            return result
        if isinstance(result, Mapping):
            return NodeResult(updates=result)
        return NodeResult(output=result)
