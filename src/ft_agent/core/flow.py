from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ft_agent.core.node import Context, Node, NodeResult


class FlowError(RuntimeError):
    pass


@dataclass(frozen=True)
class FlowRunResult:
    context: Context
    path: list[str]
    outputs: dict[str, Any]


class Flow:
    def __init__(
        self,
        *,
        nodes: Sequence[Node],
        start: str,
        transitions: Mapping[str, Mapping[str, str | None]] | None = None,
    ) -> None:
        self.nodes = self._index_nodes(nodes)
        self.start = start
        self.transitions = dict(transitions or {})
        self._validate()

    def run(self, context: Context | None = None, *, max_steps: int = 100) -> FlowRunResult:
        current = self.start
        state: Context = dict(context or {})
        path: list[str] = []
        outputs: dict[str, Any] = {}

        for _ in range(max_steps):
            node = self.nodes[current]
            result = node.run(state)
            self._apply_result(state, node.name, result, outputs)
            path.append(node.name)

            next_node = self._next_node(node.name, result.route)
            if next_node is None:
                return FlowRunResult(context=state, path=path, outputs=outputs)
            current = next_node

        raise FlowError(f"Flow exceeded max_steps={max_steps}.")

    @staticmethod
    def _index_nodes(nodes: Sequence[Node]) -> dict[str, Node]:
        indexed: dict[str, Node] = {}
        for node in nodes:
            if node.name in indexed:
                raise FlowError(f"Duplicate node name: {node.name}")
            indexed[node.name] = node
        return indexed

    def _validate(self) -> None:
        if self.start not in self.nodes:
            raise FlowError(f"Start node is not registered: {self.start}")

        for node_name, routes in self.transitions.items():
            if node_name not in self.nodes:
                raise FlowError(f"Transition references unknown node: {node_name}")
            for route, next_node in routes.items():
                if not route:
                    raise FlowError(f"Empty route on node: {node_name}")
                if next_node is not None and next_node not in self.nodes:
                    raise FlowError(
                        f"Transition {node_name}.{route} points to unknown node: {next_node}"
                    )

    @staticmethod
    def _apply_result(
        context: Context,
        node_name: str,
        result: NodeResult,
        outputs: dict[str, Any],
    ) -> None:
        if result.updates:
            context.update(result.updates)
        if result.output is not None:
            outputs[node_name] = result.output

    def _next_node(self, node_name: str, route: str) -> str | None:
        routes = self.transitions.get(node_name)
        if routes is None:
            return None
        if route not in routes:
            raise FlowError(f"Node {node_name} returned unknown route: {route}")
        return routes[route]
