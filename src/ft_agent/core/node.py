from __future__ import annotations

from collections.abc import Callable
from typing import Any
import time

Action = str
Payload = Any
ExecResult = tuple[Action, Payload]


class Node:
    def __init__(self, *, max_retries: int = 1, wait: float = 0) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        self.successors: dict[Action, Node] = {}
        self._action: Action = "default"
        self.max_retries = max_retries
        self.wait = wait

    def exec(self, payload: Payload) -> ExecResult:
        raise NotImplementedError

    def _exec(self, payload: Payload) -> ExecResult:
        for attempt in range(self.max_retries):
            try:
                return self.exec(payload)
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                if self.wait > 0:
                    time.sleep(self.wait)
        raise RuntimeError("Unexpected error in Node._exec")

    def __rshift__(self, other: Node) -> Node:
        self.successors[self._action] = other
        self._action = "default"
        return other

    def __sub__(self, action: Action) -> Node:
        if not isinstance(action, str):
            raise TypeError("action must be a string.")
        self._action = action or "default"
        return self


class CallableNode(Node):
    def __init__(
        self,
        fn: Callable[[Payload], ExecResult | Payload],
        *,
        max_retries: int = 1,
        wait: float = 0,
    ) -> None:
        super().__init__(max_retries=max_retries, wait=wait)
        self.fn = fn

    def exec(self, payload: Payload) -> ExecResult:
        result = self.fn(payload)
        if self._is_exec_result(result):
            return result
        return "default", result

    @staticmethod
    def _is_exec_result(value: Any) -> bool:
        return (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], str)
        )
