from __future__ import annotations

import json
import queue
import threading
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ft_agent import Agent
from ft_agent.core import Flow, TraceEvent, make_trace_options
from ft_agent.llm import DeepSeekLLM
from ft_agent.pipeline import (
    FinalAnswerNode,
    PlannerNode,
    RouterDecision,
    RouterNode,
    SupervisorNode,
    WriterNode,
)


STATIC_DIR = Path(__file__).with_name("static")
TOP_LEVEL_NODES = {
    "RouterNode",
    "PlannerNode",
    "WriterNode",
    "SupervisorNode",
    "FinalAnswerNode",
}


app = FastAPI(title="ft-agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    message = str(body.get("message", "")).strip()
    state = body.get("state") if isinstance(body.get("state"), Mapping) else {}

    if not message:
        return StreamingResponse(
            iter([_to_line({"type": "error", "message": "Message is required."})]),
            media_type="application/x-ndjson",
        )

    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def emit(event: dict[str, Any]) -> None:
        event_queue.put(event)

    def worker() -> None:
        try:
            payload = _build_payload(message, state, emit)
            agent = Agent(_build_flow())
            result = agent.run(
                payload,
                max_steps=24,
                trace=make_trace_options(include=["node", "tool"], on_event=_trace_emitter(emit)),
            )
            emit(
                {
                    "type": "final",
                    "answer": str(result.payload.get("answer", "")),
                    "state": _response_state(result.payload, message),
                    "path": result.path,
                }
            )
        except Exception as exc:
            emit({"type": "error", "message": str(exc)})
        finally:
            event_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            event = event_queue.get()
            if event is None:
                break
            yield _to_line(event)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _build_flow() -> Flow:
    router = RouterNode(llm=DeepSeekLLM())
    planner = PlannerNode(llm=DeepSeekLLM())
    writer = WriterNode(llm=DeepSeekLLM())
    supervisor = SupervisorNode(llm=DeepSeekLLM())
    final = FinalAnswerNode(llm=DeepSeekLLM())

    router - "irrelevant" >> final
    router - "clarify" >> final
    router - "ready" >> planner
    planner - "planned" >> writer
    writer - "written" >> supervisor
    supervisor - "revise" >> writer
    supervisor - "approved" >> final

    return Flow(router)


def _build_payload(
    message: str,
    state: Mapping[str, Any],
    emit: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": message,
        "final_chat_kwargs": {
            "stream": True,
            "on_delta": lambda delta: emit({"type": "answer_delta", "delta": delta}),
        },
    }

    if state.get("awaiting_clarification"):
        original_question = str(state.get("question") or message)
        payload["question"] = original_question
        payload["clarification_response"] = message
        if isinstance(state.get("router_context"), Mapping):
            payload["router_context"] = dict(state["router_context"])
        if isinstance(state.get("router_decision"), Mapping):
            payload["router_decision"] = dict(state["router_decision"])

    return payload


def _trace_emitter(emit: Any):
    def on_event(event: TraceEvent) -> None:
        if event.category == "node" and event.node in TOP_LEVEL_NODES:
            emit(
                {
                    "type": "node",
                    "event": event.event,
                    "node": event.node,
                    "step": event.step,
                    "action": event.action,
                    "next_node": event.data.get("next_node"),
                    "timestamp": event.timestamp,
                }
            )
            return

        if event.category == "tool":
            emit(
                {
                    "type": "activity",
                    "event": event.event,
                    "node": event.node,
                    "step": event.step,
                    "data": _safe_json(event.data),
                    "timestamp": event.timestamp,
                }
            )

    return on_event


def _response_state(payload: Mapping[str, Any], fallback_question: str) -> dict[str, Any]:
    decision = payload.get("router_decision")
    decision_data = _safe_json(decision)
    awaiting = False
    if isinstance(decision, RouterDecision):
        awaiting = decision.action == "clarify"
    elif isinstance(decision_data, Mapping):
        awaiting = bool(decision_data.get("needs_clarification"))

    return {
        "awaiting_clarification": awaiting,
        "question": payload.get("question") or fallback_question,
        "router_context": _safe_json(payload.get("router_context", {})),
        "router_decision": decision_data,
    }


def _safe_json(value: Any) -> Any:
    if is_dataclass(value):
        return _safe_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _to_line(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def main() -> None:
    uvicorn.run("ft_agent.web.app:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
