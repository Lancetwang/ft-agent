from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent_core import TraceEvent, make_trace_options
from ft_agent.pipeline import RouterDecision, build_ft_agent, to_jsonable

STATIC_DIR = Path(__file__).with_name("static")
REPORT_WORKSPACE = Path("artifacts").resolve()
TOP_LEVEL_NODES = {
    "RouterAgent",
    "PlannerAgent",
    "WriterAgent",
    "SupervisorAgent",
    "FinalAgent",
}

app = FastAPI(title="ft-agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/report")
def report(path: str = Query(..., min_length=1)) -> JSONResponse:
    resolved = _resolve_report_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")
    return JSONResponse({"path": path, "content": resolved.read_text(encoding="utf-8")})


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    message = str(body.get("message", "")).strip()
    state = body.get("state") if isinstance(body.get("state"), Mapping) else {}
    if not message:
        return StreamingResponse(
            iter([_line({"type": "error", "message": "Message is required."})]),
            media_type="application/x-ndjson",
        )

    events: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def emit(event: dict[str, Any]) -> None:
        events.put(event)

    def worker() -> None:
        started = time.perf_counter()
        try:
            payload = _payload(message, state)
            result = build_ft_agent(report_workspace=REPORT_WORKSPACE).run(
                payload,
                max_steps=24,
                trace=make_trace_options(
                    include=["node", "tool", "model", "plan"],
                    on_event=_trace_emitter(emit),
                ),
            )
            answer = str(result.payload.get("answer", ""))
            for chunk in _chunks(answer):
                emit({"type": "answer_delta", "delta": chunk})
                time.sleep(0.006)
            emit(
                {
                    "type": "final",
                    "answer": answer,
                    "report_path": result.payload.get("writer_report_path"),
                    "state": _response_state(result.payload, message),
                    "path": result.path,
                    "metrics": {"elapsed_ms": round((time.perf_counter() - started) * 1000, 2)},
                }
            )
        except Exception as exc:
            emit(
                {
                    "type": "error",
                    "message": str(exc),
                    "metrics": {"elapsed_ms": round((time.perf_counter() - started) * 1000, 2)},
                }
            )
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is None:
                break
            yield _line(event)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _payload(message: str, state: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"question": message}
    if state.get("awaiting_clarification"):
        payload["question"] = str(state.get("question") or message)
        payload["clarification_response"] = message
    return payload


def _response_state(payload: Mapping[str, Any], fallback_question: str) -> dict[str, Any]:
    decision = payload.get("router_decision")
    awaiting = isinstance(decision, RouterDecision) and decision.action == "clarify"
    return {
        "awaiting_clarification": awaiting,
        "question": payload.get("question") or fallback_question,
        "router_decision": to_jsonable(decision),
    }


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
        elif event.category in {"tool", "model", "plan"}:
            emit(
                {
                    "type": "activity",
                    "event": event.event,
                    "node": event.node,
                    "step": event.step,
                    "category": event.category,
                    "data": to_jsonable(event.data),
                    "timestamp": event.timestamp,
                }
            )

    return on_event


def _line(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def _chunks(text: str, size: int = 28) -> list[str]:
    chunks: list[str] = []
    current = ""
    for char in text:
        current += char
        if len(current) >= size or char in "\n.!?。！？":
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _resolve_report_path(path: str) -> Path:
    candidate = (REPORT_WORKSPACE / path).resolve()
    if REPORT_WORKSPACE != candidate and REPORT_WORKSPACE not in candidate.parents:
        raise HTTPException(status_code=400, detail="Report path escapes workspace.")
    return candidate


def main() -> None:
    uvicorn.run("ft_agent.web.app:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
