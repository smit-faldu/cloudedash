"""
api/server.py
=============
FastAPI application for the CloudDash Multi-Agent Customer Support System.

Endpoints
---------
GET  /health         — Liveness probe; returns 200 + version info.
POST /chat           — Main chat endpoint. Runs the full pipeline:
                         input guardrail → LangGraph → output guardrail.
GET  /history        — Returns the conversation message history for a session.

Architecture
------------
                    ┌──────────────────────────────────────┐
 Browser / client → │  POST /chat                          │
                    │                                      │
                    │  1. check_input()   ← input guardrail│
                    │       ↓ PASS                         │
                    │  2. graph.invoke()  ← LangGraph       │
                    │       ↓                               │
                    │  3. check_output()  ← output guardrail│
                    │       ↓                               │
                    │  4. Return ChatResponse               │
                    └──────────────────────────────────────┘

Multi-turn conversations
------------------------
The LangGraph graph uses ``SqliteSaver`` as its checkpointer, keyed on
``session_id`` as the LangGraph ``thread_id``.  On follow-up requests the
client sends the same ``session_id`` and the graph automatically loads the
previous state and appends the new message.

CORS
----
CORS is configured to allow requests from ``localhost`` and ``127.0.0.1``
on any port (covers the browser opening ``frontend/index.html`` directly or
via a local dev server).
"""

from __future__ import annotations

import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages import AIMessage, HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.deps import get_app_config, get_compiled_graph, setup_logging
from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryResponse,
    MessageRecord,
)
from graph.graph import create_initial_state
from guardrails.input_guard import check_input
from guardrails.output_guard import check_output
from utils.logger import (
    get_logger,
    log_agent_end,
    log_agent_start,
    log_guardrail_triggered,
)

logger = get_logger(__name__)


# ===========================================================================
# Application lifespan (startup / shutdown)
# ===========================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: configure logging and warm up the graph singleton."""
    setup_logging("INFO")
    logger.info("CloudDash API starting up — warming graph singleton…")
    # Warm the compiled graph so the first request isn't slow
    try:
        get_compiled_graph()
        logger.info("Graph singleton ready.")
    except Exception as exc:
        logger.warning(
            "Graph pre-warm failed (will retry on first request): %s", exc
        )
    yield
    logger.info("CloudDash API shutting down.")


# ===========================================================================
# FastAPI application
# ===========================================================================

app = FastAPI(
    title="CloudDash Multi-Agent Support API",
    description=(
        "Production-grade multi-agent customer support system for CloudDash. "
        "Routes queries through Triage, Technical Support, Billing, and "
        "Escalation agents using LangGraph orchestration."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow all localhost/127.0.0.1 origins so the static frontend works
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        # Allow file:// origin by including a wildcard for development
        # (browsers send Origin: null for file:// pages)
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Helper: extract the last AI response from the graph state
# ===========================================================================


def _extract_response(final_state: dict[str, Any]) -> tuple[str, str]:
    """
    Walk the message list backwards and return (content, agent_name) for
    the last AIMessage that was produced by a specialist or escalation agent
    (i.e. not the triage classification message).

    Falls back to a generic error message if nothing is found.
    """
    messages = final_state.get("messages", [])
    skip_prefixes = ("[TRIAGE]",)  # internal triage annotation messages

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = getattr(msg, "content", "")
        if not content:
            continue
        # Skip the internal [TRIAGE] annotation messages
        if any(content.startswith(p) for p in skip_prefixes):
            continue
        agent_name = getattr(msg, "name", None) or "assistant"
        return content, agent_name

    return (
        "I'm sorry, I wasn't able to process your request. "
        "Please try again or contact support@clouddash.io.",
        "system",
    )


def _extract_source_doc_ids(final_state: dict[str, Any]) -> list[str]:
    """Extract source document IDs from the last_agent_response metadata."""
    last = final_state.get("last_agent_response") or {}
    return list(last.get("source_documents", []))


# ===========================================================================
# Endpoints
# ===========================================================================


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


@app.get(
    "/",
    response_class=FileResponse,
    tags=["Frontend"],
    summary="Serve the frontend application",
)
async def serve_frontend() -> FileResponse:
    """Serves the main static index.html frontend page."""
    frontend_path = _PROJECT_ROOT / "frontend" / "index.html"
    if not frontend_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index.html not found.",
        )
    return FileResponse(frontend_path)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness probe",
)
async def health() -> HealthResponse:
    """Returns 200 OK when the service is running."""
    return HealthResponse()


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Support"],
    summary="Send a support message",
    status_code=status.HTTP_200_OK,
)
async def chat(
    body: ChatRequest,
    graph=Depends(get_compiled_graph),
) -> ChatResponse:
    """
    Main chat endpoint.  Full pipeline:

    1. **Input guardrail** — rule-based checks (prompt injection, SQL
       injection, off-topic).  Returns a safe fallback immediately if triggered;
       the LangGraph graph is never invoked.

    2. **LangGraph invocation** — routes through Triage → specialist agents
       with full conversation history via ``SqliteSaver`` checkpointer.

    3. **Output guardrail** — validates the specialist's response against
       canonical pricing, refund policy, and KB citations.  Triggers
       escalation fallback if a hallucination is detected.

    4. **Returns** a ``ChatResponse`` with the agent reply, routing metadata,
       and the session/trace IDs for follow-up turns.
    """
    session_id = body.session_id or str(uuid.uuid4())
    trace_id = str(uuid.uuid4())
    t_start = time.monotonic()

    log_agent_start(
        "api_handler",
        trace_id=trace_id,
        session_id=session_id,
        user_message_preview=body.message[:120],
    )

    # ------------------------------------------------------------------
    # Step 1 — Input guardrail
    # ------------------------------------------------------------------
    input_result = check_input(body.message, trace_id=trace_id, session_id=session_id)
    if not input_result.passed:
        log_guardrail_triggered(
            "input_guard",
            trace_id=trace_id,
            flag_reason=input_result.flag_reason,
            user_message_preview=body.message[:120],
            session_id=session_id,
        )
        return ChatResponse(
            session_id=session_id,
            trace_id=trace_id,
            response=input_result.safe_reply,
            agent="guardrail",
            guardrail_triggered=True,
            guardrail_reason=input_result.flag_reason,
        )

    # ------------------------------------------------------------------
    # Step 2 — LangGraph invocation
    # ------------------------------------------------------------------
    try:
        initial_state = create_initial_state(
            user_message=body.message,
            session_id=session_id,
            customer_id=body.customer_id,
            trace_id=trace_id,
        )

        # LangGraph config: thread_id = session_id for checkpointer persistence
        config = {"configurable": {"thread_id": session_id}}

        final_state: dict[str, Any] = graph.invoke(initial_state, config=config)

    except Exception as exc:
        logger.error(
            "Graph invocation failed",
            extra={
                "trace_id": trace_id,
                "session_id": session_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent system error: {type(exc).__name__}. Please try again.",
        )

    # ------------------------------------------------------------------
    # Step 3 — Output guardrail
    # ------------------------------------------------------------------
    response_text, agent_name = _extract_response(final_state)
    source_doc_ids = _extract_source_doc_ids(final_state)

    output_result = check_output(
        agent_name=agent_name,
        response_content=response_text,
        trace_id=trace_id,
        retrieved_doc_ids=source_doc_ids,
        use_llm_judge=False,   # Keep fast; enable per-request in production
        session_id=session_id,
    )

    if not output_result.passed:
        log_guardrail_triggered(
            "output_guard",
            trace_id=trace_id,
            flag_reason=output_result.flag_reason,
            session_id=session_id,
        )
        # Return a safe escalation message instead of the flagged response
        response_text = output_result.safe_reply
        agent_name = "guardrail_escalation"

    # ------------------------------------------------------------------
    # Step 4 — Build response
    # ------------------------------------------------------------------
    duration_ms = round((time.monotonic() - t_start) * 1000, 1)
    log_agent_end(
        agent_name,
        trace_id=trace_id,
        duration_ms=duration_ms,
        needs_handover=final_state.get("is_escalated", False),
        session_id=session_id,
    )

    return ChatResponse(
        session_id=session_id,
        trace_id=trace_id,
        response=response_text,
        agent=agent_name,
        intent=final_state.get("intent"),
        confidence=final_state.get("confidence"),
        is_escalated=final_state.get("is_escalated", False),
        handover_count=final_state.get("handover_count", 0),
        guardrail_triggered=not output_result.passed,
        guardrail_reason=output_result.flag_reason if not output_result.passed else None,
    )


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------


@app.get(
    "/history",
    response_model=HistoryResponse,
    tags=["Support"],
    summary="Get conversation history",
)
async def history(
    session_id: str,
    graph=Depends(get_compiled_graph),
) -> HistoryResponse:
    """
    Retrieve the conversation history and metadata for a given session ID.

    Uses the LangGraph ``SqliteSaver`` checkpointer to load the persisted
    state for the provided ``session_id``.

    Returns an empty message list if the session is not found.
    """
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id query parameter is required.",
        )

    try:
        config = {"configurable": {"thread_id": session_id}}
        state_snapshot = graph.get_state(config)

        if state_snapshot is None or not state_snapshot.values:
            return HistoryResponse(session_id=session_id, messages=[])

        state: dict[str, Any] = state_snapshot.values
        raw_messages = state.get("messages", [])

        # Convert LangChain messages to the API's MessageRecord format
        records: list[MessageRecord] = []
        for msg in raw_messages:
            content = getattr(msg, "content", "")
            if not content:
                continue
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = getattr(msg, "name", None) or "assistant"
            else:
                role = type(msg).__name__.lower()

            records.append(MessageRecord(role=role, content=content))

        return HistoryResponse(
            session_id=session_id,
            messages=records,
            current_agent=state.get("current_agent"),
            intent=state.get("intent"),
            confidence=state.get("confidence"),
            is_escalated=state.get("is_escalated", False),
            handover_count=state.get("handover_count", 0),
            customer_id=state.get("customer_id"),
        )

    except Exception as exc:
        logger.error(
            "History retrieval failed",
            extra={
                "session_id": session_id,
                "error": str(exc),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve session history.",
        )


# ===========================================================================
# Global exception handler — return JSON instead of HTML for all errors
# ===========================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception",
        extra={"path": request.url.path, "error": str(exc)},
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=True)