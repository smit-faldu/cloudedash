"""
api/schemas.py
==============
Pydantic request/response models for the CloudDash REST API.

These are the *external* API contract.  They are intentionally separate from
the internal models in ``models/models.py`` so the API surface can evolve
independently of the graph internals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Body for POST /chat"""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4_000,
        description="The user's support message.",
        examples=["I'm getting ERR-4012 when connecting my AWS account."],
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Stable identifier for the conversation session.  "
            "Pass the same session_id on follow-up turns to preserve history.  "
            "Auto-generated on first call if omitted."
        ),
    )
    customer_id: str | None = Field(
        default=None,
        description="Pre-authenticated customer ID (e.g. CLD-00001). Optional.",
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class ChatResponse(BaseModel):
    """Response envelope for POST /chat"""

    session_id: str = Field(description="Session ID (use this on follow-up turns).")
    trace_id: str = Field(description="Unique trace ID for this graph run.")
    response: str = Field(description="The agent's natural-language reply.")
    agent: str = Field(description="Which agent produced the final reply.")
    intent: str | None = Field(default=None, description="Triage intent label.")
    confidence: float | None = Field(
        default=None, description="Triage confidence (0–1)."
    )
    is_escalated: bool = Field(
        default=False,
        description="True if the conversation was routed to the Escalation Agent.",
    )
    handover_count: int = Field(
        default=0, description="Number of inter-agent handovers during this run."
    )
    # Set when the input/output guardrail fires instead of the graph running
    guardrail_triggered: bool = Field(default=False)
    guardrail_reason: str | None = Field(default=None)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    )


class MessageRecord(BaseModel):
    """A single turn in the conversation history."""

    role: str = Field(description="'user' | 'assistant' | 'triage_agent' | …")
    content: str
    timestamp: str | None = None


class HistoryResponse(BaseModel):
    """Response envelope for GET /history"""

    session_id: str
    messages: list[MessageRecord]
    current_agent: str | None = None
    intent: str | None = None
    confidence: float | None = None
    is_escalated: bool = False
    handover_count: int = 0
    customer_id: str | None = None


class HealthResponse(BaseModel):
    """Response for GET /health"""

    status: str = "ok"
    version: str = "1.0.0"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    )
