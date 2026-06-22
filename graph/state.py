"""
graph/state.py
==============
LangGraph-compatible GraphState TypedDict for the CloudDash Multi-Agent System.

Why a separate TypedDict?
-------------------------
LangGraph requires state to be a TypedDict (or a dataclass). Our Stage 1
``ConversationState`` is a Pydantic model — excellent for validation but
not directly usable as a LangGraph state schema. This module defines
``GraphState`` as the graph's state contract, mirroring every field of
``ConversationState`` while adding reducers via ``Annotated``.

Key reducer choices
-------------------
messages
    Uses the built-in ``add_messages`` reducer so that each node can return
    a *list of new messages* and they are **appended** to the existing list
    rather than overwriting it. This is the standard LangGraph pattern for
    chat histories and ensures no message is ever lost during handovers.

entities
    Uses a custom ``merge_entities`` reducer that does a shallow dict merge,
    so the Triage Agent can update a single field (e.g. ``customer_id``) without
    wiping fields that were extracted on earlier turns.

All other fields use LangGraph's default last-write-wins semantics (no reducer),
meaning a node only needs to return the fields it wants to change.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Custom reducer: shallow-merge entity dicts across turns
# ---------------------------------------------------------------------------

def merge_entities(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """
    Merge entity dicts, preferring *right* values for non-None entries.

    This allows the Triage Agent to update ``customer_id`` on a follow-up turn
    without erasing ``error_code`` that was extracted on the first turn.

    Examples
    --------
    >>> merge_entities({"customer_id": "CLD-00001", "error_code": "ERR-4012"},
    ...                {"customer_id": None, "product_area": "alerting"})
    {'customer_id': 'CLD-00001', 'error_code': 'ERR-4012', 'product_area': 'alerting'}
    """
    merged = dict(left)
    for k, v in right.items():
        # Only overwrite with a non-None value so partial updates don't erase data
        if v is not None:
            merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# GraphState definition
# ---------------------------------------------------------------------------

# NOTE: TypedDict must be defined at module level for LangGraph to introspect it.
# We cannot use a dataclass or Pydantic model directly here — LangGraph enforces
# TypedDict for state schemas.
from typing import TypedDict  # noqa: E402  (kept below for visual grouping)


class GraphState(TypedDict, total=False):
    """
    The shared state object that flows through every node in the LangGraph graph.

    Fields
    ------
    messages
        Full conversation history as LangChain ``BaseMessage`` objects.
        The ``add_messages`` reducer ensures messages are always **appended**.

    current_agent
        The name of the agent node currently handling the session.
        Overwritten each time control passes to a new agent.

    customer_id
        Shorthand accessor for the most recently confirmed customer ID.
        Mirrors ``entities["customer_id"]`` for convenience in routing logic.

    trace_id
        Unique session trace identifier. Set once on session creation,
        threaded through every agent call and log line for end-to-end tracing.

    session_id
        Stable identifier for the customer's support session (distinct from
        trace_id, which changes per graph run in some designs).

    intent
        Most recent intent classification from the Triage Agent.

    confidence
        Triage confidence score. Used by the router to trigger auto-escalation
        when below the configured threshold.

    entities
        Dict of extracted entities (customer_id, error_code, product_area,
        urgency). The ``merge_entities`` reducer accumulates values across turns.

    handover_count
        Monotonically increasing counter of inter-agent handovers.
        Guards against infinite routing loops.

    is_escalated
        True once the Escalation Agent has produced a handover package.
        Prevents re-routing after escalation.

    last_agent_response
        The raw ``AgentResponse`` dict from the most recently executed
        specialist node. The router reads this to decide whether to hand over.

    error
        Set by a node on unrecoverable failure. The router sends the session
        to the Escalation Agent when this field is non-empty.
    """

    # --- Core conversation data ---
    messages: Annotated[list, add_messages]
    current_agent: str
    customer_id: str | None
    trace_id: str
    session_id: str

    # --- Triage outputs ---
    intent: str | None
    confidence: float | None

    # --- Entities (merge reducer keeps best-known value per field) ---
    entities: Annotated[dict[str, Any], merge_entities]

    # --- Routing metadata ---
    handover_count: int
    is_escalated: bool
    last_agent_response: dict[str, Any] | None

    # --- Error recovery ---
    error: str | None
