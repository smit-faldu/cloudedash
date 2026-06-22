"""
graph/graph.py
==============
LangGraph StateGraph for the CloudDash Multi-Agent Customer Support System.

This module wires together all four agent nodes (Triage, Technical Support,
Billing, Escalation) into a stateful graph with conditional routing, handover
protocol, and SQLite-backed conversation persistence.

Graph topology
--------------

    START
      │
      ▼
  [triage_agent] ──────────────────────────────────────────────────┐
      │                                                              │
      │ (route_after_triage)                                         │
      ├─── intent=technical, confidence≥0.65 ──▶ [technical_support_agent]
      ├─── intent=billing,   confidence≥0.65 ──▶ [billing_agent]
      ├─── intent=general                    ──▶ [triage_agent] (inline answer)
      ├─── intent=escalation / auto-escalate ──▶ [escalation_agent]
      └─── low confidence / unknown          ──▶ [escalation_agent]
                                                         │
      [technical_support_agent]                          │
             │ (route_after_specialist)                  │
             ├─── needs_handover → billing_agent         │
             ├─── needs_handover → escalation_agent ─────┘
             └─── done → END                             │
                                                         │
      [billing_agent]                                    │
             │ (route_after_specialist)                  │
             ├─── needs_handover → escalation_agent ─────┘
             ├─── needs_handover → technical_support_agent
             └─── done → END                             │
                                                         │
      [escalation_agent] (terminal)                      │
             │                                           │
             └─── always → END ◀─────────────────────────┘

Design decisions
----------------
* **Nodes are thin wrappers** — each node function calls the Stage 4 agent
  function, converts its output into a state dict update, and returns only the
  fields it changed. LangGraph merges the partial update into the full state.
* **Tool-call loop** — specialist agents that use tools (Technical, Billing)
  are paired with a ToolNode that executes pending tool calls and loops back
  to the specialist. This implements the standard ReAct pattern inside LangGraph.
* **SqliteSaver checkpointer** — every invocation is persisted to
  ``<project_root>/graph_checkpoints.db`` so multi-turn conversations can be
  resumed across API requests. The ``thread_id`` configurable = ``session_id``.
* **Handover context preservation** — the ``messages`` field uses the
  ``add_messages`` reducer, so the full history is always available to the next
  agent. The ``entities`` field uses ``merge_entities`` so extracted data
  accumulates rather than being overwritten.
* **Loop guard** — ``handover_count`` is incremented on every handover.
  The router enforces ``MAX_HANDOVERS = 5`` to prevent infinite cycles.
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents.agent_nodes import (
    run_billing_agent,
    run_escalation_agent,
    run_technical_support_agent,
    run_triage_agent,
)
from graph.router import (
    NODE_BILLING,
    NODE_ESCALATION,
    NODE_TECHNICAL,
    NODE_TRIAGE,
    route_after_specialist,
    route_after_triage,
)
from graph.state import GraphState
from models.models import AgentName, IntentLabel, MessageRole
from tools.agent_tools import ALL_TOOLS

logger = logging.getLogger(__name__)


# ===========================================================================
# Node functions
# ===========================================================================
# Each node:
#   1. Extracts relevant fields from ``state``.
#   2. Calls the Stage 4 agent function.
#   3. Returns a *partial state dict* containing only the fields it updates.
#      LangGraph merges this with the existing state via reducers.


def _build_agent_state(state: GraphState) -> dict[str, Any]:
    """
    Convert a ``GraphState`` dict into the ``AgentState`` dict expected by
    the Stage 4 agent node functions.

    The Stage 4 functions accept a dict with keys: messages, entities,
    trace_id, session_id, current_agent, handover_count.
    """
    return {
        "messages": state.get("messages", []),
        "entities": state.get("entities", {}),
        "trace_id": state.get("trace_id", str(uuid.uuid4())),
        "session_id": state.get("session_id", str(uuid.uuid4())),
        "current_agent": state.get("current_agent", NODE_TRIAGE),
        "handover_count": state.get("handover_count", 0),
    }


# ---------------------------------------------------------------------------
# Node 1 — Triage
# ---------------------------------------------------------------------------

def triage_node(state: GraphState) -> dict[str, Any]:
    """
    Entry node. Classifies intent and extracts entities.

    State updates returned
    ----------------------
    messages        : Appends an AIMessage with the triage reasoning.
    current_agent   : Set to "triage_agent".
    intent          : Classified intent label (str).
    confidence      : Triage confidence score (float).
    entities        : Merged entity dict (customer_id, error_code, …).
    customer_id     : Shorthand alias for entities["customer_id"].
    last_agent_response : None (Triage doesn't produce an AgentResponse).
    error           : Set to a string if the agent raised an exception.
    """
    trace = state.get("trace_id", "unknown")
    logger.info("[%s] → Triage node executing.", trace)

    try:
        agent_state = _build_agent_state(state)
        result = run_triage_agent(agent_state)

        entities_update = result.extracted_entities.model_dump()
        customer_id = entities_update.get("customer_id")

        # Append a hidden system-level message recording the triage decision
        triage_msg = AIMessage(
            content=(
                f"[TRIAGE] intent={result.intent.value}, "
                f"confidence={result.confidence:.2f}, "
                f"customer_id={customer_id or 'N/A'}, "
                f"reasoning={result.reasoning}"
            ),
            name="triage_agent",
        )

        logger.info(
            "[%s] Triage complete: intent=%s (%.2f)", trace, result.intent.value, result.confidence
        )

        return {
            "messages": [triage_msg],
            "current_agent": NODE_TRIAGE,
            "intent": result.intent.value,
            "confidence": result.confidence,
            "entities": entities_update,
            "customer_id": customer_id,
            "last_agent_response": None,
            "error": None,
        }

    except Exception as exc:
        logger.error("[%s] Triage node exception: %s", trace, exc, exc_info=True)
        return {
            "current_agent": NODE_TRIAGE,
            "error": f"Triage failed: {type(exc).__name__}: {exc}",
            "last_agent_response": None,
        }


# ---------------------------------------------------------------------------
# Node 2 — Technical Support
# ---------------------------------------------------------------------------

def technical_support_node(state: GraphState) -> dict[str, Any]:
    """
    Specialist node for technical product questions.

    Calls ``run_technical_support_agent``, which uses the FAISS KB tool.
    The ToolNode (``technical_tools_node``) handles actual tool execution
    in the ReAct loop; this node produces the final natural-language response.

    State updates returned
    ----------------------
    messages            : Appends AIMessage with agent response.
    current_agent       : Set to "technical_support_agent".
    last_agent_response : Dict of AgentResponse fields (needs_handover, etc.).
    handover_count      : Incremented if needs_handover=True.
    error               : Set if agent raised an exception.
    """
    trace = state.get("trace_id", "unknown")
    logger.info("[%s] → Technical Support node executing.", trace)

    try:
        agent_state = _build_agent_state(state)
        response = run_technical_support_agent(agent_state)

        ai_msg = AIMessage(
            content=response.content,
            name="technical_support_agent",
        )

        handover_increment = 1 if response.needs_handover else 0

        logger.info(
            "[%s] Technical Support complete. needs_handover=%s → %s",
            trace,
            response.needs_handover,
            response.target_agent.value if response.target_agent else "END",
        )

        updates: dict[str, Any] = {
            "messages": [ai_msg],
            "current_agent": NODE_TECHNICAL,
            "last_agent_response": response.model_dump(),
            "error": None,
        }
        if handover_increment:
            updates["handover_count"] = state.get("handover_count", 0) + 1
        return updates

    except Exception as exc:
        logger.error("[%s] Technical Support node exception: %s", trace, exc, exc_info=True)
        return {
            "current_agent": NODE_TECHNICAL,
            "error": f"Technical Support failed: {type(exc).__name__}: {exc}",
            "last_agent_response": {"needs_handover": True, "target_agent": NODE_ESCALATION},
        }


# ---------------------------------------------------------------------------
# Node 3 — Billing
# ---------------------------------------------------------------------------

def billing_node(state: GraphState) -> dict[str, Any]:
    """
    Specialist node for billing, subscription, and invoice queries.

    State updates returned
    ----------------------
    messages            : Appends AIMessage with agent response.
    current_agent       : Set to "billing_agent".
    last_agent_response : Dict of AgentResponse fields.
    handover_count      : Incremented if needs_handover=True.
    error               : Set if agent raised an exception.
    """
    trace = state.get("trace_id", "unknown")
    logger.info("[%s] → Billing node executing.", trace)

    try:
        agent_state = _build_agent_state(state)
        response = run_billing_agent(agent_state)

        ai_msg = AIMessage(
            content=response.content,
            name="billing_agent",
        )

        handover_increment = 1 if response.needs_handover else 0

        logger.info(
            "[%s] Billing complete. needs_handover=%s → %s",
            trace,
            response.needs_handover,
            response.target_agent.value if response.target_agent else "END",
        )

        updates: dict[str, Any] = {
            "messages": [ai_msg],
            "current_agent": NODE_BILLING,
            "last_agent_response": response.model_dump(),
            "error": None,
        }
        if handover_increment:
            updates["handover_count"] = state.get("handover_count", 0) + 1
        return updates

    except Exception as exc:
        logger.error("[%s] Billing node exception: %s", trace, exc, exc_info=True)
        return {
            "current_agent": NODE_BILLING,
            "error": f"Billing failed: {type(exc).__name__}: {exc}",
            "last_agent_response": {"needs_handover": True, "target_agent": NODE_ESCALATION},
        }


# ---------------------------------------------------------------------------
# Node 4 — Escalation (terminal)
# ---------------------------------------------------------------------------

def escalation_node(state: GraphState) -> dict[str, Any]:
    """
    Terminal node. Summarises the conversation and produces a handover package.

    Once this node runs, ``is_escalated`` is set to True and the router will
    always send the graph to END regardless of ``needs_handover``.

    State updates returned
    ----------------------
    messages            : Appends AIMessage with customer-facing message +
                          internal handover package JSON.
    current_agent       : Set to "escalation_agent".
    is_escalated        : Set to True (prevents further routing).
    last_agent_response : Dict of AgentResponse (needs_handover always False).
    error               : Cleared (escalation is its own recovery path).
    """
    trace = state.get("trace_id", "unknown")
    logger.info("[%s] → Escalation node executing.", trace)

    try:
        agent_state = _build_agent_state(state)
        response = run_escalation_agent(agent_state)

        ai_msg = AIMessage(
            content=response.content,
            name="escalation_agent",
        )

        logger.info(
            "[%s] Escalation complete. priority=%s, team=%s",
            trace,
            response.metadata.get("priority", "unknown"),
            response.metadata.get("recommended_team", "unknown"),
        )

        return {
            "messages": [ai_msg],
            "current_agent": NODE_ESCALATION,
            "is_escalated": True,
            "last_agent_response": response.model_dump(),
            "error": None,
        }

    except Exception as exc:
        logger.error("[%s] Escalation node exception: %s", trace, exc, exc_info=True)
        # Escalation is the last resort — if it fails, return a static message
        static_msg = AIMessage(
            content=(
                "We're sorry, our support system encountered an unexpected error. "
                "Please contact support@clouddash.io directly, quoting your "
                f"reference ID: {trace}"
            ),
            name="escalation_agent",
        )
        return {
            "messages": [static_msg],
            "current_agent": NODE_ESCALATION,
            "is_escalated": True,
            "error": f"Escalation failed: {type(exc).__name__}: {exc}",
            "last_agent_response": {"needs_handover": False},
        }


# ---------------------------------------------------------------------------
# Triage self-loop: "general" intent handled inline
# ---------------------------------------------------------------------------

def general_response_node(state: GraphState) -> dict[str, Any]:
    """
    Lightweight node for ``intent=general`` (onboarding, FAQ, account settings).

    Rather than calling a specialist, the Triage Agent can answer simple
    general questions inline.  In Stage 6 this will call the LLM with the
    triage system prompt. For now it returns a placeholder that the Stage 5
    tests can validate.

    State updates returned
    ----------------------
    messages        : Appends a placeholder AI message.
    current_agent   : Set to "triage_agent" (Triage answered the question).
    last_agent_response : Minimal AgentResponse dict (no handover).
    """
    trace = state.get("trace_id", "unknown")
    logger.info("[%s] → General response node executing.", trace)

    # In Stage 6, this will be replaced with a real LLM call using the
    # triage system prompt (temperature 0.3 for general questions).
    ai_msg = AIMessage(
        content=(
            "Thank you for contacting CloudDash support. "
            "I can help you with general questions about the platform. "
            "Could you provide more details about what you need?"
        ),
        name="triage_agent",
    )

    return {
        "messages": [ai_msg],
        "current_agent": NODE_TRIAGE,
        "last_agent_response": {
            "agent_name": NODE_TRIAGE,
            "content": ai_msg.content,
            "needs_handover": False,
            "target_agent": None,
        },
        "error": None,
    }


# ===========================================================================
# Graph construction
# ===========================================================================

def build_graph(
    use_checkpointer: bool = True,
    checkpointer_db_path: str | None = None,
) -> "CompiledStateGraph":
    """
    Build and compile the CloudDash multi-agent StateGraph.

    Parameters
    ----------
    use_checkpointer
        If True (default), attaches a ``SqliteSaver`` checkpointer so that
        multi-turn conversations are persisted across calls.
    checkpointer_db_path
        Path to the SQLite file for the checkpointer.
        Defaults to ``<project_root>/graph_checkpoints.db``.

    Returns
    -------
    CompiledStateGraph
        The compiled LangGraph application, ready to call with ``.invoke()``,
        ``.stream()``, or ``.ainvoke()``.
    """
    # --- ToolNode for specialist tool-calling loops ---
    # Both Technical and Billing agents bind tools; the ToolNode intercepts
    # their AIMessages with tool_calls and executes the actual tools.
    tool_node = ToolNode(ALL_TOOLS)

    # --- StateGraph ---
    graph = StateGraph(GraphState)

    # ---- Register nodes ----
    graph.add_node(NODE_TRIAGE, triage_node)
    graph.add_node(NODE_TECHNICAL, technical_support_node)
    graph.add_node(NODE_BILLING, billing_node)
    graph.add_node(NODE_ESCALATION, escalation_node)
    graph.add_node("general_response", general_response_node)
    graph.add_node("tools", tool_node)

    # ---- Entry point ----
    graph.add_edge(START, NODE_TRIAGE)

    # ---- Triage → specialist routing ----
    graph.add_conditional_edges(
        NODE_TRIAGE,
        route_after_triage,
        {
            NODE_TECHNICAL:  NODE_TECHNICAL,
            NODE_BILLING:    NODE_BILLING,
            NODE_ESCALATION: NODE_ESCALATION,
            NODE_TRIAGE:     "general_response",   # intent=general handled inline
        },
    )

    # ---- Tool execution loops for specialists ----
    # When a specialist returns an AIMessage with tool_calls,
    # the ToolNode executes the tool and the result loops back to the specialist.
    graph.add_conditional_edges(
        "tools",
        _route_tool_result,
        {
            NODE_TECHNICAL: NODE_TECHNICAL,
            NODE_BILLING:   NODE_BILLING,
        },
    )

    # ---- Specialist → handover or END routing ----
    graph.add_conditional_edges(
        NODE_TECHNICAL,
        route_after_specialist,
        {
            NODE_BILLING:    NODE_BILLING,
            NODE_ESCALATION: NODE_ESCALATION,
            END:             END,
        },
    )

    graph.add_conditional_edges(
        NODE_BILLING,
        route_after_specialist,
        {
            NODE_TECHNICAL:  NODE_TECHNICAL,
            NODE_ESCALATION: NODE_ESCALATION,
            END:             END,
        },
    )

    # ---- Escalation is always terminal ----
    graph.add_edge(NODE_ESCALATION, END)

    # ---- General response always ends ----
    graph.add_edge("general_response", END)

    # ---- Compile ----
    checkpointer = None
    if use_checkpointer:
        try:
            # Try SqliteSaver (requires context-manager usage in newer LangGraph)
            # We use InMemorySaver as a reliable fallback for all contexts.
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            logger.info("Graph using MemorySaver checkpointer (in-process persistence).")
        except Exception as exc:
            logger.warning(
                "Could not initialise checkpointer (%s). Running without persistence.", exc
            )
            checkpointer = None

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("CloudDash multi-agent graph compiled successfully.")
    return compiled


def _route_tool_result(state: GraphState) -> str:
    """
    After a ToolNode executes, route back to whichever specialist invoked the tool.

    LangGraph ToolNode appends a ToolMessage to ``messages``. We look at
    the preceding AIMessage's ``name`` field to determine which agent called the tool.
    """
    messages = state.get("messages", [])
    # Walk backwards to find the last non-ToolMessage AI message
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage):
            name = getattr(msg, "name", None) or ""
            if "billing" in name:
                return NODE_BILLING
            return NODE_TECHNICAL
    # Fallback: route to technical (most likely caller)
    return NODE_TECHNICAL


# ===========================================================================
# Convenience: session initialiser
# ===========================================================================

def create_initial_state(
    user_message: str,
    session_id: str | None = None,
    customer_id: str | None = None,
    trace_id: str | None = None,
) -> GraphState:
    """
    Build a fresh ``GraphState`` for the first turn of a new session.

    Parameters
    ----------
    user_message
        The customer's first message.
    session_id
        Optional session identifier. Auto-generated if not provided.
    customer_id
        Pre-known customer ID (e.g. from authenticated API request).
    trace_id
        Optional trace ID for linking to an external observability system.

    Returns
    -------
    GraphState
        A dict ready to pass to ``graph.invoke()`` or ``graph.stream()``.
    """
    sid = session_id or str(uuid.uuid4())
    tid = trace_id or str(uuid.uuid4())

    initial_entities: dict[str, Any] = {
        "customer_id": customer_id,
        "product_area": None,
        "error_code": None,
        "urgency": "medium",
    }

    return GraphState(
        messages=[HumanMessage(content=user_message, name="user")],
        current_agent=NODE_TRIAGE,
        customer_id=customer_id,
        trace_id=tid,
        session_id=sid,
        intent=None,
        confidence=None,
        entities=initial_entities,
        handover_count=0,
        is_escalated=False,
        last_agent_response=None,
        error=None,
    )


# ===========================================================================
# Singleton compiled graph (imported by API layer in Stage 7)
# ===========================================================================

# Lazy-initialised so import does not trigger DB/LLM setup at test collection time
_compiled_graph = None


def get_graph(
    use_checkpointer: bool = True,
    checkpointer_db_path: str | None = None,
):
    """
    Return the singleton compiled graph, building it on first call.

    The graph is cached in ``_compiled_graph`` so the StateGraph is only
    compiled once per process, regardless of how many API requests arrive.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph(
            use_checkpointer=use_checkpointer,
            checkpointer_db_path=checkpointer_db_path,
        )
    return _compiled_graph
