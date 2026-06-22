"""
graph/router.py
===============
Conditional edge functions (routers) for the CloudDash LangGraph state machine.

Every function in this module is a **pure routing function** — it reads state
and returns a string node name (or END).  It has no side effects and calls
no LLMs.  This makes routing logic fully unit-testable without mocking anything.

Router map
----------
route_after_triage(state)
    Entry router called after the Triage node runs.
    Maps intent → specialist node name, respecting:
    - Auto-escalation intents (from agents_config.yaml)
    - Low-confidence fallback to escalation
    - Max-handover circuit-breaker

route_after_specialist(state)
    Exit router called after any specialist node (Technical, Billing, Escalation).
    Reads ``last_agent_response`` and decides:
    - If needs_handover=True  → route to target_agent
    - If is_escalated         → END (terminal)
    - Otherwise               → END (success)

Guard rails baked in
---------------------
- ``MAX_HANDOVERS`` constant prevents runaway agent loops.
- Any state with ``error`` set routes directly to escalation.
- Routing falls back to escalation (not a crash) for unknown intents.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END

from config.config_loader import get_config
from models.models import AgentName, IntentLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Node name constants — single source of truth so renaming a node is one change
NODE_TRIAGE = AgentName.TRIAGE.value                        # "triage_agent"
NODE_TECHNICAL = AgentName.TECHNICAL_SUPPORT.value          # "technical_support_agent"
NODE_BILLING = AgentName.BILLING.value                      # "billing_agent"
NODE_ESCALATION = AgentName.ESCALATION.value                # "escalation_agent"

# Safety circuit-breaker: maximum number of inter-agent handovers per session
MAX_HANDOVERS = 5


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _trace(state: dict[str, Any]) -> str:
    return state.get("trace_id", "unknown")


# ---------------------------------------------------------------------------
# Router 1 — after Triage node
# ---------------------------------------------------------------------------

def route_after_triage(state: dict[str, Any]) -> str:
    """
    Conditional edge function called after the Triage node completes.

    Decision priority (highest → lowest):
    1. ``error`` field set → escalation (node crashed)
    2. ``handover_count`` >= MAX_HANDOVERS → escalation (loop guard)
    3. ``intent`` in auto_escalate_intents → escalation
    4. ``confidence`` < threshold → escalation (low confidence)
    5. ``intent`` → mapped specialist node (from agents_config.yaml)
    6. Unknown / unmapped intent → re-triage (safe fallback)

    Returns
    -------
    str
        The name of the next node to execute.
    """
    trace = _trace(state)
    intent = state.get("intent") or IntentLabel.UNKNOWN.value
    confidence = state.get("confidence") or 0.0
    handover_count = state.get("handover_count", 0)
    error = state.get("error")

    # --- Circuit breaker ---
    if error:
        logger.warning("[%s] Error in triage: %s → escalating.", trace, error)
        return NODE_ESCALATION

    if handover_count >= MAX_HANDOVERS:
        logger.warning(
            "[%s] Max handovers (%d) reached → forcing escalation.", trace, MAX_HANDOVERS
        )
        return NODE_ESCALATION

    # --- Config-driven routing ---
    try:
        cfg = get_config()
        routing = cfg.routing

        # Auto-escalate check
        auto_escalate_intents = routing.auto_escalate_intents or []
        if intent in auto_escalate_intents:
            logger.info("[%s] Auto-escalate intent '%s' → escalation.", trace, intent)
            return NODE_ESCALATION

        # Low-confidence check
        if confidence < routing.confidence_threshold:
            logger.info(
                "[%s] Low confidence %.2f < %.2f for intent '%s' → escalation.",
                trace, confidence, routing.confidence_threshold, intent,
            )
            return NODE_ESCALATION

        # Map intent → node name
        target_node = cfg.resolve_intent(intent)
        logger.info(
            "[%s] Triage: intent='%s' (%.2f) → node='%s'",
            trace, intent, confidence, target_node,
        )
        return target_node

    except Exception as exc:
        logger.error("[%s] Router error in route_after_triage: %s", trace, exc)
        return NODE_ESCALATION


# ---------------------------------------------------------------------------
# Router 2 — after any specialist node
# ---------------------------------------------------------------------------

def route_after_specialist(state: dict[str, Any]) -> str:
    """
    Conditional edge function called after Technical Support, Billing, or
    Escalation nodes complete.

    Decision priority:
    1. ``error`` field set → escalation
    2. ``handover_count`` >= MAX_HANDOVERS → END (loop guard)
    3. ``is_escalated`` = True → END (terminal state)
    4. ``last_agent_response.needs_handover`` = True → target_agent
    5. Otherwise → END (conversation complete)

    Returns
    -------
    str
        The name of the next node, or ``END``.
    """
    trace = _trace(state)
    
    # --- Check for tool calls FIRST ---
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        logger.info("[%s] Tool calls detected → routing to 'tools'.", trace)
        return "tools"
    handover_count = state.get("handover_count", 0)
    is_escalated = state.get("is_escalated", False)
    error = state.get("error")
    last_response: dict[str, Any] = state.get("last_agent_response") or {}

    # --- Error recovery ---
    if error:
        logger.warning("[%s] Error in specialist: %s → escalating.", trace, error)
        if is_escalated:
            return END  # Already in escalation — don't loop
        return NODE_ESCALATION

    # --- Terminal conditions ---
    if is_escalated:
        logger.info("[%s] Session is escalated → END.", trace)
        return END

    if handover_count >= MAX_HANDOVERS:
        logger.warning(
            "[%s] Max handovers (%d) reached from specialist → END.", trace, MAX_HANDOVERS
        )
        return END

    # --- Handover requested by specialist ---
    needs_handover = last_response.get("needs_handover", False)
    if needs_handover:
        target = last_response.get("target_agent")
        if not target:
            logger.warning("[%s] Handover requested but no target_agent set → escalation.", trace)
            return NODE_ESCALATION

        # Safety: never hand over to Triage from a specialist (use escalation as
        # the "I don't know what to do" destination instead)
        if target == NODE_TRIAGE:
            logger.info("[%s] Specialist requested triage handover → escalation instead.", trace)
            return NODE_ESCALATION

        logger.info("[%s] Specialist handover → '%s'.", trace, target)
        return target

    # --- Normal completion ---
    logger.info("[%s] Specialist completed without handover → END.", trace)
    return END
