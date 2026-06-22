"""
tests/test_graph.py
===================
Stage 5 tests for the LangGraph orchestration layer.

Test strategy
-------------
All agent node functions are mocked at the ``agents.agent_nodes`` level so:
- Tests run completely offline (no Gemini API, no FAISS index needed).
- We validate graph topology, routing logic, state mutation, handover protocol,
  and loop guards — the actual LLM behaviour is covered by test_agents.py.

Structure
---------
TestMergeEntities           — reducer unit tests
TestRouteAfterTriage        — routing table + confidence/auto-escalation
TestRouteAfterSpecialist    — handover detection + loop guard
TestTriageNode              — state updates produced by the triage node fn
TestSpecialistNodes         — state updates for technical + billing nodes
TestEscalationNode          — terminal escalation + static fallback message
TestCreateInitialState      — state factory helper
TestGraphTopology           — node registration + edge compilation (smoke)
TestGraphEndToEnd           — full invoke() with mocked agents

Run
---
    pytest tests/test_graph.py -v
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from models.models import (
    AgentName,
    AgentResponse,
    EscalationPriority,
    ExtractedEntities,
    IntentLabel,
    RecommendedTeam,
    TriageResult,
    UrgencyLevel,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_triage_result(
    intent: str = "technical",
    confidence: float = 0.90,
    customer_id: str | None = "CLD-00001",
) -> TriageResult:
    return TriageResult(
        intent=IntentLabel(intent),
        confidence=confidence,
        extracted_entities=ExtractedEntities(
            customer_id=customer_id,
            urgency=UrgencyLevel.MEDIUM,
        ),
        reasoning=f"Test triage result for intent={intent}",
    )


def _make_agent_response(
    agent_name: AgentName = AgentName.TECHNICAL_SUPPORT,
    content: str = "Here is your answer. Sources: TS-001",
    needs_handover: bool = False,
    target_agent: AgentName | None = None,
) -> AgentResponse:
    return AgentResponse(
        agent_name=agent_name,
        content=content,
        needs_handover=needs_handover,
        target_agent=target_agent,
        source_documents=["TS-001"] if agent_name == AgentName.TECHNICAL_SUPPORT else [],
    )


def _make_state(
    intent: str | None = "technical",
    confidence: float = 0.90,
    handover_count: int = 0,
    is_escalated: bool = False,
    last_agent_response: dict | None = None,
    error: str | None = None,
    customer_id: str | None = "CLD-00001",
    messages: list | None = None,
) -> dict[str, Any]:
    return {
        "messages": messages or [HumanMessage(content="test message", name="user")],
        "current_agent": "triage_agent",
        "customer_id": customer_id,
        "trace_id": "test-trace-001",
        "session_id": "test-session-001",
        "intent": intent,
        "confidence": confidence,
        "entities": {"customer_id": customer_id, "urgency": "medium"},
        "handover_count": handover_count,
        "is_escalated": is_escalated,
        "last_agent_response": last_agent_response,
        "error": error,
    }


# ===========================================================================
# merge_entities reducer
# ===========================================================================

class TestMergeEntities:

    def test_right_overwrites_left_for_non_none(self):
        from graph.state import merge_entities
        left = {"customer_id": "CLD-00001", "error_code": "ERR-4012"}
        right = {"customer_id": "CLD-00002", "product_area": "alerting"}
        result = merge_entities(left, right)
        assert result["customer_id"] == "CLD-00002"
        assert result["error_code"] == "ERR-4012"   # preserved from left
        assert result["product_area"] == "alerting"

    def test_none_values_in_right_do_not_overwrite(self):
        from graph.state import merge_entities
        left = {"customer_id": "CLD-00001", "error_code": "ERR-4012"}
        right = {"customer_id": None, "product_area": "alerting"}
        result = merge_entities(left, right)
        assert result["customer_id"] == "CLD-00001"  # not overwritten by None

    def test_empty_left_takes_right(self):
        from graph.state import merge_entities
        result = merge_entities({}, {"customer_id": "CLD-00003"})
        assert result["customer_id"] == "CLD-00003"

    def test_empty_right_preserves_left(self):
        from graph.state import merge_entities
        left = {"customer_id": "CLD-00001", "urgency": "high"}
        result = merge_entities(left, {})
        assert result == left

    def test_both_empty_returns_empty(self):
        from graph.state import merge_entities
        assert merge_entities({}, {}) == {}

    def test_does_not_mutate_left(self):
        from graph.state import merge_entities
        left = {"customer_id": "CLD-00001"}
        merge_entities(left, {"customer_id": "CLD-00002"})
        assert left["customer_id"] == "CLD-00001"   # original unchanged


# ===========================================================================
# route_after_triage
# ===========================================================================

class TestRouteAfterTriage:

    def test_technical_intent_routes_to_technical(self):
        from graph.router import route_after_triage, NODE_TECHNICAL
        state = _make_state(intent="technical", confidence=0.90)
        assert route_after_triage(state) == NODE_TECHNICAL

    def test_billing_intent_routes_to_billing(self):
        from graph.router import route_after_triage, NODE_BILLING
        state = _make_state(intent="billing", confidence=0.85)
        assert route_after_triage(state) == NODE_BILLING

    def test_general_intent_routes_to_triage(self):
        """general intent is handled inline by triage_agent node."""
        from graph.router import route_after_triage, NODE_TRIAGE
        state = _make_state(intent="general", confidence=0.80)
        assert route_after_triage(state) == NODE_TRIAGE

    def test_escalation_intent_routes_to_escalation(self):
        from graph.router import route_after_triage, NODE_ESCALATION
        state = _make_state(intent="escalation", confidence=0.95)
        assert route_after_triage(state) == NODE_ESCALATION

    def test_low_confidence_routes_to_escalation(self):
        """confidence < threshold (0.65) should escalate regardless of intent."""
        from graph.router import route_after_triage, NODE_ESCALATION
        state = _make_state(intent="technical", confidence=0.40)
        assert route_after_triage(state) == NODE_ESCALATION

    def test_error_field_routes_to_escalation(self):
        from graph.router import route_after_triage, NODE_ESCALATION
        state = _make_state(error="Triage crashed")
        assert route_after_triage(state) == NODE_ESCALATION

    def test_max_handovers_reached_routes_to_escalation(self):
        from graph.router import route_after_triage, NODE_ESCALATION, MAX_HANDOVERS
        state = _make_state(intent="technical", confidence=0.90, handover_count=MAX_HANDOVERS)
        assert route_after_triage(state) == NODE_ESCALATION

    def test_unknown_intent_falls_back(self):
        """unknown intent should not crash — falls back via config resolve_intent."""
        from graph.router import route_after_triage
        state = _make_state(intent="unknown", confidence=0.90)
        # unknown → triage_agent per agents_config.yaml
        result = route_after_triage(state)
        assert result in ("triage_agent", "escalation_agent")

    def test_auto_escalate_intent(self):
        """refund_request is in auto_escalate_intents → escalation."""
        from graph.router import route_after_triage, NODE_ESCALATION
        state = _make_state(intent="refund_request", confidence=0.99)
        assert route_after_triage(state) == NODE_ESCALATION


# ===========================================================================
# route_after_specialist
# ===========================================================================

class TestRouteAfterSpecialist:

    def test_no_handover_routes_to_end(self):
        from graph.router import route_after_specialist
        from langgraph.graph import END
        state = _make_state(
            last_agent_response={"needs_handover": False, "target_agent": None}
        )
        assert route_after_specialist(state) == END

    def test_handover_to_billing_routes_to_billing(self):
        from graph.router import route_after_specialist, NODE_BILLING
        state = _make_state(
            last_agent_response={
                "needs_handover": True,
                "target_agent": "billing_agent",
            }
        )
        assert route_after_specialist(state) == NODE_BILLING

    def test_handover_to_escalation_routes_to_escalation(self):
        from graph.router import route_after_specialist, NODE_ESCALATION
        state = _make_state(
            last_agent_response={
                "needs_handover": True,
                "target_agent": "escalation_agent",
            }
        )
        assert route_after_specialist(state) == NODE_ESCALATION

    def test_handover_to_technical_routes_to_technical(self):
        from graph.router import route_after_specialist, NODE_TECHNICAL
        state = _make_state(
            last_agent_response={
                "needs_handover": True,
                "target_agent": "technical_support_agent",
            }
        )
        assert route_after_specialist(state) == NODE_TECHNICAL

    def test_is_escalated_always_ends(self):
        from graph.router import route_after_specialist
        from langgraph.graph import END
        state = _make_state(
            is_escalated=True,
            last_agent_response={"needs_handover": False, "target_agent": None},
        )
        assert route_after_specialist(state) == END

    def test_max_handovers_from_specialist_ends(self):
        from graph.router import route_after_specialist, MAX_HANDOVERS
        from langgraph.graph import END
        state = _make_state(
            handover_count=MAX_HANDOVERS,
            last_agent_response={"needs_handover": True, "target_agent": "billing_agent"},
        )
        assert route_after_specialist(state) == END

    def test_needs_handover_without_target_routes_to_escalation(self):
        from graph.router import route_after_specialist, NODE_ESCALATION
        state = _make_state(
            last_agent_response={"needs_handover": True, "target_agent": None}
        )
        assert route_after_specialist(state) == NODE_ESCALATION

    def test_error_with_is_escalated_ends(self):
        from graph.router import route_after_specialist
        from langgraph.graph import END
        state = _make_state(error="boom", is_escalated=True)
        assert route_after_specialist(state) == END

    def test_error_without_escalated_routes_to_escalation(self):
        from graph.router import route_after_specialist, NODE_ESCALATION
        state = _make_state(error="specialist crashed", is_escalated=False)
        assert route_after_specialist(state) == NODE_ESCALATION

    def test_handover_to_triage_redirected_to_escalation(self):
        """Specialists are not allowed to loop back to Triage."""
        from graph.router import route_after_specialist, NODE_ESCALATION
        state = _make_state(
            last_agent_response={"needs_handover": True, "target_agent": "triage_agent"}
        )
        assert route_after_specialist(state) == NODE_ESCALATION


# ===========================================================================
# Triage node
# ===========================================================================

class TestTriageNode:

    @patch("graph.graph.run_triage_agent")
    def test_triage_node_updates_intent_and_confidence(self, mock_triage):
        from graph.graph import triage_node
        mock_triage.return_value = _make_triage_result("billing", 0.88, "CLD-00002")
        state = _make_state()
        result = triage_node(state)
        assert result["intent"] == "billing"
        assert abs(result["confidence"] - 0.88) < 1e-6

    @patch("graph.graph.run_triage_agent")
    def test_triage_node_sets_customer_id(self, mock_triage):
        from graph.graph import triage_node
        mock_triage.return_value = _make_triage_result(customer_id="CLD-00005")
        result = triage_node(_make_state())
        assert result["customer_id"] == "CLD-00005"

    @patch("graph.graph.run_triage_agent")
    def test_triage_node_appends_ai_message(self, mock_triage):
        from graph.graph import triage_node
        mock_triage.return_value = _make_triage_result()
        result = triage_node(_make_state())
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @patch("graph.graph.run_triage_agent")
    def test_triage_node_sets_current_agent(self, mock_triage):
        from graph.graph import triage_node
        mock_triage.return_value = _make_triage_result()
        result = triage_node(_make_state())
        assert result["current_agent"] == "triage_agent"

    @patch("graph.graph.run_triage_agent")
    def test_triage_node_clears_error(self, mock_triage):
        from graph.graph import triage_node
        mock_triage.return_value = _make_triage_result()
        result = triage_node(_make_state(error="old error"))
        assert result["error"] is None

    @patch("graph.graph.run_triage_agent")
    def test_triage_node_handles_exception_gracefully(self, mock_triage):
        from graph.graph import triage_node
        mock_triage.side_effect = RuntimeError("LLM quota exceeded")
        result = triage_node(_make_state())
        assert result["error"] is not None
        assert "Triage failed" in result["error"]

    @patch("graph.graph.run_triage_agent")
    def test_triage_message_contains_intent(self, mock_triage):
        from graph.graph import triage_node
        mock_triage.return_value = _make_triage_result("escalation", 0.99)
        result = triage_node(_make_state())
        msg_content = result["messages"][0].content
        assert "escalation" in msg_content.lower()


# ===========================================================================
# Technical Support node
# ===========================================================================

class TestTechnicalSupportNode:

    @patch("graph.graph.run_technical_support_agent")
    def test_returns_ai_message_with_content(self, mock_tech):
        from graph.graph import technical_support_node
        mock_tech.return_value = _make_agent_response(
            AgentName.TECHNICAL_SUPPORT, "Step 1: ... Sources: TS-001"
        )
        result = technical_support_node(_make_state())
        assert len(result["messages"]) == 1
        assert "Step 1" in result["messages"][0].content

    @patch("graph.graph.run_technical_support_agent")
    def test_no_handover_does_not_increment_count(self, mock_tech):
        from graph.graph import technical_support_node
        mock_tech.return_value = _make_agent_response(needs_handover=False)
        result = technical_support_node(_make_state(handover_count=1))
        assert "handover_count" not in result   # not updated

    @patch("graph.graph.run_technical_support_agent")
    def test_handover_increments_count(self, mock_tech):
        from graph.graph import technical_support_node
        mock_tech.return_value = _make_agent_response(
            needs_handover=True,
            target_agent=AgentName.ESCALATION,
        )
        result = technical_support_node(_make_state(handover_count=2))
        assert result["handover_count"] == 3

    @patch("graph.graph.run_technical_support_agent")
    def test_sets_last_agent_response(self, mock_tech):
        from graph.graph import technical_support_node
        mock_tech.return_value = _make_agent_response()
        result = technical_support_node(_make_state())
        assert result["last_agent_response"] is not None
        assert result["last_agent_response"]["agent_name"] == "technical_support_agent"

    @patch("graph.graph.run_technical_support_agent")
    def test_exception_sets_error_and_escalation_handover(self, mock_tech):
        from graph.graph import technical_support_node
        mock_tech.side_effect = RuntimeError("boom")
        result = technical_support_node(_make_state())
        assert result["error"] is not None
        assert result["last_agent_response"]["target_agent"] == "escalation_agent"


# ===========================================================================
# Billing node
# ===========================================================================

class TestBillingNode:

    @patch("graph.graph.run_billing_agent")
    def test_returns_ai_message(self, mock_billing):
        from graph.graph import billing_node
        mock_billing.return_value = _make_agent_response(
            AgentName.BILLING, "Your plan is Growth."
        )
        result = billing_node(_make_state())
        assert "Growth" in result["messages"][0].content

    @patch("graph.graph.run_billing_agent")
    def test_refund_handover_increments_count(self, mock_billing):
        from graph.graph import billing_node
        mock_billing.return_value = _make_agent_response(
            AgentName.BILLING,
            needs_handover=True,
            target_agent=AgentName.ESCALATION,
        )
        result = billing_node(_make_state(handover_count=0))
        assert result["handover_count"] == 1

    @patch("graph.graph.run_billing_agent")
    def test_sets_current_agent_to_billing(self, mock_billing):
        from graph.graph import billing_node
        mock_billing.return_value = _make_agent_response(AgentName.BILLING, "Billing info.")
        result = billing_node(_make_state())
        assert result["current_agent"] == "billing_agent"

    @patch("graph.graph.run_billing_agent")
    def test_exception_triggers_escalation_fallback(self, mock_billing):
        from graph.graph import billing_node
        mock_billing.side_effect = Exception("DB down")
        result = billing_node(_make_state())
        assert result["error"] is not None
        assert result["last_agent_response"]["target_agent"] == "escalation_agent"


# ===========================================================================
# Escalation node
# ===========================================================================

class TestEscalationNode:

    @patch("graph.graph.run_escalation_agent")
    def test_sets_is_escalated_true(self, mock_esc):
        from graph.graph import escalation_node
        mock_esc.return_value = _make_agent_response(
            AgentName.ESCALATION,
            "Handover prepared. Reference ID: test-trace-001",
        )
        result = escalation_node(_make_state())
        assert result["is_escalated"] is True

    @patch("graph.graph.run_escalation_agent")
    def test_sets_current_agent_to_escalation(self, mock_esc):
        from graph.graph import escalation_node
        mock_esc.return_value = _make_agent_response(AgentName.ESCALATION, "Escalated.")
        result = escalation_node(_make_state())
        assert result["current_agent"] == "escalation_agent"

    @patch("graph.graph.run_escalation_agent")
    def test_appends_ai_message(self, mock_esc):
        from graph.graph import escalation_node
        mock_esc.return_value = _make_agent_response(AgentName.ESCALATION, "Handover done.")
        result = escalation_node(_make_state())
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @patch("graph.graph.run_escalation_agent")
    def test_exception_returns_static_fallback_message(self, mock_esc):
        from graph.graph import escalation_node
        mock_esc.side_effect = Exception("LLM error")
        result = escalation_node(_make_state())
        assert result["is_escalated"] is True
        assert "support@clouddash.io" in result["messages"][0].content

    @patch("graph.graph.run_escalation_agent")
    def test_clears_error_field(self, mock_esc):
        from graph.graph import escalation_node
        mock_esc.return_value = _make_agent_response(AgentName.ESCALATION, "Done.")
        result = escalation_node(_make_state(error="old error"))
        assert result["error"] is None


# ===========================================================================
# create_initial_state
# ===========================================================================

class TestCreateInitialState:

    def test_messages_contain_user_message(self):
        from graph.graph import create_initial_state
        state = create_initial_state("Help me with billing", customer_id="CLD-00001")
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "Help me with billing"

    def test_state_has_trace_id(self):
        from graph.graph import create_initial_state
        state = create_initial_state("test")
        assert state["trace_id"] is not None
        assert len(state["trace_id"]) > 0

    def test_customer_id_propagated(self):
        from graph.graph import create_initial_state
        state = create_initial_state("test", customer_id="CLD-00003")
        assert state["customer_id"] == "CLD-00003"
        assert state["entities"]["customer_id"] == "CLD-00003"

    def test_initial_handover_count_is_zero(self):
        from graph.graph import create_initial_state
        state = create_initial_state("test")
        assert state["handover_count"] == 0

    def test_initial_is_escalated_is_false(self):
        from graph.graph import create_initial_state
        state = create_initial_state("test")
        assert state["is_escalated"] is False

    def test_custom_session_id_used(self):
        from graph.graph import create_initial_state
        state = create_initial_state("test", session_id="my-session-123")
        assert state["session_id"] == "my-session-123"

    def test_current_agent_is_triage(self):
        from graph.graph import create_initial_state
        state = create_initial_state("test")
        assert state["current_agent"] == "triage_agent"

    def test_intent_and_confidence_start_as_none(self):
        from graph.graph import create_initial_state
        state = create_initial_state("test")
        assert state["intent"] is None
        assert state["confidence"] is None


# ===========================================================================
# Graph topology (compilation smoke test)
# ===========================================================================

class TestGraphTopology:

    def test_build_graph_compiles_without_error(self):
        from graph.graph import build_graph
        # Build without checkpointer for test speed
        compiled = build_graph(use_checkpointer=False)
        assert compiled is not None

    def test_compiled_graph_is_invocable(self):
        from graph.graph import build_graph
        compiled = build_graph(use_checkpointer=False)
        assert hasattr(compiled, "invoke")
        assert hasattr(compiled, "stream")

    def test_get_graph_returns_same_instance(self):
        """Singleton: two calls return the same compiled object."""
        from graph import graph as graph_module
        # Reset singleton for clean test
        graph_module._compiled_graph = None
        g1 = graph_module.get_graph(use_checkpointer=False)
        g2 = graph_module.get_graph(use_checkpointer=False)
        assert g1 is g2
        graph_module._compiled_graph = None  # cleanup


# ===========================================================================
# End-to-end graph invocation (fully mocked agents)
# ===========================================================================

class TestGraphEndToEnd:
    """
    Full graph.invoke() tests with all four agent functions mocked.

    These tests validate that the graph routes correctly and that state
    accumulates properly across nodes — without any real LLM calls.
    """

    @patch("graph.graph.run_technical_support_agent")
    @patch("graph.graph.run_triage_agent")
    def test_technical_intent_reaches_technical_node(self, mock_triage, mock_tech):
        from graph.graph import build_graph, create_initial_state
        mock_triage.return_value = _make_triage_result("technical", 0.92, "CLD-00001")
        mock_tech.return_value = _make_agent_response(
            AgentName.TECHNICAL_SUPPORT,
            "ERR-4012: reconfigure IAM credentials. Sources: TS-001",
            needs_handover=False,
        )
        graph = build_graph(use_checkpointer=False)
        state = create_initial_state("I keep getting ERR-4012", customer_id="CLD-00001")
        result = graph.invoke(state)
        # Should have triage message + technical message
        agent_names = [getattr(m, "name", "") for m in result["messages"]]
        assert "technical_support_agent" in agent_names
        assert result["current_agent"] == "technical_support_agent"

    @patch("graph.graph.run_billing_agent")
    @patch("graph.graph.run_triage_agent")
    def test_billing_intent_reaches_billing_node(self, mock_triage, mock_billing):
        from graph.graph import build_graph, create_initial_state
        mock_triage.return_value = _make_triage_result("billing", 0.88, "CLD-00002")
        mock_billing.return_value = _make_agent_response(
            AgentName.BILLING,
            "Your Growth plan renews on 2026-07-15.",
            needs_handover=False,
        )
        graph = build_graph(use_checkpointer=False)
        state = create_initial_state("What plan am I on?", customer_id="CLD-00002")
        result = graph.invoke(state)
        agent_names = [getattr(m, "name", "") for m in result["messages"]]
        assert "billing_agent" in agent_names
        assert result["current_agent"] == "billing_agent"

    @patch("graph.graph.run_escalation_agent")
    @patch("graph.graph.run_triage_agent")
    def test_low_confidence_escalates(self, mock_triage, mock_esc):
        from graph.graph import build_graph, create_initial_state
        # Confidence below threshold → escalation
        mock_triage.return_value = _make_triage_result("technical", 0.30)
        mock_esc.return_value = _make_agent_response(
            AgentName.ESCALATION,
            "Handover package prepared. Reference ID: test-trace.",
        )
        graph = build_graph(use_checkpointer=False)
        result = graph.invoke(create_initial_state("something unclear"))
        assert result["is_escalated"] is True

    @patch("graph.graph.run_escalation_agent")
    @patch("graph.graph.run_technical_support_agent")
    @patch("graph.graph.run_triage_agent")
    def test_technical_handover_to_escalation(self, mock_triage, mock_tech, mock_esc):
        """Technical agent requesting escalation should reach the escalation node."""
        from graph.graph import build_graph, create_initial_state
        mock_triage.return_value = _make_triage_result("technical", 0.90)
        mock_tech.return_value = _make_agent_response(
            AgentName.TECHNICAL_SUPPORT,
            "I'm unable to find docs. I'll escalate.",
            needs_handover=True,
            target_agent=AgentName.ESCALATION,
        )
        mock_esc.return_value = _make_agent_response(
            AgentName.ESCALATION,
            "Escalated. Reference ID: trace-001",
        )
        graph = build_graph(use_checkpointer=False)
        result = graph.invoke(create_initial_state("weird obscure issue"))
        assert result["is_escalated"] is True
        assert result["handover_count"] >= 1

    @patch("graph.graph.run_billing_agent")
    @patch("graph.graph.run_escalation_agent")
    @patch("graph.graph.run_triage_agent")
    def test_escalation_intent_skips_specialist(self, mock_triage, mock_esc, mock_billing):
        """direct escalation intent should never call the billing agent."""
        from graph.graph import build_graph, create_initial_state
        mock_triage.return_value = _make_triage_result("escalation", 0.99)
        mock_esc.return_value = _make_agent_response(
            AgentName.ESCALATION, "Escalated immediately."
        )
        graph = build_graph(use_checkpointer=False)
        result = graph.invoke(create_initial_state("I want to terminate my account"))
        mock_billing.assert_not_called()
        assert result["is_escalated"] is True

    @patch("graph.graph.run_escalation_agent")
    @patch("graph.graph.run_billing_agent")
    @patch("graph.graph.run_triage_agent")
    def test_message_history_preserved_on_handover(self, mock_triage, mock_billing, mock_esc):
        """Full message history should be in final state after a handover."""
        from graph.graph import build_graph, create_initial_state
        mock_triage.return_value = _make_triage_result("billing", 0.88)
        mock_billing.return_value = _make_agent_response(
            AgentName.BILLING,
            content="Refund not processable — escalating.",
            needs_handover=True,
            target_agent=AgentName.ESCALATION,
        )
        mock_esc.return_value = _make_agent_response(
            AgentName.ESCALATION, "Refund request escalated."
        )
        graph = build_graph(use_checkpointer=False)
        result = graph.invoke(create_initial_state("I want a refund", customer_id="CLD-00001"))
        # At minimum: user msg + triage msg + billing msg + escalation msg
        assert len(result["messages"]) >= 4

    @patch("graph.graph.run_escalation_agent")
    @patch("graph.graph.run_triage_agent")
    def test_general_intent_ends_without_specialist(self, mock_triage, mock_esc):
        """general intent should end at the general_response node, not call escalation."""
        from graph.graph import build_graph, create_initial_state
        mock_triage.return_value = _make_triage_result("general", 0.80)
        graph = build_graph(use_checkpointer=False)
        result = graph.invoke(create_initial_state("How do I reset my password?"))
        mock_esc.assert_not_called()
        # general_response node always sets current_agent back to triage_agent
        assert result["current_agent"] == "triage_agent"
