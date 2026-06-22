"""
tests/test_models.py
====================
Unit tests for the Stage 1 Pydantic models.
Run with:  pytest tests/test_models.py -v
"""

import pytest
from pydantic import ValidationError

from models.models import (
    AgentName,
    AgentResponse,
    ConversationMessage,
    ConversationState,
    EscalationPackage,
    EscalationPriority,
    ExtractedEntities,
    HandoverPayload,
    IntentLabel,
    MessageRole,
    RecommendedTeam,
    TriageResult,
    UrgencyLevel,
)


# ---------------------------------------------------------------------------
# ExtractedEntities
# ---------------------------------------------------------------------------

class TestExtractedEntities:
    def test_valid_customer_id(self):
        e = ExtractedEntities(customer_id="CLD-00042")
        assert e.customer_id == "CLD-00042"

    def test_auto_zero_pad_customer_id(self):
        e = ExtractedEntities(customer_id="CLD-42")
        assert e.customer_id == "CLD-00042"

    def test_invalid_customer_id_raises(self):
        with pytest.raises(ValidationError):
            ExtractedEntities(customer_id="INVALID-ID")

    def test_null_customer_id_is_allowed(self):
        e = ExtractedEntities()
        assert e.customer_id is None

    def test_default_urgency_is_medium(self):
        e = ExtractedEntities()
        assert e.urgency == UrgencyLevel.MEDIUM

    def test_extra_fields_are_forbidden(self):
        with pytest.raises(ValidationError):
            ExtractedEntities(unknown_field="oops")


# ---------------------------------------------------------------------------
# ConversationMessage
# ---------------------------------------------------------------------------

class TestConversationMessage:
    def test_valid_user_message(self):
        msg = ConversationMessage(role=MessageRole.USER, content="Hello!")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert msg.agent_name is None

    def test_whitespace_stripped_from_content(self):
        msg = ConversationMessage(role=MessageRole.USER, content="  hello  ")
        assert msg.content == "hello"

    def test_empty_content_raises(self):
        with pytest.raises(ValidationError):
            ConversationMessage(role=MessageRole.USER, content="   ")


# ---------------------------------------------------------------------------
# ConversationState
# ---------------------------------------------------------------------------

class TestConversationState:
    def test_default_state_is_valid(self):
        state = ConversationState()
        assert state.current_agent == AgentName.TRIAGE
        assert state.messages == []
        assert state.handover_count == 0
        assert state.is_escalated is False

    def test_add_message_appends(self):
        state = ConversationState()
        state.add_message(MessageRole.USER, "What's wrong with my alerts?")
        assert len(state.messages) == 1
        assert state.messages[0].content == "What's wrong with my alerts?"

    def test_get_recent_messages_limits_correctly(self):
        state = ConversationState()
        for i in range(15):
            state.add_message(MessageRole.USER, f"Message {i}")
        recent = state.get_recent_messages(5)
        assert len(recent) == 5
        assert recent[-1].content == "Message 14"

    def test_increment_handover_updates_agent(self):
        state = ConversationState()
        state.increment_handover(AgentName.BILLING)
        assert state.current_agent == AgentName.BILLING
        assert state.handover_count == 1


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------

class TestAgentResponse:
    def test_valid_response_no_handover(self):
        resp = AgentResponse(
            agent_name=AgentName.TRIAGE,
            content="I classified your intent as billing.",
        )
        assert resp.needs_handover is False
        assert resp.target_agent is None

    def test_handover_requires_target_agent(self):
        with pytest.raises(ValidationError):
            AgentResponse(
                agent_name=AgentName.TECHNICAL_SUPPORT,
                content="Handing over.",
                needs_handover=True,
                # target_agent is missing — should raise
            )

    def test_valid_handover_response(self):
        resp = AgentResponse(
            agent_name=AgentName.TECHNICAL_SUPPORT,
            content="This is a billing issue.",
            needs_handover=True,
            target_agent=AgentName.BILLING,
            handover_reason="User asked about invoice",
        )
        assert resp.target_agent == AgentName.BILLING


# ---------------------------------------------------------------------------
# HandoverPayload
# ---------------------------------------------------------------------------

class TestHandoverPayload:
    def _make_state(self) -> ConversationState:
        state = ConversationState()
        state.intent = IntentLabel.BILLING
        state.confidence = 0.92
        state.add_message(MessageRole.USER, "Why is my invoice so high?")
        return state

    def test_from_state_factory(self):
        state = self._make_state()
        payload = HandoverPayload.from_state(
            state=state,
            from_agent=AgentName.TRIAGE,
            to_agent=AgentName.BILLING,
            reason="Billing intent detected",
        )
        assert payload.from_agent == AgentName.TRIAGE
        assert payload.to_agent == AgentName.BILLING
        assert payload.intent == IntentLabel.BILLING
        assert len(payload.recent_messages) == 1

    def test_from_state_without_intent_raises(self):
        state = ConversationState()  # intent is None
        with pytest.raises(ValueError, match="intent is None"):
            HandoverPayload.from_state(
                state=state,
                from_agent=AgentName.TRIAGE,
                to_agent=AgentName.BILLING,
                reason="test",
            )


# ---------------------------------------------------------------------------
# EscalationPackage
# ---------------------------------------------------------------------------

class TestEscalationPackage:
    def test_valid_escalation_package(self):
        pkg = EscalationPackage(
            priority=EscalationPriority.P2_HIGH,
            summary_bullets=["Customer reported billing error", "Refund requested"],
            core_issue="Customer wants a refund for duplicate charge",
            recommended_team=RecommendedTeam.BILLING_TEAM,
            extracted_entities=ExtractedEntities(customer_id="CLD-00001"),
            full_trace_id="trace-abc-123",
            session_id="session-xyz-456",
            estimated_resolution_time="2–4 business hours",
        )
        assert pkg.priority == EscalationPriority.P2_HIGH
        assert pkg.recommended_team == RecommendedTeam.BILLING_TEAM
