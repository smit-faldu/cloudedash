"""
tests/test_agents.py
====================
Stage 4 tests for the four CloudDash agent node implementations.

Test strategy
-------------
All Gemini LLM calls are mocked at the ``_build_llm`` factory level so:
- Tests run completely offline (no API key needed, no cost).
- We validate agent logic (prompt loading, JSON parsing, handover signal detection,
  fallback behaviour) without the non-determinism of a live LLM.
- Actual LLM integration is validated manually / in end-to-end Stage 5 tests.

Structure
---------
TestTriageAgentParsing    — JSON parsing, fallback, intent/entity extraction
TestTriageAgentRun        — Full run_triage_agent with mocked LLM
TestTechnicalSupportAgent — run_technical_support_agent with mocked LLM
TestBillingAgent          — run_billing_agent with mocked LLM
TestEscalationAgent       — run_escalation_agent with mocked LLM + package parsing
TestAgentRegistry         — Registry completeness and type checks

Run
---
    pytest tests/test_agents.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from models.models import (
    AgentName,
    AgentResponse,
    EscalationPriority,
    ExtractedEntities,
    IntentLabel,
    MessageRole,
    RecommendedTeam,
    TriageResult,
    UrgencyLevel,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_state(
    user_message: str = "I need help with my account.",
    customer_id: str | None = "CLD-00001",
    session_id: str = "test-session-001",
    trace_id: str = "test-trace-001",
    extra_messages: list | None = None,
) -> dict:
    """Build a minimal state dict for agent node tests."""
    from models.models import ConversationMessage
    messages = [
        ConversationMessage(role=MessageRole.USER, content=user_message),
    ]
    if extra_messages:
        messages = extra_messages + messages
    entities = ExtractedEntities(
        customer_id=customer_id,
        urgency=UrgencyLevel.MEDIUM,
    )
    return {
        "messages": messages,
        "entities": entities.model_dump(),
        "session_id": session_id,
        "trace_id": trace_id,
        "current_agent": AgentName.TRIAGE.value,
        "handover_count": 0,
    }


def _mock_llm_response(content: str) -> MagicMock:
    """Return a mock LLM that returns *content* as an AIMessage."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=content)
    mock_llm.bind_tools.return_value = mock_llm  # bind_tools returns self
    return mock_llm


# ===========================================================================
# Triage Agent — JSON parsing unit tests
# ===========================================================================

class TestTriageAgentParsing:

    def test_parse_clean_json(self):
        from agents.agent_nodes import _parse_triage_json
        raw = json.dumps({
            "intent": "technical",
            "confidence": 0.92,
            "extracted_entities": {
                "customer_id": "CLD-00001",
                "product_area": "alerting",
                "error_code": "ERR-4012",
                "urgency": "high",
            },
            "reasoning": "User reported an error code in the alerting system.",
        })
        result = _parse_triage_json(raw)
        assert isinstance(result, TriageResult)
        assert result.intent == IntentLabel.TECHNICAL
        assert result.confidence == pytest.approx(0.92)
        assert result.extracted_entities.customer_id == "CLD-00001"
        assert result.extracted_entities.error_code == "ERR-4012"

    def test_parse_markdown_fenced_json(self):
        from agents.agent_nodes import _parse_triage_json
        raw = """```json
{
  "intent": "billing",
  "confidence": 0.85,
  "extracted_entities": {
    "customer_id": "CLD-00002",
    "product_area": null,
    "error_code": null,
    "urgency": "medium"
  },
  "reasoning": "Customer asked about invoice."
}
```"""
        result = _parse_triage_json(raw)
        assert result.intent == IntentLabel.BILLING
        assert result.extracted_entities.customer_id == "CLD-00002"

    def test_parse_json_embedded_in_text(self):
        from agents.agent_nodes import _parse_triage_json
        raw = (
            "Here is my classification: "
            '{"intent": "general", "confidence": 0.75, '
            '"extracted_entities": {"customer_id": null, "product_area": null, '
            '"error_code": null, "urgency": "low"}, '
            '"reasoning": "General feature question."}'
            " End."
        )
        result = _parse_triage_json(raw)
        assert result.intent == IntentLabel.GENERAL

    def test_parse_invalid_json_falls_back_to_unknown(self):
        from agents.agent_nodes import _parse_triage_json
        result = _parse_triage_json("This is not JSON at all.")
        assert result.intent == IntentLabel.UNKNOWN
        assert result.confidence == 0.0

    def test_parse_empty_string_falls_back(self):
        from agents.agent_nodes import _parse_triage_json
        result = _parse_triage_json("")
        assert result.intent == IntentLabel.UNKNOWN

    def test_parse_escalation_intent(self):
        from agents.agent_nodes import _parse_triage_json
        raw = json.dumps({
            "intent": "escalation",
            "confidence": 0.99,
            "extracted_entities": {
                "customer_id": "CLD-00004",
                "product_area": None,
                "error_code": None,
                "urgency": "critical",
            },
            "reasoning": "Customer demands a refund.",
        })
        result = _parse_triage_json(raw)
        assert result.intent == IntentLabel.ESCALATION
        assert result.extracted_entities.urgency == UrgencyLevel.CRITICAL

    def test_parse_unknown_intent(self):
        from agents.agent_nodes import _parse_triage_json
        raw = json.dumps({
            "intent": "unknown",
            "confidence": 0.30,
            "extracted_entities": {
                "customer_id": None,
                "product_area": None,
                "error_code": None,
                "urgency": "medium",
            },
            "reasoning": "Cannot determine intent.",
        })
        result = _parse_triage_json(raw)
        assert result.intent == IntentLabel.UNKNOWN


# ===========================================================================
# Triage Agent — full run tests
# ===========================================================================

class TestTriageAgentRun:

    @patch("agents.agent_nodes._build_llm")
    def test_run_returns_triage_result(self, mock_build: MagicMock):
        from agents.agent_nodes import run_triage_agent
        mock_build.return_value = _mock_llm_response(json.dumps({
            "intent": "technical",
            "confidence": 0.90,
            "extracted_entities": {
                "customer_id": "CLD-00001",
                "product_area": "monitoring",
                "error_code": "ERR-4012",
                "urgency": "high",
            },
            "reasoning": "Error code in monitoring.",
        }))
        result = run_triage_agent(_make_state("I keep getting ERR-4012"))
        assert isinstance(result, TriageResult)
        assert result.intent == IntentLabel.TECHNICAL

    @patch("agents.agent_nodes._build_llm")
    def test_run_returns_billing_intent(self, mock_build: MagicMock):
        from agents.agent_nodes import run_triage_agent
        mock_build.return_value = _mock_llm_response(json.dumps({
            "intent": "billing",
            "confidence": 0.88,
            "extracted_entities": {
                "customer_id": "CLD-00002",
                "product_area": None,
                "error_code": None,
                "urgency": "medium",
            },
            "reasoning": "Invoice question.",
        }))
        result = run_triage_agent(_make_state("What is my current invoice?"))
        assert result.intent == IntentLabel.BILLING

    @patch("agents.agent_nodes._build_llm")
    def test_run_handles_empty_state_messages(self, mock_build: MagicMock):
        """Empty messages should return UNKNOWN intent without crashing."""
        from agents.agent_nodes import run_triage_agent
        mock_build.return_value = _mock_llm_response("")
        result = run_triage_agent({"messages": [], "trace_id": "t1"})
        assert result.intent == IntentLabel.UNKNOWN

    @patch("agents.agent_nodes._build_llm")
    def test_run_extracts_customer_id(self, mock_build: MagicMock):
        from agents.agent_nodes import run_triage_agent
        mock_build.return_value = _mock_llm_response(json.dumps({
            "intent": "billing",
            "confidence": 0.80,
            "extracted_entities": {
                "customer_id": "CLD-00003",
                "product_area": None,
                "error_code": None,
                "urgency": "medium",
            },
            "reasoning": "Billing query with customer ID.",
        }))
        result = run_triage_agent(_make_state("CLD-00003 — I need help with billing"))
        assert result.extracted_entities.customer_id == "CLD-00003"

    @patch("agents.agent_nodes._build_llm")
    def test_run_gracefully_handles_llm_error(self, mock_build: MagicMock):
        """LLM that always raises should not crash — after retries, re-raises."""
        from agents.agent_nodes import run_triage_agent
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM quota exceeded")
        mock_build.return_value = mock_llm
        with pytest.raises(RuntimeError, match="LLM quota exceeded"):
            run_triage_agent(_make_state("any message"))


# ===========================================================================
# Technical Support Agent
# ===========================================================================

class TestTechnicalSupportAgent:

    @patch("agents.agent_nodes._build_llm")
    def test_run_returns_agent_response(self, mock_build: MagicMock):
        from agents.agent_nodes import run_technical_support_agent
        content = (
            "The ERR-4012 error occurs when AWS credentials expire. "
            "Steps: 1. Go to Settings. 2. Re-authenticate.\n"
            "Sources: TS-001, FAQ-003"
        )
        mock_build.return_value = _mock_llm_response(content)
        result = run_technical_support_agent(_make_state("I get ERR-4012"))
        assert isinstance(result, AgentResponse)
        assert result.agent_name == AgentName.TECHNICAL_SUPPORT

    @patch("agents.agent_nodes._build_llm")
    def test_extracts_source_citations(self, mock_build: MagicMock):
        from agents.agent_nodes import run_technical_support_agent
        content = "Steps to fix: ...\nSources: TS-001, API-002"
        mock_build.return_value = _mock_llm_response(content)
        result = run_technical_support_agent(_make_state("ERR-4012 issue"))
        assert "TS-001" in result.source_documents
        assert "API-002" in result.source_documents

    @patch("agents.agent_nodes._build_llm")
    def test_source_citations_from_references_section(self, mock_build: MagicMock):
        from agents.agent_nodes import run_technical_support_agent
        content = "The answer is here.\n\nReferences: FAQ-001, TS-003"
        mock_build.return_value = _mock_llm_response(content)
        result = run_technical_support_agent(_make_state("cloud monitoring"))
        assert "FAQ-001" in result.source_documents

    @patch("agents.agent_nodes._build_llm")
    def test_escalation_signal_sets_handover(self, mock_build: MagicMock):
        from agents.agent_nodes import run_technical_support_agent
        content = (
            "I'm unable to find specific documentation for this issue. "
            "I'll escalate this to our specialist team."
        )
        mock_build.return_value = _mock_llm_response(content)
        result = run_technical_support_agent(_make_state("obscure custom question"))
        assert result.needs_handover is True
        assert result.target_agent == AgentName.ESCALATION

    @patch("agents.agent_nodes._build_llm")
    def test_no_handover_for_clean_answer(self, mock_build: MagicMock):
        from agents.agent_nodes import run_technical_support_agent
        content = (
            "CloudDash supports AWS, GCP, and Azure.\n"
            "Sources: FAQ-002"
        )
        mock_build.return_value = _mock_llm_response(content)
        result = run_technical_support_agent(_make_state("which clouds?"))
        assert result.needs_handover is False
        assert result.target_agent is None

    @patch("agents.agent_nodes._build_llm")
    def test_response_contains_agent_name(self, mock_build: MagicMock):
        from agents.agent_nodes import run_technical_support_agent
        mock_build.return_value = _mock_llm_response("Some technical answer.")
        result = run_technical_support_agent(_make_state("test"))
        assert result.agent_name == AgentName.TECHNICAL_SUPPORT

    @patch("agents.agent_nodes._build_llm")
    def test_metadata_contains_trace_id(self, mock_build: MagicMock):
        from agents.agent_nodes import run_technical_support_agent
        mock_build.return_value = _mock_llm_response("Answer.")
        state = _make_state(trace_id="my-trace-xyz")
        result = run_technical_support_agent(state)
        assert result.metadata.get("trace_id") == "my-trace-xyz"


# ===========================================================================
# Billing Agent
# ===========================================================================

class TestBillingAgent:

    @patch("agents.agent_nodes._build_llm")
    def test_run_returns_agent_response(self, mock_build: MagicMock):
        from agents.agent_nodes import run_billing_agent
        content = "Your current plan is Growth ($149/month). Next billing: 2026-07-01."
        mock_build.return_value = _mock_llm_response(content)
        result = run_billing_agent(_make_state("What plan am I on?"))
        assert isinstance(result, AgentResponse)
        assert result.agent_name == AgentName.BILLING

    @patch("agents.agent_nodes._build_llm")
    def test_refund_signal_triggers_escalation(self, mock_build: MagicMock):
        from agents.agent_nodes import run_billing_agent
        content = (
            "I cannot process this refund autonomously. "
            "This requires human approval — I'll escalate this now."
        )
        mock_build.return_value = _mock_llm_response(content)
        result = run_billing_agent(_make_state("I want a refund for last month"))
        assert result.needs_handover is True
        assert result.target_agent == AgentName.ESCALATION

    @patch("agents.agent_nodes._build_llm")
    def test_clean_billing_answer_no_handover(self, mock_build: MagicMock):
        from agents.agent_nodes import run_billing_agent
        content = "Your Growth plan renews on 2026-07-15. Your last invoice was paid."
        mock_build.return_value = _mock_llm_response(content)
        result = run_billing_agent(_make_state("when does my plan renew?"))
        assert result.needs_handover is False

    @patch("agents.agent_nodes._build_llm")
    def test_metadata_includes_customer_id(self, mock_build: MagicMock):
        from agents.agent_nodes import run_billing_agent
        mock_build.return_value = _mock_llm_response("Your subscription is active.")
        result = run_billing_agent(_make_state("billing query", customer_id="CLD-00005"))
        assert result.metadata.get("customer_id") == "CLD-00005"

    @patch("agents.agent_nodes._build_llm")
    def test_escalation_signal_triggers_handover(self, mock_build: MagicMock):
        from agents.agent_nodes import run_billing_agent
        content = "This requires manual review by the billing team. I cannot process this automatically."
        mock_build.return_value = _mock_llm_response(content)
        result = run_billing_agent(_make_state("my account is wrong"))
        assert result.needs_handover is True

    @patch("agents.agent_nodes._build_llm")
    def test_agent_name_is_billing(self, mock_build: MagicMock):
        from agents.agent_nodes import run_billing_agent
        mock_build.return_value = _mock_llm_response("Billing info here.")
        result = run_billing_agent(_make_state("test"))
        assert result.agent_name == AgentName.BILLING


# ===========================================================================
# Escalation Agent
# ===========================================================================

class TestEscalationAgent:

    def _valid_escalation_json(self, priority: str = "P2") -> str:
        return json.dumps({
            "priority": priority,
            "summary_bullets": [
                "Customer reported billing discrepancy",
                "Requested refund of $149",
                "Account CLD-00001 verified active",
            ],
            "core_issue": "Customer requesting refund not approved autonomously.",
            "recommended_team": "billing_team",
            "extracted_entities": {
                "customer_id": "CLD-00001",
                "product_area": None,
                "error_code": None,
                "urgency": "high",
            },
            "full_trace_id": "test-trace-001",
            "estimated_resolution_time": "4–8 business hours",
        })

    @patch("agents.agent_nodes._build_llm")
    def test_run_returns_agent_response(self, mock_build: MagicMock):
        from agents.agent_nodes import run_escalation_agent
        mock_build.return_value = _mock_llm_response(self._valid_escalation_json())
        result = run_escalation_agent(_make_state("I need a refund"))
        assert isinstance(result, AgentResponse)
        assert result.agent_name == AgentName.ESCALATION

    @patch("agents.agent_nodes._build_llm")
    def test_needs_handover_is_always_false(self, mock_build: MagicMock):
        """Escalation is the terminal node — never hands over further."""
        from agents.agent_nodes import run_escalation_agent
        mock_build.return_value = _mock_llm_response(self._valid_escalation_json())
        result = run_escalation_agent(_make_state("refund request"))
        assert result.needs_handover is False
        assert result.target_agent is None

    @patch("agents.agent_nodes._build_llm")
    def test_metadata_contains_escalation_package(self, mock_build: MagicMock):
        from agents.agent_nodes import run_escalation_agent
        mock_build.return_value = _mock_llm_response(self._valid_escalation_json("P2"))
        result = run_escalation_agent(_make_state("account issue"))
        assert "escalation_package" in result.metadata
        pkg = result.metadata["escalation_package"]
        assert pkg["priority"] == "P2"
        assert pkg["recommended_team"] == "billing_team"

    @patch("agents.agent_nodes._build_llm")
    def test_p1_priority_routes_to_engineering(self, mock_build: MagicMock):
        from agents.agent_nodes import run_escalation_agent
        p1_json = json.dumps({
            "priority": "P1",
            "summary_bullets": ["Complete service outage"],
            "core_issue": "Platform unavailable.",
            "recommended_team": "engineering_oncall",
            "extracted_entities": {
                "customer_id": "CLD-00001",
                "product_area": "monitoring",
                "error_code": "ERR-5001",
                "urgency": "critical",
            },
            "full_trace_id": "trace-p1",
            "estimated_resolution_time": "1 hour",
        })
        mock_build.return_value = _mock_llm_response(p1_json)
        result = run_escalation_agent(_make_state("service is completely down"))
        pkg = result.metadata["escalation_package"]
        assert pkg["priority"] == "P1"
        assert pkg["recommended_team"] == "engineering_oncall"

    @patch("agents.agent_nodes._build_llm")
    def test_invalid_json_falls_back_gracefully(self, mock_build: MagicMock):
        """Unparseable JSON should produce a safe P3/senior_support package."""
        from agents.agent_nodes import run_escalation_agent
        mock_build.return_value = _mock_llm_response("I cannot generate a package right now.")
        result = run_escalation_agent(_make_state("urgent issue"))
        assert result.agent_name == AgentName.ESCALATION
        pkg = result.metadata["escalation_package"]
        assert pkg["priority"] == "P3"
        assert pkg["recommended_team"] == "senior_support"

    @patch("agents.agent_nodes._build_llm")
    def test_content_contains_reference_id(self, mock_build: MagicMock):
        from agents.agent_nodes import run_escalation_agent
        mock_build.return_value = _mock_llm_response(self._valid_escalation_json())
        state = _make_state(trace_id="REF-TEST-999")
        result = run_escalation_agent(state)
        assert "REF-TEST-999" in result.content

    @patch("agents.agent_nodes._build_llm")
    def test_content_contains_human_friendly_message(self, mock_build: MagicMock):
        from agents.agent_nodes import run_escalation_agent
        mock_build.return_value = _mock_llm_response(self._valid_escalation_json())
        result = run_escalation_agent(_make_state("escalated issue"))
        # Should address the customer, not just dump JSON
        assert "follow up" in result.content.lower() or "shortly" in result.content.lower()


# ===========================================================================
# Escalation package builder — unit tests
# ===========================================================================

class TestBuildEscalationPackage:

    def test_valid_p2_billing_team(self):
        from agents.agent_nodes import _build_escalation_package
        raw = json.dumps({
            "priority": "P2",
            "summary_bullets": ["Customer reported overdue invoice"],
            "core_issue": "Overdue invoice not resolved.",
            "recommended_team": "billing_team",
            "extracted_entities": {
                "customer_id": "CLD-00008",
                "product_area": None,
                "error_code": None,
                "urgency": "high",
            },
            "full_trace_id": "t1",
            "estimated_resolution_time": "4 hours",
        })
        state = {"session_id": "s1", "entities": {}}
        pkg = _build_escalation_package(raw, state, "t1")
        assert pkg.priority == EscalationPriority.P2_HIGH
        assert pkg.recommended_team == RecommendedTeam.BILLING_TEAM

    def test_markdown_fenced_json(self):
        from agents.agent_nodes import _build_escalation_package
        raw = "```json\n" + json.dumps({
            "priority": "P4",
            "summary_bullets": ["General question"],
            "core_issue": "Feature request.",
            "recommended_team": "general_support",
            "extracted_entities": {
                "customer_id": None,
                "product_area": "dashboard",
                "error_code": None,
                "urgency": "low",
            },
            "full_trace_id": "t2",
            "estimated_resolution_time": "1–2 business days",
        }) + "\n```"
        pkg = _build_escalation_package(raw, {"session_id": "s2", "entities": {}}, "t2")
        assert pkg.priority == EscalationPriority.P4_LOW
        assert pkg.recommended_team == RecommendedTeam.GENERAL_SUPPORT

    def test_garbage_input_produces_safe_fallback(self):
        from agents.agent_nodes import _build_escalation_package
        pkg = _build_escalation_package(
            "completely unparseable ###", {"session_id": "s3", "entities": {}}, "t3"
        )
        assert pkg.priority == EscalationPriority.P3_MEDIUM
        assert pkg.recommended_team == RecommendedTeam.SENIOR_SUPPORT
        assert len(pkg.summary_bullets) >= 1


# ===========================================================================
# Agent Registry
# ===========================================================================

class TestAgentRegistry:

    def test_registry_has_all_four_agents(self):
        from agents.agent_nodes import AGENT_REGISTRY
        expected = {
            AgentName.TRIAGE.value,
            AgentName.TECHNICAL_SUPPORT.value,
            AgentName.BILLING.value,
            AgentName.ESCALATION.value,
        }
        assert set(AGENT_REGISTRY.keys()) == expected

    def test_registry_values_are_callables(self):
        from agents.agent_nodes import AGENT_REGISTRY
        for name, fn in AGENT_REGISTRY.items():
            assert callable(fn), f"AGENT_REGISTRY['{name}'] is not callable"

    def test_triage_function_is_correct(self):
        from agents.agent_nodes import AGENT_REGISTRY, run_triage_agent
        assert AGENT_REGISTRY[AgentName.TRIAGE.value] is run_triage_agent

    def test_technical_function_is_correct(self):
        from agents.agent_nodes import AGENT_REGISTRY, run_technical_support_agent
        assert AGENT_REGISTRY[AgentName.TECHNICAL_SUPPORT.value] is run_technical_support_agent

    def test_billing_function_is_correct(self):
        from agents.agent_nodes import AGENT_REGISTRY, run_billing_agent
        assert AGENT_REGISTRY[AgentName.BILLING.value] is run_billing_agent

    def test_escalation_function_is_correct(self):
        from agents.agent_nodes import AGENT_REGISTRY, run_escalation_agent
        assert AGENT_REGISTRY[AgentName.ESCALATION.value] is run_escalation_agent


# ===========================================================================
# Format history helper
# ===========================================================================

class TestFormatHistory:

    def test_format_history_returns_lc_messages(self):
        from agents.agent_nodes import _format_history
        from models.models import ConversationMessage
        state = {
            "messages": [
                ConversationMessage(role=MessageRole.USER, content="Hello"),
                ConversationMessage(role=MessageRole.ASSISTANT, content="Hi there"),
            ]
        }
        result = _format_history(state)
        assert len(result) == 2

    def test_format_history_skips_system_messages(self):
        from agents.agent_nodes import _format_history
        from models.models import ConversationMessage
        state = {
            "messages": [
                ConversationMessage(role=MessageRole.SYSTEM, content="System message"),
                ConversationMessage(role=MessageRole.USER, content="User message"),
            ]
        }
        result = _format_history(state)
        assert len(result) == 1

    def test_format_history_respects_max_turns(self):
        from agents.agent_nodes import _format_history
        from models.models import ConversationMessage
        msgs = [
            ConversationMessage(role=MessageRole.USER, content=f"msg {i}")
            for i in range(30)
        ]
        state = {"messages": msgs}
        result = _format_history(state, max_turns=3)
        assert len(result) <= 6  # 3 turns * 2 messages each

    def test_latest_user_message_returns_last_user_msg(self):
        from agents.agent_nodes import _latest_user_message
        from models.models import ConversationMessage
        state = {
            "messages": [
                ConversationMessage(role=MessageRole.USER, content="First question"),
                ConversationMessage(role=MessageRole.ASSISTANT, content="Answer"),
                ConversationMessage(role=MessageRole.USER, content="Follow-up question"),
            ]
        }
        result = _latest_user_message(state)
        assert result == "Follow-up question"

    def test_latest_user_message_empty_state(self):
        from agents.agent_nodes import _latest_user_message
        assert _latest_user_message({"messages": []}) == ""

    def test_format_history_supports_lc_objects(self):
        from agents.agent_nodes import _format_history
        from langchain_core.messages import HumanMessage, AIMessage
        state = {
            "messages": [
                HumanMessage(content="Hello from human"),
                AIMessage(content="Hi from AI"),
            ]
        }
        result = _format_history(state)
        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello from human"
        assert isinstance(result[1], AIMessage)
        assert result[1].content == "Hi from AI"

    def test_latest_user_message_supports_lc_objects(self):
        from agents.agent_nodes import _latest_user_message
        from langchain_core.messages import HumanMessage, AIMessage
        state = {
            "messages": [
                HumanMessage(content="First question"),
                AIMessage(content="Answer"),
                HumanMessage(content="Follow-up question"),
            ]
        }
        result = _latest_user_message(state)
        assert result == "Follow-up question"
