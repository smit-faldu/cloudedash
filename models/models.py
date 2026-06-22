"""
models/models.py
================
Core Pydantic v2 data models for the CloudDash Multi-Agent Support System.

These models are the contracts between every layer of the application:
  - API layer  ↔  Agent layer   (via ConversationMessage, AgentResponse)
  - Agent layer ↔ LangGraph     (via ConversationState)
  - Agent layer ↔ Agent layer   (via HandoverPayload, EscalationPackage)

Design notes
------------
* All models inherit from ``CloudDashBaseModel`` which sets strict validation
  and forbids extra fields — this catches config bugs early.
* Enums are used for all bounded sets of values so that type-checkers and
  the runtime both enforce the contract.
* ``ConversationState`` is intentionally a plain Pydantic model here, not a
  TypedDict. The LangGraph ``GraphState`` (Stage 5) will inherit these field
  definitions but wrap them in a TypedDict for graph compatibility.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ===========================================================================
# Shared base model
# ===========================================================================


class CloudDashBaseModel(BaseModel):
    """
    Base for all CloudDash models.
    - ``extra = "forbid"``  → extra fields raise a ValidationError
    - ``str_strip_whitespace`` → strips accidental whitespace from strings
    """

    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
        "populate_by_name": True,
    }


# ===========================================================================
# Enumerations
# ===========================================================================


class IntentLabel(str, Enum):
    """Intent categories produced by the Triage Agent."""

    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    ESCALATION = "escalation"
    UNKNOWN = "unknown"


class AgentName(str, Enum):
    """Canonical agent identifiers that must match keys in agents_config.yaml."""

    TRIAGE = "triage_agent"
    TECHNICAL_SUPPORT = "technical_support_agent"
    BILLING = "billing_agent"
    ESCALATION = "escalation_agent"


class MessageRole(str, Enum):
    """Roles in a conversation message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"           # Tool call results injected into conversation history


class UrgencyLevel(str, Enum):
    """Urgency / severity of a customer issue."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationPriority(str, Enum):
    """Priority codes used in escalation handover packages."""

    P1_CRITICAL = "P1"
    P2_HIGH = "P2"
    P3_MEDIUM = "P3"
    P4_LOW = "P4"


class RecommendedTeam(str, Enum):
    """Internal teams to route escalated conversations to."""

    ENGINEERING_ONCALL = "engineering_oncall"
    BILLING_TEAM = "billing_team"
    SENIOR_SUPPORT = "senior_support"
    GENERAL_SUPPORT = "general_support"


# ===========================================================================
# Building-block models
# ===========================================================================


class ConversationMessage(CloudDashBaseModel):
    """
    A single turn in the conversation history.
    Used to carry messages through the LangGraph state and store them in memory.
    """

    role: MessageRole
    content: str
    agent_name: AgentName | None = Field(
        default=None,
        description="Which agent produced this message (None for user/system messages)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this message was created",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key/value metadata (e.g., token counts, source docs)",
    )

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ConversationMessage.content must not be empty")
        return v


class ExtractedEntities(CloudDashBaseModel):
    """
    Key entities pulled from the conversation by the Triage Agent.
    Fields are nullable because not every turn will contain every entity.
    """

    customer_id: str | None = Field(
        default=None,
        pattern=r"^CLD-\d{5}$",
        description="CloudDash customer identifier, e.g. CLD-00042",
        examples=["CLD-00042"],
    )
    product_area: str | None = Field(
        default=None,
        description="Product area mentioned by the customer (e.g. 'alerting')",
    )
    error_code: str | None = Field(
        default=None,
        description="An error code referenced in the conversation (e.g. ERR-4012)",
    )
    urgency: UrgencyLevel = Field(
        default=UrgencyLevel.MEDIUM,
        description="Estimated urgency of the customer's issue",
    )

    @field_validator("customer_id", mode="before")
    @classmethod
    def _normalise_customer_id(cls, v: str | None) -> str | None:
        """Accept both 'CLD-42' and 'CLD-00042' by zero-padding the numeric part."""
        if v is None:
            return None
        parts = v.strip().upper().split("-")
        if len(parts) == 2 and parts[0] == "CLD" and parts[1].isdigit():
            return f"CLD-{int(parts[1]):05d}"
        return v  # let the regex validator reject invalid formats


# ===========================================================================
# Conversation State
# ===========================================================================


class ConversationState(CloudDashBaseModel):
    """
    The full mutable state of one support session.

    This is the authoritative record that flows through the LangGraph graph.
    In Stage 5 it will be wrapped in a ``TypedDict`` (``GraphState``) for
    LangGraph compatibility, but this Pydantic model remains the validation
    contract.

    Fields
    ------
    session_id :
        Unique identifier for the support session. Immutable once set.
    trace_id :
        UUID that tags every log line for this session, enabling end-to-end tracing.
    messages :
        Ordered history of all conversation turns.
    current_agent :
        The agent node currently handling the conversation.
    intent :
        Latest intent classification produced by the Triage Agent.
    confidence :
        Triage Agent's confidence score for the current intent.
    entities :
        Extracted entities accumulated across the conversation.
    handover_count :
        How many times the conversation has been handed between agents.
    is_escalated :
        True once the Escalation Agent has packaged the conversation.
    created_at :
        UTC timestamp when the session was first created.
    updated_at :
        UTC timestamp of the most recent state mutation.
    """

    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session identifier (UUID4)",
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="End-to-end trace identifier for logging and observability",
    )
    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Ordered conversation history",
    )
    current_agent: AgentName = Field(
        default=AgentName.TRIAGE,
        description="Agent node currently handling this session",
    )
    intent: IntentLabel | None = Field(
        default=None,
        description="Most recent intent label from the Triage Agent",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Triage confidence score [0.0, 1.0]",
    )
    entities: ExtractedEntities = Field(
        default_factory=ExtractedEntities,
        description="Entities extracted and accumulated during the session",
    )
    handover_count: int = Field(
        default=0,
        ge=0,
        description="Number of inter-agent handovers in this session",
    )
    is_escalated: bool = Field(
        default=False,
        description="True once the Escalation Agent has taken over",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: MessageRole,
        content: str,
        agent_name: AgentName | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ConversationState":
        """
        Append a new message to history and bump ``updated_at``.
        Returns *self* for convenient chaining.
        """
        self.messages.append(
            ConversationMessage(
                role=role,
                content=content,
                agent_name=agent_name,
                metadata=metadata or {},
            )
        )
        self.updated_at = datetime.now(timezone.utc)
        return self

    def get_recent_messages(self, n: int = 10) -> list[ConversationMessage]:
        """Return the last *n* messages from history."""
        return self.messages[-n:]

    def increment_handover(self, target_agent: AgentName) -> "ConversationState":
        """Record a handover to *target_agent* and increment the counter."""
        self.current_agent = target_agent
        self.handover_count += 1
        self.updated_at = datetime.now(timezone.utc)
        return self


# ===========================================================================
# Agent Responses
# ===========================================================================


class TriageResult(CloudDashBaseModel):
    """
    Structured output of the Triage Agent for a single user message.
    Parsed from the Triage Agent's JSON response by the graph router.
    """

    intent: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_entities: ExtractedEntities
    reasoning: str = Field(description="One-sentence explanation of the classification")


class AgentResponse(CloudDashBaseModel):
    """
    Standardised response envelope returned by every specialist agent.
    Allows the LangGraph router to consistently read routing decisions
    without parsing free-form text.
    """

    agent_name: AgentName = Field(description="Identity of the responding agent")
    content: str = Field(description="The natural-language reply to send to the user")
    needs_handover: bool = Field(
        default=False,
        description="True if this agent wants to pass control to another agent",
    )
    target_agent: AgentName | None = Field(
        default=None,
        description="Destination agent for handover (required when needs_handover=True)",
    )
    handover_reason: str | None = Field(
        default=None,
        description="Human-readable reason for the handover",
    )
    source_documents: list[str] = Field(
        default_factory=list,
        description="KB document IDs cited in the response (Technical Agent only)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra data (e.g. SQL query results, token usage)",
    )

    @model_validator(mode="after")
    def _target_required_on_handover(self) -> "AgentResponse":
        if self.needs_handover and self.target_agent is None:
            raise ValueError(
                "target_agent must be set when needs_handover is True"
            )
        return self


# ===========================================================================
# Handover Payload
# ===========================================================================


class HandoverPayload(CloudDashBaseModel):
    """
    Context bundle passed between agents during a handover.
    Ensures the receiving agent has full situational awareness without
    re-reading the entire message history.
    """

    from_agent: AgentName
    to_agent: AgentName
    session_id: str
    trace_id: str
    reason: str = Field(description="Why the handover is happening")
    intent: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    entities: ExtractedEntities
    recent_messages: list[ConversationMessage] = Field(
        description="Last N messages for the receiving agent's context window"
    )
    handover_count: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_state(
        cls,
        state: ConversationState,
        from_agent: AgentName,
        to_agent: AgentName,
        reason: str,
        recent_n: int = 10,
    ) -> "HandoverPayload":
        """
        Factory method to build a HandoverPayload directly from a ConversationState.

        Parameters
        ----------
        state :
            The current conversation state.
        from_agent :
            The agent initiating the handover.
        to_agent :
            The agent receiving the handover.
        reason :
            Human-readable explanation for the handover.
        recent_n :
            How many recent messages to include in the payload.
        """
        if state.intent is None:
            raise ValueError(
                "Cannot create HandoverPayload: ConversationState.intent is None. "
                "The Triage Agent must have run before any handover."
            )
        if state.confidence is None:
            raise ValueError(
                "Cannot create HandoverPayload: ConversationState.confidence is None."
            )
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            session_id=state.session_id,
            trace_id=state.trace_id,
            reason=reason,
            intent=state.intent,
            confidence=state.confidence,
            entities=state.entities,
            recent_messages=state.get_recent_messages(recent_n),
            handover_count=state.handover_count,
        )


# ===========================================================================
# Escalation Package
# ===========================================================================


class EscalationPackage(CloudDashBaseModel):
    """
    Final output of the Escalation Agent.
    This is what would be sent to a human operator's ticketing system.
    """

    priority: EscalationPriority
    summary_bullets: list[str] = Field(min_length=1)
    core_issue: str
    recommended_team: RecommendedTeam
    extracted_entities: ExtractedEntities
    full_trace_id: str
    session_id: str
    estimated_resolution_time: str = Field(
        description="Human-readable ETA, e.g. '2–4 business hours'"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
