"""
agents/agent_nodes.py
=====================
LangChain agent node implementations for the CloudDash Multi-Agent Support System.

Four agents are defined here, each as a callable that accepts a dict of inputs
compatible with LangGraph's node function signature:

  run_triage_agent(state)           → TriageResult
  run_technical_support_agent(state) → AgentResponse
  run_billing_agent(state)           → AgentResponse
  run_escalation_agent(state)        → AgentResponse  (content = JSON EscalationPackage)

Architecture
------------
Each agent follows the same three-phase pattern (from @langchain-architecture skill):

  1. BUILD  — Assemble the LLM, system prompt, and tool bindings into a chain.
  2. FORMAT — Convert the current ConversationState into a prompt-ready message list.
  3. INVOKE — Call the chain, parse the structured output, and return a typed response.

Key design decisions
---------------------
* **Prompts from YAML, not hardcoded strings** — All system prompts are loaded via
  ``config.get_config().get_agent_prompt()``. This means prompt iteration never
  requires a code change.
* **``with_structured_output`` for Triage** — Gemini's native structured output
  mode is used so the Triage Agent always returns a valid ``TriageResult`` without
  manual JSON parsing. Falls back to JSON parsing on older model versions.
* **``bind_tools`` for specialist agents** — Specialist agents receive their tools
  via LangChain's standard ``llm.bind_tools()`` API so LangGraph can execute them
  in tool-calling loops (Stage 5).
* **Tenacity retry decorator** — Every LLM call is wrapped with exponential-backoff
  retries for transient Gemini API errors (429 / 503).
* **Graceful fallback on parse error** — If the Triage Agent returns malformed JSON
  (rare), a safe ``unknown`` intent result is returned so the graph can route to a
  human, rather than crashing the session.
* **All agent outputs log their trace_id** — Stage 6 will add structured JSON logging;
  the trace_id is threaded through all calls so it can be added with a single pass.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Project root on sys.path (needed when running agent_nodes as __main__)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from config.config_loader import get_config
from models.models import (
    AgentName,
    AgentResponse,
    EscalationPackage,
    EscalationPriority,
    ExtractedEntities,
    IntentLabel,
    MessageRole,
    RecommendedTeam,
    TriageResult,
    UrgencyLevel,
)
from tools.agent_tools import BILLING_TOOLS, TECHNICAL_TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias — state dict passed in from LangGraph nodes (Stage 5)
# ---------------------------------------------------------------------------
AgentState = dict[str, Any]


# ===========================================================================
# LLM factory
# ===========================================================================

def _build_llm(
    agent_name: str,
    *,
    model_override: str | None = None,
    temperature_override: float | None = None,
) -> ChatGoogleGenerativeAI:
    """
    Instantiate a ``ChatGoogleGenerativeAI`` model configured for *agent_name*.

    Parameters are sourced from ``agents_config.yaml`` unless overridden.
    The ``GEMINI_API_KEY`` environment variable must be set.

    Raises
    ------
    EnvironmentError
        If ``GEMINI_API_KEY`` is not set in the environment.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is not set. "
            "Add it to your .env file and reload the application."
        )

    cfg = get_config()
    agent_cfg = cfg.get_agent_config(agent_name)

    model = model_override or cfg.global_settings.llm_model
    temperature = temperature_override if temperature_override is not None else agent_cfg.temperature

    logger.debug(
        "Building LLM for %s: model=%s, temperature=%.1f",
        agent_name,
        model,
        temperature,
    )

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_retries=cfg.global_settings.llm_max_retries,
        timeout=cfg.global_settings.llm_timeout_seconds,
    )


# ---------------------------------------------------------------------------
# Retry decorator — wraps LLM calls for transient API errors
# ---------------------------------------------------------------------------
def _llm_retry(fn):
    """Wrap *fn* with exponential-backoff retry for rate-limit / server errors."""
    return retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )(fn)


def _get_message_role(msg: Any) -> str:
    """Extract and normalise the role of a message (object, dict, or LangChain BaseMessage)."""
    # 1. Try role attribute or dict key (handles ConversationMessage or dicts)
    role = getattr(msg, "role", None)
    if role is None and isinstance(msg, dict):
        role = msg.get("role")
    
    if role is not None:
        if hasattr(role, "value"): # in case it's an Enum
            return str(role.value).lower()
        return str(role).lower()
        
    # 2. Try type attribute (LangChain BaseMessage uses type='human'/'ai'/'system'/'tool')
    msg_type = getattr(msg, "type", None)
    if msg_type is None and isinstance(msg, dict):
        msg_type = msg.get("type")
        
    if msg_type is not None:
        msg_type_str = str(msg_type).lower()
        if msg_type_str == "human":
            return "user"
        if msg_type_str == "ai":
            return "assistant"
        return msg_type_str
        
    return ""


# ---------------------------------------------------------------------------
# Conversation history formatter
# ---------------------------------------------------------------------------
def _format_history(state: AgentState, max_turns: int = 10) -> list[HumanMessage | AIMessage]:
    """
    Convert the last *max_turns* messages from state["messages"] into a list of
    LangChain ``HumanMessage`` / ``AIMessage`` objects for injection into prompts.

    Skips system and tool messages — they are not part of the user-facing dialogue.
    """
    messages = state.get("messages", [])
    recent = messages[-max_turns * 2:]  # *2 because each turn = user + assistant
    lc_messages: list[HumanMessage | AIMessage] = []
    for msg in recent:
        role = _get_message_role(msg)
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content or ""))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content or ""))
        # skip system / tool messages
    return lc_messages


def _latest_user_message(state: AgentState) -> str:
    """Return the content of the most recent user message in state."""
    for msg in reversed(state.get("messages", [])):
        role = _get_message_role(msg)
        if role == "user":
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content", "")
            return content or ""
    return ""


def _trace_id(state: AgentState) -> str:
    """Extract trace_id from state, generating one if absent."""
    return state.get("trace_id") or str(uuid.uuid4())


# ===========================================================================
# Agent 1 — Triage Agent
# ===========================================================================

def _parse_triage_json(raw: str) -> TriageResult:
    """
    Extract and parse the JSON block from the Triage Agent's raw response.

    Handles three formats:
    1. Pure JSON string.
    2. Markdown code block (```json ... ```).
    3. JSON embedded anywhere in a longer string.
    """
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Try direct parse first
    try:
        data = json.loads(clean)
        return TriageResult.model_validate(data)
    except (json.JSONDecodeError, Exception):
        pass

    # Try extracting the first {...} block
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return TriageResult.model_validate(data)
        except Exception:
            pass

    # Ultimate fallback — return safe unknown intent
    logger.warning("Triage Agent returned unparseable output; defaulting to 'unknown' intent.")
    return TriageResult(
        intent=IntentLabel.UNKNOWN,
        confidence=0.0,
        extracted_entities=ExtractedEntities(urgency=UrgencyLevel.MEDIUM),
        reasoning="Failed to parse triage agent output.",
    )


def run_triage_agent(state: AgentState) -> TriageResult:
    trace = _trace_id(state)
    user_msg = _latest_user_message(state)

    if not user_msg:
        logger.warning("[%s] Triage called with no user message.", trace)
        return TriageResult(
            intent=IntentLabel.UNKNOWN,
            confidence=0.0,
            extracted_entities=ExtractedEntities(),
            reasoning="No user message found in conversation state.",
        )

    cfg = get_config()
    system_prompt = cfg.get_agent_prompt("triage_agent")

    # Build standard LLM, then bind the Pydantic model for structured output
    llm = _build_llm("triage_agent", model_override="gemini-3.5-flash")
    structured_llm = llm.with_structured_output(TriageResult)

    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ]

    logger.info("[%s] Triage Agent processing: '%s'", trace, user_msg[:80])

    @_llm_retry
    def _call() -> TriageResult:
        return structured_llm.invoke(prompt_messages)

    try:
        result = _call()
    except Exception as exc:
        logger.warning("[%s] Triage Agent structured parse failed (%s); defaulting to 'unknown'.", trace, exc)
        result = TriageResult(
            intent=IntentLabel.UNKNOWN,
            confidence=0.0,
            extracted_entities=ExtractedEntities(urgency=UrgencyLevel.MEDIUM),
            reasoning="Failed to parse triage agent output.",
        )

    logger.info(
        "[%s] Triage result: intent=%s, confidence=%.2f, customer_id=%s",
        trace,
        result.intent.value,
        result.confidence,
        result.extracted_entities.customer_id,
    )
    return result

# ===========================================================================
# Agent 2 — Technical Support Agent
# ===========================================================================

def run_technical_support_agent(state: AgentState) -> AgentResponse:
    """
    Technical Support Agent — answers technical questions using the FAISS KB.

    Input (from state)
    ------------------
    messages       : list  Conversation history.
    entities       : dict  Entities extracted by Triage (may include customer_id).
    trace_id       : str   Session trace identifier.

    Output
    ------
    AgentResponse
        natural-language answer with source_documents populated from KB citations.
        If the KB has no relevant docs, sets needs_handover=True → escalation_agent.
        If the question is billing-related, sets needs_handover=True → billing_agent.

    Tool binding
    ------------
    The LLM is bound to ``search_technical_knowledge_base`` via ``bind_tools()``.
    In Stage 5, LangGraph's ToolNode will handle the tool execution loop.
    """
    trace = _trace_id(state)
    user_msg = _latest_user_message(state)
    history = _format_history(state)

    cfg = get_config()
    system_prompt = cfg.get_agent_prompt("technical_support_agent")

    # Pro model for technical reasoning; citations require careful reading of KB output
    llm = _build_llm("technical_support_agent")
    llm_with_tools = llm.bind_tools(TECHNICAL_TOOLS)

    # Build prompt — system + rolling history + current user message
    messages = [SystemMessage(content=system_prompt)] + history

    logger.info("[%s] Technical Support Agent processing: '%s'", trace, user_msg[:80])

    @_llm_retry
    def _call():
        return llm_with_tools.invoke(messages)

    ai_msg = _call()
    
    # Safely parse content to avoid "[]" strings when Gemini uses tools
    if isinstance(ai_msg.content, list) and not ai_msg.content:
        response_content = ""
    else:
        response_content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
    # ---- Extract source citations from response ----
    # The agent is instructed to list source IDs in a "Sources:" section
    source_ids: list[str] = []
    src_match = re.search(
        r"(?:Sources?|Citations?|References?)\s*[:\-]\s*((?:[A-Z]{2,4}-\d{3,4}(?:[,\s]+)?)+)",
        response_content,
        re.IGNORECASE,
    )
    if src_match:
        raw_ids = src_match.group(1)
        source_ids = [s.strip().upper() for s in re.split(r"[,\s]+", raw_ids) if s.strip()]

    # ---- Detect handover signals from agent content ----
    needs_handover = False
    target_agent: AgentName | None = None
    handover_reason: str | None = None

    escalation_phrases = [
        "escalate", "I'll escalate", "unable to find specific documentation",
        "cannot resolve", "needs human", "beyond my capabilities",
    ]
    billing_phrases = ["billing", "invoice", "payment", "subscription", "refund"]

    lower_content = response_content.lower()
    if any(p.lower() in lower_content for p in billing_phrases) and "billing" in lower_content:
        # Soft signal — agent may have detected a billing question
        if "hand" in lower_content or "transfer" in lower_content or "billing agent" in lower_content:
            needs_handover = True
            target_agent = AgentName.BILLING
            handover_reason = "Billing-related question detected; routing to Billing Agent."

    if any(p.lower() in lower_content for p in escalation_phrases):
        needs_handover = True
        target_agent = AgentName.ESCALATION
        handover_reason = "Technical Agent could not resolve the issue from the knowledge base."

    logger.info(
        "[%s] Technical Support Agent response ready. Sources: %s, handover: %s",
        trace,
        source_ids,
        needs_handover,
    )

    return AgentResponse(
        agent_name=AgentName.TECHNICAL_SUPPORT,
        content=response_content,
        needs_handover=needs_handover,
        target_agent=target_agent if needs_handover else None,
        handover_reason=handover_reason,
        source_documents=source_ids,
        # ADD tool_calls HERE
        metadata={"trace_id": trace, "tool_calls": getattr(ai_msg, "tool_calls", [])}, 
    )


# ===========================================================================
# Agent 3 — Billing Agent
# ===========================================================================

def run_billing_agent(state: AgentState) -> AgentResponse:
    """
    Billing Agent — handles subscription and invoice queries using SQL tools.

    Input (from state)
    ------------------
    messages       : list  Conversation history.
    entities       : dict  Should contain customer_id if extracted by Triage.
    trace_id       : str   Session trace identifier.

    Output
    ------
    AgentResponse
        Billing response with account/plan details.
        Sets needs_handover=True → escalation_agent for refunds or blocked accounts.
        Sets needs_handover=True → technical_support_agent for technical questions.

    Tool binding
    ------------
    Bound to ``lookup_account_billing_info`` and ``process_plan_upgrade``.
    Temperature is 0.0 for deterministic financial responses.
    """
    trace = _trace_id(state)
    user_msg = _latest_user_message(state)
    history = _format_history(state)

    # Inject extracted customer_id into the context if present
    entities = state.get("entities") or {}
    if isinstance(entities, dict):
        customer_id = entities.get("customer_id")
    else:
        customer_id = getattr(entities, "customer_id", None)

    cfg = get_config()
    system_prompt = cfg.get_agent_prompt("billing_agent")

    # Prepend customer_id to the user message so the agent doesn't have to ask
    augmented_user_msg = user_msg
    if customer_id:
        augmented_user_msg = f"[Customer ID: {customer_id}]\n{user_msg}"

    llm = _build_llm("billing_agent")
    llm_with_tools = llm.bind_tools(BILLING_TOOLS)

    messages = [SystemMessage(content=system_prompt)] + history
    # Replace last human message with the augmented version if customer_id added
    if customer_id and messages:
        messages.append(HumanMessage(content=augmented_user_msg))

    logger.info(
        "[%s] Billing Agent processing for customer_id=%s: '%s'",
        trace,
        customer_id or "UNKNOWN",
        user_msg[:80],
    )

    @_llm_retry
    def _call():
        return llm_with_tools.invoke(messages)

    ai_msg = _call()
    
    # Safely parse content to avoid "[]" strings when Gemini uses tools
    if isinstance(ai_msg.content, list) and not ai_msg.content:
        response_content = ""
    else:
        response_content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
    # ---- Detect handover signals ----
    needs_handover = False
    target_agent: AgentName | None = None
    handover_reason: str | None = None

    lower = response_content.lower()

    refund_signals = ["refund", "reimburse", "money back", "chargeback"]
    technical_signals = ["technical", "error code", "integration issue", "api problem", "monitoring"]
    escalation_signals = ["cannot process", "requires human", "escalat", "billing team", "manual review"]

    if any(p in lower for p in refund_signals):
        needs_handover = True
        target_agent = AgentName.ESCALATION
        handover_reason = "Refund request detected; requires human operator approval."
    elif any(p in lower for p in escalation_signals):
        needs_handover = True
        target_agent = AgentName.ESCALATION
        handover_reason = "Billing issue requires human operator review."
    elif any(p in lower for p in technical_signals) and "technical" in lower:
        if "technical agent" in lower or "technical support" in lower:
            needs_handover = True
            target_agent = AgentName.TECHNICAL_SUPPORT
            handover_reason = "Technical question detected; routing to Technical Support Agent."

    logger.info(
        "[%s] Billing Agent response ready. handover: %s → %s",
        trace,
        needs_handover,
        target_agent.value if target_agent else "none",
    )

    return AgentResponse(
        agent_name=AgentName.BILLING,
        content=response_content,
        needs_handover=needs_handover,
        target_agent=target_agent if needs_handover else None,
        handover_reason=handover_reason,
        # ADD tool_calls HERE
        metadata={"trace_id": trace, "customer_id": customer_id, "tool_calls": getattr(ai_msg, "tool_calls", [])},
    )


# ===========================================================================
# Agent 4 — Escalation Agent
# ===========================================================================

def _build_escalation_package(
    raw_json: str,
    state: AgentState,
    trace: str,
) -> EscalationPackage:
    """
    Parse the Escalation Agent's JSON output into an ``EscalationPackage``.

    Falls back to a safe default P3/senior_support package if parsing fails
    so the handover record is always created, even on LLM output errors.
    """
    entities_raw = state.get("entities") or {}
    if isinstance(entities_raw, dict):
        try:
            entities = ExtractedEntities.model_validate(entities_raw)
        except Exception:
            entities = ExtractedEntities()
    else:
        entities = entities_raw if isinstance(entities_raw, ExtractedEntities) else ExtractedEntities()

    # Clean markdown fences
    clean = re.sub(r"```(?:json)?\s*", "", raw_json).replace("```", "").strip()

    try:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(clean)

        # Map priority string → enum
        priority_map = {
            "P1": EscalationPriority.P1_CRITICAL,
            "P2": EscalationPriority.P2_HIGH,
            "P3": EscalationPriority.P3_MEDIUM,
            "P4": EscalationPriority.P4_LOW,
        }
        priority_str = str(data.get("priority", "P3")).upper().strip()
        priority = priority_map.get(priority_str, EscalationPriority.P3_MEDIUM)

        team_map = {
            "engineering_oncall": RecommendedTeam.ENGINEERING_ONCALL,
            "billing_team": RecommendedTeam.BILLING_TEAM,
            "senior_support": RecommendedTeam.SENIOR_SUPPORT,
            "general_support": RecommendedTeam.GENERAL_SUPPORT,
        }
        team_str = str(data.get("recommended_team", "senior_support")).lower().strip()
        team = team_map.get(team_str, RecommendedTeam.SENIOR_SUPPORT)

        bullets = data.get("summary_bullets", ["No summary available."])
        if not bullets:
            bullets = ["No summary available."]

        return EscalationPackage(
            priority=priority,
            summary_bullets=bullets,
            core_issue=data.get("core_issue", "Unresolved customer issue requiring human review."),
            recommended_team=team,
            extracted_entities=entities,
            full_trace_id=trace,
            session_id=state.get("session_id", str(uuid.uuid4())),
            estimated_resolution_time=data.get("estimated_resolution_time", "2–4 business hours"),
        )

    except Exception as exc:
        logger.warning(
            "[%s] Escalation Agent JSON parse failed (%s) — using safe fallback.",
            trace,
            exc,
        )
        return EscalationPackage(
            priority=EscalationPriority.P3_MEDIUM,
            summary_bullets=["Unable to parse escalation package — manual review required."],
            core_issue="Unresolved customer issue; escalation package generation failed.",
            recommended_team=RecommendedTeam.SENIOR_SUPPORT,
            extracted_entities=entities,
            full_trace_id=trace,
            session_id=state.get("session_id", str(uuid.uuid4())),
            estimated_resolution_time="2–4 business hours",
        )


def run_escalation_agent(state: AgentState) -> AgentResponse:
    trace = _trace_id(state)
    history = _format_history(state, max_turns=20) 

    # Build human-readable transcript for the LLM
    transcript_lines: list[str] = []
    for msg in state.get("messages", []):
        role = getattr(msg, "role", "") or ""
        content = getattr(msg, "content", "") or ""
        agent = getattr(msg, "agent_name", None)
        label = f"[{role.upper()}]" + (f" ({agent})" if agent else "")
        transcript_lines.append(f"{label}: {content}")

    transcript = "\n".join(transcript_lines) if transcript_lines else "No conversation history available."

    # Session context injected into the prompt
    entities_raw = state.get("entities") or {}
    customer_id = (
        entities_raw.get("customer_id")
        if isinstance(entities_raw, dict)
        else getattr(entities_raw, "customer_id", None)
    )

    context_note = (
        f"\nSESSION CONTEXT:\n"
        f"- trace_id: {trace}\n"
        f"- session_id: {state.get('session_id', 'N/A')}\n"
        f"- customer_id: {customer_id or 'Not provided'}\n"
        f"- handover_count: {state.get('handover_count', 0)}\n"
        f"- current_agent: {state.get('current_agent', 'N/A')}\n"
    )

    cfg = get_config()
    system_prompt = cfg.get_agent_prompt("escalation_agent")
    full_prompt = f"{system_prompt}\n{context_note}"

    user_content = (
        f"Please generate the handover package for the following conversation:\n\n"
        f"--- CONVERSATION TRANSCRIPT ---\n{transcript}\n--- END TRANSCRIPT ---"
    )

    # 1. Define llm first
    llm = _build_llm("escalation_agent")
    
    # 2. Then bind structured output
    structured_llm = llm.with_structured_output(EscalationPackage)
    
    messages = [
        SystemMessage(content=full_prompt),
        HumanMessage(content=user_content),
    ]

    logger.info("[%s] Escalation Agent generating handover package.", trace)

    @_llm_retry
    def _call() -> EscalationPackage:
        return structured_llm.invoke(messages)

    try:
        package = _call()
    except Exception as exc:
        logger.warning("[%s] Escalation Agent structured parse failed (%s) — using safe fallback.", trace, exc)
        
        entities_raw = state.get("entities") or {}
        if isinstance(entities_raw, dict):
            try:
                entities = ExtractedEntities.model_validate(entities_raw)
            except Exception:
                entities = ExtractedEntities()
        else:
            entities = entities_raw if isinstance(entities_raw, ExtractedEntities) else ExtractedEntities()

        package = EscalationPackage(
            priority=EscalationPriority.P3_MEDIUM,
            summary_bullets=["Unable to parse escalation package — manual review required."],
            core_issue="Unresolved customer issue; escalation package generation failed.",
            recommended_team=RecommendedTeam.SENIOR_SUPPORT,
            extracted_entities=entities,
            full_trace_id=trace,
            session_id=state.get("session_id", str(uuid.uuid4())),
            estimated_resolution_time="2–4 business hours",
        )

    package_json = package.model_dump_json(indent=2)

    logger.info(
        "[%s] Escalation package created: priority=%s, team=%s",
        trace,
        package.priority.value,
        package.recommended_team.value,
    )

    return AgentResponse(
        agent_name=AgentName.ESCALATION,
        content=(
            f"I've prepared a handover package for our support team. "
            f"A human operator from {package.recommended_team.value.replace('_', ' ').title()} "
            f"will follow up with you shortly (estimated: {package.estimated_resolution_time}).\n\n"
            f"**Reference ID:** {trace}\n"
            f"**Priority:** {package.priority.value}\n\n"
            f"---\n*Internal handover package (not shown to customer):*\n```json\n{package_json}\n```"
        ),
        needs_handover=False,
        target_agent=None,
        metadata={
            "trace_id": trace,
            "escalation_package": package.model_dump(),
            "priority": package.priority.value,
            "recommended_team": package.recommended_team.value,
        },
    )

# ===========================================================================
# Convenience registry — used by Stage 5 LangGraph to dispatch to the right fn
# ===========================================================================

AGENT_REGISTRY: dict[str, callable] = {
    AgentName.TRIAGE.value:            run_triage_agent,
    AgentName.TECHNICAL_SUPPORT.value: run_technical_support_agent,
    AgentName.BILLING.value:           run_billing_agent,
    AgentName.ESCALATION.value:        run_escalation_agent,
}
