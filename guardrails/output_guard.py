"""
guardrails/output_guard.py
==========================
Output guardrail for the CloudDash Multi-Agent Support System.

Responsibility
--------------
This module runs AFTER a specialist agent (Technical Support or Billing)
produces its response, before the response is returned to the user.

It addresses two hallucination risk profiles that are specific to this system:

Billing Agent risks
  - Quoting pricing not in the canonical plan table
    (Starter $49, Growth $149, Scale $499, Enterprise custom)
  - Claiming a refund can be processed autonomously
  - Stating a billing cycle that contradicts policy
  - Inventing plan features or discounts not present in the knowledge base

Technical Support Agent risks
  - Citing a KB document ID that doesn't exist in the retrieved context
  - Inventing CloudDash product features not mentioned in source documents
  - Providing error codes or API endpoints that contradict the KB

Design
------
The output guardrail uses a two-tier approach:

  Tier 1 — Rule-based (fast, zero cost, always runs):
    * Regex checks for known incorrect pricing patterns
    * Source citation verification (do cited IDs exist in retrieved docs?)
    * Refund autonomy detection for the Billing Agent

  Tier 2 — LLM-as-judge (optional, slower, higher accuracy):
    * A small, focused prompt asks a separate LLM call whether the response
      contradicts the retrieved context.
    * Only triggered when Tier 1 passes but the response confidence is low,
      or when ``use_llm_judge=True`` is explicitly set.
    * Uses ``gemini-3.5-flash`` (fast, cheap) as the judge — NOT the same
      model that produced the response being evaluated.

Return contract
---------------
``check_output(agent_name, response_content, context, trace_id, ...)``
returns a ``GuardResult``:
  - ``passed=True``  → response is safe to return to user
  - ``passed=False`` → trigger escalation fallback

When ``passed=False``, the caller should:
  1. Log the event via ``log_guardrail_triggered``
  2. Route to the Escalation Agent instead of returning the flagged response
  3. NOT show the flagged content to the user
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from guardrails.input_guard import GuardResult  # re-use the same result type

logger = logging.getLogger(__name__)


# ===========================================================================
# Canonical facts — the source of truth for Tier 1 rule checks
# ===========================================================================

# Canonical CloudDash plan pricing (USD per month, from agents_config.yaml billing policy)
CANONICAL_PLANS: dict[str, float] = {
    "starter":    49.0,
    "growth":    149.0,
    "scale":     499.0,
    "enterprise": -1.0,   # custom pricing — any number is technically valid
}

# Regex to catch dollar amounts mentioned alongside plan names.
# Matches: "Growth plan costs $149", "Starter: $49/month", "Growth - $149"
# Uses a flexible gap (up to 50 chars) to find the price after the plan name.
_PRICE_MENTION_RE = re.compile(
    r"(?P<plan>starter|growth|scale|enterprise)(?:[^\n]{0,60}?)\$\s*(?P<amount>\d[\d,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

# Regex to find cited KB source IDs (e.g. TS-001, FAQ-003, API-012, BIL-005)
_CITATION_RE = re.compile(
    r"\b(?:TS|FAQ|API|BIL|ARCH|ERR)-\d{3,4}\b",
    re.IGNORECASE,
)

# Phrases indicating the Billing Agent claims to process a refund autonomously
_AUTONOMOUS_REFUND_RE = re.compile(
    r"\b(I\s+have\s+(processed|issued|approved|completed)\s+"
    r"|processing\s+your\s+refund"
    r"|refund\s+(has\s+been\s+|was\s+)?(processed|issued|approved|applied|credited)"
    r"|credited\s+\$[\d,]+\s+back"
    r"|your\s+refund\s+of\s+\$[\d,]+)\b",
    re.IGNORECASE,
)

# Known valid KB source ID prefixes
_VALID_SOURCE_PREFIXES = frozenset({"TS", "FAQ", "API", "BIL", "ARCH", "ERR"})

# Maximum number of chars checked in a response for performance
_MAX_CHECK_LENGTH = 8_000


# ===========================================================================
# Tier 1 — Rule-based checks
# ===========================================================================


def _check_billing_pricing(response: str, trace_id: str) -> GuardResult | None:
    """
    Verify that any plan price mentioned in the Billing Agent's response
    matches the canonical pricing table.

    Returns a blocked ``GuardResult`` if a mismatch is found, else ``None``.
    """
    for match in _PRICE_MENTION_RE.finditer(response[:_MAX_CHECK_LENGTH]):
        plan = match.group("plan").lower()
        raw_amount = match.group("amount").replace(",", "")
        try:
            mentioned_price = float(raw_amount)
        except ValueError:
            continue

        canonical = CANONICAL_PLANS.get(plan, None)
        if canonical is None:
            continue   # Unknown plan — let LLM judge handle

        if canonical == -1.0:
            continue   # Enterprise = custom pricing, skip

        # Allow a small tolerance for cent rounding ($149 vs $149.00)
        if abs(mentioned_price - canonical) > 0.5:
            logger.warning(
                "Output guardrail: billing price mismatch detected",
                extra={
                    "event": "guardrail_triggered",
                    "guard_name": "output_guard_billing_price",
                    "trace_id": trace_id,
                    "plan": plan,
                    "mentioned_price": mentioned_price,
                    "canonical_price": canonical,
                    "flagged": True,
                },
            )
            return GuardResult.blocked(
                flag_reason=f"billing_price_hallucination: {plan}=${mentioned_price} (canonical=${canonical})",
                safe_reply=(
                    "I need to verify some details before providing billing information. "
                    "Let me connect you with our billing team for accurate pricing."
                ),
                severity="high",
                plan=plan,
                mentioned_price=mentioned_price,
                canonical_price=canonical,
            )
    return None


def _check_autonomous_refund(response: str, trace_id: str) -> GuardResult | None:
    """
    Detect if the Billing Agent claims to have processed a refund autonomously.
    Per billing policy, refunds always require human approval.
    """
    match = _AUTONOMOUS_REFUND_RE.search(response[:_MAX_CHECK_LENGTH])
    if match:
        logger.error(
            "Output guardrail: autonomous refund claim detected",
            extra={
                "event": "guardrail_triggered",
                "guard_name": "output_guard_autonomous_refund",
                "trace_id": trace_id,
                "matched_text": match.group()[:80],
                "flagged": True,
            },
        )
        return GuardResult.blocked(
            flag_reason="autonomous_refund_claim",
            safe_reply=(
                "Refunds require manual review by our billing team. "
                "I'm escalating your request to a human operator who can "
                "process your refund directly."
            ),
            severity="critical",
            matched_text=match.group()[:80],
        )
    return None


def _check_source_citations(
    response: str,
    retrieved_doc_ids: list[str],
    trace_id: str,
) -> GuardResult | None:
    """
    Verify that every KB source ID cited in the Technical Support response
    was actually present in the retrieved context passed to the agent.

    If the agent cites a document that wasn't retrieved, it may have hallucinated
    the citation (or the citation is malformed).

    Parameters
    ----------
    response
        The agent's full response text.
    retrieved_doc_ids
        List of document IDs that were actually returned by the FAISS retriever
        and passed to the agent as context.
    trace_id
        Session trace ID.
    """
    if not retrieved_doc_ids:
        # No context was provided — we can't validate citations
        return None

    cited_ids = set(_CITATION_RE.findall(response[:_MAX_CHECK_LENGTH]))
    if not cited_ids:
        # Agent didn't cite any sources — suspicious but not a hard block
        logger.info(
            "Output guardrail: no source citations found in technical response",
            extra={
                "event": "guardrail_no_citations",
                "trace_id": trace_id,
                "flagged": False,
            },
        )
        return None

    # Normalise: upper-case all IDs for comparison
    cited_upper = {c.upper() for c in cited_ids}
    retrieved_upper = {d.upper() for d in retrieved_doc_ids}

    phantom_citations = cited_upper - retrieved_upper
    if phantom_citations:
        logger.warning(
            "Output guardrail: phantom KB citations detected",
            extra={
                "event": "guardrail_triggered",
                "guard_name": "output_guard_phantom_citations",
                "trace_id": trace_id,
                "phantom_citations": list(phantom_citations),
                "retrieved_doc_ids": retrieved_doc_ids,
                "flagged": True,
            },
        )
        return GuardResult.blocked(
            flag_reason=f"phantom_kb_citations: {phantom_citations}",
            safe_reply=(
                "I'm unable to verify the source of some information in my response. "
                "Let me escalate this to our specialist team to give you an accurate answer."
            ),
            severity="high",
            phantom_citations=list(phantom_citations),
        )
    return None


# ===========================================================================
# Tier 2 — LLM-as-judge (optional)
# ===========================================================================

_LLM_JUDGE_PROMPT = """\
You are a strict quality-assurance judge for a B2B SaaS customer support system.

Your task: determine whether the AGENT RESPONSE contains any factual claims that
CONTRADICT or are NOT SUPPORTED BY the RETRIEVED CONTEXT below.

Focus ONLY on:
- Pricing or plan details
- Product features or capabilities
- Policy statements (e.g. refund rules)
- Technical specifications, error codes, or API endpoints

Do NOT flag:
- Stylistic differences
- Reordering of information
- Minor paraphrasing that preserves meaning
- General courteous language

RETRIEVED CONTEXT (what the agent was given):
---
{context}
---

AGENT RESPONSE (what the agent said):
---
{response}
---

Respond with EXACTLY ONE of:
  PASS — the response is factually consistent with the context, OR the response
         appropriately says it cannot answer because information is missing.
  FAIL: <one-sentence reason> — the response contains a specific factual claim
        that contradicts or is absent from the context.

Your verdict (PASS or FAIL: ...):"""


def _llm_judge(
    response: str,
    context: str,
    trace_id: str,
) -> GuardResult:
    """
    Use a small, fast LLM to judge whether the agent response is grounded
    in the retrieved context.

    Only called when ``use_llm_judge=True``. Falls back to ``GuardResult.ok()``
    if the API key is not set or the call fails — the rule-based tier already
    ran, so a judge failure degrades gracefully.

    Returns
    -------
    GuardResult
        ``passed=True`` if the judge says PASS or if the judge call fails.
        ``passed=False`` with ``flag_reason`` if the judge says FAIL.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning(
            "LLM judge skipped: no API key available",
            extra={"trace_id": trace_id, "event": "llm_judge_skipped"},
        )
        return GuardResult.ok()

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        model = (
            os.environ.get("GEMINI_MODEL_LLM_JUDGE") or
            os.environ.get("GEMINI_MODEL_JUDGE") or
            os.environ.get("GEMINI_MODEL_DEFAULT") or
            os.environ.get("GEMINI_MODEL") or
            "gemini-3.5-flash"
        )

        judge_llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.0,
            max_retries=2,
        )

        prompt = _LLM_JUDGE_PROMPT.format(
            context=context[:3_000],
            response=response[:2_000],
        )

        ai_msg = judge_llm.invoke([HumanMessage(content=prompt)])
        verdict = ai_msg.content.strip() if isinstance(ai_msg.content, str) else ""

        logger.info(
            "LLM judge verdict",
            extra={
                "event": "llm_judge_result",
                "trace_id": trace_id,
                "verdict_preview": verdict[:120],
            },
        )

        if verdict.upper().startswith("FAIL"):
            reason = verdict[5:].strip(" :")
            return GuardResult.blocked(
                flag_reason=f"llm_judge_hallucination: {reason}",
                safe_reply=(
                    "I want to make sure I give you accurate information. "
                    "Let me connect you with a specialist who can verify the details."
                ),
                severity="high",
                judge_verdict=verdict[:200],
            )

        return GuardResult.ok()

    except Exception as exc:
        logger.warning(
            "LLM judge call failed — defaulting to PASS",
            extra={
                "event": "llm_judge_error",
                "trace_id": trace_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        return GuardResult.ok()   # graceful degradation


# ===========================================================================
# Public API
# ===========================================================================


def check_output(
    agent_name: str,
    response_content: str,
    trace_id: str = "",
    retrieved_doc_ids: list[str] | None = None,
    retrieved_context: str = "",
    use_llm_judge: bool = False,
    session_id: str | None = None,
) -> GuardResult:
    """
    Run all output guardrail checks for a specialist agent's response.

    Parameters
    ----------
    agent_name
        Which agent produced the response: ``"billing_agent"`` or
        ``"technical_support_agent"``.
    response_content
        The full text of the agent's response.
    trace_id
        Session trace ID for logging.
    retrieved_doc_ids
        Document IDs retrieved from the FAISS KB and passed to the Technical
        Support Agent. Used to validate source citations.
    retrieved_context
        The raw retrieved KB text passed to the agent (used by the LLM judge).
    use_llm_judge
        If True, run the Tier 2 LLM-as-judge after Tier 1 passes.
        Defaults to False to keep the fast path zero-cost.
    session_id
        Optional session identifier for logging.

    Returns
    -------
    GuardResult
        ``passed=True`` → safe to return to user.
        ``passed=False`` → do NOT return ``response_content`` to user;
                           route to Escalation Agent instead.
    """
    if not response_content.strip():
        return GuardResult.blocked(
            flag_reason="empty_response",
            safe_reply=(
                "I wasn't able to generate a response. Please try again or "
                "contact support@clouddash.io if the issue persists."
            ),
            severity="medium",
        )

    # ---- Billing Agent checks ----
    if "billing" in agent_name.lower():
        result = _check_billing_pricing(response_content, trace_id)
        if result and not result.passed:
            return result

        result = _check_autonomous_refund(response_content, trace_id)
        if result and not result.passed:
            return result

    # ---- Technical Support Agent checks ----
    if "technical" in agent_name.lower():
        result = _check_source_citations(
            response_content,
            retrieved_doc_ids or [],
            trace_id,
        )
        if result and not result.passed:
            return result

    # ---- Tier 2: LLM judge (optional) ----
    if use_llm_judge and retrieved_context:
        result = _llm_judge(response_content, retrieved_context, trace_id)
        if result and not result.passed:
            logger.warning(
                "Output guardrail: LLM judge flagged response",
                extra={
                    "event": "guardrail_triggered",
                    "guard_name": "output_guard_llm_judge",
                    "agent_name": agent_name,
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "flagged": True,
                    **result.metadata,
                },
            )
            return result

    logger.debug(
        "Output guardrail passed",
        extra={
            "event": "guardrail_passed",
            "guard_name": "output_guard",
            "agent_name": agent_name,
            "trace_id": trace_id,
            "session_id": session_id,
            "flagged": False,
        },
    )
    return GuardResult.ok()
