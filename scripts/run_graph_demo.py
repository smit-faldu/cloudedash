"""
scripts/run_graph_demo.py
=========================
Quick interactive demo of the CloudDash multi-agent graph.

Runs the compiled LangGraph graph against three scripted scenarios:
  1. Technical query (ERR-4012) → Technical Support Agent
  2. Billing query              → Billing Agent
  3. Refund request             → Billing Agent → Escalation Agent (handover)

Usage
-----
    # From the project root (with GEMINI_API_KEY set in .env)
    .venv\\Scripts\\python scripts/run_graph_demo.py

    # Skip live LLM calls — print routing only (dry-run)
    .venv\\Scripts\\python scripts/run_graph_demo.py --dry-run

Environment
-----------
    GEMINI_API_KEY or GOOGLE_API_KEY must be set in .env
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "Scenario 1 - Technical Error Query",
        "user_message": "I keep getting ERR-4012 when connecting my AWS account. CLD-00001",
        "customer_id": "CLD-00001",
    },
    {
        "name": "Scenario 2 - Billing Plan Inquiry",
        "user_message": "What plan am I on and when does it renew? My ID is CLD-00002",
        "customer_id": "CLD-00002",
    },
    {
        "name": "Scenario 3 - Refund Request (Handover to Escalation)",
        "user_message": "I was double-charged last month and I want a full refund. CLD-00003",
        "customer_id": "CLD-00003",
    },
]


# ---------------------------------------------------------------------------
# Dry-run stubs — return realistic outputs without real LLM calls
# ---------------------------------------------------------------------------

def _dry_run_triage(state):
    """Return a mock TriageResult based on keywords in the user message."""
    from models.models import AgentName, ExtractedEntities, IntentLabel, TriageResult, UrgencyLevel

    msg = ""
    for m in state.get("messages", []):
        content = getattr(m, "content", "")
        if content:
            msg = content.lower()
            break

    if "err-" in msg or "error" in msg or "aws" in msg:
        intent, confidence = "technical", 0.92
    elif "refund" in msg:
        intent, confidence = "escalation", 0.98
    elif "plan" in msg or "billing" in msg or "renew" in msg or "invoice" in msg:
        intent, confidence = "billing", 0.88
    else:
        intent, confidence = "general", 0.75

    import re
    customer_id = None
    m_obj = re.search(r"CLD-\d{5}", msg.upper())
    if m_obj:
        customer_id = m_obj.group()

    return TriageResult(
        intent=IntentLabel(intent),
        confidence=confidence,
        extracted_entities=ExtractedEntities(
            customer_id=customer_id,
            urgency=UrgencyLevel.HIGH if "err-" in msg else UrgencyLevel.MEDIUM,
        ),
        reasoning=f"[DRY-RUN] Keyword-based classification: {intent}",
    )


def _dry_run_technical(state):
    from models.models import AgentName, AgentResponse
    return AgentResponse(
        agent_name=AgentName.TECHNICAL_SUPPORT,
        content=(
            "[DRY-RUN] The ERR-4012 error indicates your AWS IAM credentials have "
            "expired or lack required permissions.\n\n"
            "Steps:\n"
            "1. Navigate to Settings > Integrations > AWS.\n"
            "2. Click 'Re-authenticate' and follow the OAuth flow.\n"
            "3. Ensure the IAM role has the CloudWatch:GetMetricData permission.\n\n"
            "Sources: TS-001, FAQ-003"
        ),
        needs_handover=False,
        source_documents=["TS-001", "FAQ-003"],
    )


def _dry_run_billing(state):
    from models.models import AgentName, AgentResponse

    # Check if messages contain refund intent
    has_refund = any(
        "refund" in getattr(m, "content", "").lower()
        for m in state.get("messages", [])
    )

    if has_refund:
        return AgentResponse(
            agent_name=AgentName.BILLING,
            content=(
                "[DRY-RUN] I can see your account history for CLD-00003. "
                "I cannot process refunds autonomously — this requires human approval. "
                "I'll escalate this to our billing team immediately."
            ),
            needs_handover=True,
            target_agent=AgentName.ESCALATION,
            handover_reason="Refund request requires human operator approval.",
        )

    return AgentResponse(
        agent_name=AgentName.BILLING,
        content=(
            "[DRY-RUN] Account CLD-00002:\n"
            "  Plan: Growth ($149/month)\n"
            "  Status: Active\n"
            "  Next billing: 2026-07-15\n"
            "  Last invoice: INV-2024-002001 — $149.00 PAID (2026-06-15)"
        ),
        needs_handover=False,
    )


def _dry_run_escalation(state):
    from models.models import AgentName, AgentResponse
    trace = state.get("trace_id", "unknown")
    return AgentResponse(
        agent_name=AgentName.ESCALATION,
        content=(
            "[DRY-RUN] I've prepared your handover package.\n"
            "A member of the Billing Team will contact you within 4 business hours.\n\n"
            f"**Reference ID:** {trace}\n"
            "**Priority:** P2\n"
        ),
        needs_handover=False,
        metadata={
            "trace_id": trace,
            "priority": "P2",
            "recommended_team": "billing_team",
            "escalation_package": {
                "priority": "P2",
                "core_issue": "Customer requesting refund for double-charge.",
                "recommended_team": "billing_team",
                "summary_bullets": [
                    "Customer reported double-charge on June 2026 invoice.",
                    "Billing Agent confirmed could not process autonomously.",
                ],
            },
        },
    )


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 70
W = 68


def _print_scenario(name: str, user_message: str, result: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"\n  USER: {textwrap.fill(user_message, W, subsequent_indent='        ')}")
    print()

    messages = result.get("messages", [])
    for msg in messages:
        role = getattr(msg, "name", None) or getattr(msg, "type", "?")
        content = getattr(msg, "content", "")
        if role == "user":
            continue   # already printed above
        if not content:
            continue
        label = role.upper().replace("_", " ")
        print(f"  [{label}]")
        for line in textwrap.wrap(content.strip(), W, subsequent_indent="    "):
            print(f"    {line}")
        print()

    print(f"  {'-' * 66}")
    print(f"  intent       : {result.get('intent', 'N/A')}")
    print(f"  confidence   : {result.get('confidence', 0):.2f}")
    print(f"  customer_id  : {result.get('customer_id', 'N/A')}")
    print(f"  handovers    : {result.get('handover_count', 0)}")
    print(f"  is_escalated : {result.get('is_escalated', False)}")
    print(f"  trace_id     : {result.get('trace_id', 'N/A')[:16]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CloudDash multi-agent graph demo.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use mock agents — no real LLM calls, no API key needed."
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  CloudDash Multi-Agent Support System - Stage 5 Demo")
    print("  Mode:", "DRY-RUN (mocked agents)" if args.dry_run else "LIVE (real Gemini API)")
    print("=" * 70)

    from graph.graph import build_graph, create_initial_state

    if args.dry_run:
        patches = [
            patch("graph.graph.run_triage_agent", side_effect=_dry_run_triage),
            patch("graph.graph.run_technical_support_agent", side_effect=_dry_run_technical),
            patch("graph.graph.run_billing_agent", side_effect=_dry_run_billing),
            patch("graph.graph.run_escalation_agent", side_effect=_dry_run_escalation),
        ]
        for p in patches:
            p.start()

    try:
        graph = build_graph(use_checkpointer=False)

        for scenario in SCENARIOS:
            state = create_initial_state(
                user_message=scenario["user_message"],
                customer_id=scenario["customer_id"],
            )
            result = graph.invoke(state)
            _print_scenario(scenario["name"], scenario["user_message"], result)

    finally:
        if args.dry_run:
            for p in patches:
                p.stop()

    print(f"\n{'=' * 70}")
    print("  Demo complete.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
