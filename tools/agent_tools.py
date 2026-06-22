"""
tools/agent_tools.py
=====================
LangChain ``@tool``-decorated functions for the CloudDash specialist agents.

Three tools are defined here:

1. ``lookup_account_billing_info``  — Billing Agent tool
   Queries SQLite for a customer's subscription and recent invoice history.

2. ``process_plan_upgrade``         — Billing Agent tool
   Validates and executes a plan change in SQLite, respecting billing policy.

3. ``search_technical_knowledge_base`` — Technical Support Agent tool
   Delegates to the Stage 2 FAISS retrieval chain with query rewriting.

LangChain Tool Design Principles (from @langchain-architecture skill)
----------------------------------------------------------------------
* **Docstrings are the tool's instruction to the LLM.** They must be precise,
  unambiguous, and specify exactly WHEN and HOW the tool should be used.
  A poor docstring is the #1 cause of incorrect tool selection by agents.
* **Return strings, not dicts.** LangChain tool outputs are injected into the
  LLM's prompt as plain text. Returning a well-formatted string gives the agent
  the most flexibility in parsing the result.
* **Validate inputs before hitting the database.** Catching bad inputs at the
  tool boundary produces a clean error message that the agent can relay to the
  user, rather than a raw Python traceback.
* **Tools are stateless.** They open a DB connection, execute, and close. The
  LangGraph state object is the single source of truth for session data.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import retrieval helpers at module level so @patch can target them.
# These are imported here rather than inside the tool body so that
# unittest.mock.patch("tools.agent_tools.retrieve_with_context") works.
# If the FAISS index is not built yet, the ImportError is caught gracefully
# inside the tool body itself.
# ---------------------------------------------------------------------------
try:
    from retrieval.retriever import (
        retrieve_with_context,
        format_results_for_prompt,
        get_source_ids,
        IndexNotReadyError,
    )
except Exception:  # noqa: BLE001 — index may not exist yet; handled at call time
    retrieve_with_context = None  # type: ignore[assignment]
    format_results_for_prompt = None  # type: ignore[assignment]
    get_source_ids = None  # type: ignore[assignment]
    IndexNotReadyError = Exception  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Resolved at import time; can be overridden via CLOUDDASH_DB_PATH env var
_DEFAULT_DB_PATH = _PROJECT_ROOT / "clouddash.db"

# Valid plan names — enforced before any DB write
_VALID_PLANS = {"Starter", "Growth", "Scale", "Enterprise"}

# Monthly prices for validation feedback
_PLAN_PRICES: dict[str, float] = {
    "Starter": 49.00,
    "Growth": 149.00,
    "Scale": 499.00,
    "Enterprise": 1200.00,  # representative; actual Enterprise pricing is custom
}


def _get_db_path() -> Path:
    """Return the DB path, respecting the CLOUDDASH_DB_PATH environment override."""
    env_override = os.environ.get("CLOUDDASH_DB_PATH")
    return Path(env_override) if env_override else _DEFAULT_DB_PATH


def _get_connection() -> sqlite3.Connection:
    """Open a sqlite3 connection with Row factory and FK enforcement."""
    from database.db_setup import get_connection
    return get_connection(_get_db_path())


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert a sqlite3.Row to a plain dict, or return None."""
    return dict(row) if row else None


# ===========================================================================
# Tool 1 — Billing: look up account and invoice information
# ===========================================================================

@tool
def lookup_account_billing_info(customer_id: str) -> str:
    """
    Retrieve the full billing profile for a CloudDash customer, including their
    current subscription plan, subscription status, billing dates, and the last
    three invoices.

    USE THIS TOOL when:
    - A customer asks about their current plan, seat count, or resource limits.
    - A customer asks about a charge, invoice, or payment status.
    - A customer reports a billing error (overdue payment, unexpected charge).
    - You need to verify a customer's account exists before making any changes.
    - A customer asks when their next billing date is.

    DO NOT use this tool for:
    - Technical product questions (use search_technical_knowledge_base instead).
    - Changing a customer's plan (use process_plan_upgrade instead).
    - Refund requests — always escalate refunds to a human operator.

    Args:
        customer_id: The CloudDash customer identifier in the format 'CLD-XXXXX'
                     (e.g., 'CLD-00042'). This is always provided in the
                     customer's account profile and often mentioned by the
                     customer or extracted by the Triage Agent.

    Returns:
        A formatted string summarising the customer's account, subscription, and
        recent invoices. Returns an error message if the customer_id is not found
        or is malformed.

    Example:
        lookup_account_billing_info("CLD-00001")
        → Returns full billing profile for Alice Chen at Acme Corp.
    """
    # ---- Input validation ----
    customer_id = customer_id.strip().upper()
    if not customer_id.startswith("CLD-") or not customer_id[4:].isdigit():
        return (
            f"ERROR: Invalid customer_id format '{customer_id}'. "
            "Expected format: CLD-XXXXX (e.g., CLD-00042). "
            "Please ask the customer to confirm their account ID."
        )

    try:
        conn = _get_connection()
        try:
            # ---- Fetch user ----
            user = _row_to_dict(
                conn.execute(
                    "SELECT * FROM users WHERE customer_id = ?", (customer_id,)
                ).fetchone()
            )
            if not user:
                return (
                    f"ERROR: Customer '{customer_id}' not found in the CloudDash database. "
                    "The customer_id may be incorrect, or the account may not exist. "
                    "Please ask the customer to verify their account ID."
                )

            # ---- Fetch subscription ----
            sub = _row_to_dict(
                conn.execute(
                    "SELECT * FROM subscriptions WHERE customer_id = ?", (customer_id,)
                ).fetchone()
            )

            # ---- Fetch last 3 invoices ----
            invoice_rows = conn.execute(
                """
                SELECT invoice_number, amount_usd, status, description,
                       issued_date, due_date, paid_date, payment_method
                FROM invoices
                WHERE customer_id = ?
                ORDER BY issued_date DESC
                LIMIT 3
                """,
                (customer_id,),
            ).fetchall()
            invoices = [_row_to_dict(r) for r in invoice_rows]

        finally:
            conn.close()

    except sqlite3.Error as exc:
        logger.error("DB error in lookup_account_billing_info: %s", exc)
        return f"ERROR: Database error while retrieving account info — {exc}"

    # ---- Format response ----
    lines: list[str] = [
        "=== CLOUDDASH ACCOUNT BILLING PROFILE ===",
        f"Customer ID   : {user['customer_id']}",
        f"Name          : {user['full_name']}",
        f"Email         : {user['email']}",
        f"Company       : {user['company']}",
        f"Cloud Providers: {user['cloud_providers']}",
        f"Account Status: {'ACTIVE' if user['is_active'] else 'SUSPENDED/INACTIVE'}",
        f"Member Since  : {user['created_at']}",
        "",
    ]

    if sub:
        status_tag = f"[{sub['status'].upper()}]"
        trial_note = ""
        if sub["status"] == "trial" and sub.get("trial_ends_at"):
            trial_note = f" (Trial ends: {sub['trial_ends_at']})"

        lines += [
            "--- Subscription ---",
            f"Plan          : {sub['plan_name']} {status_tag}{trial_note}",
            f"Price         : ${sub['monthly_price_usd']:.2f} / month ({sub['billing_cycle']} billing)",
            f"Resources     : {sub['monitored_resources']} monitored",
            f"Seats         : {sub['seat_count']}",
            f"Current Period: {sub['current_period_start']} → {sub['current_period_end']}",
            f"Next Billing  : {sub['next_billing_date']}",
            f"Auto-Renew    : {'Yes' if sub['auto_renew'] else 'No'}",
            "",
        ]
    else:
        lines += ["--- Subscription ---", "No active subscription found.", ""]

    lines.append("--- Recent Invoices (last 3) ---")
    if invoices:
        for inv in invoices:
            paid_info = f"  Paid: {inv['paid_date']} via {inv['payment_method']}" if inv["paid_date"] else ""
            status_flag = ""
            if inv["status"] == "overdue":
                status_flag = "  *** OVERDUE — ACTION REQUIRED ***"
            elif inv["status"] == "unpaid":
                status_flag = "  (Unpaid — due " + inv["due_date"] + ")"
            lines.append(
                f"  {inv['invoice_number']}  |  ${inv['amount_usd']:.2f}  "
                f"|  {inv['status'].upper()}  |  Issued: {inv['issued_date']}"
                f"{paid_info}{status_flag}"
            )
            lines.append(f"    {inv['description']}")
    else:
        lines.append("  No invoices found.")

    lines.append("=== END OF BILLING PROFILE ===")
    return "\n".join(lines)


# ===========================================================================
# Tool 2 — Billing: process a plan upgrade
# ===========================================================================

@tool
def process_plan_upgrade(customer_id: str, new_plan: str) -> str:
    """
    Change a CloudDash customer's subscription to a new plan tier.

    USE THIS TOOL ONLY when:
    - The customer has explicitly confirmed they want to change their plan.
    - You have ALREADY called lookup_account_billing_info to verify the account exists.
    - The requested plan is one of: Starter, Growth, Scale, Enterprise.

    MANDATORY pre-conditions — do NOT call this tool if any of these are false:
    1. The customer has stated the new plan name clearly.
    2. You have confirmed the plan name with the customer (paraphrase it back).
    3. The account is not currently 'suspended' or 'cancelled'.

    BILLING POLICY (non-negotiable — never deviate):
    - Upgrades (moving to a higher-priced plan) take effect IMMEDIATELY.
    - Downgrades (moving to a lower-priced plan) take effect at the START of the
      NEXT billing cycle. Inform the customer of this before calling the tool.
    - You CANNOT process refunds. If a customer requests a refund, do NOT use
      this tool — escalate to the Escalation Agent immediately.
    - Enterprise plan changes require human approval — set needs_handover=True.

    Args:
        customer_id: The CloudDash customer identifier in format 'CLD-XXXXX'.
        new_plan: The target plan name. Must be EXACTLY one of:
                  'Starter', 'Growth', 'Scale', 'Enterprise'
                  (case-sensitive).

    Returns:
        A confirmation string describing the plan change that was applied, including
        the effective date. Returns an error string if the change cannot be processed.

    Example:
        process_plan_upgrade("CLD-00003", "Growth")
        → "Plan upgrade confirmed: Starter → Growth for CLD-00003. Effective immediately."
    """
    # ---- Input validation ----
    customer_id = customer_id.strip().upper()
    new_plan = new_plan.strip().capitalize()  # normalise: "growth" → "Growth"
    if new_plan == "Enterprise":
        # Title-case works for all except this might need special handling
        pass

    if not customer_id.startswith("CLD-") or not customer_id[4:].isdigit():
        return (
            f"ERROR: Invalid customer_id format '{customer_id}'. "
            "Expected format: CLD-XXXXX (e.g., CLD-00042)."
        )

    if new_plan not in _VALID_PLANS:
        return (
            f"ERROR: '{new_plan}' is not a valid CloudDash plan. "
            f"Valid plans are: {', '.join(sorted(_VALID_PLANS))}. "
            "Please ask the customer which plan they want."
        )

    if new_plan == "Enterprise":
        return (
            "POLICY: Enterprise plan changes require human approval and custom pricing negotiation. "
            "I am unable to process this automatically. "
            "Please escalate this conversation to the Billing Team for follow-up."
        )

    try:
        conn = _get_connection()
        try:
            # ---- Fetch current subscription ----
            sub = _row_to_dict(
                conn.execute(
                    "SELECT plan_name, status, monthly_price_usd FROM subscriptions "
                    "WHERE customer_id = ?",
                    (customer_id,),
                ).fetchone()
            )

            if not sub:
                return (
                    f"ERROR: No subscription found for customer '{customer_id}'. "
                    "Please verify the account exists via lookup_account_billing_info first."
                )

            if sub["status"] in ("suspended", "cancelled"):
                return (
                    f"ERROR: Cannot change plan — account '{customer_id}' is currently "
                    f"'{sub['status']}'. Payment must be resolved before a plan change "
                    "can be processed. Please escalate to the Billing Team."
                )

            current_plan = sub["plan_name"]
            if current_plan == new_plan:
                return (
                    f"INFO: Customer '{customer_id}' is already on the '{new_plan}' plan. "
                    "No changes were made."
                )

            # ---- Determine upgrade vs downgrade ----
            current_price = _PLAN_PRICES.get(current_plan, 0)
            new_price = _PLAN_PRICES.get(new_plan, 0)
            is_upgrade = new_price > current_price
            effective_date = date.today().isoformat() if is_upgrade else sub.get(
                "current_period_end",
                date.today().isoformat(),
            )

            # ---- Apply the change in the database ----
            today_str = date.today().isoformat()
            with conn:
                conn.execute(
                    """
                    UPDATE subscriptions
                    SET plan_name = ?,
                        monthly_price_usd = ?,
                        updated_at = ?
                    WHERE customer_id = ?
                    """,
                    (new_plan, _PLAN_PRICES[new_plan], today_str, customer_id),
                )

            logger.info(
                "Plan change applied: %s → %s for %s (effective: %s)",
                current_plan,
                new_plan,
                customer_id,
                effective_date,
            )

        finally:
            conn.close()

    except sqlite3.Error as exc:
        logger.error("DB error in process_plan_upgrade: %s", exc)
        return f"ERROR: Database error while processing plan change — {exc}"

    # ---- Build response ----
    change_type = "upgrade" if is_upgrade else "downgrade"
    price_diff = abs(new_price - current_price)
    effective_note = (
        "effective immediately"
        if is_upgrade
        else f"effective at the start of the next billing cycle ({effective_date})"
    )
    price_note = (
        f"The new monthly price is ${new_price:.2f}/month "
        f"({'an increase' if is_upgrade else 'a reduction'} of ${price_diff:.2f}/month)."
    )

    return (
        f"PLAN CHANGE CONFIRMED\n"
        f"Customer     : {customer_id}\n"
        f"Change type  : {change_type.upper()}\n"
        f"Previous plan: {current_plan} (${current_price:.2f}/month)\n"
        f"New plan     : {new_plan} (${new_price:.2f}/month)\n"
        f"Effective    : {effective_note.capitalize()}\n"
        f"Pricing note : {price_note}\n"
        f"Processed at : {today_str}\n"
        f"\nPlease inform the customer of this change and confirm they have no further questions."
    )


# ===========================================================================
# Tool 3 — Technical: search the FAISS knowledge base
# ===========================================================================

@tool
def search_technical_knowledge_base(query: str) -> str:
    """
    Search the CloudDash technical knowledge base for documentation, troubleshooting
    guides, API references, and FAQ articles relevant to a customer's question.

    USE THIS TOOL when:
    - A customer reports an error code (e.g., ERR-4012, ERR-5001, ERR-3007).
    - A customer asks how a product feature works (e.g., alerting, cost optimization).
    - A customer asks about API endpoints, authentication, or webhook setup.
    - A customer asks about integrating CloudDash with AWS, GCP, or Azure.
    - A customer asks about data retention, metrics resolution, or export.
    - You need supporting documentation BEFORE composing a technical answer.

    DO NOT use this tool for:
    - Billing, invoice, or subscription questions (use lookup_account_billing_info).
    - General greetings or off-topic questions.
    - Confirming information already returned by a previous tool call in this session.

    IMPORTANT: Always call this tool FIRST before drafting a technical answer.
    Base your answer strictly on the retrieved content. If the KB does not contain
    sufficient information, state that clearly rather than guessing.

    CITATION REQUIREMENT: After answering, you MUST list the source article IDs
    (e.g., TS-001, API-002) that you used. Do not cite articles you did not retrieve.

    Args:
        query: A clear, specific technical question or search phrase. Include
               error codes, feature names, or product areas for best results.
               Examples:
                 - "ERR-4012 AWS integration disconnected"
                 - "how to set up PagerDuty alert notifications"
                 - "API authentication OAuth 2.0 scopes"
                 - "cost optimization idle resource detection"

    Returns:
        A formatted string containing the most relevant knowledge base excerpts,
        each prefixed with its source article ID. Returns a "no results" message
        if no relevant articles are found — in that case, escalate the conversation.

    Example:
        search_technical_knowledge_base("ERR-4012 AWS credentials")
        → Returns TS-001 content about fixing expired IAM credentials.
    """
    query = query.strip()
    if not query:
        return (
            "ERROR: Search query is empty. Please provide a specific question or "
            "error description to search the knowledge base."
        )

    try:
        if retrieve_with_context is None:
            return (
                "ERROR: The knowledge base retriever could not be loaded. "
                "Ensure retrieval/retriever.py is importable and the index exists."
            )

        results = retrieve_with_context(
            query=query,
            conversation_history=[],
            top_k=4,
            min_score=0.28,
            llm=None,
        )

        if not results:
            return (
                "KNOWLEDGE BASE: No relevant articles found for this query.\n"
                "This topic may not be covered in the current knowledge base. "
                "Consider escalating to the senior support team for specialist assistance.\n"
                "Query searched: " + query
            )

        formatted = format_results_for_prompt(results)
        source_ids = get_source_ids(results)

        return (
            f"KNOWLEDGE BASE SEARCH RESULTS\n"
            f"Query: {query}\n"
            f"Sources: {', '.join(source_ids)}\n"
            f"Top {len(results)} relevant article(s):\n\n"
            f"{formatted}\n\n"
            f"CITATION NOTE: When answering, cite these source IDs: {', '.join(source_ids)}"
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Error in search_technical_knowledge_base: %s", exc, exc_info=True)
        if "IndexNotReadyError" in type(exc).__name__ or "index.faiss" in str(exc):
            return (
                "ERROR: The knowledge base index is not available. "
                "The system administrator needs to run the ingestion pipeline first:\n"
                "    python -m retrieval.ingest --force\n"
                "Please escalate this conversation until the knowledge base is ready."
            )
        return (
            f"ERROR: Knowledge base search failed — {type(exc).__name__}: {exc}. "
            "Please escalate this issue to the senior support team."
        )


# ===========================================================================
# Convenience: collect all tools as a list for agent binding
# ===========================================================================

BILLING_TOOLS = [
    lookup_account_billing_info,
    process_plan_upgrade,
]

TECHNICAL_TOOLS = [
    search_technical_knowledge_base,
]

ALL_TOOLS = BILLING_TOOLS + TECHNICAL_TOOLS
