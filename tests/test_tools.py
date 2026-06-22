"""
tests/test_tools.py
===================
Stage 3 tests for the SQLite database setup and LangChain tools.

Test strategy
-------------
- DB setup tests use a temp file path so they never touch the real clouddash.db.
- Tool tests run against a freshly seeded in-memory / temp SQLite DB by
  patching ``_get_db_path`` at the module level in agent_tools.py.
- The FAISS retriever is mocked in ``search_technical_knowledge_base`` tests
  so the test suite stays fully offline.

Run
---
    pytest tests/test_tools.py -v
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from database.db_setup import (
    DEFAULT_DB_PATH,
    _INVOICES,
    _SUBSCRIPTIONS,
    _USERS,
    create_tables,
    get_connection,
    seed_invoices,
    seed_subscriptions,
    seed_users,
    setup_database,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    """Create a fully seeded temporary SQLite database and return its path."""
    db_path = tmp_path / "test_clouddash.db"
    setup_database(db_path=db_path, force=False)
    return db_path


@pytest.fixture()
def db_conn(temp_db: Path) -> sqlite3.Connection:
    """Return an open connection to the temp database."""
    conn = get_connection(temp_db)
    yield conn
    conn.close()


# ===========================================================================
# Database Setup Tests
# ===========================================================================

class TestDatabaseSetup:

    def test_setup_creates_db_file(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        result = setup_database(db_path=db_path)
        assert db_path.exists()
        assert result == db_path.resolve()

    def test_users_table_exists(self, db_conn: sqlite3.Connection):
        row = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone()
        assert row is not None

    def test_subscriptions_table_exists(self, db_conn: sqlite3.Connection):
        row = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='subscriptions'"
        ).fetchone()
        assert row is not None

    def test_invoices_table_exists(self, db_conn: sqlite3.Connection):
        row = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='invoices'"
        ).fetchone()
        assert row is not None

    def test_correct_user_count(self, db_conn: sqlite3.Connection):
        count = db_conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        assert count == len(_USERS)

    def test_correct_subscription_count(self, db_conn: sqlite3.Connection):
        count = db_conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
        assert count == len(_SUBSCRIPTIONS)

    def test_correct_invoice_count(self, db_conn: sqlite3.Connection):
        count = db_conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
        assert count == len(_INVOICES)

    def test_customer_ids_match_format(self, db_conn: sqlite3.Connection):
        rows = db_conn.execute("SELECT customer_id FROM users").fetchall()
        for row in rows:
            cid = row[0]
            assert cid.startswith("CLD-"), f"Bad customer_id: {cid}"
            assert cid[4:].isdigit(), f"Non-numeric suffix in: {cid}"

    def test_subscriptions_reference_valid_users(self, db_conn: sqlite3.Connection):
        orphans = db_conn.execute(
            """
            SELECT s.customer_id FROM subscriptions s
            LEFT JOIN users u ON s.customer_id = u.customer_id
            WHERE u.customer_id IS NULL
            """
        ).fetchall()
        assert not orphans, f"Orphan subscriptions: {[r[0] for r in orphans]}"

    def test_invoices_reference_valid_users(self, db_conn: sqlite3.Connection):
        orphans = db_conn.execute(
            """
            SELECT i.customer_id FROM invoices i
            LEFT JOIN users u ON i.customer_id = u.customer_id
            WHERE u.customer_id IS NULL
            """
        ).fetchall()
        assert not orphans, f"Orphan invoices: {[r[0] for r in orphans]}"

    def test_all_plan_names_are_valid(self, db_conn: sqlite3.Connection):
        valid = {"Starter", "Growth", "Scale", "Enterprise"}
        rows = db_conn.execute("SELECT DISTINCT plan_name FROM subscriptions").fetchall()
        for row in rows:
            assert row[0] in valid, f"Invalid plan: {row[0]}"

    def test_at_least_one_suspended_account(self, db_conn: sqlite3.Connection):
        count = db_conn.execute(
            "SELECT COUNT(*) FROM subscriptions WHERE status = 'suspended'"
        ).fetchone()[0]
        assert count >= 1

    def test_at_least_one_overdue_invoice(self, db_conn: sqlite3.Connection):
        count = db_conn.execute(
            "SELECT COUNT(*) FROM invoices WHERE status = 'overdue'"
        ).fetchone()[0]
        assert count >= 1

    def test_at_least_one_trial_subscription(self, db_conn: sqlite3.Connection):
        count = db_conn.execute(
            "SELECT COUNT(*) FROM subscriptions WHERE status = 'trial'"
        ).fetchone()[0]
        assert count >= 1

    def test_force_recreates_tables(self, tmp_path: Path):
        db_path = tmp_path / "force_test.db"
        setup_database(db_path=db_path)
        # Insert an extra row, then force-recreate — it should disappear
        conn = get_connection(db_path)
        conn.execute(
            "INSERT INTO users (customer_id, full_name, email, company, cloud_providers, created_at) "
            "VALUES ('CLD-99999', 'Test User', 'test@test.com', 'Test Co', 'AWS', '2024-01-01')"
        )
        conn.commit()
        conn.close()
        # Re-setup with force — should wipe CLD-99999
        setup_database(db_path=db_path, force=True)
        conn = get_connection(db_path)
        row = conn.execute("SELECT * FROM users WHERE customer_id = 'CLD-99999'").fetchone()
        conn.close()
        assert row is None

    def test_idempotent_seeding(self, tmp_path: Path):
        """Running setup_database twice without force should not duplicate rows."""
        db_path = tmp_path / "idempotent.db"
        setup_database(db_path=db_path)
        setup_database(db_path=db_path)  # second run
        conn = get_connection(db_path)
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        assert count == len(_USERS)


# ===========================================================================
# Tool Tests
# ===========================================================================

@pytest.fixture()
def patched_db_path(temp_db: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect all tool DB calls to the temp database."""
    monkeypatch.setenv("CLOUDDASH_DB_PATH", str(temp_db))
    yield temp_db


class TestLookupAccountBillingInfo:

    def test_returns_billing_profile(self, patched_db_path: Path):
        from tools.agent_tools import lookup_account_billing_info
        result = lookup_account_billing_info.invoke({"customer_id": "CLD-00001"})
        assert "CLD-00001" in result
        assert "Alice Chen" in result
        assert "Scale" in result

    def test_shows_subscription_plan(self, patched_db_path: Path):
        from tools.agent_tools import lookup_account_billing_info
        result = lookup_account_billing_info.invoke({"customer_id": "CLD-00002"})
        assert "Growth" in result
        assert "$149.00" in result

    def test_shows_recent_invoices(self, patched_db_path: Path):
        from tools.agent_tools import lookup_account_billing_info
        result = lookup_account_billing_info.invoke({"customer_id": "CLD-00001"})
        assert "INV-" in result

    def test_unknown_customer_returns_error(self, patched_db_path: Path):
        from tools.agent_tools import lookup_account_billing_info
        result = lookup_account_billing_info.invoke({"customer_id": "CLD-99999"})
        assert "ERROR" in result
        assert "not found" in result

    def test_invalid_format_returns_error(self, patched_db_path: Path):
        from tools.agent_tools import lookup_account_billing_info
        result = lookup_account_billing_info.invoke({"customer_id": "INVALID"})
        assert "ERROR" in result
        assert "Invalid customer_id format" in result

    def test_suspended_account_flagged(self, patched_db_path: Path):
        from tools.agent_tools import lookup_account_billing_info
        result = lookup_account_billing_info.invoke({"customer_id": "CLD-00008"})
        assert "SUSPENDED" in result or "suspended" in result.lower()

    def test_trial_account_shows_trial_info(self, patched_db_path: Path):
        from tools.agent_tools import lookup_account_billing_info
        result = lookup_account_billing_info.invoke({"customer_id": "CLD-00006"})
        assert "trial" in result.lower() or "Trial" in result

    def test_overdue_invoice_flagged(self, patched_db_path: Path):
        from tools.agent_tools import lookup_account_billing_info
        result = lookup_account_billing_info.invoke({"customer_id": "CLD-00008"})
        assert "OVERDUE" in result


class TestProcessPlanUpgrade:

    def test_valid_upgrade_changes_plan(self, patched_db_path: Path):
        from tools.agent_tools import process_plan_upgrade
        result = process_plan_upgrade.invoke({"customer_id": "CLD-00003", "new_plan": "Growth"})
        assert "CONFIRMED" in result
        assert "Starter" in result
        assert "Growth" in result

    def test_upgrade_marked_immediate(self, patched_db_path: Path):
        from tools.agent_tools import process_plan_upgrade
        result = process_plan_upgrade.invoke({"customer_id": "CLD-00003", "new_plan": "Scale"})
        assert "immediately" in result.lower() or "IMMEDIATELY" in result

    def test_downgrade_effective_next_cycle(self, patched_db_path: Path):
        from tools.agent_tools import process_plan_upgrade
        result = process_plan_upgrade.invoke({"customer_id": "CLD-00001", "new_plan": "Growth"})
        assert "next billing cycle" in result.lower()

    def test_db_actually_updated(self, patched_db_path: Path):
        from tools.agent_tools import process_plan_upgrade
        process_plan_upgrade.invoke({"customer_id": "CLD-00003", "new_plan": "Growth"})
        conn = get_connection(patched_db_path)
        row = conn.execute(
            "SELECT plan_name FROM subscriptions WHERE customer_id = 'CLD-00003'"
        ).fetchone()
        conn.close()
        assert row[0] == "Growth"

    def test_same_plan_returns_info_no_change(self, patched_db_path: Path):
        from tools.agent_tools import process_plan_upgrade
        result = process_plan_upgrade.invoke({"customer_id": "CLD-00002", "new_plan": "Growth"})
        assert "already on" in result.lower()

    def test_invalid_plan_returns_error(self, patched_db_path: Path):
        from tools.agent_tools import process_plan_upgrade
        result = process_plan_upgrade.invoke({"customer_id": "CLD-00003", "new_plan": "SuperPlan"})
        assert "ERROR" in result
        assert "not a valid" in result

    def test_enterprise_plan_blocked(self, patched_db_path: Path):
        from tools.agent_tools import process_plan_upgrade
        result = process_plan_upgrade.invoke({"customer_id": "CLD-00001", "new_plan": "Enterprise"})
        assert "POLICY" in result or "human approval" in result

    def test_suspended_account_blocked(self, patched_db_path: Path):
        from tools.agent_tools import process_plan_upgrade
        result = process_plan_upgrade.invoke({"customer_id": "CLD-00008", "new_plan": "Growth"})
        assert "suspended" in result.lower()
        assert "ERROR" in result

    def test_invalid_customer_id_returns_error(self, patched_db_path: Path):
        from tools.agent_tools import process_plan_upgrade
        result = process_plan_upgrade.invoke({"customer_id": "BADID", "new_plan": "Growth"})
        assert "ERROR" in result

    def test_normalises_plan_name_case(self, patched_db_path: Path):
        from tools.agent_tools import process_plan_upgrade
        # "growth" should be normalised to "Growth"
        result = process_plan_upgrade.invoke({"customer_id": "CLD-00003", "new_plan": "growth"})
        assert "ERROR" not in result
        assert "Growth" in result


class TestSearchTechnicalKnowledgeBase:

    @patch("tools.agent_tools.retrieve_with_context")
    def test_returns_formatted_results(self, mock_retrieve: MagicMock):
        from retrieval.retriever import RetrievalResult
        from tools.agent_tools import search_technical_knowledge_base

        mock_retrieve.return_value = [
            RetrievalResult(
                source_id="TS-001",
                category="troubleshooting",
                title="Error ERR-4012: AWS credentials expired",
                content="Go to Settings → Integrations → Re-authenticate.",
                score=0.87,
            )
        ]

        result = search_technical_knowledge_base.invoke({"query": "ERR-4012 AWS"})
        assert "TS-001" in result
        assert "ERR-4012" in result
        assert "Re-authenticate" in result

    @patch("tools.agent_tools.retrieve_with_context")
    def test_empty_results_returns_no_results_message(self, mock_retrieve: MagicMock):
        from tools.agent_tools import search_technical_knowledge_base

        mock_retrieve.return_value = []
        result = search_technical_knowledge_base.invoke({"query": "completely unknown topic"})
        assert "No relevant" in result or "not be covered" in result

    @patch("tools.agent_tools.retrieve_with_context")
    def test_citation_note_in_output(self, mock_retrieve: MagicMock):
        from retrieval.retriever import RetrievalResult
        from tools.agent_tools import search_technical_knowledge_base

        mock_retrieve.return_value = [
            RetrievalResult("FAQ-001", "faq", "What is CloudDash?", "Content.", 0.75),
            RetrievalResult("TS-002", "troubleshooting", "ERR-5001", "Content.", 0.72),
        ]
        result = search_technical_knowledge_base.invoke({"query": "monitoring features"})
        assert "CITATION" in result
        assert "FAQ-001" in result
        assert "TS-002" in result

    def test_empty_query_returns_error(self):
        from tools.agent_tools import search_technical_knowledge_base
        result = search_technical_knowledge_base.invoke({"query": "   "})
        assert "ERROR" in result
        assert "empty" in result.lower()

    @patch("tools.agent_tools.retrieve_with_context")
    def test_index_not_ready_returns_helpful_message(self, mock_retrieve: MagicMock):
        from retrieval.retriever import IndexNotReadyError
        from tools.agent_tools import search_technical_knowledge_base

        mock_retrieve.side_effect = IndexNotReadyError("index.faiss not found")
        result = search_technical_knowledge_base.invoke({"query": "alert notifications"})
        assert "not available" in result.lower() or "IndexNotReadyError" in result or "index" in result.lower()

    def test_tool_has_correct_langchain_name(self):
        from tools.agent_tools import search_technical_knowledge_base
        assert search_technical_knowledge_base.name == "search_technical_knowledge_base"

    def test_billing_tool_names_are_correct(self):
        from tools.agent_tools import lookup_account_billing_info, process_plan_upgrade
        assert lookup_account_billing_info.name == "lookup_account_billing_info"
        assert process_plan_upgrade.name == "process_plan_upgrade"

    def test_all_tools_list_has_three_entries(self):
        from tools.agent_tools import ALL_TOOLS, BILLING_TOOLS, TECHNICAL_TOOLS
        assert len(BILLING_TOOLS) == 2
        assert len(TECHNICAL_TOOLS) == 1
        assert len(ALL_TOOLS) == 3

    def test_tools_have_descriptions(self):
        from tools.agent_tools import ALL_TOOLS
        for t in ALL_TOOLS:
            assert t.description, f"Tool '{t.name}' has no description"
            assert len(t.description) > 50, f"Tool '{t.name}' description is too short"
