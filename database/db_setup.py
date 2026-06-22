"""
database/db_setup.py
====================
SQLite database initialisation and seeding script for CloudDash.

Schema
------
  users         — CloudDash customer accounts
  subscriptions — One active subscription per user (current plan, billing dates)
  invoices      — Historical invoices linked to a user

Design decisions
----------------
* Pure ``sqlite3`` stdlib — no ORM dependency in the setup script so the
  schema is immediately readable and the file can be run standalone.
* SQLAlchemy (added in requirements.txt) will be used by the Billing Agent tools
  for safe parameterised queries; the schema here is compatible with SQLAlchemy ORM.
* The ``customer_id`` field mirrors the format used in ``ExtractedEntities``
  (``CLD-XXXXX``) so the Triage Agent's extracted entity maps directly to a
  DB lookup without transformation.
* Foreign key enforcement is enabled via PRAGMA for referential integrity.

Usage
-----
    # From the project root
    python -m database.db_setup [--db-path PATH] [--force]

Arguments
---------
--db-path   Path to the SQLite file (default: clouddash.db in project root)
--force     Drop and recreate all tables (wipes existing data)
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "clouddash.db"

# ---------------------------------------------------------------------------
# DDL — table definitions
# ---------------------------------------------------------------------------

_DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     TEXT    NOT NULL UNIQUE,   -- e.g. CLD-00001
    full_name       TEXT    NOT NULL,
    email           TEXT    NOT NULL UNIQUE,
    company         TEXT    NOT NULL,
    cloud_providers TEXT    NOT NULL,          -- comma-separated: AWS,GCP,Azure
    created_at      TEXT    NOT NULL,          -- ISO-8601 date string
    is_active       INTEGER NOT NULL DEFAULT 1
);
"""

_DDL_SUBSCRIPTIONS = """
CREATE TABLE IF NOT EXISTS subscriptions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id         TEXT    NOT NULL UNIQUE REFERENCES users(customer_id),
    plan_name           TEXT    NOT NULL,      -- Starter | Growth | Scale | Enterprise
    monthly_price_usd   REAL    NOT NULL,
    billing_cycle       TEXT    NOT NULL,      -- monthly | annual
    monitored_resources INTEGER NOT NULL DEFAULT 0,
    seat_count          INTEGER NOT NULL DEFAULT 1,
    status              TEXT    NOT NULL DEFAULT 'active',
    -- 'active' | 'suspended' | 'cancelled' | 'trial'
    trial_ends_at       TEXT,                  -- NULL if not on trial
    current_period_start TEXT   NOT NULL,
    current_period_end   TEXT   NOT NULL,
    next_billing_date    TEXT   NOT NULL,
    auto_renew          INTEGER NOT NULL DEFAULT 1,
    updated_at          TEXT    NOT NULL
);
"""

_DDL_INVOICES = """
CREATE TABLE IF NOT EXISTS invoices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_number  TEXT    NOT NULL UNIQUE,   -- e.g. INV-2024-001234
    customer_id     TEXT    NOT NULL REFERENCES users(customer_id),
    amount_usd      REAL    NOT NULL,
    status          TEXT    NOT NULL,          -- paid | unpaid | overdue | void
    description     TEXT    NOT NULL,
    issued_date     TEXT    NOT NULL,
    due_date        TEXT    NOT NULL,
    paid_date       TEXT,                      -- NULL if not yet paid
    payment_method  TEXT,                      -- e.g. Visa •••• 4242
    period_start    TEXT    NOT NULL,
    period_end      TEXT    NOT NULL
);
"""

_ALL_DDL = [_DDL_USERS, _DDL_SUBSCRIPTIONS, _DDL_INVOICES]

_DROP_TABLES = [
    "DROP TABLE IF EXISTS invoices;",
    "DROP TABLE IF EXISTS subscriptions;",
    "DROP TABLE IF EXISTS users;",
]

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_today = date.today()
_fmt = lambda d: d.isoformat()  # noqa: E731


def _period(months_ago: int = 0) -> tuple[str, str, str]:
    """Return (period_start, period_end, next_billing_date) strings."""
    start = _today.replace(day=1) - timedelta(days=30 * months_ago)
    # End = last day of start's month
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
    next_billing = end + timedelta(days=1)
    return _fmt(start), _fmt(end), _fmt(next_billing)


_USERS: list[dict[str, Any]] = [
    {
        "customer_id": "CLD-00001",
        "full_name": "Alice Chen",
        "email": "alice.chen@acmecorp.io",
        "company": "Acme Corp",
        "cloud_providers": "AWS,GCP",
        "created_at": "2023-03-15",
        "is_active": 1,
    },
    {
        "customer_id": "CLD-00002",
        "full_name": "Bob Martinez",
        "email": "bob.martinez@techwave.com",
        "company": "TechWave Solutions",
        "cloud_providers": "AWS",
        "created_at": "2023-07-22",
        "is_active": 1,
    },
    {
        "customer_id": "CLD-00003",
        "full_name": "Priya Patel",
        "email": "priya.patel@cloudnative.dev",
        "company": "CloudNative Dev",
        "cloud_providers": "AWS,Azure",
        "created_at": "2024-01-10",
        "is_active": 1,
    },
    {
        "customer_id": "CLD-00004",
        "full_name": "James O'Brien",
        "email": "james.obrien@finserve.co",
        "company": "FinServe Capital",
        "cloud_providers": "Azure",
        "created_at": "2022-11-05",
        "is_active": 1,
    },
    {
        "customer_id": "CLD-00005",
        "full_name": "Sara Kim",
        "email": "sara.kim@retailedge.com",
        "company": "RetailEdge Inc.",
        "cloud_providers": "AWS,GCP,Azure",
        "created_at": "2023-05-18",
        "is_active": 1,
    },
    {
        "customer_id": "CLD-00006",
        "full_name": "David Nguyen",
        "email": "david.nguyen@startupboost.io",
        "company": "StartupBoost",
        "cloud_providers": "AWS",
        "created_at": "2024-03-01",
        "is_active": 1,
    },
    {
        "customer_id": "CLD-00007",
        "full_name": "Elena Rossi",
        "email": "elena.rossi@medisync.eu",
        "company": "MediSync EU",
        "cloud_providers": "Azure,GCP",
        "created_at": "2023-09-30",
        "is_active": 1,
    },
    {
        "customer_id": "CLD-00008",
        "full_name": "Tom Fletcher",
        "email": "tom.fletcher@logisticspro.com",
        "company": "LogisticsPro",
        "cloud_providers": "AWS",
        "created_at": "2023-02-14",
        "is_active": 0,  # suspended account — for testing
    },
    {
        "customer_id": "CLD-00009",
        "full_name": "Anya Sharma",
        "email": "anya.sharma@datadriven.ai",
        "company": "DataDriven AI",
        "cloud_providers": "AWS,GCP",
        "created_at": "2024-02-20",
        "is_active": 1,
    },
    {
        "customer_id": "CLD-00010",
        "full_name": "Carlos Mendez",
        "email": "carlos.mendez@ecommhub.mx",
        "company": "EcommHub MX",
        "cloud_providers": "AWS",
        "created_at": "2023-12-01",
        "is_active": 1,
    },
]

_p = [_period(i) for i in range(10)]  # generate 10 billing periods

_SUBSCRIPTIONS: list[dict[str, Any]] = [
    {
        "customer_id": "CLD-00001",
        "plan_name": "Scale",
        "monthly_price_usd": 499.00,
        "billing_cycle": "annual",
        "monitored_resources": 312,
        "seat_count": 18,
        "status": "active",
        "trial_ends_at": None,
        "current_period_start": _p[0][0],
        "current_period_end": _p[0][1],
        "next_billing_date": _p[0][2],
        "auto_renew": 1,
        "updated_at": _fmt(_today),
    },
    {
        "customer_id": "CLD-00002",
        "plan_name": "Growth",
        "monthly_price_usd": 149.00,
        "billing_cycle": "monthly",
        "monitored_resources": 87,
        "seat_count": 5,
        "status": "active",
        "trial_ends_at": None,
        "current_period_start": _p[1][0],
        "current_period_end": _p[1][1],
        "next_billing_date": _p[1][2],
        "auto_renew": 1,
        "updated_at": _fmt(_today),
    },
    {
        "customer_id": "CLD-00003",
        "plan_name": "Starter",
        "monthly_price_usd": 49.00,
        "billing_cycle": "monthly",
        "monitored_resources": 34,
        "seat_count": 2,
        "status": "active",
        "trial_ends_at": None,
        "current_period_start": _p[2][0],
        "current_period_end": _p[2][1],
        "next_billing_date": _p[2][2],
        "auto_renew": 1,
        "updated_at": _fmt(_today),
    },
    {
        "customer_id": "CLD-00004",
        "plan_name": "Enterprise",
        "monthly_price_usd": 1200.00,
        "billing_cycle": "annual",
        "monitored_resources": 1450,
        "seat_count": 60,
        "status": "active",
        "trial_ends_at": None,
        "current_period_start": _p[3][0],
        "current_period_end": _p[3][1],
        "next_billing_date": _p[3][2],
        "auto_renew": 1,
        "updated_at": _fmt(_today),
    },
    {
        "customer_id": "CLD-00005",
        "plan_name": "Scale",
        "monthly_price_usd": 499.00,
        "billing_cycle": "monthly",
        "monitored_resources": 520,
        "seat_count": 30,
        "status": "active",
        "trial_ends_at": None,
        "current_period_start": _p[4][0],
        "current_period_end": _p[4][1],
        "next_billing_date": _p[4][2],
        "auto_renew": 1,
        "updated_at": _fmt(_today),
    },
    {
        "customer_id": "CLD-00006",
        "plan_name": "Growth",
        "monthly_price_usd": 149.00,
        "billing_cycle": "monthly",
        "monitored_resources": 12,
        "seat_count": 3,
        "status": "trial",
        "trial_ends_at": _fmt(_today + timedelta(days=8)),
        "current_period_start": _p[5][0],
        "current_period_end": _p[5][1],
        "next_billing_date": _p[5][2],
        "auto_renew": 1,
        "updated_at": _fmt(_today),
    },
    {
        "customer_id": "CLD-00007",
        "plan_name": "Growth",
        "monthly_price_usd": 149.00,
        "billing_cycle": "annual",
        "monitored_resources": 198,
        "seat_count": 12,
        "status": "active",
        "trial_ends_at": None,
        "current_period_start": _p[6][0],
        "current_period_end": _p[6][1],
        "next_billing_date": _p[6][2],
        "auto_renew": 0,
        "updated_at": _fmt(_today),
    },
    {
        "customer_id": "CLD-00008",
        "plan_name": "Starter",
        "monthly_price_usd": 49.00,
        "billing_cycle": "monthly",
        "monitored_resources": 22,
        "seat_count": 1,
        "status": "suspended",
        "trial_ends_at": None,
        "current_period_start": _p[7][0],
        "current_period_end": _p[7][1],
        "next_billing_date": _p[7][2],
        "auto_renew": 0,
        "updated_at": _fmt(_today - timedelta(days=14)),
    },
    {
        "customer_id": "CLD-00009",
        "plan_name": "Growth",
        "monthly_price_usd": 149.00,
        "billing_cycle": "monthly",
        "monitored_resources": 95,
        "seat_count": 7,
        "status": "active",
        "trial_ends_at": None,
        "current_period_start": _p[8][0],
        "current_period_end": _p[8][1],
        "next_billing_date": _p[8][2],
        "auto_renew": 1,
        "updated_at": _fmt(_today),
    },
    {
        "customer_id": "CLD-00010",
        "plan_name": "Starter",
        "monthly_price_usd": 49.00,
        "billing_cycle": "monthly",
        "monitored_resources": 48,
        "seat_count": 3,
        "status": "active",
        "trial_ends_at": None,
        "current_period_start": _p[9][0],
        "current_period_end": _p[9][1],
        "next_billing_date": _p[9][2],
        "auto_renew": 1,
        "updated_at": _fmt(_today),
    },
]

# Generate ~3 invoices per user (30 total)
def _build_invoices() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    plans = {
        "CLD-00001": (499.00, "Scale Plan — Annual"),
        "CLD-00002": (149.00, "Growth Plan — Monthly"),
        "CLD-00003": (49.00, "Starter Plan — Monthly"),
        "CLD-00004": (1200.00, "Enterprise Plan — Annual"),
        "CLD-00005": (499.00, "Scale Plan — Monthly"),
        "CLD-00006": (149.00, "Growth Plan — Monthly (Trial)"),
        "CLD-00007": (149.00, "Growth Plan — Annual"),
        "CLD-00008": (49.00, "Starter Plan — Monthly"),
        "CLD-00009": (149.00, "Growth Plan — Monthly"),
        "CLD-00010": (49.00, "Starter Plan — Monthly"),
    }
    n = 1000
    for cust_id, (price, desc) in plans.items():
        for i in range(3):
            months_back = i + 1
            issued = _today.replace(day=1) - timedelta(days=30 * months_back)
            due = issued + timedelta(days=14)
            if i == 0 and cust_id == "CLD-00008":
                # Overdue invoice for the suspended account
                status = "overdue"
                paid_date = None
                payment_method = None
            elif i == 0 and cust_id == "CLD-00006":
                # Unpaid trial user
                status = "unpaid"
                paid_date = None
                payment_method = None
            else:
                status = "paid"
                paid_date = _fmt(due - timedelta(days=2))
                payment_method = "Visa •••• 4242" if cust_id not in ("CLD-00004",) else "ACH Bank Transfer"

            period_start = issued
            period_end = issued + timedelta(days=29)

            n += 1
            rows.append({
                "invoice_number": f"INV-{issued.year}-{n:06d}",
                "customer_id": cust_id,
                "amount_usd": price,
                "status": status,
                "description": f"{desc} — {issued.strftime('%B %Y')}",
                "issued_date": _fmt(issued),
                "due_date": _fmt(due),
                "paid_date": paid_date,
                "payment_method": payment_method,
                "period_start": _fmt(period_start),
                "period_end": _fmt(period_end),
            })
    return rows


_INVOICES = _build_invoices()


# ===========================================================================
# Database operations
# ===========================================================================

def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Return a sqlite3 connection to *db_path* with foreign key enforcement
    and ``Row`` factory for dict-like access.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_tables(conn: sqlite3.Connection, force: bool = False) -> None:
    """Create all tables. If *force* is True, drop them first."""
    if force:
        logger.warning("--force: dropping existing tables.")
        for stmt in _DROP_TABLES:
            conn.execute(stmt)
        conn.commit()
        logger.info("Existing tables dropped.")

    with conn:
        for ddl in _ALL_DDL:
            conn.execute(ddl)
    logger.info("Tables created (or already exist).")


def seed_users(conn: sqlite3.Connection) -> int:
    """Insert seed users, skipping rows that already exist. Returns insert count."""
    sql = """
    INSERT OR IGNORE INTO users
        (customer_id, full_name, email, company, cloud_providers, created_at, is_active)
    VALUES
        (:customer_id, :full_name, :email, :company, :cloud_providers, :created_at, :is_active)
    """
    with conn:
        cursor = conn.executemany(sql, _USERS)
    logger.info("Seeded %d users.", cursor.rowcount)
    return cursor.rowcount


def seed_subscriptions(conn: sqlite3.Connection) -> int:
    """Insert seed subscriptions. Returns insert count."""
    sql = """
    INSERT OR IGNORE INTO subscriptions
        (customer_id, plan_name, monthly_price_usd, billing_cycle,
         monitored_resources, seat_count, status, trial_ends_at,
         current_period_start, current_period_end, next_billing_date,
         auto_renew, updated_at)
    VALUES
        (:customer_id, :plan_name, :monthly_price_usd, :billing_cycle,
         :monitored_resources, :seat_count, :status, :trial_ends_at,
         :current_period_start, :current_period_end, :next_billing_date,
         :auto_renew, :updated_at)
    """
    with conn:
        cursor = conn.executemany(sql, _SUBSCRIPTIONS)
    logger.info("Seeded %d subscriptions.", cursor.rowcount)
    return cursor.rowcount


def seed_invoices(conn: sqlite3.Connection) -> int:
    """Insert seed invoices. Returns insert count."""
    sql = """
    INSERT OR IGNORE INTO invoices
        (invoice_number, customer_id, amount_usd, status, description,
         issued_date, due_date, paid_date, payment_method, period_start, period_end)
    VALUES
        (:invoice_number, :customer_id, :amount_usd, :status, :description,
         :issued_date, :due_date, :paid_date, :payment_method, :period_start, :period_end)
    """
    with conn:
        cursor = conn.executemany(sql, _INVOICES)
    logger.info("Seeded %d invoices.", cursor.rowcount)
    return cursor.rowcount


def setup_database(
    db_path: Path = DEFAULT_DB_PATH,
    force: bool = False,
) -> Path:
    """
    Full setup: create tables and seed all mock data.

    Parameters
    ----------
    db_path :
        Path to the SQLite file. Will be created if it doesn't exist.
    force :
        If True, wipes and recreates all tables before seeding.

    Returns
    -------
    Path
        Resolved path to the created database file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Setting up database at: %s", db_path.resolve())

    conn = get_connection(db_path)
    try:
        create_tables(conn, force=force)
        seed_users(conn)
        seed_subscriptions(conn)
        seed_invoices(conn)
    finally:
        conn.close()

    logger.info(
        "Database ready: %d users, %d subscriptions, %d invoices.",
        len(_USERS),
        len(_SUBSCRIPTIONS),
        len(_INVOICES),
    )
    return db_path.resolve()


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize and seed the CloudDash SQLite database."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate all tables (WARNING: deletes existing data)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    path = setup_database(db_path=args.db_path, force=args.force)
    print(f"OK  Database created at: {path}")
    print(f"    Users: {len(_USERS)}, Subscriptions: {len(_SUBSCRIPTIONS)}, Invoices: {len(_INVOICES)}")
