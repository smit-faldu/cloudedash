"""
database package — SQLite schema, seeding, and connection helpers.

Public API
----------
    from database.db_setup import setup_database, get_connection, DEFAULT_DB_PATH
"""
from database.db_setup import setup_database, get_connection, DEFAULT_DB_PATH

__all__ = ["setup_database", "get_connection", "DEFAULT_DB_PATH"]
