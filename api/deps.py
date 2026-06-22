"""
api/deps.py
===========
Dependency-injection helpers for the FastAPI application.

FastAPI's ``Depends()`` system is used so that the compiled graph singleton,
configuration, and logger are available to every endpoint handler without
being re-initialised on each request.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import get_config
from graph.graph import get_graph
from utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Graph singleton
# ---------------------------------------------------------------------------


def get_compiled_graph():
    """
    FastAPI dependency: returns the singleton compiled LangGraph application.

    The graph is compiled exactly once per process (lazy, on the first request)
    and cached in ``graph.graph._compiled_graph``.  Subsequent calls return the
    cached instance instantly.
    """
    return get_graph(use_checkpointer=True)


# ---------------------------------------------------------------------------
# Config singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_app_config() -> dict[str, Any]:
    """
    FastAPI dependency: returns the parsed ``agents_config.yaml`` dict.
    Cached with ``lru_cache`` so the YAML file is only read once.
    """
    return get_config()


# ---------------------------------------------------------------------------
# Logging bootstrap (called during app lifespan)
# ---------------------------------------------------------------------------


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structured JSON logging for the whole process.
    Safe to call multiple times — ``configure_logging`` is idempotent.
    """
    numeric = getattr(logging, log_level.upper(), logging.INFO)
    configure_logging(level=numeric)
    logger.info("CloudDash API logging initialised", extra={"log_level": log_level})
