"""
utils/logger.py
===============
Structured JSON logging for the CloudDash Multi-Agent Support System.

Design goals
------------
* Every log record includes ``trace_id`` so all events within one customer
  session can be retrieved with a single filter query (e.g. in Loki, Splunk,
  or a simple ``grep``).
* Output is newline-delimited JSON (NDJSON) so it streams cleanly into any
  log aggregation system without a custom parser.
* The standard Python ``logging`` hierarchy is preserved — callers keep using
  ``logger.info()``, ``logger.warning()``, etc.  The formatter does the work.
* A LangChain ``BaseCallbackHandler`` subclass (``TracedCallbackHandler``) emits
  structured log events for every LLM invocation, tool call, and chain step,
  giving full observability without modifying any agent code.

Usage
-----
    from utils.logger import get_logger, TracedCallbackHandler

    # Module-level logger (trace_id bound at call time)
    logger = get_logger(__name__)
    logger.info("Triage complete", trace_id="abc-123", intent="billing", confidence=0.90)

    # LangChain callback (attach to any LLM / chain)
    handler = TracedCallbackHandler(trace_id="abc-123", session_id="ses-456")
    llm = ChatGoogleGenerativeAI(..., callbacks=[handler])

Structured record schema
------------------------
Every JSON line contains at least:
{
    "timestamp":  "2026-06-22T03:45:00.123456Z",  // ISO-8601 UTC
    "level":      "INFO",
    "logger":     "agents.agent_nodes",
    "trace_id":   "abc-123",                        // session trace
    "session_id": "ses-456",                        // optional
    "event":      "agent_invocation_start",         // event type
    "message":    "Triage Agent processing …",
    ... (arbitrary extra fields from the log call)
}
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------


class StructuredJsonFormatter(logging.Formatter):
    """
    Converts every ``LogRecord`` into a single-line JSON object.

    Extra fields added via the ``extra`` kwarg in log calls are merged into
    the top-level JSON record so they are directly queryable.

    Example
    -------
    logger.info(
        "Triage complete",
        extra={"trace_id": "abc", "intent": "billing", "confidence": 0.88}
    )

    Produces:
    {"timestamp": "...", "level": "INFO", "logger": "...",
     "trace_id": "abc", "intent": "billing", "confidence": 0.88,
     "message": "Triage complete"}
    """

    # Fields that are always promoted to top-level
    _ALWAYS_TOP_LEVEL = frozenset({
        "trace_id", "session_id", "event", "agent", "intent",
        "confidence", "customer_id", "handover_count", "duration_ms",
        "tool_name", "tool_input", "error", "flagged", "flag_reason",
    })

    def format(self, record: logging.LogRecord) -> str:
        record_dict: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="microseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge well-known extra fields into top-level
        for key in self._ALWAYS_TOP_LEVEL:
            val = getattr(record, key, None)
            if val is not None:
                record_dict[key] = val

        # Merge all other extra fields that aren't standard LogRecord attrs
        _STANDARD_ATTRS = frozenset({
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs", "message",
            "pathname", "process", "processName", "relativeCreated",
            "thread", "threadName", "stack_info", "exc_info", "exc_text",
        })
        for key, val in record.__dict__.items():
            if key.startswith("_") or key in _STANDARD_ATTRS:
                continue
            if key in record_dict:
                continue
            try:
                json.dumps(val)   # cheap serialisability check
                record_dict[key] = val
            except (TypeError, ValueError):
                record_dict[key] = str(val)

        # Attach exception info if present
        if record.exc_info:
            record_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(record_dict, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Root logger bootstrap
# ---------------------------------------------------------------------------

def configure_logging(
    level: int = logging.INFO,
    stream=None,
    fmt: str = "json",
) -> None:
    """
    Configure the root logger once for the whole process.

    Call this at application startup (e.g. in ``main.py`` or the FastAPI
    lifespan).  Subsequent calls are idempotent — if the root logger already
    has a handler the function returns immediately.

    Parameters
    ----------
    level
        Logging level (default: INFO).
    stream
        Output stream (default: sys.stdout).
    fmt
        ``"json"`` for structured NDJSON (production), ``"text"`` for
        human-readable output (development / tests).
    """
    root = logging.getLogger()
    if root.handlers:
        return   # already configured

    handler = logging.StreamHandler(stream or sys.stdout)
    if fmt == "json":
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s trace=%(trace_id)s: %(message)s",
                defaults={"trace_id": "-"},
            )
        )
    handler.setLevel(level)
    root.setLevel(level)
    root.addHandler(handler)


# ---------------------------------------------------------------------------
# Per-module logger factory
# ---------------------------------------------------------------------------


class _TraceAdapter(logging.LoggerAdapter):
    """
    A ``LoggerAdapter`` that merges a static ``trace_id`` (and optional
    ``session_id``) into every log record emitted through this adapter.

    This lets module-level loggers carry a bound trace_id without needing
    to pass it on every call — while still allowing overrides via ``extra``.
    """

    def process(
        self, msg: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        extra = {**self.extra, **kwargs.get("extra", {})}
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(
    name: str,
    trace_id: str | None = None,
    session_id: str | None = None,
) -> _TraceAdapter:
    """
    Return a module-level ``LoggerAdapter`` with an optional bound trace_id.

    Parameters
    ----------
    name
        Logger name — pass ``__name__`` from the calling module.
    trace_id
        Session trace identifier. When provided, it is injected into every
        log record emitted by this adapter.
    session_id
        Optional stable session identifier (distinct from trace_id).

    Returns
    -------
    _TraceAdapter
        A ``LoggerAdapter`` wrapping the standard ``logging.Logger``.

    Example
    -------
    logger = get_logger(__name__, trace_id=state["trace_id"])
    logger.info("Triage Agent processing", extra={"intent": "billing"})
    """
    base_logger = logging.getLogger(name)
    extra: dict[str, Any] = {}
    if trace_id:
        extra["trace_id"] = trace_id
    if session_id:
        extra["session_id"] = session_id
    return _TraceAdapter(base_logger, extra)


# ---------------------------------------------------------------------------
# Event helpers — emit strongly-typed structured log events
# ---------------------------------------------------------------------------

_MODULE_LOGGER = logging.getLogger("clouddash.events")


def log_agent_start(
    agent_name: str,
    trace_id: str,
    session_id: str | None = None,
    user_message_preview: str = "",
    **extra: Any,
) -> None:
    """Emit a structured event when an agent node begins execution."""
    _MODULE_LOGGER.info(
        "Agent invocation started",
        extra={
            "event": "agent_invocation_start",
            "agent": agent_name,
            "trace_id": trace_id,
            "session_id": session_id,
            "user_message_preview": user_message_preview[:120],
            **extra,
        },
    )


def log_agent_end(
    agent_name: str,
    trace_id: str,
    duration_ms: float,
    needs_handover: bool = False,
    target_agent: str | None = None,
    session_id: str | None = None,
    **extra: Any,
) -> None:
    """Emit a structured event when an agent node finishes execution."""
    _MODULE_LOGGER.info(
        "Agent invocation completed",
        extra={
            "event": "agent_invocation_end",
            "agent": agent_name,
            "trace_id": trace_id,
            "session_id": session_id,
            "duration_ms": round(duration_ms, 2),
            "needs_handover": needs_handover,
            "target_agent": target_agent,
            **extra,
        },
    )


def log_handover(
    from_agent: str,
    to_agent: str,
    trace_id: str,
    reason: str,
    handover_count: int,
    session_id: str | None = None,
    **extra: Any,
) -> None:
    """Emit a structured event for every inter-agent handover."""
    _MODULE_LOGGER.info(
        "Agent handover",
        extra={
            "event": "agent_handover",
            "from_agent": from_agent,
            "to_agent": to_agent,
            "trace_id": trace_id,
            "session_id": session_id,
            "reason": reason,
            "handover_count": handover_count,
            **extra,
        },
    )


def log_triage_result(
    trace_id: str,
    intent: str,
    confidence: float,
    customer_id: str | None,
    session_id: str | None = None,
    **extra: Any,
) -> None:
    """Emit a structured event with the Triage Agent's classification result."""
    _MODULE_LOGGER.info(
        "Triage classification",
        extra={
            "event": "triage_result",
            "trace_id": trace_id,
            "session_id": session_id,
            "intent": intent,
            "confidence": confidence,
            "customer_id": customer_id,
            **extra,
        },
    )


def log_guardrail_triggered(
    guard_name: str,
    trace_id: str,
    flag_reason: str,
    user_message_preview: str = "",
    session_id: str | None = None,
    **extra: Any,
) -> None:
    """Emit a structured event when any guardrail fires."""
    _MODULE_LOGGER.warning(
        "Guardrail triggered",
        extra={
            "event": "guardrail_triggered",
            "guard_name": guard_name,
            "trace_id": trace_id,
            "session_id": session_id,
            "flag_reason": flag_reason,
            "flagged": True,
            "user_message_preview": user_message_preview[:120],
            **extra,
        },
    )


def log_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    trace_id: str,
    agent: str = "",
    session_id: str | None = None,
    **extra: Any,
) -> None:
    """Emit a structured event when an agent invokes a LangChain tool."""
    _MODULE_LOGGER.info(
        "Tool call",
        extra={
            "event": "tool_call",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "trace_id": trace_id,
            "session_id": session_id,
            "agent": agent,
            **extra,
        },
    )


def log_kb_retrieval(
    query: str,
    num_docs: int,
    doc_ids: list[str],
    trace_id: str,
    session_id: str | None = None,
    **extra: Any,
) -> None:
    """Emit a structured event for every FAISS KB retrieval."""
    _MODULE_LOGGER.info(
        "KB retrieval",
        extra={
            "event": "kb_retrieval",
            "query_preview": query[:120],
            "num_docs": num_docs,
            "doc_ids": doc_ids,
            "trace_id": trace_id,
            "session_id": session_id,
            **extra,
        },
    )


# ---------------------------------------------------------------------------
# LangChain Callback Handler
# ---------------------------------------------------------------------------

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    class TracedCallbackHandler(BaseCallbackHandler):
        """
        A LangChain ``BaseCallbackHandler`` that emits structured JSON log
        events for every LLM call, tool invocation, and chain step.

        Attach this handler to any LLM, chain, or agent to get automatic
        observability without modifying agent code.

        Parameters
        ----------
        trace_id
            Session trace ID to tag every event.
        session_id
            Optional stable session identifier.
        agent_name
            Human-readable agent name for log context.

        Example
        -------
        handler = TracedCallbackHandler(trace_id="abc-123", agent_name="billing_agent")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", callbacks=[handler])
        """

        def __init__(
            self,
            trace_id: str = "",
            session_id: str | None = None,
            agent_name: str = "unknown",
        ) -> None:
            super().__init__()
            self.trace_id = trace_id or str(uuid.uuid4())
            self.session_id = session_id
            self.agent_name = agent_name
            self._llm_start_times: dict[str, float] = {}
            self._chain_start_times: dict[str, float] = {}
            self._logger = logging.getLogger("clouddash.langchain")

        # ---- LLM events ----

        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            self._llm_start_times[str(run_id)] = time.monotonic()
            model = serialized.get("kwargs", {}).get("model", "unknown")
            self._logger.info(
                "LLM invocation started",
                extra={
                    "event": "llm_start",
                    "trace_id": self.trace_id,
                    "session_id": self.session_id,
                    "agent": self.agent_name,
                    "model": model,
                    "num_prompts": len(prompts),
                    "run_id": str(run_id),
                },
            )

        def on_llm_end(
            self,
            response: "LLMResult",
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            start = self._llm_start_times.pop(str(run_id), None)
            duration_ms = round((time.monotonic() - start) * 1000, 2) if start else None

            # Extract token usage if available
            token_usage: dict[str, Any] = {}
            if response.llm_output:
                usage = response.llm_output.get("token_usage") or response.llm_output.get("usage_metadata")
                if usage:
                    token_usage = dict(usage)

            self._logger.info(
                "LLM invocation completed",
                extra={
                    "event": "llm_end",
                    "trace_id": self.trace_id,
                    "session_id": self.session_id,
                    "agent": self.agent_name,
                    "duration_ms": duration_ms,
                    "token_usage": token_usage,
                    "run_id": str(run_id),
                },
            )

        def on_llm_error(
            self,
            error: Exception,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            self._llm_start_times.pop(str(run_id), None)
            self._logger.error(
                "LLM invocation error",
                extra={
                    "event": "llm_error",
                    "trace_id": self.trace_id,
                    "session_id": self.session_id,
                    "agent": self.agent_name,
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "run_id": str(run_id),
                },
            )

        # ---- Tool events ----

        def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            tool_name = serialized.get("name", "unknown")
            self._logger.info(
                "Tool call started",
                extra={
                    "event": "tool_start",
                    "trace_id": self.trace_id,
                    "session_id": self.session_id,
                    "agent": self.agent_name,
                    "tool_name": tool_name,
                    "tool_input": input_str[:200],
                    "run_id": str(run_id),
                },
            )

        def on_tool_end(
            self,
            output: str,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            self._logger.info(
                "Tool call completed",
                extra={
                    "event": "tool_end",
                    "trace_id": self.trace_id,
                    "session_id": self.session_id,
                    "agent": self.agent_name,
                    "tool_output_preview": str(output)[:200],
                    "run_id": str(run_id),
                },
            )

        def on_tool_error(
            self,
            error: Exception,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            self._logger.error(
                "Tool call error",
                extra={
                    "event": "tool_error",
                    "trace_id": self.trace_id,
                    "session_id": self.session_id,
                    "agent": self.agent_name,
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "run_id": str(run_id),
                },
            )

        # ---- Chain events ----

        def on_chain_start(
            self,
            serialized: dict[str, Any],
            inputs: dict[str, Any],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            self._chain_start_times[str(run_id)] = time.monotonic()
            chain_name = serialized.get("name", "unknown")
            self._logger.debug(
                "Chain started",
                extra={
                    "event": "chain_start",
                    "trace_id": self.trace_id,
                    "session_id": self.session_id,
                    "agent": self.agent_name,
                    "chain_name": chain_name,
                    "run_id": str(run_id),
                },
            )

        def on_chain_end(
            self,
            outputs: dict[str, Any],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            start = self._chain_start_times.pop(str(run_id), None)
            duration_ms = round((time.monotonic() - start) * 1000, 2) if start else None
            self._logger.debug(
                "Chain completed",
                extra={
                    "event": "chain_end",
                    "trace_id": self.trace_id,
                    "session_id": self.session_id,
                    "agent": self.agent_name,
                    "duration_ms": duration_ms,
                    "run_id": str(run_id),
                },
            )

except ImportError:
    # langchain_core not available (e.g. during pure unit tests)
    class TracedCallbackHandler:  # type: ignore[no-redef]
        """Stub when langchain_core is unavailable."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
