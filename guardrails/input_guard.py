"""
guardrails/input_guard.py
=========================
Input guardrail for the CloudDash Multi-Agent Support System.

Responsibility
--------------
This module runs BEFORE a user message enters the LangGraph workflow.
It is intentionally **rule-based and zero-latency** — it does NOT call an LLM
so it cannot be bypassed by prompt injection targeting the guardrail itself,
and it adds zero API cost.

What it checks
--------------
1. **Prompt injection / jailbreak patterns** — phrases that try to override
   the system prompt (e.g. "ignore all previous instructions", "act as",
   "DAN mode").

2. **PII leakage attempts** — messages asking the agent to reveal internal
   system prompts, configuration, or other users' data.

3. **Severely off-topic queries** — content clearly unrelated to a B2B SaaS
   support context (e.g. creative writing requests, explicit content).

4. **Excessive length** — messages > 4,000 characters are rejected to
   prevent token stuffing / context overflow attacks.

5. **Repeated injection sequences** — multiple known attack phrases in a
   single message.

What it does NOT check
----------------------
* Factual correctness or hallucinations — that is the output guardrail's job.
* Sentiment or tone — out of scope.
* Language detection — the system accepts all languages.

Return contract
---------------
``check_input(message, trace_id)`` returns a ``GuardResult`` dataclass:
  - ``passed: bool``       — True means the message is safe to process.
  - ``flag_reason: str``   — Human-readable reason if passed=False.
  - ``safe_reply: str``    — Customer-facing fallback message if triggered.
  - ``severity: str``      — "low" | "medium" | "high" | "critical"

The FastAPI layer (Stage 7) returns the ``safe_reply`` to the user and logs
the event without forwarding the message to the LangGraph graph.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    """
    Return type for all guardrail check functions.

    Attributes
    ----------
    passed
        True if the content is safe to proceed; False if blocked.
    flag_reason
        Internal reason code / description (not shown to user).
    safe_reply
        Customer-facing response to return when the guardrail fires.
        Empty string when ``passed=True``.
    severity
        "low" | "medium" | "high" | "critical"
    metadata
        Optional extra data for logging (matched pattern, category, etc.).
    """

    passed: bool
    flag_reason: str = ""
    safe_reply: str = ""
    severity: str = "low"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls) -> "GuardResult":
        """Convenience factory for a passing result."""
        return cls(passed=True)

    @classmethod
    def blocked(
        cls,
        flag_reason: str,
        safe_reply: str,
        severity: str = "medium",
        **metadata: Any,
    ) -> "GuardResult":
        """Convenience factory for a blocked result."""
        return cls(
            passed=False,
            flag_reason=flag_reason,
            safe_reply=safe_reply,
            severity=severity,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Pattern catalogues
# ---------------------------------------------------------------------------

# Prompt injection / jailbreak — phrases that attempt to override the system prompt.
# Patterns are pre-compiled for speed; case-insensitive matching.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+(all\s+)?previous\s+instructions?",
        r"forget\s+(all\s+)?previous\s+instructions?",
        r"override\s+(all\s+)?instructions?",
        r"you\s+are\s+now\s+(?!a\s+CloudDash)",   # "you are now [different persona]"
        r"act\s+as\s+(?!a\s+CloudDash)",            # "act as [other role]"
        r"\bDAN\s+mode\b",
        r"\bjailbreak\b",
        r"pretend\s+(you\s+are|to\s+be)\s+(?!a\s+CloudDash)",
        r"roleplay\s+as",
        r"your\s+(new|real)\s+(instructions?|rules?|persona)",
        r"bypass\s+(all\s+)?(safety|guardrail|filter|restriction)",
        r"do\s+anything\s+now",
        r"enable\s+developer\s+mode",
        r"ignore\s+ethical\s+(guidelines?|constraints?)",
        r"respond\s+only\s+in\s+\[?json\]?",       # output format hijacking
        r"output\s+your\s+system\s+prompt",
        r"repeat\s+the\s+(above|following|system|initial)\s+(prompt|instruction)",
        r"what\s+(are\s+)?your\s+(instructions?|system\s+prompt|directives?)",
        r"reveal\s+your\s+(instructions?|prompt|training)",
        r"print\s+your\s+(instructions?|prompt|configuration)",
        r"show\s+me\s+your\s+(system\s+)?prompt",
        r"translate\s+the\s+above\s+into",          # indirect extraction
        r"\btoken\s+budget\s+hack\b",
        r"\bprompt\s+leak\b",
        r"disallow\s+mode",
    ]
]

# PII / data exfiltration attempts
_EXFIL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"other\s+customers?'?s?\s+(data|details|info|account)",
        r"all\s+customer\s+(ids?|records?|emails?|data)",
        r"list\s+all\s+(users?|accounts?|customers?)",
        r"dump\s+(the\s+)?(database|db|table|all\s+data)",
        r"select\s+\*\s+from",           # raw SQL injection attempts
        r"drop\s+table",
        r"insert\s+into\s+\w+",
        r"union\s+select",
        r"--\s*$",                        # SQL comment terminator
        r";\s*(drop|delete|update|insert)",
        r"<script\b",                     # XSS probe
        r"javascript\s*:",
    ]
]

# Off-topic content (B2B SaaS support context violations)
_OFF_TOPIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bwrite\s+(me\s+)?(a\s+)?(poem|story|essay|novel|song|rap|haiku)\b",
        r"\bgenerate\s+(a\s+|an\s+)?(image|picture|photo|artwork)\b",
        r"\b(illegal|hack|exploit|crack|pirate|torrent)\b",
        r"\b(adult|explicit|nsfw|pornograph)\b",
        r"\bhow\s+(?:to|do\s+I|can\s+I)\s+(?:make|build|create|construct|synthesize)\s+(?:a\s+|an\s+)?(bomb|weapon|drug|virus|malware)\b",
        r"\bstocks?\s+(tips?|prediction|investment)\b",
        r"\bplay\s+(chess|poker|a\s+game)\b",
    ]
]

# Maximum permitted message length (characters)
_MAX_MESSAGE_LENGTH = 4_000

# If this many injection phrases appear in a single message, escalate severity
_MULTI_INJECTION_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_input(
    message: str,
    trace_id: str = "",
    session_id: str | None = None,
) -> GuardResult:
    """
    Run all input checks on *message* and return a ``GuardResult``.

    Checks run in priority order (fastest / most critical first):
    1. Empty message
    2. Excessive length
    3. Prompt injection / jailbreak
    4. Data exfiltration / SQL injection
    5. Off-topic content

    Parameters
    ----------
    message
        The raw user message string.
    trace_id
        Session trace ID (used in log events only).
    session_id
        Optional session identifier (used in log events only).

    Returns
    -------
    GuardResult
        ``passed=True`` → safe to forward to the LangGraph graph.
        ``passed=False`` → return ``safe_reply`` to the user immediately.
    """
    # --- 1. Empty message ---
    stripped = message.strip()
    if not stripped:
        return GuardResult.blocked(
            flag_reason="empty_message",
            safe_reply=(
                "I didn't receive any message. "
                "Please describe your issue and I'll be happy to help."
            ),
            severity="low",
        )

    # --- 2. Excessive length ---
    if len(stripped) > _MAX_MESSAGE_LENGTH:
        _log_blocked(trace_id, "excessive_length", "medium", session_id=session_id,
                     message_length=len(stripped))
        return GuardResult.blocked(
            flag_reason="excessive_length",
            safe_reply=(
                "Your message is too long for me to process. "
                "Please summarise your issue in a few sentences and try again."
            ),
            severity="low",
            message_length=len(stripped),
        )

    # --- 3. Prompt injection / jailbreak ---
    injection_matches = _find_matches(stripped, _INJECTION_PATTERNS)
    if injection_matches:
        severity = "critical" if len(injection_matches) >= _MULTI_INJECTION_THRESHOLD else "high"
        _log_blocked(trace_id, "prompt_injection", severity, session_id=session_id,
                     matched_patterns=injection_matches[:5])
        return GuardResult.blocked(
            flag_reason="prompt_injection",
            safe_reply=(
                "I'm only able to help with CloudDash support topics. "
                "Please describe your support issue and I'll assist you."
            ),
            severity=severity,
            matched_patterns=injection_matches[:5],
        )

    # --- 4. Data exfiltration / SQL injection ---
    exfil_matches = _find_matches(stripped, _EXFIL_PATTERNS)
    if exfil_matches:
        _log_blocked(trace_id, "data_exfiltration_attempt", "critical",
                     session_id=session_id, matched_patterns=exfil_matches[:5])
        return GuardResult.blocked(
            flag_reason="data_exfiltration_attempt",
            safe_reply=(
                "I'm not able to process that request. "
                "If you need account information, please log in to the CloudDash "
                "portal or contact support@clouddash.io."
            ),
            severity="critical",
            matched_patterns=exfil_matches[:5],
        )

    # --- 5. Off-topic content ---
    off_topic_matches = _find_matches(stripped, _OFF_TOPIC_PATTERNS)
    if off_topic_matches:
        _log_blocked(trace_id, "off_topic_content", "low",
                     session_id=session_id, matched_patterns=off_topic_matches[:3])
        return GuardResult.blocked(
            flag_reason="off_topic_content",
            safe_reply=(
                "I'm a CloudDash support assistant and can only help with questions "
                "about the CloudDash platform. How can I help you with your "
                "monitoring, billing, or account needs today?"
            ),
            severity="low",
            matched_patterns=off_topic_matches[:3],
        )

    # --- All checks passed ---
    return GuardResult.ok()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_matches(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    """Return a list of pattern strings that match *text*."""
    return [p.pattern for p in patterns if p.search(text)]


def _log_blocked(
    trace_id: str,
    flag_reason: str,
    severity: str,
    session_id: str | None = None,
    **extra: Any,
) -> None:
    """Log a blocked input event at the appropriate level."""
    log_level = logging.WARNING if severity in ("low", "medium") else logging.ERROR
    logger.log(
        log_level,
        "Input guardrail triggered",
        extra={
            "event": "guardrail_triggered",
            "guard_name": "input_guard",
            "trace_id": trace_id,
            "session_id": session_id,
            "flag_reason": flag_reason,
            "severity": severity,
            "flagged": True,
            **extra,
        },
    )
