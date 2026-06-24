"""
config/config_loader.py
=======================
Loads and validates the agents_config.yaml file at startup.
Provides typed, cached access to agent prompts, routing rules, and global settings.

Usage
-----
    from config.config_loader import get_config

    cfg = get_config()
    prompt   = cfg.get_agent_prompt("triage_agent")
    target   = cfg.routing.intent_to_agent["billing"]
    model    = cfg.global_settings.llm_model
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default path — resolves to <project_root>/config/agents_config.yaml
# regardless of where the process is started from.
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "agents_config.yaml"


# ===========================================================================
# Pydantic models that mirror the YAML schema
# ===========================================================================


class GlobalSettings(BaseModel):
    """Top-level global settings shared across all agents."""

    max_history_turns: int = Field(20, ge=1, le=200)
    default_temperature: float = Field(0.2, ge=0.0, le=2.0)
    llm_model: str = "gemini-3.1-flash-lite"
    llm_timeout_seconds: int = Field(30, ge=5, le=300)
    llm_max_retries: int = Field(3, ge=0, le=10)

    @model_validator(mode="before")
    @classmethod
    def _override_from_env(cls, data: Any) -> Any:
        if isinstance(data, dict):
            env_model = os.environ.get("GEMINI_MODEL_DEFAULT") or os.environ.get("GEMINI_MODEL")
            if env_model:
                data["llm_model"] = env_model
        return data


class RoutingConfig(BaseModel):
    """Routing rules used by the Triage Agent and the LangGraph router."""

    intent_to_agent: dict[str, str] = Field(default_factory=dict)
    auto_escalate_intents: list[str] = Field(default_factory=list)
    confidence_threshold: float = Field(0.65, ge=0.0, le=1.0)

    @field_validator("intent_to_agent")
    @classmethod
    def _required_intents_present(cls, v: dict[str, str]) -> dict[str, str]:
        required = {"technical", "billing", "general", "escalation", "unknown"}
        missing = required - v.keys()
        if missing:
            raise ValueError(
                f"agents_config.yaml routing.intent_to_agent is missing keys: {missing}"
            )
        return v


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    display_name: str
    description: str = ""
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    system_prompt: str

    @field_validator("system_prompt")
    @classmethod
    def _prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("system_prompt must not be empty")
        return v.strip()


class CloudDashConfig(BaseModel):
    """Root configuration object parsed from agents_config.yaml."""

    global_settings: GlobalSettings = Field(alias="global")
    routing: RoutingConfig
    agents: dict[str, AgentConfig]

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _agents_match_routing_targets(self) -> "CloudDashConfig":
        """Ensure every agent referenced in routing rules actually exists."""
        all_targets = set(self.routing.intent_to_agent.values())
        defined_agents = set(self.agents.keys())
        missing = all_targets - defined_agents
        if missing:
            raise ValueError(
                f"Routing references undefined agents: {missing}. "
                f"Defined agents: {defined_agents}"
            )
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Return the AgentConfig for *agent_name*, raising KeyError if absent."""
        if agent_name not in self.agents:
            raise KeyError(
                f"No agent named '{agent_name}' in configuration. "
                f"Available: {list(self.agents.keys())}"
            )
        return self.agents[agent_name]

    def get_agent_prompt(self, agent_name: str) -> str:
        """Shorthand — return just the system prompt string for *agent_name*."""
        return self.get_agent_config(agent_name).system_prompt

    def get_agent_temperature(self, agent_name: str) -> float:
        """Return the per-agent temperature (falls back to global default)."""
        cfg = self.get_agent_config(agent_name)
        return cfg.temperature

    def resolve_intent(self, intent: str) -> str:
        """Map an intent label to its target agent node name."""
        return self.routing.intent_to_agent.get(
            intent.lower(), self.routing.intent_to_agent["unknown"]
        )


# ===========================================================================
# Loader
# ===========================================================================


def load_config(config_path: Path | str | None = None) -> CloudDashConfig:
    """
    Parse *config_path* (defaults to ``config/agents_config.yaml``) and return
    a validated :class:`CloudDashConfig` instance.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist at the resolved path.
    pydantic.ValidationError
        If the YAML content fails schema validation.
    yaml.YAMLError
        If the file cannot be parsed as valid YAML.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path.resolve()}\n"
            "Ensure agents_config.yaml exists in the config/ directory."
        )

    logger.info("Loading CloudDash configuration from: %s", path.resolve())
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    config = CloudDashConfig.model_validate(raw)
    logger.info(
        "Configuration loaded successfully — %d agents, %d routing rules.",
        len(config.agents),
        len(config.routing.intent_to_agent),
    )
    return config


@functools.lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> CloudDashConfig:
    """
    Cached singleton accessor.  Call this anywhere in the application to obtain
    the loaded configuration without re-parsing the YAML on every request.

    Parameters
    ----------
    config_path:
        Optional override path for the YAML file (useful in tests).
        Pass an explicit path as a *string* (not Path) for hashability.
    """
    return load_config(config_path)
