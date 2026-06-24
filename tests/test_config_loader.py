"""
tests/test_config_loader.py
============================
Unit tests for the Stage 1 YAML configuration loader.
Run with:  pytest tests/test_config_loader.py -v
"""

import textwrap
from pathlib import Path

import pytest
import yaml

from config.config_loader import CloudDashConfig, load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_VALID_CONFIG = textwrap.dedent("""
    global:
      max_history_turns: 20
      default_temperature: 0.2
      llm_model: "gemini-3.1-flash-lite"
      llm_timeout_seconds: 30
      llm_max_retries: 3

    routing:
      intent_to_agent:
        technical:   "technical_support_agent"
        billing:     "billing_agent"
        general:     "triage_agent"
        escalation:  "escalation_agent"
        unknown:     "triage_agent"
      auto_escalate_intents:
        - "refund_request"
      confidence_threshold: 0.65

    agents:
      triage_agent:
        display_name: "Triage Agent"
        temperature: 0.0
        system_prompt: "You are the triage agent."
      technical_support_agent:
        display_name: "Technical Support Agent"
        temperature: 0.1
        system_prompt: "You are the technical support agent."
      billing_agent:
        display_name: "Billing Agent"
        temperature: 0.0
        system_prompt: "You are the billing agent."
      escalation_agent:
        display_name: "Escalation Agent"
        temperature: 0.3
        system_prompt: "You are the escalation agent."
""")


@pytest.fixture()
def valid_config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "agents_config.yaml"
    cfg.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_config(self, valid_config_file: Path):
        cfg = load_config(valid_config_file)
        assert isinstance(cfg, CloudDashConfig)

    def test_global_settings(self, valid_config_file: Path):
        cfg = load_config(valid_config_file)
        assert cfg.global_settings.max_history_turns == 20
        assert cfg.global_settings.llm_model == "gemini-3.1-flash-lite"

    def test_routing_loaded(self, valid_config_file: Path):
        cfg = load_config(valid_config_file)
        assert cfg.routing.intent_to_agent["billing"] == "billing_agent"
        assert cfg.routing.confidence_threshold == 0.65

    def test_agents_loaded(self, valid_config_file: Path):
        cfg = load_config(valid_config_file)
        assert "triage_agent" in cfg.agents
        assert cfg.agents["triage_agent"].display_name == "Triage Agent"

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_missing_intent_key_raises(self, tmp_path: Path):
        bad = yaml.safe_load(MINIMAL_VALID_CONFIG)
        del bad["routing"]["intent_to_agent"]["billing"]  # remove required key
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(yaml.dump(bad), encoding="utf-8")
        with pytest.raises(Exception):  # pydantic.ValidationError
            load_config(cfg_file)

    def test_routing_references_undefined_agent_raises(self, tmp_path: Path):
        bad = yaml.safe_load(MINIMAL_VALID_CONFIG)
        bad["routing"]["intent_to_agent"]["billing"] = "nonexistent_agent"
        cfg_file = tmp_path / "bad2.yaml"
        cfg_file.write_text(yaml.dump(bad), encoding="utf-8")
        with pytest.raises(Exception):
            load_config(cfg_file)

    def test_empty_system_prompt_raises(self, tmp_path: Path):
        bad = yaml.safe_load(MINIMAL_VALID_CONFIG)
        bad["agents"]["triage_agent"]["system_prompt"] = "   "
        cfg_file = tmp_path / "bad3.yaml"
        cfg_file.write_text(yaml.dump(bad), encoding="utf-8")
        with pytest.raises(Exception):
            load_config(cfg_file)


class TestConfigHelpers:
    def test_get_agent_prompt(self, valid_config_file: Path):
        cfg = load_config(valid_config_file)
        prompt = cfg.get_agent_prompt("triage_agent")
        assert "triage" in prompt.lower()

    def test_get_agent_config_missing_raises(self, valid_config_file: Path):
        cfg = load_config(valid_config_file)
        with pytest.raises(KeyError):
            cfg.get_agent_config("nonexistent_agent")

    def test_resolve_intent_known(self, valid_config_file: Path):
        cfg = load_config(valid_config_file)
        assert cfg.resolve_intent("billing") == "billing_agent"

    def test_resolve_intent_unknown_fallback(self, valid_config_file: Path):
        cfg = load_config(valid_config_file)
        assert cfg.resolve_intent("gibberish") == "triage_agent"

    def test_get_agent_temperature(self, valid_config_file: Path):
        cfg = load_config(valid_config_file)
        assert cfg.get_agent_temperature("triage_agent") == 0.0
