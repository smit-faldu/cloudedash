import os
import unittest
from unittest.mock import patch, MagicMock
import pytest

from config.config_loader import get_config, load_config
from agents.agent_nodes import _build_llm
from retrieval.retriever import rewrite_query
from guardrails.output_guard import _llm_judge

@pytest.fixture(autouse=True)
def clean_env():
    """Ensure a clean environment for each test so overrides don't leak."""
    keys_to_remove = [
        "GEMINI_MODEL", "GEMINI_MODEL_DEFAULT",
        "GEMINI_MODEL_TRIAGE", "GEMINI_MODEL_TRIAGE_AGENT",
        "GEMINI_MODEL_TECHNICAL_SUPPORT", "GEMINI_MODEL_TECHNICAL_SUPPORT_AGENT",
        "GEMINI_MODEL_BILLING", "GEMINI_MODEL_BILLING_AGENT",
        "GEMINI_MODEL_ESCALATION", "GEMINI_MODEL_ESCALATION_AGENT",
        "GEMINI_MODEL_QUERY_REWRITE", "GEMINI_MODEL_REWRITE",
        "GEMINI_MODEL_LLM_JUDGE", "GEMINI_MODEL_JUDGE"
    ]
    saved = {}
    for k in keys_to_remove:
        if k in os.environ:
            saved[k] = os.environ[k]
            del os.environ[k]
    yield
    for k in keys_to_remove:
        if k in saved:
            os.environ[k] = saved[k]
        elif k in os.environ:
            del os.environ[k]

def test_global_settings_env_override():
    os.environ["GEMINI_MODEL_DEFAULT"] = "gemini-test-default"
    cfg = load_config()
    assert cfg.global_settings.llm_model == "gemini-test-default"

    del os.environ["GEMINI_MODEL_DEFAULT"]
    os.environ["GEMINI_MODEL"] = "gemini-test-global"
    cfg = load_config()
    assert cfg.global_settings.llm_model == "gemini-test-global"

@patch("agents.agent_nodes.ChatGoogleGenerativeAI")
def test_agent_nodes_env_override(mock_chat: MagicMock):
    # Temporarily set API key for test verification if not present
    original_key = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = "fake-key"
    try:
        # 1. Test triage agent specific override (with "_agent" suffix)
        os.environ["GEMINI_MODEL_TRIAGE_AGENT"] = "gemini-test-triage-agent"
        _build_llm("triage_agent")
        mock_chat.assert_called_with(
            model="gemini-test-triage-agent",
            google_api_key="fake-key",
            temperature=pytest.approx(0.0),
            max_retries=3,
            timeout=30
        )
        mock_chat.reset_mock()
        del os.environ["GEMINI_MODEL_TRIAGE_AGENT"]

        # 2. Test triage agent specific override (without "_agent" suffix)
        os.environ["GEMINI_MODEL_TRIAGE"] = "gemini-test-triage"
        _build_llm("triage_agent")
        mock_chat.assert_called_with(
            model="gemini-test-triage",
            google_api_key="fake-key",
            temperature=pytest.approx(0.0),
            max_retries=3,
            timeout=30
        )
        mock_chat.reset_mock()
        del os.environ["GEMINI_MODEL_TRIAGE"]

        # 3. Test tech support agent fallback to global override
        os.environ["GEMINI_MODEL"] = "gemini-test-global-fallback"
        _build_llm("technical_support_agent")
        mock_chat.assert_called_with(
            model="gemini-test-global-fallback",
            google_api_key="fake-key",
            temperature=pytest.approx(0.1),
            max_retries=3,
            timeout=30
        )
        mock_chat.reset_mock()
    finally:
        if original_key is not None:
            os.environ["GEMINI_API_KEY"] = original_key
        elif "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_query_rewriter_env_override(mock_chat: MagicMock):
    original_key = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = "fake-key"
    try:
        os.environ["GEMINI_MODEL_QUERY_REWRITE"] = "gemini-rewrite-test"
        try:
            rewrite_query("hello", [{"role": "user", "content": "hi"}])
        except Exception:
            pass
        mock_chat.assert_called_with(
            model="gemini-rewrite-test",
            google_api_key="fake-key",
            temperature=0.0
        )
        mock_chat.reset_mock()
        del os.environ["GEMINI_MODEL_QUERY_REWRITE"]

        os.environ["GEMINI_MODEL_REWRITE"] = "gemini-rewrite-test-2"
        try:
            rewrite_query("hello", [{"role": "user", "content": "hi"}])
        except Exception:
            pass
        mock_chat.assert_called_with(
            model="gemini-rewrite-test-2",
            google_api_key="fake-key",
            temperature=0.0
        )
    finally:
        if original_key is not None:
            os.environ["GEMINI_API_KEY"] = original_key
        elif "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_llm_judge_env_override(mock_chat: MagicMock):
    original_key = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = "fake-key"
    try:
        os.environ["GEMINI_MODEL_LLM_JUDGE"] = "gemini-judge-test"
        try:
            _llm_judge("response", "context", "trace")
        except Exception:
            pass
        mock_chat.assert_called_with(
            model="gemini-judge-test",
            google_api_key="fake-key",
            temperature=0.0,
            max_retries=2
        )
        mock_chat.reset_mock()
        del os.environ["GEMINI_MODEL_LLM_JUDGE"]

        os.environ["GEMINI_MODEL_JUDGE"] = "gemini-judge-test-2"
        try:
            _llm_judge("response", "context", "trace")
        except Exception:
            pass
        mock_chat.assert_called_with(
            model="gemini-judge-test-2",
            google_api_key="fake-key",
            temperature=0.0,
            max_retries=2
        )
    finally:
        if original_key is not None:
            os.environ["GEMINI_API_KEY"] = original_key
        elif "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
