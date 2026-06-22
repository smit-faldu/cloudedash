"""
tests package — Pytest test suite.

Structure mirrors the main package layout:
tests/
  test_models.py          — Unit tests for Pydantic models (Stage 1)
  test_config_loader.py   — Unit tests for YAML config loader (Stage 1)
  test_rag.py             — Tests for RAG pipeline (Stage 2)
  test_tools.py           — Tests for LangChain tools (Stage 3)
  test_agents.py          — Tests for agent nodes (Stage 4)
  test_graph.py           — Integration tests for LangGraph flow (Stage 5)
  test_api.py             — FastAPI endpoint tests (Stage 7)
"""
