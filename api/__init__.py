"""
api — FastAPI application for CloudDash Multi-Agent Support.

Modules
-------
api/server.py   — FastAPI app, CORS, /health, /chat, /history endpoints
api/schemas.py  — Pydantic request/response models
api/deps.py     — Dependency-injection helpers (graph, config, logging)
"""
from api.server import app

__all__ = ["app"]
