"""
knowledge_base package — KB article generation and article data.

Modules
-------
generate_kb.py   — Generates the 18 sample KB articles as JSON files.

Usage
-----
    python -m knowledge_base.generate_kb
"""
from knowledge_base.generate_kb import generate_articles, ARTICLES

__all__ = ["generate_articles", "ARTICLES"]
