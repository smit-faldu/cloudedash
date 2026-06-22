"""
retrieval package — FAISS ingestion pipeline and retrieval chain.

Public API
----------
    from retrieval.ingest import run_ingestion
    from retrieval.retriever import retrieve_with_context, RetrievalResult, format_results_for_prompt
"""
from retrieval.retriever import (
    RetrievalResult,
    IndexNotReadyError,
    retrieve_with_context,
    format_results_for_prompt,
    get_source_ids,
)

__all__ = [
    "RetrievalResult",
    "IndexNotReadyError",
    "retrieve_with_context",
    "format_results_for_prompt",
    "get_source_ids",
]
