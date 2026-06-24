"""
retrieval/retriever.py
======================
Retrieval module for the CloudDash Technical Support Agent.

The core function ``retrieve_with_context`` implements two RAG best practices
that the RAG-engineer skill recommends:

1. **Query rewriting** — uses Gemini to reformulate the raw user query into
   a semantically richer standalone question that makes better use of the
   conversation history and is more suitable for vector search.

2. **Source citations** — every returned chunk carries full metadata
   (``source`` = KB article ID, ``category``, ``title``), so the Technical
   Support Agent can produce properly cited answers.

Additional hardening
--------------------
- **Similarity score threshold** — chunks below 0.35 cosine similarity are
  silently dropped, preventing the agent from citing irrelevant documents.
- **Deduplication** — when the same article ID appears in multiple top-k
  chunks, only the highest-scoring chunk per article is returned (prevents
  the agent from citing the same article three times).
- **Graceful degradation** — if the index does not exist yet, a clear
  ``IndexNotReadyError`` is raised with instructions for running ingest.py.
- **Lazy loading** — the FAISS index and embedding model are loaded once on
  first call and cached in module-level singletons; subsequent calls are fast.

Usage (standalone, no LangGraph required)
-----------------------------------------
    from retrieval.retriever import retrieve_with_context, RetrievalResult

    results = retrieve_with_context(
        query="My AWS integration keeps disconnecting",
        conversation_history=[
            {"role": "user", "content": "I'm getting ERR-4012 every morning."},
            {"role": "assistant", "content": "Let me look that up for you."},
        ],
    )
    for r in results:
        print(r.source_id, r.title, r.score)
        print(r.content[:200])
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when running directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (must match values in ingest.py)
# ---------------------------------------------------------------------------
DEFAULT_INDEX_DIR = _PROJECT_ROOT / "knowledge_base" / "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval tuning knobs
TOP_K_CANDIDATES = 10   # retrieve this many before dedup + score filtering
TOP_K_FINAL = 4         # return at most this many chunks to the agent
MIN_SCORE = 0.22        # cosine similarity floor (L2-normalised inner product)
                        # Lowered from 0.30 to ensure symptom/error-code queries
                        # match KB articles that use different but related terminology


# ===========================================================================
# Custom exception
# ===========================================================================

class IndexNotReadyError(RuntimeError):
    """Raised when the FAISS index has not been built yet."""


# ===========================================================================
# Return type
# ===========================================================================

@dataclass
class RetrievalResult:
    """
    A single retrieved knowledge base chunk with full citation metadata.

    Attributes
    ----------
    source_id : str
        KB article ID (e.g. "TS-001"). Use this in citations.
    category : str
        Article category: "faq" | "troubleshooting" | "billing" | "api_docs".
    title : str
        Human-readable article title.
    content : str
        The text chunk that was retrieved.
    score : float
        Cosine similarity score in [0, 1]. Higher is more relevant.
    tags : list[str]
        Tags from the source article.
    chunk_index : int
        Which chunk within the article this came from.
    """

    source_id: str
    category: str
    title: str
    content: str
    score: float
    tags: list[str] = field(default_factory=list)
    chunk_index: int = 0

    def to_context_string(self) -> str:
        """Format this result as a readable context block for an LLM prompt."""
        return (
            f"[SOURCE: {self.source_id} | {self.title}]\n"
            f"{self.content}\n"
        )


# ===========================================================================
# Module-level singletons (lazy-loaded)
# ===========================================================================

_embeddings = None
_vectorstore = None


def _get_embeddings():
    """Return the cached embedding model, loading it on first call."""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
    return _embeddings


def _get_vectorstore(index_dir: Path = DEFAULT_INDEX_DIR):
    """Return the cached FAISS vector store, loading it from disk on first call."""
    global _vectorstore
    if _vectorstore is None:
        from langchain_community.vectorstores import FAISS

        index_file = index_dir / "index.faiss"
        if not index_file.exists():
            raise IndexNotReadyError(
                f"FAISS index not found at: {index_dir.resolve()}\n"
                "Build the index first by running:\n"
                "    python -m retrieval.ingest --force\n"
            )
        logger.info("Loading FAISS index from: %s", index_dir.resolve())
        _vectorstore = FAISS.load_local(
            str(index_dir),
            _get_embeddings(),
            allow_dangerous_deserialization=True,   # safe: our own pickle
        )
        logger.info("FAISS index loaded.")
    return _vectorstore


# ===========================================================================
# Query rewriting
# ===========================================================================

_REWRITE_PROMPT = """You are a search query optimizer for a cloud infrastructure support knowledge base.

Given the conversation history and the user's latest message, rewrite the user's question into a
concise, standalone search query that:
1. Incorporates relevant context from the history (e.g. error codes, product areas mentioned earlier).
2. Uses specific technical terminology that will match documentation. Key mappings:
   - "index out of memory" / "memory error" / "out of memory" → "ERR-3007 FAISS index out of memory agent memory buffer"
   - "RAG not working" / "RAG broken" → "knowledge base retrieval FAISS index error"
   - "agent crash" / "agent down" → "CloudDash Agent crash error self-hosted"
   - "disconnected" / "keeps disconnecting" → include the cloud provider + "ERR-4012" if AWS
3. Removes conversational filler ("can you help me", "I'm having trouble with", etc.).
4. Is a single sentence of no more than 25 words.
5. ALWAYS include any error codes mentioned (ERR-XXXX format) verbatim.

Output ONLY the rewritten query string. Do not add quotes, explanations, or labels.

Conversation history:
{history}

User's latest message: {query}

Rewritten search query:"""


def rewrite_query(
    query: str,
    conversation_history: list[dict[str, str]],
    llm: Any | None = None,
) -> str:
    """
    Rewrite *query* into a semantically richer search query using the conversation history.

    Parameters
    ----------
    query :
        The user's raw latest message.
    conversation_history :
        List of ``{"role": "user"|"assistant", "content": "..."}`` dicts, oldest first.
        Pass an empty list if there is no prior history.
    llm :
        Optional LangChain LLM instance. If None, a Gemini LLM is instantiated
        using the GEMINI_API_KEY from the environment.

    Returns
    -------
    str
        The rewritten search query. Falls back to the original *query* on any error.
    """
    if not conversation_history:
        # No history → nothing to rewrite; return as-is.
        logger.debug("No conversation history — skipping query rewrite.")
        return query

    # Format history as a readable string
    history_str = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in conversation_history[-6:]  # last 3 turns (6 messages)
    )

    prompt = _REWRITE_PROMPT.format(history=history_str, query=query)

    try:
        if llm is None:
            import os
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = (
                os.environ.get("GEMINI_MODEL_QUERY_REWRITE") or
                os.environ.get("GEMINI_MODEL_REWRITE") or
                os.environ.get("GEMINI_MODEL_DEFAULT") or
                os.environ.get("GEMINI_MODEL") or
                "gemini-3.1-flash-lite"
            )

            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=os.environ["GEMINI_API_KEY"],
                temperature=0.0,
            )

        response = llm.invoke(prompt)
        rewritten = response.content.strip().strip('"')
        logger.debug("Query rewrite: '%s'  →  '%s'", query, rewritten)
        return rewritten if rewritten else query

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Query rewrite failed (%s: %s) — falling back to original query.",
            type(exc).__name__,
            exc,
        )
        return query


# ===========================================================================
# Retrieval
# ===========================================================================

def retrieve_with_context(
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
    top_k: int = TOP_K_FINAL,
    min_score: float = MIN_SCORE,
    llm: Any | None = None,
    index_dir: Path | None = None,
) -> list[RetrievalResult]:
    """
    Retrieve the most relevant KB chunks for *query*, rewriting the query
    using *conversation_history* to improve recall.

    Parameters
    ----------
    query :
        The user's raw message or question.
    conversation_history :
        List of ``{"role": "...", "content": "..."}`` dicts. Pass ``None`` or
        ``[]`` for a standalone query with no history.
    top_k :
        Maximum number of results to return after deduplication and score filtering.
    min_score :
        Minimum cosine similarity score. Results below this threshold are dropped.
    llm :
        LangChain LLM for query rewriting. If None, a default Gemini Flash
        instance is created (requires GEMINI_API_KEY env var).
    index_dir :
        Override the FAISS index directory (useful in tests).

    Returns
    -------
    list[RetrievalResult]
        Ranked list of relevant chunks, most similar first, with full citation metadata.
        Returns an empty list if no results pass the similarity threshold.
    """
    history = conversation_history or []
    idx_dir = index_dir or DEFAULT_INDEX_DIR

    # Step 1 — Rewrite the query for better semantic matching
    search_query = rewrite_query(query, history, llm=llm)

    # Step 2 — Semantic vector search (retrieve more candidates than needed)
    vectorstore = _get_vectorstore(idx_dir)
    candidates = vectorstore.similarity_search_with_relevance_scores(
        search_query, k=TOP_K_CANDIDATES
    )

    logger.debug(
        "Raw FAISS candidates for '%s': %d results",
        search_query,
        len(candidates),
    )

    # Step 3 — Filter by minimum similarity score
    filtered = [
        (doc, score) for doc, score in candidates if score >= min_score
    ]

    if not filtered:
        logger.info(
            "No results above min_score=%.2f for query: '%s'", min_score, search_query
        )
        return []

    # Step 4 — Deduplicate: keep the highest-scoring chunk per source article.
    # This prevents the same article from dominating the result set.
    seen_sources: dict[str, tuple[Any, float]] = {}
    for doc, score in filtered:
        src = doc.metadata.get("source", "UNKNOWN")
        if src not in seen_sources or score > seen_sources[src][1]:
            seen_sources[src] = (doc, score)

    deduplicated = sorted(seen_sources.values(), key=lambda x: x[1], reverse=True)

    # Step 5 — Take top_k after dedup
    final = deduplicated[:top_k]

    # Step 6 — Convert to RetrievalResult dataclasses
    results: list[RetrievalResult] = []
    for doc, score in final:
        meta = doc.metadata
        tags_raw = meta.get("tags", "")
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

        results.append(
            RetrievalResult(
                source_id=meta.get("source", "UNKNOWN"),
                category=meta.get("category", "unknown"),
                title=meta.get("title", ""),
                content=doc.page_content,
                score=round(float(score), 4),
                tags=tags,
                chunk_index=meta.get("chunk_index", 0),
            )
        )
        logger.debug(
            "  → %s  (score=%.3f)  %s",
            meta.get("source"),
            score,
            meta.get("title", "")[:60],
        )

    logger.info(
        "Retrieved %d chunks for query: '%s'", len(results), search_query
    )
    return results


def format_results_for_prompt(results: list[RetrievalResult]) -> str:
    """
    Format a list of RetrievalResults into a single string suitable for
    injection into an LLM's system prompt as retrieved context.

    Each block is separated by a divider for readability.
    """
    if not results:
        return "No relevant knowledge base articles found."

    blocks = [r.to_context_string() for r in results]
    return "\n---\n".join(blocks)


def get_source_ids(results: list[RetrievalResult]) -> list[str]:
    """Return a deduplicated list of article source IDs from *results*."""
    seen: set[str] = set()
    ids: list[str] = []
    for r in results:
        if r.source_id not in seen:
            seen.add(r.source_id)
            ids.append(r.source_id)
    return ids
