"""
retrieval/ingest.py
===================
Ingestion pipeline for the CloudDash knowledge base.

Responsibilities
----------------
1. Load all KB articles from ``knowledge_base/articles/*.json``.
2. Convert each article into a LangChain ``Document`` with rich metadata.
3. Chunk documents using a sentence-aware splitter (RecursiveCharacterTextSplitter)
   with deliberate overlap so context is never severed at a chunk boundary.
4. Embed all chunks with ``all-MiniLM-L6-v2`` (local, no API key required).
5. Build a FAISS index and save it to disk for later reuse by the retriever.

Design decisions (per RAG-engineer skill)
------------------------------------------
- **Sentence-boundary splitting over fixed-token splitting** – RecursiveCharacterTextSplitter
  splits on ``\\n\\n`` → ``\\n`` → ``. `` → `` `` in that priority order, which closely
  follows natural paragraph/sentence boundaries.
- **200-token overlap** – prevents the retriever from missing context that spans
  two consecutive chunks.
- **Metadata preserved per chunk** – source article ID, category, title, tags, and
  chunk index are stored so the retriever can return citations.
- **Minimum similarity score filter** – enforced at retrieval time (retriever.py),
  not here, keeping ingestion simple and deterministic.

Run
---
    python -m retrieval.ingest [--articles-dir PATH] [--index-dir PATH] [--force]

Arguments
---------
--articles-dir  Path to the directory containing *.json KB articles.
                Default: knowledge_base/articles/
--index-dir     Directory where the FAISS index is saved.
                Default: knowledge_base/faiss_index/
--force         Re-generate articles before ingesting (runs generate_kb.py first).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root so this module works whether run as __main__ or imported
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ARTICLES_DIR = _PROJECT_ROOT / "knowledge_base" / "articles"
DEFAULT_INDEX_DIR = _PROJECT_ROOT / "knowledge_base" / "faiss_index"

# Splitter configuration — tuned for support KB articles (medium-length prose)
CHUNK_SIZE = 600         # characters (not tokens); keeps chunks under ~150 tokens
CHUNK_OVERLAP = 120      # ~20% overlap to preserve cross-boundary context
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

# Embedding model — matches the task spec and runs fully offline
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ===========================================================================
# Document loading
# ===========================================================================

def load_articles(articles_dir: Path) -> list[Document]:
    """
    Load every ``*.json`` file from *articles_dir* (excluding ``index.json``)
    and return a list of LangChain ``Document`` objects, one per article.

    Metadata stored on each Document:
    - ``source``    : article ID (e.g. "TS-001")
    - ``category``  : "faq" | "troubleshooting" | "billing" | "api_docs"
    - ``title``     : human-readable article title
    - ``tags``      : comma-separated tag string (FAISS metadata must be str/int/float)
    - ``version``   : doc version string
    - ``file``      : original filename
    """
    article_files = sorted(
        f for f in articles_dir.glob("*.json") if f.name != "index.json"
    )
    if not article_files:
        raise FileNotFoundError(
            f"No KB article JSON files found in: {articles_dir.resolve()}\n"
            "Run 'python -m knowledge_base.generate_kb' first."
        )

    docs: list[Document] = []
    for fpath in article_files:
        raw = json.loads(fpath.read_text(encoding="utf-8"))
        docs.append(
            Document(
                page_content=raw["content"],
                metadata={
                    "source": raw["id"],
                    "category": raw["category"],
                    "title": raw["title"],
                    "tags": ", ".join(raw.get("tags", [])),
                    "version": raw.get("version", "unknown"),
                    "file": fpath.name,
                },
            )
        )

    logger.info("Loaded %d articles from %s", len(docs), articles_dir)
    return docs


# ===========================================================================
# Chunking
# ===========================================================================

def chunk_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split *docs* into smaller, overlapping chunks using a sentence-aware splitter.

    Each chunk inherits the parent article's metadata and adds:
    - ``chunk_index`` : zero-based position of this chunk within its parent article

    This lets the retriever cite the originating article even when it only finds
    a sub-section of it.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: list[Document] = []
    for doc in docs:
        chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(chunks)
            all_chunks.append(chunk)

    logger.info(
        "Chunked %d articles → %d chunks  (size=%d, overlap=%d)",
        len(docs),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks


# ===========================================================================
# Embedding model
# ===========================================================================

def build_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Instantiate the HuggingFace sentence-transformers embedding model.
    The model is downloaded once and cached by sentence-transformers in
    ~/.cache/huggingface/ on subsequent runs.

    ``all-MiniLM-L6-v2`` produces 384-dimensional vectors and runs
    efficiently on CPU — ideal for a prototype with < 10k chunks.
    """
    logger.info("Loading embedding model: %s", model_name)
    t0 = time.perf_counter()

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,   # cosine similarity → dot product
            "batch_size": 32,
        },
    )

    elapsed = time.perf_counter() - t0
    logger.info("Embedding model loaded in %.1fs", elapsed)
    return embeddings


# ===========================================================================
# FAISS index
# ===========================================================================

def build_and_save_index(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    index_dir: Path,
) -> FAISS:
    """
    Embed *chunks* and save a FAISS ``IndexFlatIP`` (inner-product / cosine
    similarity because embeddings are L2-normalised) to *index_dir*.

    FAISS saves two files:
    - ``index.faiss`` — the raw vector index
    - ``index.pkl``   — pickled docstore and metadata

    Returns the in-memory FAISS vector store for immediate use.
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Embedding %d chunks and building FAISS index …", len(chunks))
    t0 = time.perf_counter()

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    elapsed = time.perf_counter() - t0
    logger.info("Index built in %.1fs", elapsed)

    vectorstore.save_local(str(index_dir))
    logger.info("FAISS index saved to: %s", index_dir.resolve())

    # Write a small manifest alongside the index for auditability
    manifest = {
        "chunk_count": len(chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "build_time_seconds": round(elapsed, 2),
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return vectorstore


# ===========================================================================
# Public orchestration function
# ===========================================================================

def run_ingestion(
    articles_dir: Path = DEFAULT_ARTICLES_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    force_regenerate: bool = False,
) -> FAISS:
    """
    Full ingestion pipeline: load → chunk → embed → save FAISS index.

    Parameters
    ----------
    articles_dir :
        Directory containing the *.json KB articles.
    index_dir :
        Directory where the FAISS index will be written.
    force_regenerate :
        If True, re-runs ``generate_kb.py`` before ingesting.

    Returns
    -------
    FAISS
        The in-memory vector store, ready for querying.
    """
    if force_regenerate:
        logger.info("--force flag set: regenerating KB articles …")
        from knowledge_base.generate_kb import generate_articles
        generate_articles(articles_dir)

    docs = load_articles(articles_dir)
    chunks = chunk_documents(docs)
    embeddings = build_embedding_model()
    vectorstore = build_and_save_index(chunks, embeddings, index_dir)

    logger.info(
        "✅  Ingestion complete — %d chunks indexed from %d articles",
        len(chunks),
        len(docs),
    )
    return vectorstore


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest CloudDash KB articles into a local FAISS vector store."
    )
    parser.add_argument(
        "--articles-dir",
        type=Path,
        default=DEFAULT_ARTICLES_DIR,
        help=f"Path to KB article JSON files (default: {DEFAULT_ARTICLES_DIR})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"Path to save FAISS index (default: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate KB articles before ingesting",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    run_ingestion(
        articles_dir=args.articles_dir,
        index_dir=args.index_dir,
        force_regenerate=args.force,
    )
