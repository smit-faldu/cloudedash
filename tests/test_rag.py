"""
tests/test_rag.py
=================
Stage 2 tests for the Knowledge Base generation and RAG retrieval pipeline.

Test strategy
-------------
- KB generation tests are pure I/O — no model needed.
- Ingestion tests mock the embedding model to avoid downloading weights in CI.
- Retriever tests use a pre-built in-memory FAISS store (no disk I/O needed).
- Query rewrite tests stub the LLM so tests run offline.

Run
---
    pytest tests/test_rag.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge_base.generate_kb import ARTICLES, generate_articles


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def generated_articles_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate articles once per session into a temp directory."""
    out = tmp_path_factory.mktemp("articles")
    generate_articles(out)
    return out


@pytest.fixture(scope="session")
def article_files(generated_articles_dir: Path) -> list[Path]:
    return sorted(f for f in generated_articles_dir.glob("*.json") if f.name != "index.json")


# ===========================================================================
# KB Generation Tests
# ===========================================================================

class TestKBGeneration:

    def test_correct_article_count(self, article_files: list[Path]):
        """Should generate exactly as many files as ARTICLES list."""
        assert len(article_files) == len(ARTICLES)

    def test_no_article_fewer_than_15(self):
        """Task spec requires at least 15 articles."""
        assert len(ARTICLES) >= 15

    def test_all_categories_represented(self):
        categories = {a["category"] for a in ARTICLES}
        assert categories == {"faq", "troubleshooting", "billing", "api_docs"}

    def test_each_article_has_required_fields(self):
        required = {"id", "category", "title", "content", "tags", "version"}
        for article in ARTICLES:
            missing = required - article.keys()
            assert not missing, f"Article {article.get('id')} missing fields: {missing}"

    def test_article_ids_are_unique(self):
        ids = [a["id"] for a in ARTICLES]
        assert len(ids) == len(set(ids)), "Duplicate article IDs found"

    def test_article_content_is_non_empty(self):
        for article in ARTICLES:
            assert article["content"].strip(), f"Empty content in article {article['id']}"

    def test_index_json_written(self, generated_articles_dir: Path):
        index_path = generated_articles_dir / "index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text())
        assert len(index) == len(ARTICLES)

    def test_index_has_correct_fields(self, generated_articles_dir: Path):
        index = json.loads((generated_articles_dir / "index.json").read_text())
        required = {"id", "category", "title", "tags", "version", "file"}
        for entry in index:
            assert required <= entry.keys()

    def test_article_json_is_valid(self, article_files: list[Path]):
        for fpath in article_files:
            data = json.loads(fpath.read_text())
            assert "id" in data
            assert "content" in data

    def test_faq_articles_exist(self):
        faq = [a for a in ARTICLES if a["category"] == "faq"]
        assert len(faq) >= 3

    def test_troubleshooting_articles_have_error_codes(self):
        ts = [a for a in ARTICLES if a["category"] == "troubleshooting"]
        assert len(ts) >= 3
        # At least some troubleshooting articles should mention error codes
        with_codes = [a for a in ts if any("ERR-" in tag for tag in a["tags"])]
        assert len(with_codes) >= 2

    def test_billing_articles_exist(self):
        billing = [a for a in ARTICLES if a["category"] == "billing"]
        assert len(billing) >= 3

    def test_api_docs_articles_exist(self):
        api = [a for a in ARTICLES if a["category"] == "api_docs"]
        assert len(api) >= 3


# ===========================================================================
# Ingestion Tests
# ===========================================================================

class TestIngestionPipeline:

    def test_load_articles_returns_documents(self, generated_articles_dir: Path):
        from retrieval.ingest import load_articles
        docs = load_articles(generated_articles_dir)
        assert len(docs) == len(ARTICLES)

    def test_document_metadata_is_correct(self, generated_articles_dir: Path):
        from retrieval.ingest import load_articles
        docs = load_articles(generated_articles_dir)
        for doc in docs:
            assert "source" in doc.metadata
            assert "category" in doc.metadata
            assert "title" in doc.metadata
            assert doc.page_content.strip()

    def test_chunk_documents_produces_more_chunks_than_articles(self, generated_articles_dir: Path):
        from retrieval.ingest import load_articles, chunk_documents
        docs = load_articles(generated_articles_dir)
        chunks = chunk_documents(docs)
        # Long articles should produce multiple chunks
        assert len(chunks) > len(docs)

    def test_chunks_inherit_source_metadata(self, generated_articles_dir: Path):
        from retrieval.ingest import load_articles, chunk_documents
        docs = load_articles(generated_articles_dir)
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "source" in chunk.metadata, "Chunk missing 'source' metadata"
            assert "chunk_index" in chunk.metadata, "Chunk missing 'chunk_index'"

    def test_chunk_size_respected(self, generated_articles_dir: Path):
        from retrieval.ingest import load_articles, chunk_documents, CHUNK_SIZE
        docs = load_articles(generated_articles_dir)
        # Use a very small chunk size to force many splits
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        oversized = [c for c in chunks if len(c.page_content) > 200 + 50]  # 50-char tolerance
        assert not oversized, f"{len(oversized)} chunks exceed target size"

    def test_load_articles_raises_on_empty_dir(self, tmp_path: Path):
        from retrieval.ingest import load_articles
        with pytest.raises(FileNotFoundError):
            load_articles(tmp_path)

    @patch("retrieval.ingest.HuggingFaceEmbeddings")
    @patch("retrieval.ingest.FAISS")
    def test_build_and_save_index_saves_manifest(
        self,
        mock_faiss_cls: MagicMock,
        mock_emb_cls: MagicMock,
        tmp_path: Path,
        generated_articles_dir: Path,
    ):
        """Verify manifest.json is written alongside the FAISS index."""
        import retrieval.ingest  # ensure module is loaded before patch resolves  # noqa: F401
        from langchain_core.documents import Document
        from retrieval.ingest import build_and_save_index

        # Mock FAISS to avoid actually building an index
        mock_vs = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_vs

        fake_chunks = [
            Document(
                page_content="test",
                metadata={"source": "FAQ-001", "category": "faq", "title": "Test"},
            )
        ]
        fake_embeddings = MagicMock()

        build_and_save_index(fake_chunks, fake_embeddings, tmp_path)

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists(), "manifest.json was not created"
        manifest = json.loads(manifest_path.read_text())
        assert "chunk_count" in manifest
        assert manifest["chunk_count"] == 1
        assert manifest["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"


# ===========================================================================
# Retriever Tests
# ===========================================================================

class TestRetriever:

    @pytest.fixture()
    def mock_vectorstore(self):
        """A mock FAISS vectorstore that returns predictable results."""
        from langchain_core.documents import Document

        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="ERR-4012 occurs when IAM credentials expire.",
                    metadata={
                        "source": "TS-001",
                        "category": "troubleshooting",
                        "title": "Error ERR-4012: AWS credentials expired",
                        "tags": "ERR-4012, aws, iam",
                        "chunk_index": 0,
                    },
                ),
                0.87,
            ),
            (
                Document(
                    page_content="To fix ERR-4012, re-authenticate in Settings.",
                    metadata={
                        "source": "TS-001",
                        "category": "troubleshooting",
                        "title": "Error ERR-4012: AWS credentials expired",
                        "tags": "ERR-4012, aws",
                        "chunk_index": 1,
                    },
                ),
                0.80,
            ),
            (
                Document(
                    page_content="CloudDash supports AWS, GCP, and Azure.",
                    metadata={
                        "source": "FAQ-002",
                        "category": "faq",
                        "title": "Which cloud providers does CloudDash support?",
                        "tags": "aws, gcp, azure",
                        "chunk_index": 0,
                    },
                ),
                0.72,
            ),
        ]
        return mock_vs

    @patch("retrieval.retriever._get_vectorstore")
    def test_retrieve_returns_results(self, mock_get_vs: MagicMock, mock_vectorstore: MagicMock):
        from retrieval.retriever import retrieve_with_context
        mock_get_vs.return_value = mock_vectorstore

        results = retrieve_with_context("ERR-4012 error", llm=None)

        assert len(results) > 0

    @patch("retrieval.retriever._get_vectorstore")
    def test_deduplication_keeps_highest_score(
        self, mock_get_vs: MagicMock, mock_vectorstore: MagicMock
    ):
        """TS-001 appears twice in candidates — only the highest-score chunk should survive."""
        from retrieval.retriever import retrieve_with_context
        mock_get_vs.return_value = mock_vectorstore

        results = retrieve_with_context("ERR-4012 error", llm=None)
        source_ids = [r.source_id for r in results]

        # TS-001 appears twice in candidates but should only appear once in results
        assert source_ids.count("TS-001") == 1

    @patch("retrieval.retriever._get_vectorstore")
    def test_results_sorted_by_score_descending(
        self, mock_get_vs: MagicMock, mock_vectorstore: MagicMock
    ):
        from retrieval.retriever import retrieve_with_context
        mock_get_vs.return_value = mock_vectorstore

        results = retrieve_with_context("ERR-4012 error", llm=None)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("retrieval.retriever._get_vectorstore")
    def test_results_contain_citation_fields(
        self, mock_get_vs: MagicMock, mock_vectorstore: MagicMock
    ):
        from retrieval.retriever import retrieve_with_context
        mock_get_vs.return_value = mock_vectorstore

        results = retrieve_with_context("my AWS integration keeps disconnecting", llm=None)
        for r in results:
            assert r.source_id
            assert r.title
            assert r.content
            assert 0.0 <= r.score <= 1.0

    @patch("retrieval.retriever._get_vectorstore")
    def test_min_score_filter_removes_low_confidence(self, mock_get_vs: MagicMock):
        """All candidates below min_score should be dropped."""
        from langchain_core.documents import Document
        from retrieval.retriever import retrieve_with_context

        low_score_vs = MagicMock()
        low_score_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Unrelated result.",
                    metadata={
                        "source": "FAQ-001",
                        "category": "faq",
                        "title": "What is CloudDash?",
                        "tags": "overview",
                        "chunk_index": 0,
                    },
                ),
                0.10,  # Below MIN_SCORE of 0.30
            )
        ]
        mock_get_vs.return_value = low_score_vs

        results = retrieve_with_context("unrelated query", min_score=0.30, llm=None)
        assert results == []

    def test_format_results_for_prompt_contains_source_ids(self, mock_vectorstore: MagicMock):
        from retrieval.retriever import RetrievalResult, format_results_for_prompt

        results = [
            RetrievalResult(
                source_id="TS-001",
                category="troubleshooting",
                title="Error ERR-4012",
                content="Check your IAM role.",
                score=0.87,
            )
        ]
        formatted = format_results_for_prompt(results)
        assert "TS-001" in formatted
        assert "Check your IAM role." in formatted

    def test_format_results_empty_returns_no_data_message(self):
        from retrieval.retriever import format_results_for_prompt
        result = format_results_for_prompt([])
        assert "No relevant" in result

    def test_get_source_ids_deduplicates(self):
        from retrieval.retriever import RetrievalResult, get_source_ids

        results = [
            RetrievalResult("TS-001", "troubleshooting", "T1", "c1", 0.9),
            RetrievalResult("TS-001", "troubleshooting", "T1", "c2", 0.8),
            RetrievalResult("FAQ-001", "faq", "F1", "c3", 0.7),
        ]
        ids = get_source_ids(results)
        assert ids == ["TS-001", "FAQ-001"]
        assert len(ids) == 2

    def test_index_not_ready_error_on_missing_index(self, tmp_path: Path):
        from retrieval.retriever import IndexNotReadyError, _get_vectorstore

        # Reset the module singleton so we don't use a cached store
        import retrieval.retriever as rmod
        original = rmod._vectorstore
        rmod._vectorstore = None

        try:
            with pytest.raises(IndexNotReadyError, match="python -m retrieval.ingest"):
                _get_vectorstore(tmp_path)  # empty dir — no index.faiss
        finally:
            rmod._vectorstore = original  # restore

    def test_query_rewrite_falls_back_on_llm_error(self):
        from retrieval.retriever import rewrite_query

        failing_llm = MagicMock()
        failing_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        result = rewrite_query(
            query="ERR-4012 keeps happening",
            conversation_history=[{"role": "user", "content": "I see ERR-4012"}],
            llm=failing_llm,
        )
        # Should return the original query unchanged
        assert result == "ERR-4012 keeps happening"

    def test_query_rewrite_no_history_returns_original(self):
        from retrieval.retriever import rewrite_query

        result = rewrite_query(
            query="What plans are available?",
            conversation_history=[],
            llm=None,
        )
        assert result == "What plans are available?"
