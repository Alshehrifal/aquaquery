"""Tests for ChromaDB knowledge base indexer."""

import tempfile
from pathlib import Path

import pytest

from backend.config import Settings
from backend.data.indexer import ARGO_KNOWLEDGE_DOCS, ArgoKnowledgeIndexer


@pytest.fixture
def temp_settings(tmp_path):
    """Create settings with temporary directories."""
    return Settings(
        anthropic_api_key="test-key",
        embeddings_dir=tmp_path / "embeddings",
        sample_data_dir=tmp_path / "sample",
        data_dir=tmp_path,
    )


@pytest.fixture
def indexer(temp_settings):
    """Create an indexer with temporary storage."""
    temp_settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    idx = ArgoKnowledgeIndexer(settings=temp_settings)
    return idx


class TestKnowledgeDocs:
    def test_docs_not_empty(self):
        assert len(ARGO_KNOWLEDGE_DOCS) > 0

    def test_docs_have_required_fields(self):
        for doc in ARGO_KNOWLEDGE_DOCS:
            assert "id" in doc
            assert "category" in doc
            assert "content" in doc
            assert len(doc["content"]) > 50

    def test_docs_have_unique_ids(self):
        ids = [doc["id"] for doc in ARGO_KNOWLEDGE_DOCS]
        assert len(ids) == len(set(ids))

    def test_docs_cover_categories(self):
        categories = {doc["category"] for doc in ARGO_KNOWLEDGE_DOCS}
        assert "argo_program" in categories
        assert "variables" in categories
        assert "ocean_concepts" in categories
        assert "ocean_basins" in categories


class TestArgoKnowledgeIndexer:
    def test_index_knowledge_base(self, indexer):
        count = indexer.index_knowledge_base()
        assert count == len(ARGO_KNOWLEDGE_DOCS)

    def test_index_is_idempotent(self, indexer):
        count1 = indexer.index_knowledge_base()
        count2 = indexer.index_knowledge_base()
        assert count1 == count2

    def test_search_returns_results(self, indexer):
        indexer.index_knowledge_base()
        results = indexer.search("What is the Argo program?")
        assert len(results) > 0
        assert len(results) <= 3  # default top_k

    def test_search_result_has_fields(self, indexer):
        indexer.index_knowledge_base()
        results = indexer.search("temperature")
        assert len(results) > 0
        result = results[0]
        assert "id" in result
        assert "content" in result
        assert "category" in result
        assert "distance" in result

    def test_search_relevance(self, indexer):
        indexer.index_knowledge_base()
        results = indexer.search("What is ocean temperature?")
        contents = " ".join(r["content"].lower() for r in results)
        assert "temperature" in contents or "temp" in contents

    def test_search_with_category_filter(self, indexer):
        indexer.index_knowledge_base()
        results = indexer.search("ocean", category="ocean_basins")
        for r in results:
            assert r["category"] == "ocean_basins"

    def test_search_custom_top_k(self, indexer):
        indexer.index_knowledge_base()
        results = indexer.search("ocean data", top_k=5)
        assert len(results) == 5

    def test_collection_property(self, indexer):
        collection = indexer.collection
        assert collection is not None
        assert collection.name == "argo_knowledge"

    def test_reset_clears_collection(self, indexer):
        indexer.index_knowledge_base()
        assert indexer.collection.count() > 0
        indexer.reset()
        new_collection = indexer.collection
        assert new_collection.count() == 0
