"""Tests for plot encoding and FAISS index building."""
import pytest
import numpy as np


class TestPrependMetadata:
    def test_prepends_metadata(self):
        from pipeline.encode_plots import prepend_metadata
        result = prepend_metadata("Neo discovers...", "Film", "Action, Sci-Fi", 1999, "The Matrix")
        assert result == "Film. Action, Sci-Fi. 1999. The Matrix. Neo discovers..."

    def test_handles_missing_year(self):
        from pipeline.encode_plots import prepend_metadata
        result = prepend_metadata("A story...", "Film", "Drama", None, "Unknown")
        assert result == "Film. Drama. Unknown. A story..."

    def test_handles_missing_genre(self):
        from pipeline.encode_plots import prepend_metadata
        result = prepend_metadata("A story...", "Film", None, 2020, "Movie")
        assert result == "Film. 2020. Movie. A story..."


class TestBuildFaissIndex:
    def test_build_index_shape(self):
        from pipeline.encode_plots import build_faiss_index
        import faiss
        embeddings = np.random.randn(10, 1024).astype(np.float32)
        index = build_faiss_index(embeddings)
        assert index.ntotal == 10
        assert index.d == 1024

    def test_search_returns_correct_k(self):
        from pipeline.encode_plots import build_faiss_index
        embeddings = np.random.randn(20, 1024).astype(np.float32)
        # Normalize for inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        index = build_faiss_index(embeddings)
        query = embeddings[0:1]
        scores, ids = index.search(query, 5)
        assert ids.shape == (1, 5)
        assert ids[0, 0] == 0  # closest to itself
