"""Tests for dual-arm inference engine."""
import pytest
import numpy as np
import faiss
import polars as pl
from unittest.mock import MagicMock, patch


def _mock_content_tower(n=50, dim=1024):
    """Create a mock ContentTower."""
    from content_tower import ContentTower
    vecs = np.random.randn(n, dim).astype(np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    metadata = pl.DataFrame({
        "faiss_idx": list(range(n)),
        "movieId": list(range(1, n + 1)),
        "title": [f"Movie {i}" for i in range(1, n + 1)],
    })
    return ContentTower(index, metadata)


def _mock_mamba_model():
    """Create a mock Mamba model that returns random scores."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    return model


class TestDualArmEngine:
    def test_has_recommend_method(self):
        from inference import DualArmEngine
        assert hasattr(DualArmEngine, "recommend")

    def test_has_recommend_group_method(self):
        from inference import DualArmEngine
        assert hasattr(DualArmEngine, "recommend_group")

    def test_content_only_recommendation(self):
        """When no Mamba model, should still return content-based results."""
        from inference import DualArmEngine
        tower = _mock_content_tower()
        engine = DualArmEngine(content_tower=tower, mamba_model=None)

        query_vec = np.random.randn(1024).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        results = engine.recommend(
            query_embedding=query_vec,
            top_k=5,
        )
        assert len(results) == 5

    def test_result_has_required_fields(self):
        from inference import DualArmEngine
        tower = _mock_content_tower()
        engine = DualArmEngine(content_tower=tower, mamba_model=None)

        query_vec = np.random.randn(1024).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        results = engine.recommend(query_embedding=query_vec, top_k=3)
        for r in results:
            assert "movie_id" in r
            assert "score" in r
