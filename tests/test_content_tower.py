"""Tests for content tower."""
import numpy as np
import faiss
import polars as pl
import tempfile
from pathlib import Path


def _make_test_index(n=50, dim=1024):
    """Create a test FAISS index with random normalized vectors."""
    vecs = np.random.randn(n, dim).astype(np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index, vecs


def _make_test_metadata(n=50):
    """Create test metadata DataFrame."""
    return pl.DataFrame({
        "faiss_idx": list(range(n)),
        "movieId": list(range(1, n + 1)),
        "title": [f"Movie {i}" for i in range(1, n + 1)],
    })


class TestContentTower:
    def test_search_by_query(self):
        from content_tower import ContentTower
        index, vecs = _make_test_index(50)
        metadata = _make_test_metadata(50)
        tower = ContentTower(index, metadata)

        query_vec = vecs[0]  # should match itself
        results = tower.search(query_embedding=query_vec, top_k=5)
        assert len(results) == 5
        assert results[0]["movieId"] == 1  # closest to itself

    def test_search_by_profile(self):
        from content_tower import ContentTower
        index, vecs = _make_test_index(50)
        metadata = _make_test_metadata(50)
        tower = ContentTower(index, metadata)

        profile = vecs[10]
        results = tower.search(profile_embedding=profile, top_k=5)
        assert len(results) == 5
        assert results[0]["movieId"] == 11

    def test_search_blended(self):
        from content_tower import ContentTower
        index, vecs = _make_test_index(50)
        metadata = _make_test_metadata(50)
        tower = ContentTower(index, metadata)

        results = tower.search(
            query_embedding=vecs[0],
            profile_embedding=vecs[10],
            alpha=0.5,
            top_k=5,
        )
        assert len(results) == 5

    def test_search_returns_scores(self):
        from content_tower import ContentTower
        index, vecs = _make_test_index(50)
        metadata = _make_test_metadata(50)
        tower = ContentTower(index, metadata)

        results = tower.search(query_embedding=vecs[0], top_k=3)
        for r in results:
            assert "score" in r
            assert "movieId" in r

    def test_add_movie(self):
        from content_tower import ContentTower
        index, vecs = _make_test_index(50)
        metadata = _make_test_metadata(50)
        tower = ContentTower(index, metadata)

        new_vec = np.random.randn(1024).astype(np.float32)
        new_vec /= np.linalg.norm(new_vec)
        tower.add_movie(new_vec, {"movieId": 999, "title": "New Movie"})
        assert tower.index.ntotal == 51

    def test_save_and_load(self):
        from content_tower import ContentTower
        index, vecs = _make_test_index(10)
        metadata = _make_test_metadata(10)
        tower = ContentTower(index, metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.faiss"
            meta_path = Path(tmpdir) / "test_meta.parquet"
            tower.save(idx_path, meta_path)

            loaded = ContentTower.load(idx_path, meta_path)
            assert loaded.index.ntotal == 10
