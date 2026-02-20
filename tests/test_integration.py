"""Integration test for the full dual-arm pipeline."""
import pytest
import numpy as np
import faiss
import polars as pl

from content_tower import ContentTower
from reranker import Reranker, compute_alpha
from inference import DualArmEngine
from embedding_store import EmbeddingManager, InMemoryProfileStore
from graduation import GraduationManager
from llm_encoder import IntentParser
from chat_provider import format_prompt


def _build_test_content_tower(n=100, dim=1024):
    vecs = np.random.randn(n, dim).astype(np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    metadata = pl.DataFrame({
        "faiss_idx": list(range(n)),
        "movieId": list(range(1, n + 1)),
        "title": [f"Movie {i}" for i in range(1, n + 1)],
        "genres": ["Action"] * 50 + ["Comedy"] * 50,
    })
    return ContentTower(index, metadata), vecs


class TestFullPipeline:
    def test_content_only_flow(self):
        """User with no history, just a query."""
        tower, vecs = _build_test_content_tower()
        engine = DualArmEngine(content_tower=tower, mamba_model=None)

        query_vec = np.random.randn(1024).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        results = engine.recommend(query_embedding=query_vec, top_k=10)
        assert len(results) == 10
        # Scores should be descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_profile_builds_from_interactions(self):
        """Interactions should build a user profile."""
        tower, vecs = _build_test_content_tower()
        engine = DualArmEngine(content_tower=tower, mamba_model=None)

        # Simulate some positive interactions
        engine.record_interaction(user_id=1, movie_id=1, rating_weight=2.0)
        engine.record_interaction(user_id=1, movie_id=2, rating_weight=1.0)

        profile = engine.embedding_manager.profile_store.get(1)
        assert profile is not None
        assert profile.interaction_count == 2
        assert profile.vector.shape == (1024,)

    def test_graduation_flow(self):
        """New movie should graduate after enough interactions."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            gm = GraduationManager(
                queue_path=Path(tmpdir) / "queue.json",
                graduation_threshold=5,
                mamba_catalog=set(),
            )

            for i in range(4):
                assert gm.record_interaction("tt_new_001") is False
            assert gm.record_interaction("tt_new_001") is True

            pending = gm.get_pending_graduations()
            assert "tt_new_001" in pending

            gm.mark_retrained(["tt_new_001"], batch_id="test_batch")
            assert len(gm.get_pending_graduations()) == 0
            assert len(gm.get_completed()) == 1

    def test_group_recommendation(self):
        """Group recommendation should work with multiple users."""
        tower, vecs = _build_test_content_tower()
        engine = DualArmEngine(content_tower=tower, mamba_model=None)

        users = [
            {"user_id": 1, "profile_embedding": vecs[0]},
            {"user_id": 2, "profile_embedding": vecs[50]},
        ]

        results = engine.recommend_group(
            users=users,
            query_text=None,
            top_k=5,
            fairness_lambda=0.5,
        )
        assert len(results) <= 5

    def test_intent_parser_integration(self):
        parser = IntentParser()
        result = parser.parse("I want a scary thriller from the 90s")
        assert "scary" in result["mood"]
        assert "thriller" in result["genre"]
        assert "90s" in result["era"]

    def test_format_prompt_integration(self):
        recs = [
            {"title": "Alien", "year": 1979, "genres": "Horror, Sci-Fi", "plot_snippet": "In space..."},
        ]
        prompt = format_prompt("something scary in space", recs)
        assert "Alien" in prompt
        assert "something scary in space" in prompt

    def test_alpha_computation_spectrum(self):
        """Alpha should increase with query specificity."""
        a_none = compute_alpha(query_text=None, has_profile=False)
        a_profile = compute_alpha(query_text=None, has_profile=True)
        a_short = compute_alpha(query_text="fun movies")
        a_long = compute_alpha(
            query_text="a dark psychological thriller set in a small town in the 1990s about a detective"
        )
        assert a_none == 0.0
        assert a_profile > 0.0
        assert a_long > a_short
