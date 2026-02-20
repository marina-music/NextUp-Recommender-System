"""Tests for reranker module."""
import pytest


class TestScoreNormalization:
    def test_min_max_normalize(self):
        from reranker import min_max_normalize
        scores = {"a": 10.0, "b": 20.0, "c": 30.0}
        normed = min_max_normalize(scores)
        assert normed["a"] == pytest.approx(0.0)
        assert normed["b"] == pytest.approx(0.5)
        assert normed["c"] == pytest.approx(1.0)

    def test_min_max_single_value(self):
        from reranker import min_max_normalize
        scores = {"a": 5.0}
        normed = min_max_normalize(scores)
        assert normed["a"] == pytest.approx(1.0)


class TestComputeAlpha:
    def test_no_query_no_profile(self):
        from reranker import compute_alpha
        alpha = compute_alpha(query_text=None, has_profile=False)
        assert alpha == 0.0

    def test_no_query_with_profile(self):
        from reranker import compute_alpha
        alpha = compute_alpha(query_text=None, has_profile=True, home_feed_alpha=0.2)
        assert alpha == pytest.approx(0.2)

    def test_short_query(self):
        from reranker import compute_alpha
        alpha = compute_alpha(query_text="something fun", has_profile=True)
        assert 0.3 <= alpha <= 0.9

    def test_long_specific_query(self):
        from reranker import compute_alpha
        alpha = compute_alpha(
            query_text="a dark psychological thriller set in the 90s about a detective",
            has_profile=True,
        )
        assert alpha > 0.5  # should lean toward content


class TestReranker:
    def test_blend_scores(self):
        from reranker import Reranker
        rr = Reranker()
        mamba_scores = {1: 0.9, 2: 0.5, 3: 0.1}
        content_scores = {1: 0.2, 2: 0.8, 4: 0.7}
        result = rr.blend(mamba_scores, content_scores, alpha=0.5, top_k=3)
        assert len(result) == 3
        # Each result should have movie_id and score
        assert all("movie_id" in r and "score" in r for r in result)

    def test_content_only_for_new_movies(self):
        from reranker import Reranker
        rr = Reranker()
        mamba_scores = {}
        content_scores = {1: 0.9, 2: 0.7, 3: 0.5}
        result = rr.blend(mamba_scores, content_scores, alpha=0.8, top_k=3)
        assert len(result) == 3
        assert result[0]["movie_id"] == 1

    def test_mamba_only_for_no_plot(self):
        from reranker import Reranker
        rr = Reranker()
        mamba_scores = {1: 0.9, 2: 0.7}
        content_scores = {}
        result = rr.blend(mamba_scores, content_scores, alpha=0.5, top_k=2)
        assert len(result) == 2


class TestGroupRanking:
    def test_group_penalizes_disagreement(self):
        from reranker import Reranker
        rr = Reranker()
        # Movie A: user1 loves (0.9), user2 hates (0.1) -> high std
        # Movie B: both like (0.6, 0.7) -> low std
        per_user_scores = {
            "A": [0.9, 0.1],
            "B": [0.6, 0.7],
        }
        result = rr.rank_group(per_user_scores, fairness_lambda=0.5, top_k=2)
        # B should rank higher than A despite lower mean
        assert result[0]["movie_id"] == "B"
