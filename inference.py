"""Dual-arm recommendation engine.

Orchestrates Mamba (behavioral) and Content Tower (semantic) arms,
computes alpha, calls the reranker, handles single and group recommendations.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

from content_tower import ContentTower
from embedding_store import EmbeddingManager
from llm_encoder import IntentParser, LLMEncoder
from reranker import Reranker, compute_alpha, min_max_normalize


@dataclass
class RecommendationResult:
    recommendations: List[Dict]
    alpha: float
    parsed_intent: Optional[Dict] = None
    mode: str = "dual"  # "dual", "content_only", "mamba_only"


class DualArmEngine:
    """Orchestrates Mamba + Content Tower for recommendations."""

    def __init__(
        self,
        content_tower: Optional[ContentTower] = None,
        mamba_model: Optional[Any] = None,
        encoder: Optional[LLMEncoder] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        reranker: Optional[Reranker] = None,
        mamba_catalog: Optional[Set[int]] = None,
        device: str = "cpu",
        home_feed_alpha: float = 0.2,
        alpha_min: float = 0.3,
        alpha_max: float = 0.9,
    ):
        self.content_tower = content_tower
        self.mamba_model = mamba_model
        self.encoder = encoder or LLMEncoder()
        self.embedding_manager = embedding_manager or EmbeddingManager(hidden_size=1024)
        self.reranker = reranker or Reranker()
        self.intent_parser = IntentParser()
        self.mamba_catalog = mamba_catalog or set()
        self.device = device
        self.home_feed_alpha = home_feed_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def recommend(
        self,
        user_id: Optional[int] = None,
        item_history: Optional[List[int]] = None,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        profile_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        alpha_override: Optional[float] = None,
        exclude_history: bool = True,
    ) -> List[Dict]:
        """Generate recommendations using both arms.

        Args:
            user_id: User ID for Mamba and profile lookup.
            item_history: User's interaction history (item IDs) for Mamba.
            query_text: Natural language query (for chatbot mode).
            query_embedding: Pre-computed query embedding (optional).
            profile_embedding: Pre-computed profile embedding (optional).
            top_k: Number of recommendations to return.
            alpha_override: Override automatic alpha computation.
            exclude_history: Whether to exclude already-seen items.

        Returns:
            List of recommendation dicts with movie_id, score, etc.
        """
        parsed_intent = None

        # Encode query if text provided
        if query_text and query_embedding is None:
            query_embedding = self.encoder.encode_query(query_text).cpu().numpy()
            parsed_intent = self.intent_parser.parse(query_text)

        # Get profile if user_id provided
        if user_id is not None and profile_embedding is None:
            profile_vec = self.embedding_manager.get_profile_vector(user_id)
            if profile_vec is not None:
                profile_embedding = profile_vec.numpy() if isinstance(profile_vec, torch.Tensor) else profile_vec

        has_profile = profile_embedding is not None

        # Compute alpha
        if alpha_override is not None:
            alpha = alpha_override
        else:
            alpha = compute_alpha(
                query_text=query_text,
                has_profile=has_profile,
                home_feed_alpha=self.home_feed_alpha,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
            )

        # Content arm
        content_scores = {}
        if self.content_tower and (query_embedding is not None or profile_embedding is not None):
            content_results = self.content_tower.search(
                query_embedding=query_embedding,
                profile_embedding=profile_embedding,
                alpha=alpha,
                top_k=50,
            )
            content_scores = {r["movieId"]: r["score"] for r in content_results}

        # Mamba arm
        mamba_scores = {}
        if self.mamba_model is not None and item_history:
            mamba_scores = self._get_mamba_scores(item_history, top_k=50)

        # Exclude history
        if exclude_history and item_history:
            history_set = set(item_history)
            content_scores = {k: v for k, v in content_scores.items() if k not in history_set}
            mamba_scores = {k: v for k, v in mamba_scores.items() if k not in history_set}

        # Determine mode
        if content_scores and mamba_scores:
            mode = "dual"
        elif content_scores:
            mode = "content_only"
        elif mamba_scores:
            mode = "mamba_only"
        else:
            return []

        # Rerank
        results = self.reranker.blend(mamba_scores, content_scores, alpha=alpha, top_k=top_k)

        return results

    def recommend_group(
        self,
        users: List[Dict],
        query_text: Optional[str] = None,
        top_k: int = 10,
        fairness_lambda: float = 0.5,
    ) -> List[Dict]:
        """Generate group recommendations.

        Args:
            users: List of user dicts, each with 'user_id' and optionally
                   'item_history' and 'profile_embedding'.
            query_text: Shared group query text.
            top_k: Number of recommendations.
            fairness_lambda: Penalty for score disagreement.

        Returns:
            List of recommendation dicts.
        """
        query_embedding = None
        if query_text:
            query_embedding = self.encoder.encode_query(query_text).cpu().numpy()

        # Gather candidates from all users
        all_candidates = set()

        for user in users:
            user_id = user.get("user_id")
            history = user.get("item_history", [])
            profile = user.get("profile_embedding")

            if self.mamba_model and history:
                mamba_top = self._get_mamba_scores(history, top_k=50)
                all_candidates |= set(mamba_top.keys())

            if self.content_tower and profile is not None:
                profile_results = self.content_tower.search(
                    profile_embedding=profile, top_k=30
                )
                all_candidates |= {r["movieId"] for r in profile_results}

        if self.content_tower and query_embedding is not None:
            query_results = self.content_tower.search(
                query_embedding=query_embedding, top_k=50
            )
            all_candidates |= {r["movieId"] for r in query_results}

        if not all_candidates:
            return []

        # Score each candidate per user
        per_movie_scores = {}
        for movie_id in all_candidates:
            user_scores = []
            for user in users:
                score = self._score_movie_for_user(movie_id, user, query_embedding)
                user_scores.append(score)
            per_movie_scores[movie_id] = user_scores

        return self.reranker.rank_group(
            per_movie_scores, fairness_lambda=fairness_lambda, top_k=top_k
        )

    def record_interaction(
        self,
        user_id: int,
        movie_id: int,
        rating_weight: float,
    ):
        """Record a user interaction and update their content profile.

        Args:
            user_id: User ID.
            movie_id: Movie they interacted with.
            rating_weight: Signal strength (+2 for 5-star, -2 for 1-star,
                          +1 for right swipe, -1 for left swipe).
        """
        if self.content_tower:
            plot_emb = self.content_tower.get_movie_embedding(movie_id)
            if plot_emb is not None:
                self.embedding_manager.profile_store.update_with_rating(
                    user_id=user_id,
                    plot_embedding=plot_emb,
                    rating_weight=rating_weight,
                )

    def _get_mamba_scores(self, item_history: List[int], top_k: int = 50) -> Dict[int, float]:
        """Get top-K scores from Mamba for a user's history."""
        if self.mamba_model is None:
            return {}

        with torch.no_grad():
            item_seq = torch.tensor([item_history], dtype=torch.long, device=self.device)
            item_seq_len = torch.tensor([len(item_history)], device=self.device)
            seq_output = self.mamba_model.forward(item_seq, item_seq_len)
            all_item_emb = self.mamba_model.item_embedding.weight
            scores = torch.matmul(seq_output, all_item_emb.T).squeeze(0)

        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        return {
            int(idx): float(score)
            for idx, score in zip(top_indices.cpu(), top_scores.cpu())
        }

    def _score_movie_for_user(
        self,
        movie_id: int,
        user: Dict,
        query_embedding: Optional[np.ndarray],
    ) -> float:
        """Score a single movie for a single user (for group ranking)."""
        alpha = compute_alpha(
            query_text=None if query_embedding is None else "query",
            has_profile=user.get("profile_embedding") is not None,
            home_feed_alpha=self.home_feed_alpha,
        )

        content_score = 0.0
        if self.content_tower:
            movie_emb = self.content_tower.get_movie_embedding(movie_id)
            if movie_emb is not None:
                profile = user.get("profile_embedding")
                if profile is not None:
                    content_score = float(np.dot(movie_emb, profile))
                elif query_embedding is not None:
                    content_score = float(np.dot(movie_emb, query_embedding))

        mamba_score = 0.0
        if self.mamba_model and user.get("item_history"):
            full_scores = self._get_mamba_scores(user["item_history"], top_k=100)
            mamba_score = full_scores.get(movie_id, 0.0)

        return alpha * content_score + (1 - alpha) * mamba_score
