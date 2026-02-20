"""Reranker -- blends Mamba and Content Tower scores."""
import math
from typing import Dict, List, Optional

import numpy as np


def min_max_normalize(scores: Dict, epsilon: float = 1e-8) -> Dict:
    """Min-max normalize a dict of scores to [0, 1]."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    spread = hi - lo
    if spread < epsilon:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / spread for k, v in scores.items()}


def compute_alpha(
    query_text: Optional[str] = None,
    has_profile: bool = False,
    home_feed_alpha: float = 0.2,
    alpha_min: float = 0.3,
    alpha_max: float = 0.9,
) -> float:
    """Compute blend alpha from query specificity.

    Returns:
        Alpha in [0, 1] where higher = trust content more.
    """
    if query_text is None:
        if has_profile:
            return home_feed_alpha
        return 0.0

    words = query_text.split()
    word_count = len(words)

    # Simple heuristic: more words -> more specific -> higher alpha
    # Sigmoid mapping: 2 words -> ~0.4, 5 words -> ~0.6, 10+ words -> ~0.8
    raw = 0.3 * word_count - 1.0
    alpha = 1.0 / (1.0 + math.exp(-raw))

    return max(alpha_min, min(alpha_max, alpha))


class Reranker:
    """Blends scores from Mamba (behavioral) and Content Tower (semantic)."""

    def blend(
        self,
        mamba_scores: Dict,
        content_scores: Dict,
        alpha: float,
        top_k: int = 10,
    ) -> List[Dict]:
        """Blend Mamba and content scores.

        Args:
            mamba_scores: {movie_id: raw_score} from Mamba arm.
            content_scores: {movie_id: cosine_sim} from content arm.
            alpha: Blend weight (1.0 = content only, 0.0 = mamba only).
            top_k: Number of results.

        Returns:
            Sorted list of {movie_id, score, mamba_score, content_score}.
        """
        # Normalize mamba scores
        mamba_normed = min_max_normalize(mamba_scores)

        # Content scores are already cosine similarity [0, 1]
        all_ids = set(mamba_normed.keys()) | set(content_scores.keys())

        scored = []
        for mid in all_ids:
            c_score = content_scores.get(mid, 0.0)
            m_score = mamba_normed.get(mid, 0.0)

            has_content = mid in content_scores
            has_mamba = mid in mamba_normed

            if has_content and has_mamba:
                final = alpha * c_score + (1 - alpha) * m_score
            elif has_content:
                final = alpha * c_score
            else:
                final = (1 - alpha) * m_score

            scored.append({
                "movie_id": mid,
                "score": final,
                "mamba_score": m_score if has_mamba else None,
                "content_score": c_score if has_content else None,
            })

        scored.sort(key=lambda x: -x["score"])
        return scored[:top_k]

    def rank_group(
        self,
        per_user_scores: Dict[str, List[float]],
        fairness_lambda: float = 0.5,
        top_k: int = 10,
    ) -> List[Dict]:
        """Rank movies for a group using fairness-weighted aggregation.

        Args:
            per_user_scores: {movie_id: [user1_score, user2_score, ...]}.
            fairness_lambda: Penalty weight for score disagreement.
            top_k: Number of results.

        Returns:
            Sorted list of {movie_id, score, mean, std}.
        """
        scored = []
        for mid, user_scores in per_user_scores.items():
            arr = np.array(user_scores)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            group_score = mean - fairness_lambda * std
            scored.append({
                "movie_id": mid,
                "score": group_score,
                "mean": mean,
                "std": std,
            })

        scored.sort(key=lambda x: -x["score"])
        return scored[:top_k]
