"""
Inference Server for Mamba4Rec with LLM Fusion

This module provides a high-level inference API for getting recommendations
that incorporate both sequential behavior and LLM-derived mood signals.

Usage:
    from inference import RecommendationEngine

    engine = RecommendationEngine.load("mamba_production.pt")
    recommendations = engine.recommend(
        user_id=123,
        session_id="abc-def-ghi",
        mood_text="I want something cozy from the early 2000s"
    )
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from mamba4rec import Mamba4RecFusion
from llm_projection import LLMProjectionWithAlignment
from llm_encoder import LLMEncoder, IntentParser
from embedding_store import EmbeddingManager, InMemoryMoodStore, InMemoryProfileStore


@dataclass
class Recommendation:
    """Single recommendation with metadata."""
    item_id: int
    score: float
    rank: int


@dataclass
class RecommendationResult:
    """Full recommendation result with explanations."""
    recommendations: List[Recommendation]
    gate_value: Optional[float]  # How much the model trusted Mamba vs LLM
    parsed_intent: Optional[Dict[str, Any]]
    raw_mood_text: Optional[str]


class RecommendationEngine:
    """
    High-level engine for mood-aware movie recommendations.

    Combines:
    - Mamba4Rec for sequential behavior modeling
    - LLM encoder for mood/intent understanding
    - Embedding stores for mood and profile persistence
    """

    def __init__(
        self,
        model: Mamba4RecFusion,
        llm_encoder: LLMEncoder,
        embedding_manager: EmbeddingManager,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        item_id_mapping: Optional[Dict[int, Any]] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.llm_encoder = llm_encoder
        self.embedding_manager = embedding_manager
        self.device = device
        self.item_id_mapping = item_id_mapping or {}

        # Intent parser for structured extraction
        self.intent_parser = IntentParser()

        # Cache item embeddings
        with torch.no_grad():
            item_emb = model.item_embedding.weight.detach()
            self.embedding_manager.set_item_embeddings(item_emb)

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        llm_model: str = "all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> "RecommendationEngine":
        """
        Load engine from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to config (uses checkpoint config if not provided)
            llm_model: Sentence transformer model name
            device: Device to run on

        Returns:
            Initialized RecommendationEngine
        """
        from recbole.config import Config
        from recbole.data import create_dataset

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load config
        if config_path:
            config = Config(model=Mamba4RecFusion, config_file_list=[config_path])
        else:
            config = checkpoint.get('config')
            if config is None:
                raise ValueError("No config in checkpoint and config_path not provided")

        # Create dataset to get item count
        dataset = create_dataset(config)

        # Initialize model
        model = Mamba4RecFusion(config, dataset)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Initialize LLM encoder
        llm_encoder = LLMEncoder(model_name=llm_model, device=device)

        # Initialize embedding manager
        embedding_manager = EmbeddingManager(
            hidden_size=config['hidden_size']
        )

        return cls(
            model=model,
            llm_encoder=llm_encoder,
            embedding_manager=embedding_manager,
            device=device
        )

    def recommend(
        self,
        user_id: int,
        session_id: str,
        item_history: List[int],
        mood_text: Optional[str] = None,
        top_k: int = 10,
        exclude_history: bool = True
    ) -> RecommendationResult:
        """
        Get personalized recommendations.

        Args:
            user_id: User identifier
            session_id: Session identifier
            item_history: List of item IDs user has interacted with
            mood_text: Optional mood/intent text from user
            top_k: Number of recommendations to return
            exclude_history: Whether to exclude already-watched items

        Returns:
            RecommendationResult with top-k recommendations
        """
        parsed_intent = None

        # Process mood text if provided
        if mood_text:
            # Parse intent
            parsed_intent = self.intent_parser.parse(mood_text)

            # Encode with LLM
            raw_mood_emb = self.llm_encoder.encode_mood(mood_text)

            # Project to Mamba space (if model has projection)
            if self.model.llm_projection is not None:
                with torch.no_grad():
                    projected_mood = self.model.llm_projection(
                        raw_mood_emb.unsqueeze(0).to(self.device)
                    )

                # Update mood store
                self.embedding_manager.update_mood(
                    session_id=session_id,
                    projected_mood=projected_mood.squeeze(0),
                    raw_text=mood_text,
                    parsed_intent=parsed_intent
                )

        # Prepare interaction dict
        interaction = self._prepare_interaction(
            item_history=item_history,
            session_id=session_id,
            user_id=user_id
        )

        # Get scores
        with torch.no_grad():
            scores = self.model.full_sort_predict(interaction)
            scores = scores.squeeze(0)  # [n_items]

            # Get gate value for interpretability
            gate_value = None
            if self.model.use_llm_fusion and hasattr(self.model, 'get_fusion_weights'):
                gate = self.model.get_fusion_weights(interaction)
                if gate is not None:
                    gate_value = gate.mean().item()

        # Exclude history if requested
        if exclude_history:
            for item_id in item_history:
                if 0 <= item_id < len(scores):
                    scores[item_id] = float('-inf')

        # Get top-k
        top_scores, top_indices = torch.topk(scores, k=top_k)

        recommendations = [
            Recommendation(
                item_id=idx.item(),
                score=score.item(),
                rank=i + 1
            )
            for i, (idx, score) in enumerate(zip(top_indices, top_scores))
        ]

        return RecommendationResult(
            recommendations=recommendations,
            gate_value=gate_value,
            parsed_intent=parsed_intent,
            raw_mood_text=mood_text
        )

    def record_feedback(
        self,
        user_id: int,
        session_id: str,
        item_id: int,
        feedback_type: str
    ):
        """
        Record user feedback on a recommendation.

        Args:
            user_id: User identifier
            session_id: Session identifier
            item_id: Item that received feedback
            feedback_type: "click", "like", "dislike", or "ignore"
        """
        feedback_map = {
            "click": 0.5,
            "like": 1.0,
            "dislike": -1.0,
            "ignore": 0.0
        }
        feedback_signal = feedback_map.get(feedback_type, 0.0)

        self.embedding_manager.record_feedback(
            user_id=user_id,
            session_id=session_id,
            feedback=feedback_signal
        )

    def _prepare_interaction(
        self,
        item_history: List[int],
        session_id: str,
        user_id: int
    ) -> Dict[str, torch.Tensor]:
        """Prepare interaction dict for model."""
        # Pad or truncate history
        max_len = self.model.max_seq_length if hasattr(self.model, 'max_seq_length') else 200

        if len(item_history) > max_len:
            item_history = item_history[-max_len:]

        seq_len = len(item_history)
        padded = [0] * (max_len - seq_len) + item_history

        interaction = {
            self.model.ITEM_SEQ: torch.tensor([padded], device=self.device),
            self.model.ITEM_SEQ_LEN: torch.tensor([seq_len], device=self.device),
        }

        # Add LLM embeddings if available
        llm_embeddings = self.embedding_manager.prepare_interaction_dict(
            session_id=session_id,
            user_id=user_id,
            device=self.device
        )
        interaction.update(llm_embeddings)

        return interaction

    def explain_recommendation(
        self,
        result: RecommendationResult
    ) -> str:
        """
        Generate human-readable explanation of recommendations.

        Args:
            result: RecommendationResult from recommend()

        Returns:
            Explanation string
        """
        lines = []

        if result.raw_mood_text:
            lines.append(f"Based on your mood: \"{result.raw_mood_text}\"")

        if result.parsed_intent:
            if result.parsed_intent.get("mood"):
                lines.append(f"  Detected mood: {', '.join(result.parsed_intent['mood'])}")
            if result.parsed_intent.get("genre"):
                lines.append(f"  Detected genre: {', '.join(result.parsed_intent['genre'])}")
            if result.parsed_intent.get("era"):
                lines.append(f"  Detected era: {', '.join(result.parsed_intent['era'])}")

        if result.gate_value is not None:
            mamba_pct = result.gate_value * 100
            llm_pct = (1 - result.gate_value) * 100
            lines.append(f"\nBlending: {mamba_pct:.0f}% watch history + {llm_pct:.0f}% current mood")

        lines.append("\nTop recommendations:")
        for rec in result.recommendations[:5]:
            item_name = self.item_id_mapping.get(rec.item_id, f"Item {rec.item_id}")
            lines.append(f"  {rec.rank}. {item_name} (score: {rec.score:.3f})")

        return "\n".join(lines)


class BatchRecommender:
    """
    Batch recommendation engine for offline processing.

    Efficiently processes many users at once for batch inference.
    """

    def __init__(
        self,
        model: Mamba4RecFusion,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def recommend_batch(
        self,
        item_sequences: List[List[int]],
        sequence_lengths: List[int],
        mood_embeddings: Optional[torch.Tensor] = None,
        profile_embeddings: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> List[List[Tuple[int, float]]]:
        """
        Get recommendations for a batch of users.

        Args:
            item_sequences: List of item sequences [B, max_len]
            sequence_lengths: Length of each sequence [B]
            mood_embeddings: Optional mood embeddings [B, hidden_size]
            profile_embeddings: Optional profile embeddings [B, hidden_size]
            top_k: Number of recommendations per user

        Returns:
            List of recommendations, each a list of (item_id, score) tuples
        """
        # Pad sequences
        max_len = max(len(seq) for seq in item_sequences)
        padded = [
            [0] * (max_len - len(seq)) + seq
            for seq in item_sequences
        ]

        interaction = {
            self.model.ITEM_SEQ: torch.tensor(padded, device=self.device),
            self.model.ITEM_SEQ_LEN: torch.tensor(sequence_lengths, device=self.device),
        }

        if mood_embeddings is not None:
            interaction["LLM_MOOD_EMB"] = mood_embeddings.to(self.device)
        if profile_embeddings is not None:
            interaction["LLM_PROFILE_EMB"] = profile_embeddings.to(self.device)

        with torch.no_grad():
            scores = self.model.full_sort_predict(interaction)  # [B, n_items]
            top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)

        results = []
        for i in range(len(item_sequences)):
            user_recs = [
                (top_indices[i, j].item(), top_scores[i, j].item())
                for j in range(top_k)
            ]
            results.append(user_recs)

        return results
