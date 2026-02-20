"""
Embedding Store for Mamba4Rec Fusion

This module provides storage interfaces for:
- Mood vectors (session-based, short-lived)
- Profile vectors (user-based, persistent)
- Item embeddings cache

Supports multiple backends:
- In-memory (for development/testing)
- Redis (for production mood vectors)
- PostgreSQL with pgvector (for production profile vectors)
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import torch
import numpy as np


@dataclass
class MoodEntry:
    """Mood vector entry with metadata."""
    vector: np.ndarray
    raw_text: str
    parsed_intent: Dict[str, Any]
    timestamp: float
    session_id: str


@dataclass
class ProfileEntry:
    """User profile entry with metadata."""
    vector: np.ndarray
    interaction_count: int
    last_updated: float
    created_at: float
    user_id: int


class BaseMoodStore(ABC):
    """Abstract base class for mood vector storage."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[MoodEntry]:
        """Retrieve mood vector for session."""
        pass

    @abstractmethod
    def set(self, session_id: str, entry: MoodEntry, ttl: int = 1800):
        """Store mood vector with TTL (default 30 minutes)."""
        pass

    @abstractmethod
    def delete(self, session_id: str):
        """Delete mood vector for session."""
        pass


class BaseProfileStore(ABC):
    """Abstract base class for profile vector storage."""

    @abstractmethod
    def get(self, user_id: int) -> Optional[ProfileEntry]:
        """Retrieve profile vector for user."""
        pass

    @abstractmethod
    def set(self, user_id: int, entry: ProfileEntry):
        """Store/update profile vector."""
        pass

    @abstractmethod
    def update_with_feedback(
        self,
        user_id: int,
        mood_vector: np.ndarray,
        feedback_signal: float
    ):
        """Update profile based on mood and feedback."""
        pass


# ==================== In-Memory Implementations ====================

class InMemoryMoodStore(BaseMoodStore):
    """
    In-memory mood store for development and testing.

    Entries expire after TTL seconds.
    """

    def __init__(self):
        self._store: Dict[str, tuple] = {}  # session_id -> (entry, expire_time)

    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired = [k for k, (_, exp) in self._store.items() if exp < current_time]
        for k in expired:
            del self._store[k]

    def get(self, session_id: str) -> Optional[MoodEntry]:
        self._cleanup_expired()
        if session_id in self._store:
            entry, expire_time = self._store[session_id]
            if expire_time > time.time():
                return entry
            else:
                del self._store[session_id]
        return None

    def set(self, session_id: str, entry: MoodEntry, ttl: int = 1800):
        expire_time = time.time() + ttl
        self._store[session_id] = (entry, expire_time)

    def delete(self, session_id: str):
        if session_id in self._store:
            del self._store[session_id]


class InMemoryProfileStore(BaseProfileStore):
    """
    In-memory profile store for development and testing.
    """

    def __init__(self, decay: float = 0.95, base_lr: float = 0.1, dim: int = 1024):
        self._store: Dict[int, ProfileEntry] = {}
        self._mood_history: Dict[int, List[tuple]] = {}  # user_id -> [(mood, feedback, time)]
        self.decay = decay
        self.base_lr = base_lr
        self.dim = dim

    def get(self, user_id: int) -> Optional[ProfileEntry]:
        return self._store.get(user_id)

    def set(self, user_id: int, entry: ProfileEntry):
        self._store[user_id] = entry

    def update_with_feedback(
        self,
        user_id: int,
        mood_vector: np.ndarray,
        feedback_signal: float
    ):
        """
        Update profile with weighted mood based on feedback.

        Args:
            user_id: User ID
            mood_vector: Current mood embedding
            feedback_signal: +1 (liked), -1 (disliked), 0 (ignored)
        """
        current_time = time.time()

        # Initialize profile if not exists
        if user_id not in self._store:
            self._store[user_id] = ProfileEntry(
                vector=np.zeros_like(mood_vector),
                interaction_count=0,
                last_updated=current_time,
                created_at=current_time,
                user_id=user_id
            )

        profile = self._store[user_id]

        # Weight the mood by feedback
        weighted_mood = feedback_signal * mood_vector

        # Adaptive learning rate: faster early, slower later
        lr = self.base_lr / (1 + 0.01 * profile.interaction_count)

        # Exponential moving average update
        new_vector = self.decay * profile.vector + lr * weighted_mood

        # Normalize to prevent drift
        norm = np.linalg.norm(new_vector)
        if norm > 1e-8:
            new_vector = new_vector / norm

        # Update profile
        profile.vector = new_vector
        profile.interaction_count += 1
        profile.last_updated = current_time

        # Log mood history
        if user_id not in self._mood_history:
            self._mood_history[user_id] = []
        self._mood_history[user_id].append((mood_vector.copy(), feedback_signal, current_time))

    def update_with_rating(self, user_id: int, plot_embedding: np.ndarray, rating_weight: float):
        """Update user profile from a plot embedding weighted by rating.

        rating_weight = rating - 3 for 5-star scale (so 5->+2, 1->-2)
        rating_weight = +1 for right swipe, -1 for left swipe
        """
        existing = self.get(user_id)
        weighted = rating_weight * plot_embedding

        if existing is None:
            norm = np.linalg.norm(weighted)
            vector = weighted / norm if norm > 0 else weighted
            self.set(user_id, ProfileEntry(
                vector=vector,
                interaction_count=1,
                last_updated=time.time(),
                created_at=time.time(),
                user_id=user_id,
            ))
        else:
            lr = self.base_lr / (1 + 0.01 * existing.interaction_count)
            new_vector = self.decay * existing.vector + lr * weighted
            norm = np.linalg.norm(new_vector)
            if norm > 0:
                new_vector = new_vector / norm
            self.set(user_id, ProfileEntry(
                vector=new_vector,
                interaction_count=existing.interaction_count + 1,
                last_updated=time.time(),
                created_at=existing.created_at,
                user_id=user_id,
            ))

    def get_mood_history(self, user_id: int, limit: int = 10) -> List[tuple]:
        """Get recent mood history for a user."""
        history = self._mood_history.get(user_id, [])
        return history[-limit:] if limit else history


# ==================== Redis Implementation ====================

class RedisMoodStore(BaseMoodStore):
    """
    Redis-based mood store for production use.

    Requires redis-py: pip install redis
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        try:
            import redis
            self._redis = redis.Redis(host=host, port=port, db=db)
        except ImportError:
            raise ImportError("redis is required. Install with: pip install redis")

        self._prefix = "mood:"

    def _serialize(self, entry: MoodEntry) -> str:
        """Serialize entry to JSON."""
        return json.dumps({
            "vector": entry.vector.tolist(),
            "raw_text": entry.raw_text,
            "parsed_intent": entry.parsed_intent,
            "timestamp": entry.timestamp,
            "session_id": entry.session_id
        })

    def _deserialize(self, data: str) -> MoodEntry:
        """Deserialize JSON to entry."""
        obj = json.loads(data)
        return MoodEntry(
            vector=np.array(obj["vector"], dtype=np.float32),
            raw_text=obj["raw_text"],
            parsed_intent=obj["parsed_intent"],
            timestamp=obj["timestamp"],
            session_id=obj["session_id"]
        )

    def get(self, session_id: str) -> Optional[MoodEntry]:
        data = self._redis.get(f"{self._prefix}{session_id}")
        if data is not None:
            return self._deserialize(data.decode())
        return None

    def set(self, session_id: str, entry: MoodEntry, ttl: int = 1800):
        self._redis.setex(
            f"{self._prefix}{session_id}",
            ttl,
            self._serialize(entry)
        )

    def delete(self, session_id: str):
        self._redis.delete(f"{self._prefix}{session_id}")


# ==================== Unified Embedding Manager ====================

class EmbeddingManager:
    """
    Unified manager for all embedding storage operations.

    Coordinates between mood store, profile store, and item embeddings.
    """

    def __init__(
        self,
        mood_store: Optional[BaseMoodStore] = None,
        profile_store: Optional[BaseProfileStore] = None,
        hidden_size: int = 1024
    ):
        self.mood_store = mood_store or InMemoryMoodStore()
        self.profile_store = profile_store or InMemoryProfileStore()
        self.hidden_size = hidden_size

        # Item embeddings cache (updated when model is loaded)
        self._item_embeddings: Optional[np.ndarray] = None

    def set_item_embeddings(self, embeddings: torch.Tensor):
        """Cache item embeddings from model."""
        self._item_embeddings = embeddings.detach().cpu().numpy()

    def get_mood_vector(self, session_id: str) -> Optional[torch.Tensor]:
        """Get mood vector as PyTorch tensor."""
        entry = self.mood_store.get(session_id)
        if entry is not None:
            return torch.from_numpy(entry.vector).float()
        return None

    def update_mood(
        self,
        session_id: str,
        projected_mood: torch.Tensor,
        raw_text: str,
        parsed_intent: Dict[str, Any],
        blend_factor: float = 0.7
    ):
        """
        Update session mood with exponential moving average blending.

        Args:
            session_id: Session identifier
            projected_mood: New mood vector (already projected to hidden_size)
            raw_text: Original user message
            parsed_intent: Parsed intent dictionary
            blend_factor: Weight for new mood (0-1)
        """
        current_time = time.time()
        mood_np = projected_mood.detach().cpu().numpy()

        # Blend with existing mood if present
        existing = self.mood_store.get(session_id)
        if existing is not None:
            mood_np = blend_factor * mood_np + (1 - blend_factor) * existing.vector

        entry = MoodEntry(
            vector=mood_np,
            raw_text=raw_text,
            parsed_intent=parsed_intent,
            timestamp=current_time,
            session_id=session_id
        )
        self.mood_store.set(session_id, entry)

    def get_profile_vector(self, user_id: int) -> Optional[torch.Tensor]:
        """Get profile vector as PyTorch tensor."""
        entry = self.profile_store.get(user_id)
        if entry is not None:
            return torch.from_numpy(entry.vector).float()
        return None

    def record_feedback(
        self,
        user_id: int,
        session_id: str,
        feedback: float
    ):
        """
        Record user feedback and update profile.

        Args:
            user_id: User identifier
            session_id: Session identifier
            feedback: Feedback signal (+1 liked, -1 disliked, 0 ignored)
        """
        mood_entry = self.mood_store.get(session_id)
        if mood_entry is not None:
            self.profile_store.update_with_feedback(
                user_id=user_id,
                mood_vector=mood_entry.vector,
                feedback_signal=feedback
            )

    def prepare_interaction_dict(
        self,
        session_id: str,
        user_id: int,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare LLM embedding fields for interaction dict.

        This creates the LLM_MOOD_EMB and LLM_PROFILE_EMB fields that
        the Mamba4RecFusion model expects.

        Args:
            session_id: Session identifier
            user_id: User identifier
            device: Device to place tensors on

        Returns:
            Dict with LLM_MOOD_EMB and LLM_PROFILE_EMB keys (or empty dict)
        """
        result = {}

        mood_vector = self.get_mood_vector(session_id)
        if mood_vector is not None:
            result["LLM_MOOD_EMB"] = mood_vector.unsqueeze(0).to(device)

        profile_vector = self.get_profile_vector(user_id)
        if profile_vector is not None:
            result["LLM_PROFILE_EMB"] = profile_vector.unsqueeze(0).to(device)

        return result
