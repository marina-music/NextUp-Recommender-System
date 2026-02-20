"""Graduation manager — tracks new movie interactions and retraining triggers."""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set


class GraduationManager:
    """Manages the lifecycle of new movies graduating into Mamba's catalog."""

    def __init__(
        self,
        queue_path: Path,
        graduation_threshold: int = 50,
        mamba_catalog: Optional[Set[str]] = None,
        retrain_on_graduation_count: int = 100,
    ):
        self._queue_path = queue_path
        self._threshold = graduation_threshold
        self._mamba_catalog = mamba_catalog or set()
        self._retrain_count = retrain_on_graduation_count

        self._interaction_counts: Dict[str, int] = {}
        self._pending: Dict[str, dict] = {}
        self._completed: List[dict] = []

        self._load()

    def _load(self):
        """Load state from disk if available."""
        if not self._queue_path.exists():
            return
        with open(self._queue_path) as f:
            data = json.load(f)
        for item in data.get("pending", []):
            self._pending[item["movie_id"]] = item
            self._interaction_counts[item["movie_id"]] = item.get("interaction_count", self._threshold)
        self._completed = data.get("completed", [])

    def save(self):
        """Persist state to disk."""
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pending": list(self._pending.values()),
            "completed": self._completed,
        }
        with open(self._queue_path, "w") as f:
            json.dump(data, f, indent=2)

    def record_interaction(self, movie_id: str) -> bool:
        """Record an interaction with a movie.

        Returns True if the movie just graduated (crossed threshold).
        """
        if movie_id in self._mamba_catalog:
            return False
        if movie_id in self._pending:
            self._interaction_counts[movie_id] = self._interaction_counts.get(movie_id, 0) + 1
            self._pending[movie_id]["interaction_count"] = self._interaction_counts[movie_id]
            return False

        self._interaction_counts[movie_id] = self._interaction_counts.get(movie_id, 0) + 1

        if self._interaction_counts[movie_id] >= self._threshold:
            self._pending[movie_id] = {
                "movie_id": movie_id,
                "graduated_at": time.strftime("%Y-%m-%d"),
                "interaction_count": self._interaction_counts[movie_id],
            }
            return True
        return False

    def get_interaction_count(self, movie_id: str) -> int:
        return self._interaction_counts.get(movie_id, 0)

    def get_pending_graduations(self) -> List[str]:
        return list(self._pending.keys())

    def get_completed(self) -> List[dict]:
        return list(self._completed)

    def should_retrain_by_threshold(self) -> bool:
        return len(self._pending) >= self._retrain_count

    def mark_retrained(self, movie_ids: List[str], batch_id: str):
        """Mark movies as retrained and move from pending to completed."""
        for mid in movie_ids:
            if mid in self._pending:
                entry = self._pending.pop(mid)
                entry["retrained_at"] = time.strftime("%Y-%m-%d")
                entry["batch"] = batch_id
                self._completed.append(entry)
                self._mamba_catalog.add(mid)

    def trigger_retrain(self, reason: str = "manual") -> Optional[List[str]]:
        """Get the list of movies to include in retraining.

        Returns None if no movies are pending.
        """
        pending = self.get_pending_graduations()
        if not pending:
            return None
        return pending
