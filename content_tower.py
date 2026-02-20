"""Content Tower — FAISS-based semantic movie retrieval."""
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import polars as pl


class ContentTower:
    """Manages a FAISS index of movie plot embeddings for content retrieval."""

    def __init__(self, index: faiss.IndexFlatIP, metadata: pl.DataFrame):
        self.index = index
        self.metadata = metadata

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "ContentTower":
        """Load a saved FAISS index and metadata."""
        index = faiss.read_index(str(index_path))
        metadata = pl.read_parquet(metadata_path)
        return cls(index, metadata)

    def save(self, index_path: Path, metadata_path: Path):
        """Save the FAISS index and metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        self.metadata.write_parquet(metadata_path)

    def search(
        self,
        query_embedding: Optional[np.ndarray] = None,
        profile_embedding: Optional[np.ndarray] = None,
        alpha: float = 0.5,
        top_k: int = 50,
    ) -> List[Dict]:
        """Search the content index.

        Args:
            query_embedding: From user's text query (1024-dim, normalized).
            profile_embedding: From user's taste profile (1024-dim, normalized).
            alpha: Blend weight — 1.0 = query only, 0.0 = profile only.
            top_k: Number of results to return.

        Returns:
            List of dicts with movieId, title, score, and faiss_idx.
        """
        if query_embedding is None and profile_embedding is None:
            return []

        if query_embedding is not None and profile_embedding is not None:
            search_vec = alpha * query_embedding + (1 - alpha) * profile_embedding
        elif query_embedding is not None:
            search_vec = query_embedding
        else:
            search_vec = profile_embedding

        # Normalize the blended vector
        norm = np.linalg.norm(search_vec)
        if norm > 0:
            search_vec = search_vec / norm

        search_vec = search_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(search_vec, top_k)

        results = []
        for i in range(top_k):
            idx = int(indices[0, i])
            if idx < 0:
                continue
            row = self.metadata.filter(pl.col("faiss_idx") == idx)
            if row.is_empty():
                continue
            entry = row.to_dicts()[0]
            entry["score"] = float(scores[0, i])
            results.append(entry)

        return results

    def get_embedding(self, faiss_idx: int) -> np.ndarray:
        """Retrieve the stored embedding for a given FAISS index position."""
        return self.index.reconstruct(faiss_idx)

    def add_movie(self, embedding: np.ndarray, metadata_row: Dict):
        """Add a new movie to the index at runtime.

        Args:
            embedding: Normalized 1024-dim embedding.
            metadata_row: Dict with at least movieId and title.
        """
        new_idx = self.index.ntotal
        vec = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(vec)

        metadata_row["faiss_idx"] = new_idx
        new_row = pl.DataFrame([metadata_row])
        self.metadata = pl.concat(
            [self.metadata, new_row], how="diagonal_relaxed"
        )

    def movie_id_to_faiss_idx(self, movie_id: int) -> Optional[int]:
        """Look up the FAISS index position for a movieId."""
        row = self.metadata.filter(pl.col("movieId") == movie_id)
        if row.is_empty():
            return None
        return row["faiss_idx"][0]

    def get_movie_embedding(self, movie_id: int) -> Optional[np.ndarray]:
        """Get the plot embedding for a movie by its movieId."""
        idx = self.movie_id_to_faiss_idx(movie_id)
        if idx is None:
            return None
        return self.get_embedding(idx)
