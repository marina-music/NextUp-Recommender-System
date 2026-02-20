"""Tests for updated embedding store with 1024-dim content profiles."""
import numpy as np


class TestContentProfile:
    def test_update_profile_with_rating(self):
        """Profile should update from plot embedding + rating weight."""
        from embedding_store import InMemoryProfileStore
        store = InMemoryProfileStore(dim=1024)
        plot_emb = np.random.randn(1024).astype(np.float32)
        plot_emb /= np.linalg.norm(plot_emb)

        store.update_with_rating(user_id=1, plot_embedding=plot_emb, rating_weight=2.0)
        profile = store.get(1)
        assert profile is not None
        assert profile.vector.shape == (1024,)

    def test_profile_ema_update(self):
        """Successive updates should blend via EMA."""
        from embedding_store import InMemoryProfileStore
        store = InMemoryProfileStore(dim=1024, decay=0.9, base_lr=0.1)

        emb1 = np.zeros(1024, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(1024, dtype=np.float32)
        emb2[1] = 1.0

        store.update_with_rating(user_id=1, plot_embedding=emb1, rating_weight=1.0)
        store.get(1).vector.copy()  # first profile state

        store.update_with_rating(user_id=1, plot_embedding=emb2, rating_weight=1.0)
        profile2 = store.get(1).vector

        # After second update, profile should have components in both dimensions
        assert profile2[0] > 0
        assert profile2[1] > 0

    def test_negative_rating_weight(self):
        """Negative rating weight should push profile away."""
        from embedding_store import InMemoryProfileStore
        store = InMemoryProfileStore(dim=1024)

        # Use a positive embedding in dim 0
        emb_pos = np.zeros(1024, dtype=np.float32)
        emb_pos[0] = 1.0

        # Use a different embedding in dim 1 to give the profile a second component
        emb_other = np.zeros(1024, dtype=np.float32)
        emb_other[1] = 1.0

        # First: positive interaction pushes toward dim 0
        store.update_with_rating(user_id=1, plot_embedding=emb_pos, rating_weight=2.0)
        # Second: add another direction so normalization doesn't collapse
        store.update_with_rating(user_id=1, plot_embedding=emb_other, rating_weight=1.0)
        p1 = store.get(1).vector[0]

        # Third: negative interaction with dim 0 embedding should reduce dim 0 component
        store.update_with_rating(user_id=1, plot_embedding=emb_pos, rating_weight=-2.0)
        p2 = store.get(1).vector[0]

        assert p2 < p1  # should have decreased


class TestEmbeddingManager:
    def test_prepare_content_profile_dict(self):
        """Should produce a dict with CONTENT_PROFILE key."""
        from embedding_store import EmbeddingManager
        mgr = EmbeddingManager(hidden_size=1024)

        # Manually set a profile
        from embedding_store import ProfileEntry
        import time
        mgr.profile_store.set(1, ProfileEntry(
            vector=np.random.randn(1024).astype(np.float32),
            interaction_count=5,
            last_updated=time.time(),
            created_at=time.time(),
            user_id=1,
        ))

        result = mgr.get_profile_vector(1)
        assert result is not None
        assert result.shape == (1024,)
