"""Tests for single-phase training module."""
import pytest
from unittest.mock import MagicMock, patch


class TestTrainModule:
    def test_no_ewc_class(self):
        """EWCRegularizer should not exist."""
        import train
        assert not hasattr(train, "EWCRegularizer")

    def test_no_phase2_phase3(self):
        """Multi-phase functions should not exist."""
        import train
        assert not hasattr(train, "run_phase2")
        assert not hasattr(train, "run_phase3")

    def test_has_train_function(self):
        """Should have a train() function."""
        import train
        assert callable(getattr(train, "train_mamba", None))

    def test_has_expand_embeddings(self):
        """Should have a function to expand item embeddings for retraining."""
        import train
        assert callable(getattr(train, "expand_item_embeddings", None))
