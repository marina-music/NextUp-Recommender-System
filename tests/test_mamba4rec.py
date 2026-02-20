"""Tests for pure Mamba4Rec (no fusion)."""
import pytest
import torch
from unittest.mock import MagicMock


def _make_config(hidden_size=64, num_layers=1):
    """Create a minimal mock config for Mamba4Rec.

    Must satisfy both Mamba4Rec's own config reads and RecBole's
    SequentialRecommender.__init__ which needs USER_ID_FIELD,
    ITEM_ID_FIELD, LIST_SUFFIX, ITEM_LIST_LENGTH_FIELD, NEG_PREFIX,
    MAX_ITEM_LIST_LENGTH, and device.
    """
    _data = {
        # Mamba4Rec hyperparameters
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout_prob": 0.1,
        "loss_type": "CE",
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        # RecBole SequentialRecommender fields
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "LIST_SUFFIX": "_list",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "NEG_PREFIX": "neg_",
        "MAX_ITEM_LIST_LENGTH": 50,
        "device": "cpu",
    }
    config = MagicMock()
    config.__getitem__ = lambda self, key: _data[key]
    config.__contains__ = lambda self, key: key in _data
    config.final_config_dict = {
        "hidden_size": hidden_size,
        "loss_type": "CE",
    }
    return config


def _make_dataset(n_items=100):
    """Create a minimal mock dataset.

    dataset.num("item_id") must return the number of items.
    """
    dataset = MagicMock()
    dataset.num = MagicMock(return_value=n_items)
    return dataset


class TestMamba4Rec:
    def test_no_fusion_attributes(self):
        """Model should not have any fusion-related attributes."""
        from mamba4rec import Mamba4Rec

        model = Mamba4Rec(_make_config(), _make_dataset())
        assert not hasattr(model, "fusion")
        assert not hasattr(model, "llm_projection")
        assert not hasattr(model, "_current_phase")
        assert not hasattr(model, "use_llm_fusion")

    def test_forward_shape(self):
        """Forward pass should return (B, hidden_size)."""
        from mamba4rec import Mamba4Rec

        model = Mamba4Rec(_make_config(), _make_dataset())
        model.eval()
        item_seq = torch.randint(0, 100, (4, 10))
        item_seq_len = torch.tensor([10, 8, 5, 3])
        with torch.no_grad():
            output = model.forward(item_seq, item_seq_len)
        assert output.shape == (4, 64)

    def test_full_sort_predict_shape(self):
        """full_sort_predict should return (B, n_items) scores."""
        from mamba4rec import Mamba4Rec

        model = Mamba4Rec(_make_config(), _make_dataset())
        model.eval()
        interaction = {
            "item_id_list": torch.randint(0, 100, (4, 10)),
            "item_length": torch.tensor([10, 8, 5, 3]),
        }
        with torch.no_grad():
            scores = model.full_sort_predict(interaction)
        assert scores.shape[0] == 4

    def test_no_set_training_phase(self):
        """set_training_phase should not exist."""
        from mamba4rec import Mamba4Rec

        model = Mamba4Rec(_make_config(), _make_dataset())
        assert not hasattr(model, "set_training_phase")
