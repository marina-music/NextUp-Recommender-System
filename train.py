"""Single-phase Mamba4Rec training with retraining support."""
import sys
import os
import argparse
import logging
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from mamba4rec import Mamba4Rec

# PyTorch 2.6+ compat
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

logger = logging.getLogger(__name__)


def expand_item_embeddings(model, new_num_items):
    """Expand the item embedding table for retraining with graduated movies.

    Args:
        model: Mamba4Rec model with existing embeddings.
        new_num_items: New total number of items (must be >= current).

    Returns:
        Model with expanded embedding table (new rows randomly initialized).
    """
    old_emb = model.item_embedding
    old_num, emb_dim = old_emb.weight.shape

    if new_num_items <= old_num:
        return model

    new_emb = torch.nn.Embedding(new_num_items, emb_dim, padding_idx=0)
    with torch.no_grad():
        new_emb.weight[:old_num] = old_emb.weight
        torch.nn.init.normal_(new_emb.weight[old_num:], mean=0.0, std=0.02)

    model.item_embedding = new_emb
    logger.info(f"Expanded embeddings: {old_num} -> {new_num_items} items")
    return model


def train_mamba(config_path, save_dir="checkpoints", checkpoint=None):
    """Train Mamba4Rec from scratch or continue from checkpoint.

    Args:
        config_path: Path to YAML config file.
        save_dir: Directory to save the trained model.
        checkpoint: Optional path to existing checkpoint for retraining.

    Returns:
        Tuple of (model, test_result).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = Config(model=Mamba4Rec, config_file_list=[config_path])
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = Mamba4Rec(config, dataset).to(config["device"])

    if checkpoint:
        state = torch.load(checkpoint, map_location=config["device"])
        model_state = state.get("state_dict", state)
        # Handle embedding size mismatch for retraining
        old_num = model_state["item_embedding.weight"].shape[0]
        new_num = model.item_embedding.weight.shape[0]
        if new_num > old_num:
            model = expand_item_embeddings(model, new_num)
            model.load_state_dict(model_state, strict=False)
        else:
            model.load_state_dict(model_state)
        logger.info(f"Loaded checkpoint from {checkpoint}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total:,} total, {trainable:,} trainable")

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=True
    )

    test_result = trainer.evaluate(test_data)
    logger.info(f"Best valid: {best_valid_result}")
    logger.info(f"Test result: {test_result}")

    save_path = save_dir / "mamba4rec.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, save_path)
    logger.info(f"Saved model to {save_path}")

    return model, test_result


def main():
    parser = argparse.ArgumentParser(description="Train Mamba4Rec")
    parser.add_argument("--config", default="config_ml32m.yaml")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    train_mamba(args.config, args.save_dir, args.checkpoint)


if __name__ == "__main__":
    main()
