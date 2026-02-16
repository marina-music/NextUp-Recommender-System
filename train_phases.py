"""
Phased Training Orchestrator for Mamba4Rec with LLM Fusion

This module implements the three-phase training strategy:
- Phase 1: Train vanilla Mamba4Rec (establishes item geometry)
- Phase 2: Train LLM projection + fusion (aligns LLM to Mamba space)
- Phase 3: Joint fine-tuning (optional, adapts to explicit intent)

Usage:
    python train_phases.py --phase 1  # Train vanilla Mamba
    python train_phases.py --phase 2 --checkpoint mamba_phase1.pt  # Train fusion
    python train_phases.py --phase 3 --checkpoint mamba_phase2.pt  # Fine-tune
"""

import sys
import os
import argparse
import logging
from logging import getLogger
from pathlib import Path

import torch

# RecBole's Trainer uses torch.load without weights_only=False,
# which fails on PyTorch 2.6+ where the default changed to True.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from recbole.utils import init_logger, init_seed, set_color
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import get_flops, get_environment

from mamba4rec import Mamba4RecFusion


class EWCRegularizer:
    """
    Elastic Weight Consolidation for Phase 3 fine-tuning.

    Penalizes changes to parameters that were important for Phase 1/2 performance.
    """

    def __init__(self, model, dataloader, device, lambda_ewc=0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.device = device

        # Compute Fisher information and store optimal parameters
        self.fisher = {}
        self.optimal_params = {}
        self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        """Compute Fisher information matrix diagonal."""
        self.model.eval()

        # Initialize Fisher with zeros
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
                self.optimal_params[name] = param.data.clone()

        # Accumulate gradients
        num_samples = 0
        for batch in dataloader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            self.model.zero_grad()
            loss = self.model.calculate_loss(batch)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2

            num_samples += 1

        # Normalize
        for name in self.fisher:
            self.fisher[name] /= num_samples

        self.model.train()

    def penalty(self):
        """Compute EWC penalty term."""
        loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher:
                loss += (self.fisher[name] *
                         (param - self.optimal_params[name]) ** 2).sum()
        return self.lambda_ewc * loss


class PhasedTrainer(Trainer):
    """
    Extended Trainer with phase-specific training logic.
    """

    def __init__(self, config, model, phase=1, ewc_regularizer=None):
        super().__init__(config, model)
        self.phase = phase
        self.ewc_regularizer = ewc_regularizer
        self.logger = getLogger()

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        """Override to add EWC regularization in Phase 3."""
        self.model.train()
        total_loss = 0.0
        iter_data = train_data if not show_progress else tqdm(train_data)

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()

            # Standard loss
            loss = self.model.calculate_loss(interaction)

            # Add EWC penalty in Phase 3
            if self.phase == 3 and self.ewc_regularizer is not None:
                loss += self.ewc_regularizer.penalty()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / (batch_idx + 1)


def run_phase1(config_path, save_dir):
    """
    Phase 1: Train vanilla Mamba4Rec.

    Establishes stable item geometry without LLM fusion.
    """
    config = Config(model=Mamba4RecFusion, config_file_list=[config_path])

    # Disable fusion for Phase 1
    config["use_llm_fusion"] = False

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    logger.info("=" * 60)
    logger.info("PHASE 1: Training Vanilla Mamba4Rec")
    logger.info("=" * 60)

    # Dataset
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Model
    model = Mamba4RecFusion(config, train_data.dataset).to(config['device'])
    model.set_training_phase(1)

    logger.info(f"Trainable parameters: {model.get_trainable_params():,}")
    logger.info(f"Frozen parameters: {model.get_frozen_params():,}")

    # Training
    trainer = PhasedTrainer(config, model, phase=1)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # Evaluation
    test_result = trainer.evaluate(test_data, show_progress=config["show_progress"])

    logger.info(set_color("Phase 1 Best Valid", "yellow") + f": {best_valid_result}")
    logger.info(set_color("Phase 1 Test Result", "yellow") + f": {test_result}")

    # Save checkpoint
    save_path = Path(save_dir) / "mamba_phase1.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }, save_path)
    logger.info(f"Phase 1 checkpoint saved to {save_path}")

    return model, test_result


def run_phase2(config_path, checkpoint_path, save_dir):
    """
    Phase 2: Train LLM Projection + Fusion.

    Freezes Mamba layers and item embeddings.
    Trains projection to align LLM embeddings with item geometry.
    """
    config = Config(model=Mamba4RecFusion, config_file_list=[config_path])

    # Enable fusion for Phase 2
    config["use_llm_fusion"] = True

    # Use Phase 2 learning rate
    config["learning_rate"] = config.get("phase2_learning_rate", 0.0005)
    config["epochs"] = config.get("phase2_epochs", 50)

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    logger.info("=" * 60)
    logger.info("PHASE 2: Training LLM Projection + Fusion")
    logger.info("=" * 60)

    # Dataset
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Model
    model = Mamba4RecFusion(config, train_data.dataset).to(config['device'])

    # Load Phase 1 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logger.info(f"Loaded Phase 1 checkpoint from {checkpoint_path}")

    # Configure for Phase 2
    model.set_training_phase(2)

    logger.info(f"Trainable parameters: {model.get_trainable_params():,}")
    logger.info(f"Frozen parameters: {model.get_frozen_params():,}")

    # Training
    trainer = PhasedTrainer(config, model, phase=2)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # Evaluation
    test_result = trainer.evaluate(test_data, show_progress=config["show_progress"])

    logger.info(set_color("Phase 2 Best Valid", "yellow") + f": {best_valid_result}")
    logger.info(set_color("Phase 2 Test Result", "yellow") + f": {test_result}")

    # Save checkpoint
    save_path = Path(save_dir) / "mamba_phase2.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }, save_path)
    logger.info(f"Phase 2 checkpoint saved to {save_path}")

    return model, test_result


def run_phase3(config_path, checkpoint_path, save_dir):
    """
    Phase 3: Joint Fine-tuning with EWC.

    Unfreezes top Mamba layer for adaptation.
    Uses EWC regularization to prevent catastrophic forgetting.
    """
    config = Config(model=Mamba4RecFusion, config_file_list=[config_path])

    # Enable fusion
    config["use_llm_fusion"] = True

    # Use Phase 3 learning rate (very low)
    config["learning_rate"] = config.get("phase3_learning_rate", 0.0001)
    config["epochs"] = config.get("phase3_epochs", 20)

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    logger.info("=" * 60)
    logger.info("PHASE 3: Joint Fine-tuning with EWC")
    logger.info("=" * 60)

    # Dataset
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Model
    model = Mamba4RecFusion(config, train_data.dataset).to(config['device'])

    # Load Phase 2 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded Phase 2 checkpoint from {checkpoint_path}")

    # Configure for Phase 3
    model.set_training_phase(3)

    logger.info(f"Trainable parameters: {model.get_trainable_params():,}")
    logger.info(f"Frozen parameters: {model.get_frozen_params():,}")

    # EWC Regularizer
    ewc_lambda = config.get("ewc_lambda", 0.4)
    ewc_regularizer = EWCRegularizer(
        model, train_data, config['device'], lambda_ewc=ewc_lambda
    )
    logger.info(f"EWC regularization enabled with lambda={ewc_lambda}")

    # Training with EWC
    trainer = PhasedTrainer(config, model, phase=3, ewc_regularizer=ewc_regularizer)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # Evaluation
    test_result = trainer.evaluate(test_data, show_progress=config["show_progress"])

    logger.info(set_color("Phase 3 Best Valid", "yellow") + f": {best_valid_result}")
    logger.info(set_color("Phase 3 Test Result", "yellow") + f": {test_result}")

    # Save final checkpoint
    save_path = Path(save_dir) / "mamba_production.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }, save_path)
    logger.info(f"Production checkpoint saved to {save_path}")

    return model, test_result


def main():
    parser = argparse.ArgumentParser(description="Phased Training for Mamba4Rec Fusion")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                        help="Training phase (1, 2, or 3)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (required for phase 2 and 3)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    args = parser.parse_args()

    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.phase == 1:
        run_phase1(args.config, args.save_dir)
    elif args.phase == 2:
        if args.checkpoint is None:
            print("Error: Phase 2 requires --checkpoint pointing to Phase 1 checkpoint")
            sys.exit(1)
        run_phase2(args.config, args.checkpoint, args.save_dir)
    elif args.phase == 3:
        if args.checkpoint is None:
            print("Error: Phase 3 requires --checkpoint pointing to Phase 2 checkpoint")
            sys.exit(1)
        run_phase3(args.config, args.checkpoint, args.save_dir)


if __name__ == "__main__":
    main()
