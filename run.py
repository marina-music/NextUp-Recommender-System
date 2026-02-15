"""
Run script for Mamba4Rec with LLM Fusion

This script supports both vanilla Mamba4Rec training and fusion-enabled training.
For phased training (recommended), use train_phases.py instead.

Usage:
    python run.py                    # Train vanilla Mamba4Rec
    python run.py --fusion           # Train with fusion enabled (requires LLM embeddings)
"""

import sys
import argparse
import logging
from logging import getLogger

from recbole.utils import init_logger, init_seed, set_color
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import get_flops, get_environment

from mamba4rec import Mamba4RecFusion


def main():
    parser = argparse.ArgumentParser(description="Run Mamba4Rec with optional fusion")
    parser.add_argument("--fusion", action="store_true",
                        help="Enable LLM fusion (requires embedding data)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    config = Config(model=Mamba4RecFusion, config_file_list=[args.config])

    # Override fusion setting if specified
    if args.fusion:
        config["use_llm_fusion"] = True
    elif "use_llm_fusion" not in config:
        config["use_llm_fusion"] = False

    init_seed(config['seed'], config['reproducibility'])

    # Logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # Dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # Dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = Mamba4RecFusion(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if config["use_llm_fusion"]:
        logger.info(set_color("LLM Fusion", "green") + ": ENABLED")
        logger.info(f"  LLM dimension: {config.get('llm_dim', 768)}")
        logger.info(f"  Vector gate: {config.get('vector_gate', False)}")
    else:
        logger.info(set_color("LLM Fusion", "yellow") + ": DISABLED")

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # Trainer loading and initialization
    trainer = Trainer(config, model)

    # Model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # Model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")


if __name__ == '__main__':
    main()
