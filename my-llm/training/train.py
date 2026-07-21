#!/usr/bin/env python3
"""
GPT Training Script.
Author: Assistant
"""

import argparse
import sys
from pathlib import Path
import torch

# Ensure the project root is in the system path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.config import ConfigLoader, load_model_from_config
from data import LLMDataset, create_dataloader
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model.")

    parser.add_argument(
        "--config",
        default="llm.config.js",
        help="Configuration file path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device (auto, cpu, or cuda)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Specific checkpoint path to resume training from (overrides auto-resume)",
    )

    return parser.parse_args()


def get_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_datasets(config, device):
    data_cfg = config.data()
    train_cfg = config.training()

    train_path = Path("data/processed/train.pt")
    val_path = Path("data/processed/val.pt")

    if not train_path.exists():
        raise FileNotFoundError("Run data/prepare.py first.")

    # Training Dataloader
    train_dataset = LLMDataset(
        str(train_path),
        max_length=data_cfg["max_length"],
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    # Validation Dataloader (Optional)
    val_loader = None
    if val_path.exists():
        val_dataset = LLMDataset(
            str(val_path),
            max_length=data_cfg["max_length"],
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            pin_memory=(device == "cuda"),
        )

    return train_loader, val_loader


def main():
    args = parse_args()

    print("=" * 60)
    print("GPT Training Pipeline")
    print("=" * 60)

    try:
        # 1. Configuration Setup
        print("\n[1/5] Loading configuration...")
        config = ConfigLoader(args.config)
        print("Configuration loaded successfully.")

        # 2. Hardware Diagnostics
        device = get_device(args.device)
        print(f"\n[2/5] Training Device: {device}")
        if device == "cuda":
            print(f"GPU Model: {torch.cuda.get_device_name(0)}")

        # 3. Model Initialization
        print("\n[3/5] Instantiating model architecture...")
        model = load_model_from_config(args.config)
        print(f"Total Parameters: {model.count_parameters():,}")

        # 4. Dataset Pipeline Setup
        print("\n[4/5] Loading and tokenizing dataset split...")
        train_loader, val_loader = load_datasets(config, device)
        print(f"Training Batches: {len(train_loader):,}")
        if val_loader is not None:
            print(f"Validation Batches: {len(val_loader):,}")

        # 5. Trainer Initialization
        print("\n[5/5] Preparing Trainer instance...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.config,
            device=device,
        )
        trainer.summary()

        # ----------------------------------------------------
        # Resume Training & Loop Start
        # ----------------------------------------------------
        if args.resume:
            print(f"\nLoading explicit checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
            trainer.train(auto_resume=False)
        else:
            trainer.train(auto_resume=True)

        print("\nTraining Completed Successfully!")
        print("Final model checkpoints saved inside checkpoints/ directory.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining Failed: {e}")
        raise


if __name__ == "__main__":
    main()