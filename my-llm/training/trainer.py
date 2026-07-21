"""
GPT Trainer

Handles:
- Training
- Validation
- Mixed Precision
- Checkpoint Saving
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


class Trainer:

    def __init__(

        self,

        model: nn.Module,

        train_loader,

        val_loader,

        config,

        device="cpu",

    ):

        self.model = model

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.config = config

        self.device = torch.device(device)

        self.model.to(self.device)

        training = config["training"]

        self.learning_rate = training["learning_rate"]

        self.max_steps = training["max_steps"]

        self.warmup_steps = training["warmup_steps"]

        self.eval_interval = training["eval_interval"]

        self.save_interval = training["save_interval"]

        self.gradient_clip = training["gradient_clip"]

        self.use_amp = training["mixed_precision"]

        self.global_step = 0

        self.best_loss = float("inf")

        self.optimizer = self.create_optimizer()

        self.scheduler = self.create_scheduler()

        self.scaler = GradScaler(enabled=self.use_amp)

        print("\nTrainer Initialized")

        print(f"Device          : {self.device}")

        print(f"Learning Rate   : {self.learning_rate}")

        print(f"Mixed Precision : {self.use_amp}")

    # ----------------------------------------------------
    # Optimizer
    # ----------------------------------------------------

    def create_optimizer(self):

        return AdamW(

            self.model.parameters(),

            lr=self.learning_rate,

            weight_decay=0.01,

        )

    # ----------------------------------------------------
    # Scheduler
    # ----------------------------------------------------

    def create_scheduler(self):

        def schedule(step):

            if step < self.warmup_steps:

                return step / max(1, self.warmup_steps)

            progress = (

                step - self.warmup_steps

            ) / max(

                1,

                self.max_steps - self.warmup_steps,

            )

            cosine = 0.5 * (

                1

                + torch.cos(

                    torch.tensor(progress * 3.14159265)

                )

            )

            return max(

                0.1,

                cosine.item(),

            )

        return LambdaLR(

            self.optimizer,

            schedule,

        )

    # ----------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------

    def save_checkpoint(

        self,

        name="checkpoint.pt",

    ):

        checkpoint = {

            "model": self.model.state_dict(),

            "optimizer": self.optimizer.state_dict(),

            "scheduler": self.scheduler.state_dict(),

            "step": self.global_step,

            "best_loss": self.best_loss,

        }

        Path("checkpoints").mkdir(

            exist_ok=True

        )

        torch.save(

            checkpoint,

            f"checkpoints/{name}",

        )

        print(

            f"\nCheckpoint Saved -> checkpoints/{name}"

        )
            # ----------------------------------------------------
    # Train
    # ----------------------------------------------------

    def train(self):

        print("\nStarting Training...\n")

        self.model.train()

        while self.global_step < self.max_steps:

            train_loss = self.train_epoch()

            print(
                f"\nStep {self.global_step} | "
                f"Train Loss : {train_loss:.4f}"
            )

            if (

                self.val_loader is not None

                and self.global_step % self.eval_interval == 0

            ):

                val_loss = self.evaluate()

                if val_loss < self.best_loss:

                    self.best_loss = val_loss

                    self.save_checkpoint(
                        "checkpoint-best.pt"
                    )

            if (

                self.global_step % self.save_interval == 0

                and self.global_step > 0

            ):

                self.save_checkpoint(
                    f"checkpoint-{self.global_step}.pt"
                )

        self.save_checkpoint(
            "checkpoint-final.pt"
        )

        print("\nTraining Finished!")

    # ----------------------------------------------------
    # One Epoch
    # ----------------------------------------------------

    def train_epoch(self):

        progress = tqdm(

            self.train_loader,

            desc="Training",

            leave=False,

        )

        total_loss = 0

        batches = 0

        for batch in progress:

            loss = self.training_step(batch)

            total_loss += loss

            batches += 1

            progress.set_postfix(

                {

                    "loss": f"{loss:.4f}",

                    "step": self.global_step,

                }

            )

            if self.global_step >= self.max_steps:

                break

        return total_loss / batches

    # ----------------------------------------------------
    # One Training Step
    # ----------------------------------------------------

    def training_step(

        self,

        batch,

    ):

        batch = {

            key: value.to(self.device)

            for key, value in batch.items()

        }

        self.optimizer.zero_grad()

        with autocast(

            enabled=self.use_amp

        ):

            outputs = self.model(

                **batch

            )

            loss = outputs["loss"]

        self.scaler.scale(

            loss

        ).backward()

        self.scaler.unscale_(

            self.optimizer

        )

        torch.nn.utils.clip_grad_norm_(

            self.model.parameters(),

            self.gradient_clip,

        )

        self.scaler.step(

            self.optimizer

        )

        self.scaler.update()

        self.scheduler.step()

        self.global_step += 1

        return loss.item()
        # ----------------------------------------------------
    # Validation
    # ----------------------------------------------------

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()

        total_loss = 0.0
        batches = 0

        progress = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
        )

        for batch in progress:

            batch = {
                key: value.to(self.device)
                for key, value in batch.items()
            }

            with autocast(enabled=self.use_amp):

                outputs = self.model(**batch)

                loss = outputs["loss"]

            total_loss += loss.item()

            batches += 1

        average_loss = total_loss / max(1, batches)

        perplexity = torch.exp(
            torch.tensor(average_loss)
        ).item()

        print(
            f"\nValidation Loss : {average_loss:.4f}"
        )

        print(
            f"Perplexity      : {perplexity:.2f}"
        )

        self.model.train()

        return average_loss

    # ----------------------------------------------------
    # Load Checkpoint
    # ----------------------------------------------------

    def load_checkpoint(
        self,
        checkpoint_path,
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        self.global_step = checkpoint["step"]

        self.best_loss = checkpoint.get(
            "best_loss",
            float("inf"),
        )

        print(
            f"\nCheckpoint Loaded"
        )

        print(
            f"Step : {self.global_step}"
        )

    # ----------------------------------------------------
    # Model Summary
    # ----------------------------------------------------

    def summary(self):

        parameters = sum(
            p.numel()
            for p in self.model.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

        print("\nModel Summary")

        print("-" * 40)

        print(
            f"Parameters : {parameters:,}"
        )

        print(
            f"Trainable  : {trainable:,}"
        )

        print(
            f"Device      : {self.device}"
        )

        print(
            f"AMP         : {self.use_amp}"
        )

        print(
            f"Max Steps   : {self.max_steps}"
        )

        print("-" * 40)