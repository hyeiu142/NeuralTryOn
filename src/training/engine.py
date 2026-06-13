"""Reusable PyTorch training engine with tracking and checkpoint hooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from src.tracking import RunTracker


BatchStep = Callable[[torch.nn.Module, object, torch.device], torch.Tensor]


@dataclass
class EarlyStopping:
    """Track validation improvement and signal when training should stop."""

    patience: int
    min_delta: float = 0.0
    best: float = float("inf")
    bad_epochs: int = 0

    def update(self, value: float) -> bool:
        """Return true when the configured patience has been exhausted."""
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class TrainingEngine:
    """Run a conventional train/validation loop around model-specific steps."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        tracker: RunTracker,
        train_step: BatchStep,
        validation_step: BatchStep,
        scheduler: object | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.tracker = tracker
        self.train_step = train_step
        self.validation_step = validation_step
        self.scheduler = scheduler
        self.early_stopping = early_stopping

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int) -> dict:
        """Train, validate, track metrics, and retain the best checkpoint."""
        best_loss = float("inf")
        best_epoch = 0
        checkpoint = self.tracker.run_dir / "checkpoints" / "best.pt"

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, training=True)
            validation_loss = self._run_epoch(validation_loader, training=False)
            self.tracker.log_metrics(
                {"train_loss": train_loss, "validation_loss": validation_loss},
                step=epoch,
                split="epoch",
            )
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                self._save_checkpoint(checkpoint, epoch, validation_loss)
            if self.scheduler is not None:
                _step_scheduler(self.scheduler, validation_loss)
            if self.early_stopping and self.early_stopping.update(validation_loss):
                break

        return {"best_epoch": best_epoch, "best_validation_loss": best_loss}

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        total = 0.0
        count = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss = self.train_step(self.model, batch, self.device)
                    loss.backward()
                    self.optimizer.step()
                else:
                    loss = self.validation_step(self.model, batch, self.device)
                total += float(loss.detach().item())
                count += 1
        if not count:
            raise ValueError("DataLoader produced no batches")
        return total / count

    def _save_checkpoint(self, path: Path, epoch: int, validation_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "validation_loss": validation_loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )


def _step_scheduler(scheduler: object, validation_loss: float) -> None:
    try:
        scheduler.step(validation_loss)
    except TypeError:
        scheduler.step()

