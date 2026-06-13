"""Checkpoint persistence compatible with the completed Model 3 notebook."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from .components import Model3Components


def save_checkpoint(
    components: Model3Components,
    output_dir: str | Path,
    history: dict,
    accelerator=None,
    scheduler=None,
    best: bool = False,
) -> Path:
    """Save LoRA and conditioning modules using the notebook checkpoint layout."""
    output_dir = Path(output_dir)
    checkpoint = output_dir / "checkpoint_latest"
    checkpoint.mkdir(parents=True, exist_ok=True)
    unwrap = accelerator.unwrap_model if accelerator is not None else (lambda model: model)
    unet = unwrap(components.unet)
    perceiver = unwrap(components.perceiver)
    cloth_spatial = unwrap(components.cloth_spatial)
    unet.save_pretrained(checkpoint / "unet_lora")
    torch.save(unet.base_model.model.conv_in.state_dict(), checkpoint / "conv_in.pt")
    torch.save(perceiver.state_dict(), checkpoint / "perceiver.pt")
    torch.save(cloth_spatial.state_dict(), checkpoint / "cloth_spatial.pt")
    if scheduler is not None:
        raw_scheduler = getattr(scheduler, "scheduler", scheduler)
        torch.save(raw_scheduler.state_dict(), checkpoint / "lr_scheduler.pt")
    (output_dir / "loss_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    if best:
        best_dir = output_dir / "checkpoint_best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(checkpoint, best_dir)
    return checkpoint

