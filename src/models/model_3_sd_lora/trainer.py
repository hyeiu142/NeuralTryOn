"""Executable Model 3 training adapter for Kaggle GPU environments."""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from peft import set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.reproducibility import seed_everything

from .checkpoints import save_checkpoint
from .components import Model3Components, load_training_components, optimizer_parameter_groups
from .dataset import Model3Dataset
from .objective import diffusion_loss
from .settings import Model3Settings


def train_model3(
    config: dict,
    viton_root: str | Path,
    csv_root: str | Path,
    caption_root: str | Path,
    output_dir: str | Path,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    epochs: int | None = None,
    resume_dir: str | Path | None = None,
    use_wandb: bool = False,
) -> dict:
    """Train Model 3 using the same architecture, objective, and checkpoint layout."""
    if not torch.cuda.is_available():
        raise RuntimeError("Model 3 production training requires a CUDA GPU.")

    settings = Model3Settings.from_config(config)
    seed_everything(settings.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.jsonl"

    train_dataset = Model3Dataset(
        settings,
        viton_root,
        csv_root,
        caption_root,
        split="train",
        augment=True,
        max_samples=max_train_samples,
    )
    validation_dataset = Model3Dataset(
        settings,
        viton_root,
        csv_root,
        caption_root,
        split="test",
        augment=False,
        max_samples=max_validation_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=settings.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=True,
    )
    if not len(train_loader) or not len(validation_loader):
        raise ValueError("Training and validation loaders must contain at least one batch.")

    components = load_training_components(settings)
    parameter_groups, clipping_parameters = optimizer_parameter_groups(components, settings)
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=settings.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    total_epochs = epochs or settings.epochs
    total_steps = max(1, (len(train_loader) // settings.gradient_accumulation_steps) * total_epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=settings.warmup_steps,
        num_training_steps=total_steps,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        mixed_precision="fp16",
        log_with="wandb" if use_wandb else None,
    )
    if use_wandb:
        accelerator.init_trackers(
            project_name=config["notes"]["wandb_project"],
            config=config,
            init_kwargs={"wandb": {"name": config["notes"]["wandb_run_name"]}},
        )

    (
        components.unet,
        components.perceiver,
        components.cloth_spatial,
        optimizer,
        train_loader,
        validation_loader,
        scheduler,
    ) = accelerator.prepare(
        components.unet,
        components.perceiver,
        components.cloth_spatial,
        optimizer,
        train_loader,
        validation_loader,
        scheduler,
    )
    device = accelerator.device
    components.vae = components.vae.to(device)
    components.text_encoder = components.text_encoder.to(device)
    components.image_encoder = components.image_encoder.to(device)

    state = _initial_state()
    if resume_dir:
        state = _load_resume(
            components,
            scheduler,
            accelerator,
            Path(resume_dir),
            output_dir,
        )

    ema_unet = AveragedModel(
        accelerator.unwrap_model(components.unet),
        multi_avg_fn=get_ema_multi_avg_fn(settings.ema_decay),
    )
    ema_perceiver = AveragedModel(
        accelerator.unwrap_model(components.perceiver),
        multi_avg_fn=get_ema_multi_avg_fn(settings.ema_decay),
    )
    ema_spatial = AveragedModel(
        accelerator.unwrap_model(components.cloth_spatial),
        multi_avg_fn=get_ema_multi_avg_fn(settings.ema_decay),
    )

    for epoch in range(state["epochs_done"], total_epochs):
        components.unet.train()
        components.perceiver.train()
        components.cloth_spatial.train()
        train_losses = []
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{total_epochs} [train]",
            disable=not accelerator.is_local_main_process,
        )
        for batch in progress:
            with accelerator.accumulate(
                components.unet, components.perceiver, components.cloth_spatial
            ):
                loss = diffusion_loss(batch, components, settings, device, training=True)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(clipping_parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    ema_unet.update_parameters(accelerator.unwrap_model(components.unet))
                    ema_perceiver.update_parameters(
                        accelerator.unwrap_model(components.perceiver)
                    )
                    ema_spatial.update_parameters(
                        accelerator.unwrap_model(components.cloth_spatial)
                    )
                    state["global_step"] += 1
            value = float(loss.detach().item())
            train_losses.append(value)
            state["step_losses"].append(value)
            progress.set_postfix(loss=f"{value:.4f}", step=state["global_step"])

        components.unet.eval()
        components.perceiver.eval()
        components.cloth_spatial.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in tqdm(
                validation_loader,
                desc=f"Epoch {epoch + 1} [validation]",
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):
                value = diffusion_loss(batch, components, settings, device, training=False)
                validation_losses.append(float(value.item()))

        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        state["epoch_train_losses"].append(train_loss)
        state["epoch_val_losses"].append(validation_loss)
        state["epochs_done"] = epoch + 1
        improved = validation_loss < state["best_val_loss"]
        if improved:
            state["best_val_loss"] = validation_loss
            state["no_improve_count"] = 0
        else:
            state["no_improve_count"] += 1
        state["config"] = _compatibility_config(settings)

        accelerator.log(
            {
                "epoch/train_loss": train_loss,
                "epoch/validation_loss": validation_loss,
                "epoch/learning_rate": scheduler.get_last_lr()[0],
            },
            step=state["global_step"],
        )
        if accelerator.is_main_process:
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "global_step": state["global_step"],
                            "train_loss": train_loss,
                            "validation_loss": validation_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                        }
                    )
                    + "\n"
                )
            (output_dir / "checkpoint_latest").mkdir(parents=True, exist_ok=True)
            _save_ema(components, ema_unet, ema_perceiver, ema_spatial, output_dir)
            save_checkpoint(
                components,
                output_dir,
                state,
                accelerator=accelerator,
                scheduler=scheduler,
                best=improved,
            )
        accelerator.wait_for_everyone()
        if state["no_improve_count"] >= settings.early_stopping_patience:
            break
        gc.collect()
        torch.cuda.empty_cache()

    if use_wandb:
        accelerator.end_training()
    if accelerator.is_main_process:
        (output_dir / "summary.json").write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
    return state


def _initial_state() -> dict:
    return {
        "epoch_train_losses": [],
        "epoch_val_losses": [],
        "step_losses": [],
        "best_val_loss": float("inf"),
        "global_step": 0,
        "epochs_done": 0,
        "no_improve_count": 0,
        "config": {},
    }


def _compatibility_config(settings: Model3Settings) -> dict:
    return {
        "unet_in_channels": str(settings.unet_input_channels),
        "lora_rank": str(settings.lora_rank),
        "lora_alpha": str(settings.lora_alpha),
        "num_image_tokens": str(settings.image_tokens),
        "perceiver_depth": str(settings.perceiver_depth),
        "perceiver_heads": str(settings.perceiver_heads),
        "lr": str(settings.learning_rate),
        "conv_in_lr": str(settings.conv_in_learning_rate),
        "x0_latent_loss_weight": str(settings.x0_latent_loss_weight),
        "x0_mask_loss_weight": str(settings.x0_mask_loss_weight),
        "cfg_drop_cloth": str(settings.cfg_drop_cloth),
    }


def _load_resume(
    components: Model3Components,
    scheduler,
    accelerator: Accelerator,
    resume_dir: Path,
    output_dir: Path,
) -> dict:
    checkpoint = (
        resume_dir if resume_dir.name == "checkpoint_latest" else resume_dir / "checkpoint_latest"
    )
    required = ("unet_lora/adapter_model.safetensors", "conv_in.pt", "perceiver.pt", "cloth_spatial.pt")
    missing = [name for name in required if not (checkpoint / name).exists()]
    if missing:
        raise FileNotFoundError(f"Resume checkpoint missing: {missing}")
    unet = accelerator.unwrap_model(components.unet)
    lora_state = load_safetensors(str(checkpoint / "unet_lora/adapter_model.safetensors"))
    set_peft_model_state_dict(unet, lora_state, adapter_name="default")
    unet.base_model.model.conv_in.load_state_dict(torch.load(checkpoint / "conv_in.pt"))
    accelerator.unwrap_model(components.perceiver).load_state_dict(
        torch.load(checkpoint / "perceiver.pt")
    )
    accelerator.unwrap_model(components.cloth_spatial).load_state_dict(
        torch.load(checkpoint / "cloth_spatial.pt")
    )
    scheduler_path = checkpoint / "lr_scheduler.pt"
    if scheduler_path.exists():
        getattr(scheduler, "scheduler", scheduler).load_state_dict(torch.load(scheduler_path))
    history_path = resume_dir / "loss_history.json"
    if not history_path.exists():
        history_path = checkpoint.parent / "loss_history.json"
    if not history_path.exists():
        raise FileNotFoundError("Resume requires loss_history.json")
    state = json.loads(history_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return state


def _save_ema(
    components: Model3Components,
    ema_unet: AveragedModel,
    ema_perceiver: AveragedModel,
    ema_spatial: AveragedModel,
    output_dir: Path,
) -> None:
    checkpoint = output_dir / "checkpoint_latest"
    lora = {
        key: value
        for key, value in ema_unet.module.state_dict().items()
        if "lora_A" in key or "lora_B" in key
    }
    torch.save(lora, checkpoint / "ema_unet_lora.pt")
    torch.save(ema_unet.module.base_model.model.conv_in.state_dict(), checkpoint / "ema_conv_in.pt")
    torch.save(ema_perceiver.module.state_dict(), checkpoint / "ema_perceiver.pt")
    torch.save(ema_spatial.module.state_dict(), checkpoint / "ema_cloth_spatial.pt")
