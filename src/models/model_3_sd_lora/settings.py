"""Typed settings derived from the resolved Model 3 experiment config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Model3Settings:
    """Runtime settings matching the completed Model 3 notebook."""

    model_id: str
    ip_adapter_id: str
    height: int
    width: int
    trigger_word: str
    seed: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    conv_in_learning_rate: float
    weight_decay: float
    warmup_steps: int
    early_stopping_patience: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    image_tokens: int
    perceiver_depth: int
    perceiver_heads: int
    unet_input_channels: int
    mask_loss_weight: float
    x0_latent_loss_weight: float
    x0_mask_loss_weight: float
    min_snr_gamma: float
    cfg_drop_text: float
    cfg_drop_cloth: float
    ema_decay: float
    vae_scale: float
    num_workers: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Model3Settings":
        """Build settings from a resolved hierarchical experiment config."""
        model = config["model"]
        architecture = model["architecture"]
        training = config["training"]
        image_size = architecture["image_size"]
        return cls(
            model_id=model["base_model"],
            ip_adapter_id=model["ip_adapter"],
            height=int(image_size[0]),
            width=int(image_size[1]),
            trigger_word=architecture["trigger_word"],
            seed=int(training["seed"]),
            batch_size=int(training["batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            epochs=int(training["configured_epochs"]),
            learning_rate=float(training["learning_rate"]),
            conv_in_learning_rate=float(training["conv_in_learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            warmup_steps=int(training["scheduler"]["warmup_steps"]),
            early_stopping_patience=int(training["early_stopping"]["patience"]),
            lora_rank=int(model["lora"]["rank"]),
            lora_alpha=int(model["lora"]["alpha"]),
            lora_dropout=float(model["lora"]["dropout"]),
            lora_targets=tuple(model["lora"]["targets"]),
            image_tokens=int(architecture["image_tokens"]),
            perceiver_depth=int(architecture["perceiver_depth"]),
            perceiver_heads=int(architecture["perceiver_heads"]),
            unet_input_channels=int(architecture["unet_input_channels"]),
            mask_loss_weight=float(training["loss_weights"]["mask"]),
            x0_latent_loss_weight=float(training["loss_weights"]["x0_latent"]),
            x0_mask_loss_weight=float(training["loss_weights"]["x0_mask"]),
            min_snr_gamma=float(training["min_snr_gamma"]),
            cfg_drop_text=float(training["condition_dropout"]["text"]),
            cfg_drop_cloth=float(training["condition_dropout"]["cloth"]),
            ema_decay=float(training["ema_decay"]),
            vae_scale=float(config["inference"]["vae_scale"]),
            num_workers=int(config["data"]["num_workers"]),
        )

