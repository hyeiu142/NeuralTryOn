"""Focused tests for the migrated Model 3 production adapter."""

import torch

from src.config import load_experiment_config
from src.models.model_3_sd_lora.modules import ClothSpatialProjector, PerceiverResampler
from src.models.model_3_sd_lora.settings import Model3Settings


def test_model_3_settings_match_completed_notebook() -> None:
    config = load_experiment_config("configs/experiments/model_3_default.yaml")
    settings = Model3Settings.from_config(config)

    assert settings.unet_input_channels == 17
    assert settings.lora_rank == 16
    assert settings.image_tokens == 8
    assert settings.gradient_accumulation_steps == 8


def test_model_3_conditioning_module_shapes() -> None:
    perceiver = PerceiverResampler(
        input_dim=32,
        output_dim=24,
        num_queries=8,
        depth=2,
        num_heads=8,
    )
    spatial = ClothSpatialProjector(in_channels=4, output_dim=24)

    assert perceiver(torch.randn(2, 16, 32)).shape == (2, 8, 24)
    assert spatial(torch.randn(2, 4, 64, 48)).shape == (2, 64, 24)

