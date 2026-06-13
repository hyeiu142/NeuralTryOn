"""Declarative registry describing the three implemented project models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Stable identity, capabilities, and migration status for one model."""

    key: str
    display_name: str
    family: str
    source_notebook: str
    evaluation_package: str
    supports_training: bool
    supports_paired_evaluation: bool
    supports_unpaired_inference: bool
    production_adapter_status: str


MODEL_SPECS = {
    "model_1": ModelSpec(
        key="model_1",
        display_name="Lightweight U-Net + GMM + TOM",
        family="convolutional_virtual_try_on",
        source_notebook="notebooks/02_models/model_1_lightweight_unet.ipynb",
        evaluation_package="evaluation/model_1_lightweight_unet",
        supports_training=True,
        supports_paired_evaluation=True,
        supports_unpaired_inference=True,
        production_adapter_status="notebook_reference",
    ),
    "model_2": ModelSpec(
        key="model_2",
        display_name="GMM + Shape Generation + Pix2Pix",
        family="conditional_gan",
        source_notebook="notebooks/02_models/model_2_pix2pix.ipynb",
        evaluation_package="evaluation/model_2_pix2pix",
        supports_training=True,
        supports_paired_evaluation=True,
        supports_unpaired_inference=True,
        production_adapter_status="notebook_reference",
    ),
    "model_3": ModelSpec(
        key="model_3",
        display_name="Stable Diffusion Inpainting + LoRA",
        family="latent_diffusion",
        source_notebook="notebooks/02_models/model_3_sd_lora.ipynb",
        evaluation_package="evaluation/model_3_sd_lora",
        supports_training=True,
        supports_paired_evaluation=True,
        supports_unpaired_inference=True,
        production_adapter_status="production_adapter_gpu_smoke_pending",
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    """Return a registered model specification."""
    try:
        return MODEL_SPECS[key]
    except KeyError as error:
        raise KeyError(f"Unknown model '{key}'. Choose from {sorted(MODEL_SPECS)}") from error
