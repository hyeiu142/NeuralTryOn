"""Load Model 3 pretrained and trainable components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPVisionModelWithProjection

from .modules import ClothSpatialProjector, PerceiverResampler, expand_unet_conv_in
from .settings import Model3Settings


@dataclass
class Model3Components:
    """All neural components required by Model 3 training."""

    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    image_encoder: CLIPVisionModelWithProjection
    unet: torch.nn.Module
    perceiver: PerceiverResampler
    cloth_spatial: ClothSpatialProjector
    noise_scheduler: DDPMScheduler


def load_training_components(
    settings: Model3Settings,
    dtype: torch.dtype = torch.float16,
    enable_xformers: bool = True,
) -> Model3Components:
    """Load pretrained models and attach trainable LoRA conditioning modules."""
    vae = AutoencoderKL.from_pretrained(settings.model_id, subfolder="vae", torch_dtype=dtype)
    vae.enable_slicing()
    vae.enable_tiling()
    vae.requires_grad_(False).eval()

    text_encoder = CLIPTextModel.from_pretrained(
        settings.model_id, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder.requires_grad_(False).eval()

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        settings.ip_adapter_id,
        subfolder="models/image_encoder",
        torch_dtype=dtype,
    )
    image_encoder.requires_grad_(False).eval()

    unet = UNet2DConditionModel.from_pretrained(
        settings.model_id, subfolder="unet", torch_dtype=dtype
    )
    unet.enable_gradient_checkpointing()
    unet = get_peft_model(
        unet,
        LoraConfig(
            r=settings.lora_rank,
            lora_alpha=settings.lora_alpha,
            target_modules=list(settings.lora_targets),
            lora_dropout=settings.lora_dropout,
            bias="none",
        ),
    )
    unet = expand_unet_conv_in(unet, settings.unet_input_channels)
    unet.base_model.model.conv_in.requires_grad_(True)
    if enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    noise_scheduler = DDPMScheduler.from_pretrained(settings.model_id, subfolder="scheduler")
    cross_attention_dim = unet.base_model.model.config.cross_attention_dim
    clip_dim = image_encoder.config.hidden_size
    perceiver = PerceiverResampler(
        input_dim=clip_dim,
        output_dim=cross_attention_dim,
        num_queries=settings.image_tokens,
        depth=settings.perceiver_depth,
        num_heads=settings.perceiver_heads,
    ).float()
    cloth_spatial = ClothSpatialProjector(4, cross_attention_dim).float()
    return Model3Components(
        vae=vae,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        unet=unet,
        perceiver=perceiver,
        cloth_spatial=cloth_spatial,
        noise_scheduler=noise_scheduler,
    )


def optimizer_parameter_groups(
    components: Model3Components,
    settings: Model3Settings,
) -> tuple[list[dict], list[torch.nn.Parameter]]:
    """Create notebook-compatible optimizer groups and the clipping parameter list."""
    conv_in = list(components.unet.base_model.model.conv_in.parameters())
    conv_ids = {id(parameter) for parameter in conv_in}
    lora = [
        parameter
        for parameter in components.unet.parameters()
        if parameter.requires_grad and id(parameter) not in conv_ids
    ]
    perceiver = list(components.perceiver.parameters())
    spatial = list(components.cloth_spatial.parameters())
    groups = [
        {"params": lora, "lr": settings.learning_rate, "name": "unet_lora"},
        {"params": conv_in, "lr": settings.conv_in_learning_rate, "name": "conv_in"},
        {"params": perceiver, "lr": settings.learning_rate, "name": "perceiver"},
        {"params": spatial, "lr": settings.learning_rate, "name": "cloth_spatial"},
    ]
    return groups, lora + conv_in + perceiver + spatial
