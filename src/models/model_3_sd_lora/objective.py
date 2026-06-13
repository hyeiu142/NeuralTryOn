"""Notebook-compatible conditioned diffusion training objective."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .components import Model3Components
from .settings import Model3Settings


def diffusion_loss(
    batch: dict,
    components: Model3Components,
    settings: Model3Settings,
    device: torch.device,
    training: bool = True,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Compute masked noise loss, x0 reconstruction loss, and Min-SNR weighting."""
    person = batch["image"].to(device, dtype=dtype)
    agnostic = batch["agnostic"].to(device, dtype=dtype)
    pose = batch["pose"].to(device, dtype=dtype)
    cloth = batch["cloth"].to(device, dtype=dtype)
    mask = batch["inpaint_mask"].to(device, dtype=dtype)
    clip_cloth = batch["clip_cloth"].to(device, dtype=dtype)
    input_ids = batch["input_ids"].to(device)

    with torch.no_grad():
        text_embedding = components.text_encoder(input_ids)[0]
        clip_patches = components.image_encoder(
            pixel_values=clip_cloth, output_hidden_states=True
        ).last_hidden_state
        person_latent = components.vae.encode(person).latent_dist.sample() * settings.vae_scale
        agnostic_latent = (
            components.vae.encode(agnostic).latent_dist.sample() * settings.vae_scale
        )
        pose_latent = components.vae.encode(pose).latent_dist.sample() * settings.vae_scale
        cloth_latent = components.vae.encode(cloth).latent_dist.mode() * settings.vae_scale

    if training and random.random() < settings.cfg_drop_text:
        text_embedding = torch.zeros_like(text_embedding)

    cloth_tokens = components.perceiver(clip_patches.float()).to(dtype)
    cloth_spatial = components.cloth_spatial(cloth_latent.float()).to(dtype)
    if training and random.random() < settings.cfg_drop_cloth:
        cloth_tokens = torch.zeros_like(cloth_tokens)
        cloth_spatial = torch.zeros_like(cloth_spatial)
    encoder_hidden_states = torch.cat([text_embedding, cloth_tokens, cloth_spatial], dim=1)

    noise = torch.randn_like(person_latent)
    timesteps = torch.randint(
        0,
        components.noise_scheduler.config.num_train_timesteps,
        (person_latent.shape[0],),
        device=device,
    ).long()
    noisy_latent = components.noise_scheduler.add_noise(person_latent, noise, timesteps)
    mask_down = F.interpolate(mask, size=person_latent.shape[-2:], mode="nearest")
    model_input = torch.cat(
        [noisy_latent, mask_down, agnostic_latent, pose_latent, cloth_latent], dim=1
    )
    noise_prediction = components.unet(
        model_input, timesteps, encoder_hidden_states=encoder_hidden_states
    ).sample

    loss_map = F.mse_loss(noise_prediction.float(), noise.float(), reduction="none")
    noise_loss = (
        loss_map * (1.0 + settings.mask_loss_weight * mask_down.float())
    ).mean(dim=(1, 2, 3))

    alphas = components.noise_scheduler.alphas_cumprod.to(device, dtype=torch.float32)
    alpha = alphas[timesteps].view(-1, 1, 1, 1).clamp(min=1e-8)
    beta = (1.0 - alpha).clamp(min=1e-8)
    predicted_x0 = (
        noisy_latent.float() - beta.sqrt() * noise_prediction.float()
    ) / alpha.sqrt()
    x0_weight = 0.2 + settings.x0_mask_loss_weight * mask_down.float()
    x0_loss = (
        F.l1_loss(predicted_x0, person_latent.float(), reduction="none") * x0_weight
    ).mean(dim=(1, 2, 3))
    pixel_loss = noise_loss + settings.x0_latent_loss_weight * x0_loss

    if settings.min_snr_gamma <= 0:
        return pixel_loss.mean()
    alpha_cumulative = alphas[timesteps]
    snr = alpha_cumulative / (1.0 - alpha_cumulative)
    snr_weight = torch.minimum(
        snr, torch.full_like(snr, settings.min_snr_gamma)
    ) / snr
    return (pixel_loss * snr_weight).mean()

