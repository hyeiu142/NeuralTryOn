"""Standalone Model 3 checkpoint loading and try-on inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import DDIMScheduler
from peft import set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from transformers import CLIPImageProcessor, CLIPTokenizer

from src.models import TryOnModel

from .components import Model3Components, load_training_components
from .dataset import Model3Dataset
from .settings import Model3Settings


class Model3InferenceAdapter(TryOnModel):
    """Load a notebook-compatible checkpoint and generate try-on images."""

    def __init__(
        self,
        config: dict,
        viton_root: str | Path,
        csv_root: str | Path,
        caption_root: str | Path,
        device: str = "cuda",
    ) -> None:
        self.settings = Model3Settings.from_config(config)
        self.config = config
        self.viton_root = Path(viton_root)
        self.csv_root = Path(csv_root)
        self.caption_root = Path(caption_root)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.components: Model3Components | None = None
        self.scheduler: DDIMScheduler | None = None
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.settings.model_id, subfolder="tokenizer"
        )
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.preprocessor = Model3Dataset(
            self.settings,
            self.viton_root,
            self.csv_root,
            self.caption_root,
            split="test",
            augment=False,
            max_samples=1,
        )

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        """Rebuild Model 3 and load the existing checkpoint format."""
        checkpoint = Path(checkpoint)
        if checkpoint.name != "checkpoint_latest":
            checkpoint = checkpoint / "checkpoint_latest"
        required = ("unet_lora/adapter_model.safetensors", "conv_in.pt", "perceiver.pt", "cloth_spatial.pt")
        missing = [name for name in required if not (checkpoint / name).exists()]
        if missing:
            raise FileNotFoundError(f"Checkpoint missing: {missing}")

        components = load_training_components(
            self.settings,
            dtype=self.dtype,
            enable_xformers=self.device.type == "cuda",
        )
        lora_state = load_safetensors(str(checkpoint / "unet_lora/adapter_model.safetensors"))
        set_peft_model_state_dict(components.unet, lora_state, adapter_name="default")
        components.unet.base_model.model.conv_in.load_state_dict(
            torch.load(checkpoint / "conv_in.pt", map_location="cpu")
        )
        components.perceiver.load_state_dict(
            torch.load(checkpoint / "perceiver.pt", map_location="cpu")
        )
        components.cloth_spatial.load_state_dict(
            torch.load(checkpoint / "cloth_spatial.pt", map_location="cpu")
        )
        for module in (
            components.vae,
            components.text_encoder,
            components.image_encoder,
            components.unet,
            components.perceiver,
            components.cloth_spatial,
        ):
            module.requires_grad_(False).eval().to(self.device)
        components.unet.to(self.device, dtype=self.dtype)
        components.vae.to(self.device, dtype=self.dtype)
        components.text_encoder.to(self.device, dtype=self.dtype)
        components.image_encoder.to(self.device, dtype=self.dtype)
        self.components = components
        self.scheduler = DDIMScheduler.from_pretrained(
            self.settings.model_id, subfolder="scheduler"
        )
        self.scheduler.set_timesteps(self.config["inference"]["steps"], device=self.device)

    @torch.no_grad()
    def predict(self, sample: dict) -> Image.Image:
        """Generate one try-on result from person_id, cloth_id, split, and optional seed."""
        if self.components is None or self.scheduler is None:
            raise RuntimeError("Call load_checkpoint before predict.")
        person_id = str(sample["person_id"])
        cloth_id = str(sample["cloth_id"])
        split = str(sample.get("split", "test"))
        seed = int(sample.get("seed", self.settings.seed))
        batch = self._load_sample(person_id, cloth_id, split)
        components = self.components

        person = batch["image"].to(self.device, self.dtype)
        agnostic = batch["agnostic"].to(self.device, self.dtype)
        pose = batch["pose"].to(self.device, self.dtype)
        cloth = batch["cloth"].to(self.device, self.dtype)
        mask = batch["inpaint_mask"].to(self.device, self.dtype)
        clip_cloth = batch["clip_cloth"].to(self.device, self.dtype)
        input_ids = batch["input_ids"].to(self.device)

        person_latent = self._encode(person)
        agnostic_latent = self._encode(agnostic)
        pose_latent = self._encode(pose)
        cloth_latent = self._encode(cloth)
        mask_latent = F.interpolate(mask, size=person_latent.shape[-2:], mode="nearest")
        text = components.text_encoder(input_ids)[0]
        patches = components.image_encoder(
            pixel_values=clip_cloth, output_hidden_states=True
        ).last_hidden_state
        global_tokens = components.perceiver(patches.float()).to(self.dtype)
        spatial_tokens = components.cloth_spatial(cloth_latent.float()).to(self.dtype)
        condition = torch.cat([text, global_tokens, spatial_tokens], dim=1)
        uncondition = torch.cat(
            [self._encode_text(""), torch.zeros_like(global_tokens), torch.zeros_like(spatial_tokens)],
            dim=1,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(
            person_latent.shape, generator=generator, device=self.device, dtype=self.dtype
        )
        latent = noise * self.scheduler.init_noise_sigma
        mask_four = mask_latent.expand_as(person_latent)
        first_timestep = self.scheduler.timesteps[0].reshape(1).to(self.device)
        original_noisy = self.scheduler.add_noise(person_latent, noise, first_timestep)
        latent = mask_four * latent + (1.0 - mask_four) * original_noisy

        guidance = float(self.config["inference"]["guidance_scale"])
        for index, timestep in enumerate(self.scheduler.timesteps):
            model_input = torch.cat(
                [latent, mask_latent, agnostic_latent, pose_latent, cloth_latent], dim=1
            )
            prediction = components.unet(
                torch.cat([model_input, model_input]),
                timestep,
                encoder_hidden_states=torch.cat([uncondition, condition]),
            ).sample
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + guidance * (conditional - unconditional)
            latent = self.scheduler.step(prediction, timestep, latent).prev_sample
            if index < len(self.scheduler.timesteps) - 1:
                next_timestep = self.scheduler.timesteps[index + 1].reshape(1).to(self.device)
                original_noisy = self.scheduler.add_noise(person_latent, noise, next_timestep)
                latent = mask_four * latent + (1.0 - mask_four) * original_noisy

        latent = mask_four * latent + (1.0 - mask_four) * person_latent
        decoded = components.vae.decode((latent / self.settings.vae_scale).to(self.dtype)).sample
        generated = (
            ((decoded[0].float().cpu() + 1.0) / 2.0)
            .clamp(0, 1)
            .permute(1, 2, 0)
            .numpy()
        )
        original = ((person[0].float().cpu() + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).numpy()
        mask_array = mask[0, 0].float().cpu().numpy()
        composite_mask = cv2.GaussianBlur(mask_array, (13, 13), 0)[..., None]
        result = composite_mask * generated + (1.0 - composite_mask) * original
        return Image.fromarray((result.clip(0, 1) * 255).astype(np.uint8))

    def _load_sample(self, person_id: str, cloth_id: str, split: str) -> dict:
        root = self.viton_root / split
        helper = self.preprocessor
        helper.root = root
        person = helper._crop_resize(Image.open(root / "image" / f"{person_id}.jpg").convert("RGB"))
        agnostic = helper._crop_resize(
            Image.open(root / "agnostic-v3.2" / f"{person_id}.jpg").convert("RGB")
        )
        pose = helper._crop_resize(helper._load_pose(person_id))
        cloth_image = Image.open(root / "cloth" / f"{cloth_id}.jpg").convert("RGB")
        cloth_mask = Image.open(root / "cloth-mask" / f"{cloth_id}.jpg").convert("L")
        cloth = helper._process_cloth(cloth_image, cloth_mask)
        mask = helper._compute_parse_mask(person_id)
        caption = self._caption(cloth_id, split)
        tokens = self.tokenizer(
            caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).input_ids
        return {
            "image": helper._normalize(TF.to_tensor(person)).unsqueeze(0),
            "agnostic": helper._normalize(TF.to_tensor(agnostic)).unsqueeze(0),
            "pose": helper._normalize(TF.to_tensor(pose)).unsqueeze(0),
            "cloth": helper._normalize(TF.to_tensor(cloth)).unsqueeze(0),
            "inpaint_mask": TF.to_tensor(mask).unsqueeze(0),
            "clip_cloth": self.clip_processor(
                images=cloth, return_tensors="pt"
            ).pixel_values,
            "input_ids": tokens,
        }

    def _caption(self, cloth_id: str, split: str) -> str:
        path = self.caption_root / "cloth-captions" / split / f"{cloth_id}.txt"
        raw = path.read_text(encoding="utf-8").strip() if path.exists() else "a photo of a garment"
        return raw if self.settings.trigger_word in raw else f"{self.settings.trigger_word}, {raw}"

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.components.vae.encode(image).latent_dist.mode() * self.settings.vae_scale

    def _encode_text(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer(
            [prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        return self.components.text_encoder(ids)[0]
