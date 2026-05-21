# File: src/models/builder.py
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPVisionModelWithProjection
from peft import get_peft_model, LoraConfig

class ImageProjModel(nn.Module):
    def __init__(self, clip_dim, cross_attn_dim, num_tokens=4):
        super().__init__()
        self.clip_dim = clip_dim
        self.cross_attn_dim = cross_attn_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(clip_dim, cross_attn_dim * num_tokens),
            nn.LayerNorm(cross_attn_dim * num_tokens),
        )

    def forward(self, image_embeds):
        x = self.proj(image_embeds)
        x = x.view(image_embeds.shape[0], self.num_tokens, self.cross_attn_dim)
        return x

def build_models(cfg):
    """Khởi tạo toàn bộ pipeline từ config"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading Stable Diffusion Inpainting components...")
    vae = AutoencoderKL.from_pretrained(cfg.model.model_id, subfolder="vae", torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(cfg.model.model_id, subfolder="text_encoder", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(cfg.model.model_id, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.model_id, subfolder="scheduler")

    print("Loading CLIP image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        cfg.model.ip_adapter_id, subfolder="models/image_encoder", torch_dtype=dtype
    )

    # Đóng băng các base module
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    vae.eval()
    text_encoder.eval()
    image_encoder.eval()

    # Memory optimization
    if device == "cuda":
        try:
            vae.enable_slicing()
            vae.enable_tiling()
            unet.enable_gradient_checkpointing()
        except:
            pass

    # Thêm LoRA vào UNet
    lora_config = LoraConfig(
        r=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    # Khởi tạo Image Projection
    dummy_clip = torch.zeros(1, 3, 224, 224, dtype=dtype)
    dummy_out = image_encoder(pixel_values=dummy_clip)
    clip_dim = dummy_out.image_embeds.shape[-1]
    cross_attn_dim = unet.base_model.model.config.cross_attention_dim

    image_proj = ImageProjModel(
        clip_dim=clip_dim,
        cross_attn_dim=cross_attn_dim,
        num_tokens=cfg.model.num_image_tokens
    ).to(dtype)
    image_proj.train()

    # Chuyển tất cả lên device
    vae.to(device)
    text_encoder.to(device)
    image_encoder.to(device)
    unet.to(device)
    image_proj.to(device)

    return vae, text_encoder, unet, noise_scheduler, image_encoder, image_proj
