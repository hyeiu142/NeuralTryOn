# ================================================================
# CELL 13: LOAD MODELS FOR V2 17CH INFERENCE
# ================================================================

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection
from peft import LoraConfig, get_peft_model


class PerceiverResampler(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, num_queries=8, depth=2, num_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, input_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(input_dim),
                nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=0.0),
                nn.LayerNorm(input_dim),
                nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=0.0),
                nn.LayerNorm(input_dim),
                nn.Sequential(
                    nn.Linear(input_dim, input_dim * 4),
                    nn.GELU(),
                    nn.Linear(input_dim * 4, input_dim),
                ),
            ])
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(input_dim)
        self.proj_out = nn.Linear(input_dim, output_dim)

    def forward(self, clip_patches):
        x = self.latents.expand(clip_patches.shape[0], -1, -1)
        for norm1, sa, norm2, ca, norm3, ff in self.layers:
            x = x + sa(norm1(x), norm1(x), norm1(x))[0]
            x = x + ca(norm2(x), clip_patches, clip_patches)[0]
            x = x + ff(norm3(x))
        return self.proj_out(self.norm_out(x))


class ClothSpatialProjector(nn.Module):
    def __init__(self, in_ch=4, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.proj = nn.Linear(256, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, cloth_lat):
        x = self.conv(cloth_lat.float())
        x = x.flatten(2).permute(0, 2, 1)
        return self.norm(self.proj(x))


def expand_unet_conv_in(unet, new_channels=17):
    base = unet.base_model.model if hasattr(unet, "base_model") else unet
    old_conv = base.conv_in
    new_conv = nn.Conv2d(
        new_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight[:, :9] = old_conv.weight.clone()
        new_conv.weight[:, 9:] = 0.0
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    base.conv_in = new_conv
    base.config["in_channels"] = new_channels
    print(f"UNet conv_in: 9ch -> {new_channels}ch")
    return unet


print("=" * 55)
print("Loading Models for V2 Inference")
print("=" * 55)

print("\n[1/6] Loading VAE...")
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=DTYPE)
vae.enable_slicing()
vae.enable_tiling()
vae.requires_grad_(False).eval().to(DEVICE)
print("  VAE loaded")

print("\n[2/6] Loading Tokenizer + Text Encoder...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE)
text_encoder.requires_grad_(False).eval().to(DEVICE)
print("  Text Encoder loaded")

print("\n[3/6] Loading UNet base + LoRA...")
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=DTYPE)
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGETS,
    lora_dropout=LORA_DROPOUT,
    bias="none",
)
unet = get_peft_model(unet, lora_config)
unet = expand_unet_conv_in(unet, new_channels=UNET_IN_CHANNELS)

adapter_name = "v2_infer"
try:
    unet.load_adapter(str(CKPT_DIR / "unet_lora"), adapter_name=adapter_name, is_trainable=False)
except TypeError:
    unet.load_adapter(str(CKPT_DIR / "unet_lora"), adapter_name=adapter_name)
unet.set_adapter(adapter_name)

conv_in_path = CKPT_DIR / "conv_in.pt"
if not conv_in_path.exists():
    raise FileNotFoundError(
        f"Thieu {conv_in_path}. Checkpoint 17ch phai duoc train lai va luu conv_in.pt."
    )
unet.base_model.model.conv_in.load_state_dict(torch.load(conv_in_path, map_location="cpu"))
print("  conv_in.pt loaded")

ema_lora_path = CKPT_DIR / "ema_unet_lora.pt"
if USE_EMA and ema_lora_path.exists():
    ema_lora_dict = torch.load(ema_lora_path, map_location="cpu")
    cur_state = unet.state_dict()
    cur_state.update(ema_lora_dict)
    unet.load_state_dict(cur_state)
    ema_conv_in_path = CKPT_DIR / "ema_conv_in.pt"
    if ema_conv_in_path.exists():
        unet.base_model.model.conv_in.load_state_dict(torch.load(ema_conv_in_path, map_location="cpu"))
        print("  ema_conv_in.pt loaded")
    print("  UNet + EMA LoRA loaded")
else:
    print("  UNet + live LoRA loaded")

unet.requires_grad_(False).eval().to(DEVICE, DTYPE)
try:
    unet.enable_xformers_memory_efficient_attention()
    print("  xformers enabled")
except Exception as e:
    print(f"  xformers not enabled: {e}")

print("\n[4/6] Loading CLIP Vision Encoder...")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    IP_ADAPTER,
    subfolder="models/image_encoder",
    torch_dtype=DTYPE,
)
image_encoder.requires_grad_(False).eval().to(DEVICE)
print("  CLIP Vision loaded")

print("\n[5/6] Loading Perceiver + ClothSpatial...")
with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224, dtype=DTYPE, device=DEVICE)
    clip_dim = image_encoder(pixel_values=dummy, output_hidden_states=True).last_hidden_state.shape[-1]
cross_attn_dim = unet.base_model.model.config.cross_attention_dim
print(f"  CLIP hidden dim: {clip_dim}")

perceiver = PerceiverResampler(
    input_dim=clip_dim,
    output_dim=cross_attn_dim,
    num_queries=NUM_IMAGE_TOKENS,
    depth=PERCEIVER_DEPTH,
    num_heads=PERCEIVER_HEADS,
).to(DEVICE)
perceiver.load_state_dict(torch.load(CKPT_DIR / "perceiver.pt", map_location=DEVICE))
perceiver.requires_grad_(False).eval()
print("  Perceiver loaded")

cloth_spatial_proj = ClothSpatialProjector(in_ch=4, out_dim=cross_attn_dim).to(DEVICE)
cloth_spatial_proj.load_state_dict(torch.load(CKPT_DIR / "cloth_spatial.pt", map_location=DEVICE))
cloth_spatial_proj.requires_grad_(False).eval()
print("  ClothSpatialProj loaded")

print("\n[6/6] Loading DDIM Scheduler...")
scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
print("  Scheduler loaded")

print("\n" + "=" * 55)
print("All models loaded. Ready for holdout inference.")
print(f"  Adapter: {adapter_name}")
print(f"  Perceiver: tokens={NUM_IMAGE_TOKENS}, depth={PERCEIVER_DEPTH}")
print("=" * 55)
