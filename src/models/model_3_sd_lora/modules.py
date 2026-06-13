"""Trainable conditioning modules used by Model 3."""

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """Convert CLIP garment patches into compact global conditioning tokens."""

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 768,
        num_queries: int = 8,
        depth: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, input_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.LayerNorm(input_dim),
                        nn.MultiheadAttention(input_dim, num_heads, batch_first=True),
                        nn.LayerNorm(input_dim),
                        nn.MultiheadAttention(input_dim, num_heads, batch_first=True),
                        nn.LayerNorm(input_dim),
                        nn.Sequential(
                            nn.Linear(input_dim, input_dim * 4),
                            nn.GELU(),
                            nn.Linear(input_dim * 4, input_dim),
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(input_dim)
        self.proj_out = nn.Linear(input_dim, output_dim)

    def forward(self, clip_patches: torch.Tensor) -> torch.Tensor:
        x = self.latents.expand(clip_patches.shape[0], -1, -1)
        for norm1, self_attention, norm2, cross_attention, norm3, feed_forward in self.layers:
            normalized = norm1(x)
            x = x + self_attention(normalized, normalized, normalized)[0]
            x = x + cross_attention(norm2(x), clip_patches, clip_patches)[0]
            x = x + feed_forward(norm3(x))
        return self.proj_out(self.norm_out(x))


class ClothSpatialProjector(nn.Module):
    """Project garment latent features into spatial conditioning tokens."""

    def __init__(self, in_channels: int = 4, output_dim: int = 768) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.proj = nn.Linear(256, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, cloth_latent: torch.Tensor) -> torch.Tensor:
        features = self.conv(cloth_latent.float()).flatten(2).permute(0, 2, 1)
        return self.norm(self.proj(features))


def expand_unet_conv_in(unet: nn.Module, new_channels: int = 17) -> nn.Module:
    """Expand the inpainting UNet input convolution with zero-initialized channels."""
    base = unet.base_model.model if hasattr(unet, "base_model") else unet
    old_conv = base.conv_in
    new_conv = nn.Conv2d(
        new_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, : old_conv.in_channels] = old_conv.weight.clone()
        new_conv.weight[:, old_conv.in_channels :] = 0.0
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    base.conv_in = new_conv
    base.config["in_channels"] = new_channels
    return unet

