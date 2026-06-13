# Model Architecture Summary

## Model 1: Lightweight U-Net Pipeline

The first pipeline predicts a clothing-region mask with a lightweight U-Net,
warps the garment using a Geometric Matching Module, and synthesizes the final
image using a Try-On Module.

```text
Person conditions → Lightweight U-Net mask → GMM garment warp → TOM result
```

## Model 2: Pix2Pix Pipeline

The second pipeline combines geometric garment alignment, shape generation,
and adversarial texture fusion.

```text
Person conditions → GMM → Shape Generation → Pix2Pix Generator + PatchGAN
```

## Model 3: Stable Diffusion + LoRA

The third pipeline adapts Stable Diffusion Inpainting with parameter-efficient
LoRA weights. The UNet input is expanded to 17 channels:

```text
noisy latent (4) + mask (1) + agnostic latent (4)
+ pose latent (4) + cloth latent (4) = 17 channels
```

CLIP Vision garment features are converted into global and spatial cloth tokens
through a Perceiver Resampler and Cloth Spatial Projector. Trainable components
are LoRA weights, the expanded input convolution, Perceiver, and spatial
projector; pretrained VAE, text encoder, and CLIP Vision remain frozen.
