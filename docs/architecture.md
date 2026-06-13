# Model Architecture and Training Summary

The project compares three Virtual Try-On approaches with different resource
and modeling trade-offs.

## Shared Inputs

All models use conditions derived from VITON-HD:

```text
Person image
Target garment and garment mask
Clothing-agnostic person image
Human parsing map
OpenPose keypoints or pose heatmaps
DensePose representation
```

Images are processed at `384 x 512` resolution. RGB tensors are normalized to
`[-1, 1]`, while masks remain in `[0, 1]`.

## Model 1: Lightweight U-Net + GMM + TOM

```text
Person conditions
    -> Lightweight U-Net clothing-region mask
    -> GMM/TPS target-garment warping
    -> TOM rendering and composition
    -> final try-on image
```

The Lightweight U-Net predicts the garment replacement region. The Geometric
Matching Module estimates Thin Plate Spline control points and warps the flat
garment into the person's pose. The Try-On Module renders missing content and
composites it with the warped garment.

Training is split into independent modules and includes early stopping,
learning-rate scheduling, checkpointing, and TOM ablation experiments.

## Model 2: GMM + Shape Generation + Pix2Pix

```text
Person conditions
    -> GMM/TPS target-garment warping
    -> Stage 1 shape-generation network
    -> Stage 2 Pix2Pix generator
    -> PatchGAN discriminator during training
    -> final try-on image
```

The GMM aligns the garment using agnostic, DensePose, pose-heatmap, garment,
and garment-mask inputs. Stage 1 predicts the target clothing and body
generation region. Stage 2 uses a U-Net-like generator to render missing
content and blend the warped garment.

Stage 2 combines reconstruction, perceptual, edge, region, mask, and LSGAN
adversarial losses.

The PatchGAN discriminator evaluates local image patches. Training includes
mixed precision, gradient clipping, checkpoint resume, early stopping, and
learning-rate scheduling.

## Model 3: Stable Diffusion Inpainting + LoRA

Model 3 adapts `runwayml/stable-diffusion-inpainting` using Parameter-Efficient
Fine-Tuning.

The UNet input convolution is expanded to 17 channels:

```text
Noisy latent       4
Inpainting mask    1
Agnostic latent    4
Pose latent        4
Garment latent     4
                  --
Total             17
```

Garment appearance is represented by a frozen CLIP Vision encoder. A Perceiver
Resampler creates compact global garment tokens, while a Cloth Spatial
Projector creates spatial garment features.

Trainable components:

```text
UNet LoRA adapters
Expanded UNet input convolution
Perceiver Resampler
Cloth Spatial Projector
```

Frozen components:

```text
VAE
Text encoder
CLIP Vision encoder
Base Stable Diffusion weights
```

This model is the project's main transfer-learning and PEFT experiment.

## Architecture Comparison

| Property | Model 1 | Model 2 | Model 3 |
| --- | --- | --- | --- |
| Main approach | Lightweight CNN pipeline | Conditional GAN pipeline | Latent diffusion with LoRA |
| Garment alignment | GMM/TPS | GMM/TPS | Garment latent and CLIP conditioning |
| Main generator | TOM | Pix2Pix-style generator | Stable Diffusion UNet |
| Adversarial training | No | PatchGAN/LSGAN | No |
| Transfer learning | Limited | VGG perceptual features | Stable Diffusion, CLIP, LoRA |
| Resource strategy | Small custom modules | Staged training | PEFT with frozen pretrained modules |
| Expected strength | Speed and structure | Texture and sharpness | Perceptual realism and garment semantics |
