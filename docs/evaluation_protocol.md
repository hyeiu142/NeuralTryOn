# Evaluation Protocol

## Evaluation Goals

The project uses two complementary evaluation modes.

### Paired Reconstruction

The target garment is the garment already worn by the person:

```text
cloth_id = person_id
```

This mode has a pixel-aligned ground-truth person image and supports:

```text
SSIM   higher is better
PSNR   higher is better
LPIPS  lower is better
```

### Unpaired Holdout

The person is combined with a different target garment. Because no
pixel-aligned ground-truth result exists, holdout evaluation is primarily
qualitative:

```text
Person Input | Target Cloth | Try-On Result
```

Review criteria include garment identity, body and face preservation, boundary
artifacts, texture loss, color bleeding, and pose consistency.

## Current Completed Results

| Model | Paired samples | Paired source | LPIPS backbone | Holdout |
| --- | ---: | --- | --- | ---: |
| Model 1: Lightweight U-Net + GMM + TOM | 2032 | All files in `VITON-HD/test/image` | VGG | 665 |
| Model 3: Stable Diffusion + LoRA | 996 | `clean_vto_dataset_test.csv` | AlexNet | 665 |

The current values document each completed experiment. They must not be used
as a strict model ranking because their paired manifests and LPIPS backbones
differ.

## Final Common Comparison Contract

For the final comparison of Models 1, 2, and 3:

1. Use `clean_vto_dataset_test.csv`.
2. Force `cloth_id = person_id`.
3. Use the same image resolution and RGB range.
4. Use the same SSIM and PSNR implementation.
5. Use LPIPS-AlexNet for every model.
6. Save one row per sample and report mean plus standard deviation.
7. Use the same holdout person-cloth manifest for qualitative comparison.

Required per-sample columns:

```text
model, sample_index, person_id, cloth_id,
ssim, psnr, lpips, seconds, status, error
```
