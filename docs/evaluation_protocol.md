# Evaluation Protocol

The project uses two complementary evaluation modes.

## Paired Reconstruction

The target garment is the garment already worn by the person:

```text
cloth_id = person_id
```

This mode has pixel-aligned ground truth. SSIM and PSNR are higher-is-better;
LPIPS is lower-is-better.

## Unpaired Holdout

The person is combined with a different target garment. Because no
pixel-aligned ground-truth result exists, holdout evaluation is primarily
qualitative:

```text
Person Input | Target Cloth | Try-On Result
```

Review criteria include garment identity, body and face preservation, boundary
artifacts, texture loss, color bleeding, and pose consistency.

## Shared Protocol

For the final comparison of Models 1, 2, and 3:

1. Use `clean_vto_dataset_test.csv`.
2. Force `cloth_id = person_id`.
3. Use the same image resolution and RGB range.
4. Use the same SSIM and PSNR implementation.
5. Save one row per sample and report mean plus standard deviation.
6. Use the same holdout person-cloth manifest for qualitative comparison.

Required per-sample columns:

```text
model, sample_index, person_id, cloth_id,
ssim, psnr, lpips, seconds, status, error
```
