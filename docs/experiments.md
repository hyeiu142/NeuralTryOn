# Experiment Status and Results

## Training Status

| Model | Training Status | Evaluation Status |
| --- | --- | --- |
| Model 1: Lightweight U-Net + GMM + TOM | Completed | Paired and holdout completed |
| Model 2: GMM + Shape Generation + Pix2Pix | Completed through Stage 2 epoch 30 | Paired completed; full holdout pending |
| Model 3: Stable Diffusion Inpainting + LoRA | Completed through epoch 12 | Paired and holdout completed |

## Common 996-Sample Paired Results

| Model | Samples | SSIM | PSNR | LPIPS | Backbone |
| --- | ---: | ---: | ---: | ---: | --- |
| Model 1 | 996 | 0.8932 | 21.39 dB | 0.1455 | VGG |
| Model 2 | 996 | 0.8951 | 21.49 dB | 0.1145 | VGG |
| Model 3 | 996 | 0.8733 +/- 0.0615 | 21.32 +/- 3.78 dB | 0.1055 +/- 0.0455 | AlexNet |

The report uses `clean_vto_dataset_test.csv` as the common 996-sample
evaluation set. Model 1 and Model 2 values are retained from their completed
legacy runs and require confirmation on the common manifest. LPIPS backbone
differences must also be considered when interpreting the results.

## Training Observations

### Model 1

The lightweight pipeline completed all core stages and generated the full
holdout result set. Its SSIM and PSNR indicate strong structural preservation,
while LPIPS shows that perceptual details remain more difficult.

### Model 2

GMM and Stage 1 stopped at epoch 10 through early stopping. Stage 2 resumed
from saved checkpoints and completed epoch 30. The reported paired metrics are
the strongest SSIM and PSNR values among the current runs. The full 665-sample
holdout export remains to be executed.

The notebook's validation dataset uses `stage="train"`, which means
synchronized augmentation is also applied during validation. This should be
reported as a limitation when interpreting convergence and early stopping.

### Model 3

The final evaluated checkpoint is epoch 12. Model 3 reports the lowest current
LPIPS value, indicating strong perceptual similarity under its evaluation
protocol. It also provides the most complete unpaired proxy evaluation,
including CLIP garment similarity and outside-mask preservation.

The Model 3 training notebook uses `clean_vto_dataset_test.csv` for validation
and checkpoint selection. Its final paired metrics are therefore
validation-seen and should not be described as evaluation on a completely
untouched test set.

## Final Comparison Requirement

A strict final comparison requires:

1. The same 996-sample clean paired manifest for every model.
2. The same LPIPS backbone and metric implementation.
3. The same fixed holdout person-cloth pairs.
4. Per-sample metrics and inference time.
5. Fixed-pair qualitative galleries and cross-model error analysis.
