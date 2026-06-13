# Experiment Status and Results

## Training Status

| Model | Training Status | Evaluation Status |
| --- | --- | --- |
| Model 1: Lightweight U-Net + GMM + TOM | Completed | Paired and holdout completed |
| Model 2: GMM + Shape Generation + Pix2Pix | Completed through Stage 2 epoch 30 | Paired completed; full holdout pending |
| Model 3: Stable Diffusion Inpainting + LoRA | Completed through epoch 12 | Paired and holdout completed |

## Reported Paired Results

| Model | Samples | SSIM | PSNR | LPIPS | Backbone |
| --- | ---: | ---: | ---: | ---: | --- |
| Model 1 | 2032 | 0.8932 | 21.39 dB | 0.1455 | VGG |
| Model 2 | 2032 | 0.8951 | 21.49 dB | 0.1145 | VGG |
| Model 3 | 996 | 0.8733 +/- 0.0615 | 21.32 +/- 3.78 dB | 0.1055 +/- 0.0455 | AlexNet |

These results document completed runs but are not a strict ranking because
Model 3 uses the cleaned 996-sample paired manifest and LPIPS-AlexNet, while
Models 1 and 2 scan all 2032 test images and use LPIPS-VGG.

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

## Final Comparison Requirement

A strict final comparison requires:

1. The same 996-sample clean paired manifest for every model.
2. The same LPIPS backbone and metric implementation.
3. The same fixed holdout person-cloth pairs.
4. Per-sample metrics and inference time.
5. Fixed-pair qualitative galleries and cross-model error analysis.
