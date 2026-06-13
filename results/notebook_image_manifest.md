# Notebook Image Extraction Manifest

These report-ready images were extracted from embedded notebook outputs without
modifying the notebooks.

## Convergence

| Output | Source |
| --- | --- |
| `convergence/model_1_unet_training_validation_loss.png` | Model 1, Cell 39 |
| `convergence/model_1_gmm_training_validation_loss.png` | Model 1, Cell 42 |
| `convergence/model_1_tom_baseline_no_dropout.png` | Model 1, Cell 45 |
| `convergence/model_1_tom_with_dropout_03.png` | Model 1, Cell 45 |
| `convergence/model_1_tom_low_lr_5e5.png` | Model 1, Cell 45 |
| `convergence/model_1_tom_high_lr_5e4.png` | Model 1, Cell 45 |
| `convergence/model_2_gmm_training_validation_loss.png` | Model 2, Cell 32 |
| `convergence/model_2_stage1_training_validation_loss.png` | Model 2, Cell 35 |
| `convergence/model_2_stage2_training_validation_loss.png` | Model 2, Cell 40 |

## Model Diagnostics and Comparisons

```text
comparisons/model_1_lightweight_unet/
    U-Net mask, GMM warp, TOM-stage diagnostics, and ten holdout samples

comparisons/model_2_pix2pix/
    GMM, Stage 1, and Stage 2 validation diagnostics
    Thirteen manually selected unpaired try-on demonstrations
```

Model 1 holdout figures show the notebook's agnostic input rather than the
original person image. Use them as Model 1 qualitative diagnostics, not as the
final cross-model three-column comparison.

## EDA and Preprocessing

```text
eda/model_2_pix2pix/
    Human parsing labels, agnostic preprocessing, and pose heatmap

eda/model_3_sd_lora/
    Preprocessing examples, dataset samples, cloth-region statistics,
    and caption statistics
```

No notebook images were automatically placed under `error_analysis/`. Failure
cases should be selected and labeled manually to avoid unsupported claims.
