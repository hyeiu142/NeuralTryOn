# Archived Image Classification Manifest

Images from `archive/image/` were copied into report-ready result directories.
The original archive files remain unchanged.

## Model 3 Convergence

| Destination | Archive Source | Purpose |
| --- | --- | --- |
| `convergence/model_3_epoch12_convergence_analysis.png` | `checkpoint_epoch_12_convergence_analysis.png` | Combined train/validation, step-level loss, and generalization-gap figure |
| `convergence/model_3_epoch12_train_validation_loss.png` | `checkpoint_epoch_12_train_validation_loss.png` | Report-ready train and validation loss chart |
| `convergence/model_3_epoch12_generalization_gap.png` | `checkpoint_epoch_12_generalization_gap.png` | Generalization-gap analysis |

`checkpoint_epoch_12_convergence_analysis (1).png` was excluded because it is
byte-identical to `checkpoint_epoch_12_convergence_analysis.png`.

`checkpoint_epoch_12_convergence_analysis (2).png` was excluded because it is a
reduced two-panel version of information already available in the combined and
individual charts.

## Model 3 Metrics

| Destination | Archive Source | Purpose |
| --- | --- | --- |
| `metrics/model_3_epoch12_ssim_psnr_lpips_distributions.png` | `ssim_psnr_lpips_distributions.png` | Final 996-sample paired-test SSIM, PSNR, and LPIPS distributions |
| `metrics/model_3_epoch12_auxiliary_metric_distributions.jpg` | `checkpoint_epoch_12_metric_distributions.jpg` | Auxiliary CLIP, masked-LPIPS, and SSIM distributions |

The auxiliary metric figure belongs to an earlier or additional paired
analysis and should not replace the final 996-sample SSIM/PSNR/LPIPS result.

## Model 3 Comparisons

| Destination | Archive Source |
| --- | --- |
| `comparisons/model_3_sd_lora/best_paired_samples.jpg` | `checkpoint_epoch_12_best_paired_samples.jpg` |
| `comparisons/model_3_sd_lora/random_paired_samples.jpg` | `checkpoint_epoch_12_random_paired_samples.jpg` |
| `comparisons/model_3_sd_lora/random5_unpaired_three_columns.jpg` | `random5_unpaired_three_columns.jpg` |

## Model 3 Error Analysis

| Destination | Archive Source |
| --- | --- |
| `error_analysis/model_3_sd_lora/worst_paired_samples.jpg` | `checkpoint_epoch_12_worst_paired_samples.jpg` |

The worst paired gallery demonstrates garment identity and graphic/pattern
reconstruction failures and can be used directly in the report's error
analysis section.
