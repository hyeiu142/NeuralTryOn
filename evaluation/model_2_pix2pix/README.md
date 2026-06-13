# Model 2 Evaluation: GMM + Shape Generation + Pix2Pix

This directory organizes the evaluation workflow for the current Model 2
notebook without modifying that notebook.

## Pipeline

```text
Person conditions
    -> GMM and TPS garment warping
    -> Stage 1 shape-generation network
    -> Stage 2 Pix2Pix generator and PatchGAN training
    -> final try-on image
```

The evaluated generator uses L1, VGG perceptual, Sobel edge, skin rendering,
full rendering, composition-mask, and adversarial losses during training.

## Structure

```text
model_2_pix2pix/
├── runtime/
│   ├── 00_install_dependencies.py
│   ├── 01_configure_paths.py
│   ├── 02_prepare_runtime.py
│   └── 03_inference.py
├── paired_test/
│   ├── 01_evaluate_metrics.py
│   ├── 02_generate_report.py
│   ├── 03_export_three_column_gallery.py
│   └── 04_publish_kaggle_dataset.py
├── unpaired_holdout/
│   ├── 01_generate_holdout.py
│   ├── 02_export_three_column_gallery.py
│   ├── 03_generate_report.py
│   └── 04_publish_kaggle_dataset.py
└── workflow.py
```

## Current Completed Results

```text
Training split       8568 samples
Validation split      952 samples
Paired test          2032 samples
SSIM                 0.8951
PSNR                 21.49 dB
LPIPS-VGG            0.1145
Manual unpaired demos  13 pairs
Full holdout          not yet executed
```

GMM and Stage 1 stopped at epoch 10. Stage 2 completed epoch 30 through
checkpoint resume. Training includes fixed random seeds, validation,
checkpointing, schedulers, early stopping, mixed precision, and convergence
charts.

## Execution

Run these stages in the same namespace after the Model 2 architecture cells:

```python
from evaluation.model_2_pix2pix.workflow import run_stage

for stage in [
    "runtime/00_install_dependencies.py",
    "runtime/01_configure_paths.py",
    "runtime/02_prepare_runtime.py",
    "runtime/03_inference.py",
    "paired_test/01_evaluate_metrics.py",
    "paired_test/02_generate_report.py",
    "paired_test/03_export_three_column_gallery.py",
    "unpaired_holdout/01_generate_holdout.py",
    "unpaired_holdout/02_export_three_column_gallery.py",
    "unpaired_holdout/03_generate_report.py",
]:
    run_stage(stage, globals())
```

Publishing stages are optional.

## Current Limitations

- The notebook validation dataset uses `stage="train"`, so synchronized
  augmentation is also applied during validation.
- The paired evaluation scans all 2032 raw test images rather than the
  996-sample clean paired manifest.
- LPIPS uses the VGG backbone.
- The notebook contains manual holdout demonstrations but has not yet saved a
  complete full-holdout result set.
- No explicit multi-configuration ablation table is currently recorded.
