# Model 3 Evaluation: Stable Diffusion + LoRA

This directory contains the complete inference and evaluation workflow for the
final SD Inpainting + LoRA checkpoint. The source was consolidated from the
successful Kaggle evaluation cells while preserving their behavior.

## Evaluation Contract

```text
Base model       runwayml/stable-diffusion-inpainting
Checkpoint       /kaggle/input/datasets/khoaanh1234/ckpt-epoch-12-yen
Resolution       384 x 512
Inference steps  45 DDIM steps
CFG scale        1.3
```

The runtime loads:

```text
UNet LoRA adapter
17-channel UNet conv_in
Perceiver Resampler
Cloth Spatial Projector
VAE, text encoder, CLIP Vision encoder, DDIM scheduler
```

## Directory Structure

```text
model_3_sd_lora/
├── runtime/
│   ├── 00_install_dependencies.py
│   ├── 01_configure_paths.py
│   ├── 02_load_models.py
│   └── 03_inference.py
├── paired_test/
│   ├── 01_evaluate_metrics.py
│   ├── 02_export_three_column_gallery.py
│   └── 03_publish_kaggle_dataset.py
├── unpaired_holdout/
│   ├── 01_evaluate_holdout.py
│   ├── 02_generate_report.py
│   └── 03_publish_kaggle_dataset.py
└── review_publish/
    ├── 01_review_published_dataset.py
    ├── 02_review_manual_shortlist.py
    └── 03_export_shortlist_gallery.py
```

Run the workflow through:

```text
notebooks/03_evaluation/model_3_sd_lora_evaluation.ipynb
```

## Runtime Stages

The runtime stages must execute in order and share the same notebook namespace.

1. `00_install_dependencies.py`
   - Checks and installs missing Kaggle dependencies.
   - Restart the session only when new packages were installed.
2. `01_configure_paths.py`
   - Locates VITON-HD, cleaned CSVs, captions, and epoch-12 checkpoint.
   - Copies checkpoint files into `/kaggle/working/checkpoint_latest`.
   - Verifies checkpoint metadata against the 17-channel architecture.
3. `02_load_models.py`
   - Rebuilds and loads all model components.
4. `03_inference.py`
   - Defines `run_inference(person_id, cloth_id, split="test")`.
   - Returns an RGB NumPy image in `[0, 1]`.
   - Preserves pixels outside the clothing mask.

## Paired Test Evaluation

Source:

```text
clean_vto_dataset_test.csv
```

Evaluation forces:

```text
cloth_id = person_id
```

This provides a ground-truth person image for full-image reconstruction metrics:

```text
SSIM   higher is better
PSNR   higher is better
LPIPS  lower is better
```

The evaluator supports resume and writes results after every sample:

```text
/kaggle/working/infer_results_v2_epoch12/paired_test_metrics/
├── paired_test_manifest.csv
├── paired_test_metrics.csv
├── paired_test_summary.json
├── ssim_psnr_lpips_distributions.png
├── images/
└── comparisons_3_columns/
```

Final measured result on 996 samples:

```text
SSIM   0.8733 +/- 0.0615
PSNR   21.32 +/- 3.78 dB
LPIPS  0.1055 +/- 0.0455
```

## Unpaired Holdout Evaluation

Source:

```text
holdout_test.csv
```

The original `person_id` and `cloth_id` pairs are preserved. Because these
cross-garment pairs have no pixel-aligned ground truth, the workflow reports:

```text
CLIP garment similarity
outside-mask MAE
raw outside-mask MAE
mask area
inference time
```

It also saves five-panel comparisons:

```text
Person | Target Cloth | Mask | Raw Model | Final Result
```

## Review and Publishing

Publishing stages package full result sets as private Kaggle Datasets. Review
stages load a published unpaired dataset, rank results, inspect a manual
shortlist, and export report-ready three-column galleries.

Publishing is optional. Always verify the Kaggle Dataset link before stopping a
session because files under `/kaggle/working` disappear when the session ends.

## Excluded Legacy Cells

The local archive retains older paired-holdout reporting experiments and an
empty convergence placeholder. They are intentionally excluded from this public
pipeline because paired quantitative evaluation is now correctly performed on
`clean_vto_dataset_test.csv`.
