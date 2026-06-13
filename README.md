# Resource-Efficient Virtual Try-On with CNN, GAN, and Diffusion Models

This repository contains a Deep Learning final project that studies
resource-efficient image-based Virtual Try-On using the
[VITON-HD dataset](https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset).

The project implements three different modeling approaches:

1. A lightweight convolutional pipeline built from U-Net, geometric cloth
   warping, and a Try-On Module.
2. A conditional GAN pipeline based on Pix2Pix and PatchGAN.
3. A Stable Diffusion Inpainting pipeline adapted with LoRA and garment-aware
   conditioning.

The main objective is not only to generate try-on images, but also to document
the complete deep learning lifecycle:

```text
Data preparation
    -> preprocessing and augmentation
    -> model construction
    -> training and hyperparameter experiments
    -> checkpoint management
    -> quantitative evaluation
    -> qualitative comparison
    -> error analysis
```

The implementation is designed for limited GPU environments and is intended to
run primarily on Kaggle.

## Project Objectives

The project investigates the following questions:

- How well can a lightweight CNN-based pipeline preserve person structure and
  garment appearance?
- How does adversarial training affect texture quality and visual realism?
- Can Parameter-Efficient Fine-Tuning adapt Stable Diffusion for Virtual
  Try-On without fully fine-tuning the base model?
- What trade-offs exist between reconstruction quality, perceptual quality,
  inference cost, and GPU memory usage?
- How should paired reconstruction and practical unpaired try-on be evaluated
  when only the former has pixel-aligned ground truth?

The project satisfies the course requirements through custom PyTorch dataset
classes, explicit training loops, transfer learning, LoRA, checkpointing,
reproducibility controls, ablation experiments, convergence analysis, and
quantitative and qualitative evaluation.

## Implemented Models

| Model | Architecture | Main Purpose | Status |
| --- | --- | --- | --- |
| Model 1 | Lightweight U-Net + GMM/TPS + TOM | Lightweight custom CNN baseline | Training and evaluation completed |
| Model 2 | GMM + Shape Generation + Pix2Pix + PatchGAN | Adversarial image synthesis experiment | Training notebook available; final evaluation pending |
| Model 3 | Stable Diffusion Inpainting + LoRA + garment conditioning | Transfer learning and PEFT experiment | Training and evaluation completed |

### Model 1: Lightweight U-Net + GMM + TOM

Model 1 is a multi-stage custom pipeline:

```text
Person conditions
    -> Lightweight U-Net predicts the clothing region
    -> GMM and Thin Plate Spline warp the target garment
    -> TOM renders and composites the final try-on image
```

Inputs include the person image, agnostic representation, OpenPose heatmaps,
DensePose representation, target cloth, and cloth mask.

This model demonstrates:

- Custom encoder-decoder architecture design.
- Geometric garment alignment.
- Multi-stage training.
- Early stopping and checkpoint recovery.
- Lightweight inference suitable for constrained hardware.

### Model 2: Pix2Pix + PatchGAN

Model 2 introduces adversarial training:

```text
Person conditions
    -> geometric alignment
    -> shape generation
    -> Pix2Pix generator
    -> PatchGAN discriminator
    -> final try-on image
```

This model is intended to study whether a conditional GAN improves local
texture realism compared with a reconstruction-focused CNN pipeline.

### Model 3: Stable Diffusion Inpainting + LoRA

Model 3 adapts `runwayml/stable-diffusion-inpainting` using LoRA and additional
garment-aware conditions.

The Stable Diffusion UNet input is expanded to 17 channels:

```text
Noisy latent       4 channels
Inpainting mask    1 channel
Agnostic latent    4 channels
Pose latent        4 channels
Garment latent     4 channels
                  -----------
Total             17 channels
```

Garment appearance is additionally represented using:

- A frozen CLIP Vision encoder.
- A Perceiver Resampler for compact garment tokens.
- A Cloth Spatial Projector for spatial garment features.
- Text captions generated for the target garments.

Only a small subset of parameters is trained:

```text
LoRA adapter weights
Expanded UNet input convolution
Perceiver Resampler
Cloth Spatial Projector
```

The pretrained VAE, text encoder, and CLIP Vision encoder remain frozen. This
provides the project's main PEFT and resource-optimization experiment.

## Dataset

All models use VITON-HD:

```text
/kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset
```

Important VITON-HD components include:

```text
image/             Person images
cloth/             Flat garment images
cloth-mask/        Garment masks
agnostic-v3.2/     Clothing-agnostic person images
image-parse-v3/    Human parsing maps
image-densepose/   DensePose conditions
openpose_img/      Pose visualization
openpose_json/     Pose keypoints
```

### Cleaned Data Splits

The data pipeline produces the following CSV files:

| Split | Samples | Purpose |
| --- | ---: | --- |
| `clean_vto_dataset_train.csv` | 9,520 | Training and train/validation splitting |
| `clean_vto_dataset_test.csv` | 996 | Clean paired reconstruction evaluation |
| `holdout_test.csv` | 665 | Unpaired try-on and qualitative analysis |

The preprocessing pipeline performs dataset auditing, quality filtering,
garment captioning, mask inspection, and Stable Diffusion input validation.

See [docs/dataset.md](docs/dataset.md) for additional details.

## Evaluation Protocol

The project uses two complementary evaluation modes.

### Paired Reconstruction Evaluation

For paired reconstruction, the target garment is forced to match the garment
already worn by the person:

```text
cloth_id = person_id
```

The original person image can therefore be used as pixel-aligned ground truth.
The following metrics are reported:

| Metric | Interpretation | Better Direction |
| --- | --- | --- |
| SSIM | Structural similarity and shape preservation | Higher |
| PSNR | Pixel-level reconstruction quality and noise | Higher |
| LPIPS | Deep perceptual similarity | Lower |

### Unpaired Holdout Evaluation

For unpaired evaluation, each person is combined with a different target
garment. There is no pixel-aligned ground-truth try-on image, so SSIM, PSNR,
and LPIPS against the original person would not represent garment-transfer
quality correctly.

Unpaired results are reviewed using:

```text
Person Input | Target Cloth | Try-On Result
```

Qualitative review focuses on:

- Garment identity and color preservation.
- Body shape and pose consistency.
- Face and hair preservation.
- Boundary artifacts and color bleeding.
- Missing logos, patterns, or garment details.
- Unrealistic folds and texture degradation.

Model 3 additionally reports CLIP garment similarity and outside-mask
preservation metrics.

See [docs/evaluation_protocol.md](docs/evaluation_protocol.md) for the complete
evaluation contract.

## Current Reported Results

| Model | Paired Samples | SSIM | PSNR | LPIPS | LPIPS Backbone | Holdout Generated |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Lightweight U-Net + GMM + TOM | 2,032 | 0.8932 | 21.39 dB | 0.1455 | VGG | 665 |
| Stable Diffusion Inpainting + LoRA | 996 | 0.8733 | 21.32 dB | 0.1055 | AlexNet | 665 |

These values document the completed Model 1 and Model 3 experiments, but they
must not yet be interpreted as a strict model ranking:

- Model 1 evaluated all 2,032 images found in `VITON-HD/test/image`.
- Model 3 evaluated the 996 samples in `clean_vto_dataset_test.csv`.
- Model 1 used LPIPS-VGG, while Model 3 used LPIPS-AlexNet.

The final three-model comparison should rerun all models using the same clean
paired manifest and the same LPIPS backbone.

The machine-readable current results are stored in
[results/metrics/reported_metrics.csv](results/metrics/reported_metrics.csv).

## Repository Structure

```text
VTO/
├── notebooks/
│   ├── 01_data_pipeline/
│   │   ├── 01_eda_and_cleaning.ipynb
│   │   ├── 02_blip_captioning.ipynb
│   │   └── 03_sd_preprocessing_validation.ipynb
│   ├── 02_models/
│   │   ├── model_1_lightweight_unet.ipynb
│   │   ├── model_2_pix2pix.ipynb
│   │   └── model_3_sd_lora.ipynb
│   ├── 03_evaluation/
│   │   └── model_3_sd_lora_evaluation.ipynb
│   └── 04_demo/
├── evaluation/
│   ├── model_1_lightweight_unet/
│   │   ├── runtime/
│   │   ├── paired_test/
│   │   └── unpaired_holdout/
│   ├── model_3_sd_lora/
│   │   ├── runtime/
│   │   ├── paired_test/
│   │   ├── unpaired_holdout/
│   │   └── review_publish/
│   └── common_comparison/
├── src/
│   ├── metrics.py
│   ├── reproducibility.py
│   └── visualization.py
├── results/
│   ├── eda/
│   ├── metrics/
│   ├── convergence/
│   ├── comparisons/
│   └── error_analysis/
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   ├── evaluation_protocol.md
│   ├── checkpoints.md
│   └── project_requirements.pdf
├── requirements.txt
├── pyproject.toml
└── README.md
```

Large checkpoints, local datasets, archived draft cells, and full-resolution
evaluation galleries are intentionally excluded from Git.

## Notebook Execution Order

### Data Preparation

Run the data notebooks in numerical order:

```text
notebooks/01_data_pipeline/01_eda_and_cleaning.ipynb
notebooks/01_data_pipeline/02_blip_captioning.ipynb
notebooks/01_data_pipeline/03_sd_preprocessing_validation.ipynb
```

Outputs include cleaned CSV manifests, EDA figures, garment captions, and
validated model inputs.

### Model Training

Each model notebook contains its own:

- PyTorch dataset and data loaders.
- Model definitions.
- Training and validation loops.
- Loss functions.
- Checkpoint loading and saving.
- Early stopping or learning-rate scheduling where applicable.
- Visual diagnostics and convergence charts.

Run the required model notebook:

```text
notebooks/02_models/model_1_lightweight_unet.ipynb
notebooks/02_models/model_2_pix2pix.ipynb
notebooks/02_models/model_3_sd_lora.ipynb
```

### Model Evaluation

Model-specific evaluation code lives under `evaluation/`.

Model 1 evaluation reuses the model objects loaded by its training notebook:

```python
from evaluation.model_1_lightweight_unet.workflow import run_stage

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

Model 3 has a complete evaluation notebook:

```text
notebooks/03_evaluation/model_3_sd_lora_evaluation.ipynb
```

The evaluation workflows support resumable inference, per-sample metric files,
JSON summaries, report-ready galleries, error analysis, and optional Kaggle
Dataset publishing.

## Common Result Comparison

The current Model 1 and Model 3 reported results can be summarized with:

```bash
python evaluation/common_comparison/01_compare_reported_metrics.py
```

This generates:

```text
results/metrics/reported_metrics_table.md
results/metrics/reported_metrics_comparison.png
```

The comparison script is ready to include Model 2 after its final evaluation
row is added to `results/metrics/reported_metrics.csv`.

## Kaggle Setup

### Required Kaggle Inputs

At minimum, attach:

```text
VITON-HD dataset
DLP cleaned CSV dataset
DLP garment caption dataset for Model 3
Required model checkpoint dataset
```

Model 3's final evaluated checkpoint is:

```text
/kaggle/input/datasets/khoaanh1234/ckpt-epoch-12-yen
```

Expected Model 3 checkpoint structure:

```text
checkpoint_latest/
├── unet_lora/
├── conv_in.pt
├── perceiver.pt
└── cloth_spatial.pt
```

### Important Kaggle Behavior

Files under `/kaggle/working` are temporary. Before ending a Kaggle session:

1. Verify the generated metric CSV and JSON summary.
2. Verify the comparison galleries.
3. Publish or download required artifacts.
4. Confirm the Kaggle Dataset link before stopping the session.

The evaluation publishing scripts create or version Kaggle Datasets without
using the unsupported `--private` CLI argument.

## Local Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency set includes:

```text
PyTorch and torchvision
Diffusers and Transformers
PEFT and Accelerate
LPIPS and scikit-image
OpenCV, NumPy, Pandas, and Matplotlib
Weights & Biases
```

Large-scale training and diffusion inference require a CUDA GPU. Local CPU
execution is primarily suitable for documentation, small utility scripts, and
result inspection.

## Reproducibility

The project applies fixed random seeds for Python, NumPy, and PyTorch.
Reusable helpers are available in
[src/reproducibility.py](src/reproducibility.py).

For reproducible experiments:

- Keep the same cleaned CSV manifests.
- Record the exact checkpoint and epoch.
- Record model hyperparameters and inference settings.
- Use the same metric implementation and LPIPS backbone.
- Preserve per-sample metric CSV files.
- Report both mean and standard deviation.
- Keep qualitative comparison pairs fixed across models.

## Checkpoints and External Artifacts

Large model weights are not committed to Git. They should be stored as Kaggle
Datasets and documented in [docs/checkpoints.md](docs/checkpoints.md).

Current published Model 3 paired result dataset:

```text
khoaanh1234/vto-sd-lora-epoch12-paired-three-column-gallery
```

Full-resolution result galleries should also remain external. Only compact,
report-ready figures should be committed under `results/`.

## Known Limitations

- The current Model 1 and Model 3 paired results use different evaluation
  manifests and LPIPS backbones.
- Model 2 does not yet have a finalized standalone evaluation workflow.
- Unpaired Virtual Try-On has no pixel-aligned ground truth; qualitative review
  remains necessary.
- Stable Diffusion inference is slower and more resource intensive than the
  lightweight CNN pipeline.
- Fine garment details, logos, hands, boundaries, and occlusions remain
  difficult failure cases.
- The final user-facing demo has not yet been integrated.

## Remaining Work

The main remaining tasks are:

1. Complete the standalone Model 2 evaluation workflow.
2. Rerun all three models using one shared paired-test manifest and LPIPS
   backbone.
3. Produce a final three-model quantitative comparison.
4. Export fixed-pair qualitative comparison galleries.
5. Complete cross-model error analysis.
6. Integrate the final Gradio demo.
7. Add final checkpoint links and the completed project report.

## Documentation

- [Dataset and splits](docs/dataset.md)
- [Architecture summary](docs/architecture.md)
- [Evaluation protocol](docs/evaluation_protocol.md)
- [Checkpoints and external artifacts](docs/checkpoints.md)
- [Model 1 evaluation](evaluation/model_1_lightweight_unet/README.md)
- [Model 3 evaluation](evaluation/model_3_sd_lora/README.md)
- [Current comparison workflow](evaluation/common_comparison/README.md)

## Academic Context

This repository was developed as a final project for the Deep Learning course,
academic year 2026. It is intended for educational and experimental use.
