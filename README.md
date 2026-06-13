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
| Model 2 | GMM + Shape Generation + Pix2Pix + PatchGAN | Adversarial image synthesis experiment | Training and paired evaluation completed; full holdout pending |
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

The architecture and completed experiment summaries are documented in
[docs/architecture.md](docs/architecture.md) and
[docs/experiments.md](docs/experiments.md).

## Common 996-Sample Evaluation Results

| Model | Paired Samples | SSIM | PSNR | LPIPS | Holdout Generated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lightweight U-Net + GMM + TOM | 996 | 0.8932 | 21.39 dB | 0.1455 | 665 |
| GMM + Shape Generation + Pix2Pix | 996 | 0.8951 | 21.49 dB | 0.1145 | 13 manual demos |
| Stable Diffusion Inpainting + LoRA | 996 | 0.8733 | 21.32 dB | 0.1055 | 665 |

The report defines `clean_vto_dataset_test.csv` as the common 996-sample paired
test set for all models. Model 1 and Model 2 metrics shown above are retained
from their completed legacy runs and should be confirmed on this common
manifest before interpreting small metric differences as a strict ranking.
LPIPS backbone differences are reported explicitly.

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
│   ├── model_2_pix2pix/
│   │   ├── runtime/
│   │   ├── paired_test/
│   │   └── unpaired_holdout/
│   ├── model_3_sd_lora/
│   │   ├── runtime/
│   │   ├── paired_test/
│   │   ├── unpaired_holdout/
│   │   └── review_publish/
│   └── common_comparison/
├── configs/
│   ├── data/
│   ├── models/
│   ├── experiments/
│   └── tracking/
├── experiments/
│   ├── hyperparameter_summary.csv
│   ├── registry.csv
│   └── runs/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── vto.py
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── config.py
│   ├── metrics.py
│   ├── reproducibility.py
│   ├── tracking.py
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
│   ├── experiments.md
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

The currently reported results for all three models can be summarized with:

```bash
python evaluation/common_comparison/01_compare_reported_metrics.py
```

This generates:

```text
results/metrics/reported_metrics_table.md
results/metrics/reported_metrics_comparison.png
```

The generated overview keeps the evaluation protocol and LPIPS-backbone
limitations visible so that the reported metrics are not mistaken for a strict
ranking.

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

[Model 3 Best Checkpoint](https://www.kaggle.com/datasets/hypivepiu/model-3-bestbest?select=checkpoint_latest)

```text
/kaggle/input/datasets/hypivepiu/model-3-bestbest
```

The completed full unpaired holdout results are available at:

[Epoch 12 Full Unpaired Holdout Evaluation](https://www.kaggle.com/datasets/khoaanh1234/vto-v2-epoch12-unpaired-full-eval-20260612)

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

## Reproducibility and Artifacts

Experiments use fixed random seeds and documented dataset manifests. Large
checkpoints and full-resolution galleries are stored as external Kaggle
Datasets. See [docs/checkpoints.md](docs/checkpoints.md) for available artifacts.

Hierarchical configurations are recorded in [configs/](configs/). The
run-oriented tracking structure, run registry, and cross-model hyperparameter
summary are documented in [experiments/](experiments/).

The production-oriented software architecture and remote GPU workflow are
documented in [docs/production_architecture.md](docs/production_architecture.md)
and [docs/kaggle_operations.md](docs/kaggle_operations.md).

Configuration validation and unit tests run automatically through
`.github/workflows/ci.yml`. Common local checks are available through
`make validate`, `make compile`, and `make test`.

## Production CLI

The installed `vto` command provides configuration validation, environment
preflight, run initialization, model inspection, and artifact packaging:

```bash
vto models
vto validate --config configs/experiments/model_3_default.yaml
vto preflight --config configs/experiments/model_3_default.yaml
vto init-run --config configs/experiments/model_3_default.yaml
vto package-run --run-dir experiments/runs/<run_id>
```

The script-based train, evaluate, and infer entry points support guarded dry
runs while the completed model implementations remain preserved in notebooks:

```bash
python scripts/train.py --config configs/experiments/model_3_default.yaml --dry-run
python scripts/evaluate.py --config configs/experiments/model_3_default.yaml --dry-run
python scripts/infer.py --config configs/experiments/model_3_default.yaml --dry-run
```

Model 3 also provides a real script-based training adapter. Start with the
small Kaggle GPU smoke test documented in
[docs/model_3_production_training.md](docs/model_3_production_training.md).

Evaluation protocols and known limitations are documented in
[docs/evaluation_protocol.md](docs/evaluation_protocol.md).
