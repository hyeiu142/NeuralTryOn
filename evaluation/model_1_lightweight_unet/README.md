# Model 1 Evaluation: Lightweight U-Net + GMM + TOM

This directory organizes the complete evaluation workflow extracted from the
updated Model 1 notebook. The notebook remains unchanged.

## Model Pipeline

```text
Person + Agnostic + Pose + Cloth + DensePose
    -> Lightweight U-Net clothing-region mask
    -> Lightweight GMM / TPS cloth warping
    -> TOM rendering and composition
    -> Final try-on result
```

## Directory Structure

```text
model_1_lightweight_unet/
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

## Execution

The evaluation scripts are notebook-oriented and share one namespace. After
the Model 1 notebook has created or loaded `model_unet`, `model_gmm`, and the
TOM architecture, run:

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

Publishing stages are optional and should run only after verifying all output
files.

## Completed Notebook Result

The updated notebook reports:

```text
Paired test samples   2032
SSIM                  0.8932
PSNR                  21.39 dB
LPIPS-VGG             0.1455
Unpaired holdout      665/665 generated
```

The paired evaluator scans all images in `VITON-HD/test/image`, matching the
protocol that produced these reported values. The unpaired workflow preserves
the notebook's shifted-cloth pairing logic.
