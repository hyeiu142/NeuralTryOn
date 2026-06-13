"""Resolve Kaggle inputs and Model 2 evaluation output directories."""

from pathlib import Path

import torch


MODEL_NAME = "model_2_pix2pix"
CHECKPOINT_NAME = "gmm_shape_generation_pix2pix_best"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIDTH, HEIGHT = 384, 512


def _find_root_with(relative_path, candidates):
    for candidate in candidates:
        root = Path(candidate)
        if (root / relative_path).exists():
            return root
    for match in Path("/kaggle/input").rglob(Path(relative_path).name):
        if str(match).endswith(str(relative_path)):
            root = match
            for _ in Path(relative_path).parts:
                root = root.parent
            return root
    return None


VITON_ROOT = _find_root_with(
    "test/image",
    [
        "/kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset",
        "/kaggle/input/high-resolution-viton-zalando-dataset",
    ],
)
CSV_ROOT = _find_root_with(
    "holdout_test.csv",
    [
        "/kaggle/input/datasets/cthnhoddt/dlp-cleandatacsv",
        "/kaggle/input/dlp-cleandatacsv",
        "/home/yennguyen/VTO/data/DLP_CleanDataCSV",
    ],
)
if VITON_ROOT is None or CSV_ROOT is None:
    raise FileNotFoundError("Khong tim thay VITON-HD hoac DLP_CleanDataCSV.")

TEST_DIR = VITON_ROOT / "test"
OUTPUT_DIR = Path("/kaggle/working/model_2_pix2pix_evaluation")
PAIRED_DIR = OUTPUT_DIR / "paired_test"
UNPAIRED_DIR = OUTPUT_DIR / "unpaired_holdout"
for directory in [OUTPUT_DIR, PAIRED_DIR, UNPAIRED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"Device   : {DEVICE}")
print(f"VITON-HD : {VITON_ROOT}")
print(f"CSV root : {CSV_ROOT}")
print(f"Output   : {OUTPUT_DIR}")
