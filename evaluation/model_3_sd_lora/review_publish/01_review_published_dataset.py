"""Browse pages of a published unpaired holdout result dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


DATASET_SLUG = "vto-v2-epoch12-unpaired-full-eval-20260612"
DATASET_ROOT = Path(f"/kaggle/input/{DATASET_SLUG}")
PAGE = 1
IMAGES_PER_PAGE = 8
SORT_BY = "sample_index"
ASCENDING = True


def _find_dataset_root():
    candidates = [
        DATASET_ROOT,
        Path(f"/kaggle/input/datasets/khoaanh1234/{DATASET_SLUG}"),
    ]
    for candidate in candidates:
        print(f"Checking: {candidate}")
        if (candidate / "unpaired_metrics.csv").exists():
            return candidate

    # Only inspect direct Kaggle Input children; never recursively scan VITON-HD.
    for candidate in Path("/kaggle/input").glob("*"):
        if candidate.is_dir() and (candidate / "unpaired_metrics.csv").exists():
            return candidate
    return None


print("Locating private unpaired dataset...")
dataset_root = _find_dataset_root()
if dataset_root is None:
    available = [path.name for path in Path("/kaggle/input").glob("*") if path.is_dir()]
    raise FileNotFoundError(
        f"Khong tim thay dataset {DATASET_SLUG}. Hay Add Input dataset vao notebook.\n"
        f"Input datasets hien co: {available}"
    )

print(f"Loading metrics from: {dataset_root}")
metrics_df = pd.read_csv(dataset_root / "unpaired_metrics.csv")
metrics_df = metrics_df[metrics_df["status"] == "ok"].copy()

if "quality_score" not in metrics_df.columns:
    clip_std = max(float(metrics_df["clip_garment_similarity"].std()), 1e-12)
    outside_std = max(float(metrics_df["outside_mae"].std()), 1e-12)
    metrics_df["quality_score"] = (
        (metrics_df["clip_garment_similarity"] - metrics_df["clip_garment_similarity"].mean()) / clip_std
        - (metrics_df["outside_mae"] - metrics_df["outside_mae"].mean()) / outside_std
    )

valid_sort_columns = {
    "sample_index",
    "quality_score",
    "clip_garment_similarity",
    "outside_mae",
    "mask_area",
}
if SORT_BY not in valid_sort_columns:
    raise ValueError(f"SORT_BY phai la mot trong: {sorted(valid_sort_columns)}")

metrics_df = metrics_df.sort_values(SORT_BY, ascending=ASCENDING).reset_index(drop=True)
total_pages = max(1, int(np.ceil(len(metrics_df) / IMAGES_PER_PAGE)))
page = max(1, min(int(PAGE), total_pages))
start = (page - 1) * IMAGES_PER_PAGE
page_df = metrics_df.iloc[start:start + IMAGES_PER_PAGE]
print(f"Preparing page {page}/{total_pages}: {len(page_df)} images")


def _find_comparison(row):
    old_path = Path(str(getattr(row, "comparison_path", "")))
    filename = old_path.name
    stem = f"{int(row.sample_index):04d}_{row.person_id}__{row.cloth_id}"
    filenames = [filename, f"{stem}_comparison.jpg"]
    folders = [dataset_root / "comparisons", dataset_root / "images"]

    for folder in folders:
        for name in filenames:
            if name:
                candidate = folder / name
                if candidate.exists():
                    return candidate
    return None


columns = 2
rows = int(np.ceil(len(page_df) / columns))
fig, axes = plt.subplots(rows, columns, figsize=(20, 5.8 * rows), squeeze=False)

for axis in axes.flat:
    axis.axis("off")

for axis, row in zip(axes.flat, page_df.itertuples(index=False)):
    comparison_path = _find_comparison(row)
    if comparison_path is not None:
        with Image.open(comparison_path) as image:
            axis.imshow(image.convert("RGB").copy())
    else:
        axis.text(0.5, 0.5, "Comparison image not found", ha="center", va="center")
    axis.set_title(
        f"{int(row.sample_index):03d} | person={row.person_id} | cloth={row.cloth_id}\n"
        f"score={row.quality_score:.2f} | CLIP={row.clip_garment_similarity:.3f}",
        fontsize=9,
    )
    axis.axis("off")

fig.suptitle(
    f"Unpaired Holdout Results | page {page}/{total_pages} | "
    f"sort={SORT_BY} {'asc' if ASCENDING else 'desc'}",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

print(f"Dataset root: {dataset_root}")
print(f"Showing samples {start + 1}-{start + len(page_df)} of {len(metrics_df)}")
print("Doi PAGE de xem trang khac.")
print("SORT_BY='quality_score', ASCENDING=False de xem anh duoc xep hang tot nhat.")
