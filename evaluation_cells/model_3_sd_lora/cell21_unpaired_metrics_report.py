import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image


TOP_N = 10
REPORT_DIR = UNPAIRED_DIR / "metric_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

metrics_path = UNPAIRED_DIR / "unpaired_metrics.csv"
if not metrics_path.exists():
    raise FileNotFoundError(f"Chua co ket qua Cell 20: {metrics_path}")

metrics_df = pd.read_csv(metrics_path)
metrics_df = metrics_df[metrics_df["status"] == "ok"].copy()
if len(metrics_df) == 0:
    raise RuntimeError("Khong co unpaired sample thanh cong de lap bao cao.")

# Move comparison files produced by the older Cell 20 into images/.
old_comparison_dir = UNPAIRED_DIR / "comparisons"
if old_comparison_dir.exists():
    for source in old_comparison_dir.glob("*_comparison.jpg"):
        destination = UNPAIRED_IMAGE_DIR / source.name
        if not destination.exists():
            shutil.copy2(source, destination)


def _zscore(series):
    std = float(series.std())
    if std <= 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


metrics_df["quality_score"] = (
    _zscore(metrics_df["clip_garment_similarity"])
    - _zscore(metrics_df["outside_mae"])
)
metrics_df = metrics_df.sort_values("quality_score", ascending=False)
metrics_df.to_csv(REPORT_DIR / "unpaired_metrics_ranked.csv", index=False)

metric_specs = [
    ("clip_garment_similarity", "CLIP Garment Similarity", True),
    ("outside_mae", "Outside-mask MAE", False),
    ("raw_outside_mae", "Raw Outside-mask MAE", False),
    ("mask_area", "Mask Area", None),
    ("seconds", "Inference Time (seconds)", False),
]

summary_rows = []
for column, label, higher_better in metric_specs:
    values = metrics_df[column].dropna()
    summary_rows.append(
        {
            "metric": column,
            "label": label,
            "better": (
                "higher" if higher_better is True
                else "lower" if higher_better is False
                else "analysis only"
            ),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(REPORT_DIR / "metric_summary.csv", index=False)
display(summary_df)

for column, label, _ in metric_specs:
    values = metrics_df[column].dropna()
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=24, color="#4C78A8", edgecolor="white", alpha=0.9)
    axis.axvline(values.mean(), color="#D62728", linestyle="--", label=f"Mean={values.mean():.4f}")
    axis.axvline(values.median(), color="#16A34A", linestyle=":", label=f"Median={values.median():.4f}")
    axis.set_title(f"{label} Distribution")
    axis.set_xlabel(label)
    axis.set_ylabel("Number of samples")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = REPORT_DIR / f"{column}_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _comparison_path(row):
    original = Path(str(row.comparison_path))
    migrated = UNPAIRED_IMAGE_DIR / original.name
    return migrated if migrated.exists() else original


def _save_gallery(rows, title, filename):
    valid_rows = [row for row in rows.itertuples(index=False) if _comparison_path(row).exists()]
    fig, axes = plt.subplots(len(valid_rows), 1, figsize=(18, 3.3 * len(valid_rows)), squeeze=False)
    for index, row in enumerate(valid_rows):
        axes[index, 0].imshow(Image.open(_comparison_path(row)).convert("RGB"))
        axes[index, 0].axis("off")
        axes[index, 0].set_title(
            f"person={row.person_id} | cloth={row.cloth_id} | "
            f"score={row.quality_score:.3f} | CLIP={row.clip_garment_similarity:.3f} | "
            f"outside MAE={row.outside_mae:.6f}",
            fontsize=10,
        )
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = REPORT_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


best_rows = metrics_df.head(TOP_N)
worst_rows = metrics_df.tail(TOP_N).sort_values("quality_score")
_save_gallery(best_rows, f"{CHECKPOINT_NAME}: Best Unpaired Samples", "best_unpaired_samples.png")
_save_gallery(worst_rows, f"{CHECKPOINT_NAME}: Worst Unpaired Samples", "worst_unpaired_samples.png")

report_summary = {
    "checkpoint": CHECKPOINT_NAME,
    "samples_evaluated": int(len(metrics_df)),
    "ranking_note": "Automatic rank uses CLIP garment similarity and outside-mask MAE.",
    "best_pairs": best_rows[["person_id", "cloth_id"]].astype(str).to_dict("records"),
    "worst_pairs": worst_rows[["person_id", "cloth_id"]].astype(str).to_dict("records"),
    "metric_summary": summary_rows,
}
with open(REPORT_DIR / "metric_report_summary.json", "w", encoding="utf-8") as file:
    json.dump(report_summary, file, indent=2)

print(f"Unpaired metric report ready: {REPORT_DIR}")
