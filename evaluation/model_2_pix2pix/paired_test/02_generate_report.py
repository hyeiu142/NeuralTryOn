"""Generate paired-test distributions and best/worst error-analysis galleries."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw


REPORT_DIR = PAIRED_DIR / "metric_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
metrics_df = pd.read_csv(PAIRED_DIR / "paired_test_metrics.csv")
metrics_df = metrics_df[metrics_df["status"] == "ok"].copy()
if metrics_df.empty:
    raise RuntimeError("Khong co paired-test result thanh cong.")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for axis, (column, title, color) in zip(
    axes,
    [
        ("ssim", "SSIM Distribution", "#4C78A8"),
        ("psnr", "PSNR Distribution", "#59A14F"),
        ("lpips", "LPIPS Distribution", "#E15759"),
    ],
):
    axis.hist(metrics_df[column], bins=30, color=color, edgecolor="white")
    axis.axvline(metrics_df[column].mean(), color="black", linestyle="--")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
fig.tight_layout()
distribution_path = REPORT_DIR / "ssim_psnr_lpips_distributions.png"
fig.savefig(distribution_path, dpi=180, bbox_inches="tight")
plt.show()
plt.close(fig)

ranked = metrics_df.copy()
ranked["quality_rank"] = (
    ranked["ssim"].rank(pct=True)
    + ranked["psnr"].rank(pct=True)
    + ranked["lpips"].rank(pct=True, ascending=False)
)
ranked = ranked.sort_values("quality_rank", ascending=False)
ranked.to_csv(REPORT_DIR / "paired_metrics_ranked.csv", index=False)


def _comparison(row):
    person_id = str(row.person_id)
    person = Image.open(TEST_DIR / "image" / f"{person_id}.jpg").convert("RGB").resize((WIDTH, HEIGHT))
    cloth = Image.open(TEST_DIR / "cloth" / f"{person_id}.jpg").convert("RGB").resize((WIDTH, HEIGHT))
    result = Image.open(Path(row.result_path)).convert("RGB").resize((WIDTH, HEIGHT))
    header = 42
    gap = 8
    canvas = Image.new("RGB", (3 * WIDTH + 2 * gap, HEIGHT + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(
        [("Person / Ground Truth", person), ("Target Cloth", cloth), ("Try-On Result", result)]
    ):
        x = index * (WIDTH + gap)
        canvas.paste(image, (x, header))
        draw.text((x + 6, 8), title, fill="black")
    return canvas


def _gallery(rows, title, filename):
    comparisons = [_comparison(row) for row in rows.itertuples(index=False)]
    gallery = Image.new("RGB", (comparisons[0].width, len(comparisons) * comparisons[0].height), "white")
    for index, image in enumerate(comparisons):
        gallery.paste(image, (0, index * image.height))
    path = REPORT_DIR / filename
    gallery.save(path, quality=92)
    print(f"{title}: {path}")


top_n = min(10, len(ranked))
_gallery(ranked.head(top_n), "Best samples", "best_paired_samples.jpg")
_gallery(ranked.tail(top_n).iloc[::-1], "Worst samples", "worst_paired_samples.jpg")

report = {
    "samples": int(len(ranked)),
    "ranking": "percentile rank of SSIM + PSNR + inverse LPIPS",
    "best_ids": ranked.head(top_n)["person_id"].astype(str).tolist(),
    "worst_ids": ranked.tail(top_n)["person_id"].astype(str).tolist(),
}
(REPORT_DIR / "error_analysis_summary.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(f"Report ready: {REPORT_DIR}")
