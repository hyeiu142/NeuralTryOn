"""Generate a report-ready overview of all currently completed model results."""

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
INPUT_PATH = METRICS_DIR / "reported_metrics.csv"
TABLE_PATH = METRICS_DIR / "reported_metrics_table.md"
CHART_PATH = METRICS_DIR / "reported_metrics_comparison.png"

with INPUT_PATH.open(encoding="utf-8", newline="") as file:
    completed = [
        row for row in csv.DictReader(file)
        if row["status"] in {"completed", "paired_completed"}
    ]
if not completed:
    raise RuntimeError(f"No completed model results found in {INPUT_PATH}")

warning = (
    "> Note: this is a reported-results overview, not a strict ranking. "
    "The completed models currently use different paired manifests and "
    "LPIPS backbones.\n\n"
)
headers = [
    "Model", "Paired samples", "SSIM", "PSNR (dB)", "LPIPS",
    "LPIPS backbone", "Holdout samples",
]
table_rows = [
    [
        row["display_name"],
        row["paired_samples"],
        f'{float(row["ssim_mean"]):.4f}',
        f'{float(row["psnr_mean"]):.2f}',
        f'{float(row["lpips_mean"]):.4f}',
        row["lpips_backbone"],
        row["unpaired_samples"],
    ]
    for row in completed
]
markdown_lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
    *["| " + " | ".join(row) + " |" for row in table_rows],
]
TABLE_PATH.write_text(warning + "\n".join(markdown_lines) + "\n", encoding="utf-8")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    specs = [
        ("ssim_mean", "SSIM (higher is better)", "#4C78A8"),
        ("psnr_mean", "PSNR dB (higher is better)", "#59A14F"),
        ("lpips_mean", "LPIPS (lower is better)", "#E15759"),
    ]
    labels = [row["display_name"].replace(" + ", "\n+ ") for row in completed]
    for axis, (column, title, color) in zip(axes, specs):
        values = [float(row[column]) for row in completed]
        bars = axis.bar(labels, values, color=color)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelrotation=8)
        for bar, value in zip(bars, values):
            label = f"{value:.4f}" if column != "psnr_mean" else f"{value:.2f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle(
        "Reported Results Overview: protocols differ, not a strict ranking",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart : {CHART_PATH}")
except ImportError:
    print("Chart skipped: matplotlib is not installed in this environment.")

for row in table_rows:
    print(" | ".join(row))
print(f"Table : {TABLE_PATH}")
