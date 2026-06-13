"""Create a deterministic qualitative shortlist from holdout results."""

import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image


REPORT_DIR = UNPAIRED_DIR / "qualitative_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
results_df = pd.read_csv(UNPAIRED_DIR / "unpaired_results.csv")
results_df = results_df[results_df["status"] == "ok"].copy()
if results_df.empty:
    raise RuntimeError("Khong co holdout result thanh cong.")

random.seed(42)
sample_indices = random.sample(range(len(results_df)), min(10, len(results_df)))
sample_df = results_df.iloc[sample_indices].copy()
sample_df.to_csv(REPORT_DIR / "random_shortlist.csv", index=False)

gallery_dir = UNPAIRED_DIR / "comparisons_3_columns"
images = []
for row in sample_df.itertuples(index=False):
    path = gallery_dir / f"{int(row.sample_index):04d}_{row.person_id}_to_{row.cloth_id}.jpg"
    if path.exists():
        images.append(Image.open(path).convert("RGB"))

if images:
    gallery = Image.new("RGB", (images[0].width, len(images) * images[0].height), "white")
    for index, image in enumerate(images):
        gallery.paste(image, (0, index * image.height))
    gallery.save(REPORT_DIR / "random_10_holdout_samples.jpg", quality=92)

summary = {
    "samples_generated": int(len(results_df)),
    "qualitative_shortlist_size": int(len(sample_df)),
    "note": "Unpaired holdout has no pixel-aligned ground truth; inspect garment fidelity, body preservation, and artifacts manually.",
}
(REPORT_DIR / "qualitative_report_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(f"Qualitative report ready: {REPORT_DIR}")
