"""Export a three-column comparison for every successful paired-test result."""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from tqdm.auto import tqdm


GALLERY_DIR = PAIRED_DIR / "comparisons_3_columns"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)
metrics_df = pd.read_csv(PAIRED_DIR / "paired_test_metrics.csv")
metrics_df = metrics_df[metrics_df["status"] == "ok"]

for row in tqdm(metrics_df.itertuples(index=False), total=len(metrics_df), desc="Paired gallery"):
    person_id = str(row.person_id)
    panels = [
        ("Person / Ground Truth", Image.open(TEST_DIR / "image" / f"{person_id}.jpg").convert("RGB")),
        ("Target Cloth", Image.open(TEST_DIR / "cloth" / f"{person_id}.jpg").convert("RGB")),
        ("Try-On Result", Image.open(Path(row.result_path)).convert("RGB")),
    ]
    header, gap = 42, 8
    canvas = Image.new("RGB", (3 * WIDTH + 2 * gap, HEIGHT + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        x = index * (WIDTH + gap)
        canvas.paste(image.resize((WIDTH, HEIGHT)), (x, header))
        draw.text((x + 6, 8), title, fill="black")
    canvas.save(GALLERY_DIR / f"{person_id}_comparison.jpg", quality=92)

print(f"Saved {len(metrics_df)} paired comparisons: {GALLERY_DIR}")
