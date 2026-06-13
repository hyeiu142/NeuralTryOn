"""Export Person Input | Target Cloth | Try-On Result holdout comparisons."""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from tqdm.auto import tqdm


GALLERY_DIR = UNPAIRED_DIR / "comparisons_3_columns"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)
results_df = pd.read_csv(UNPAIRED_DIR / "unpaired_results.csv")
results_df = results_df[results_df["status"] == "ok"]

for row in tqdm(results_df.itertuples(index=False), total=len(results_df), desc="Holdout gallery"):
    panels = [
        ("Person Input", Image.open(TEST_DIR / "image" / f"{row.person_id}.jpg").convert("RGB")),
        ("Target Cloth", Image.open(TEST_DIR / "cloth" / f"{row.cloth_id}.jpg").convert("RGB")),
        ("Try-On Result", Image.open(Path(row.result_path)).convert("RGB")),
    ]
    header, gap = 42, 8
    canvas = Image.new("RGB", (3 * WIDTH + 2 * gap, HEIGHT + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        x = index * (WIDTH + gap)
        canvas.paste(image.resize((WIDTH, HEIGHT)), (x, header))
        draw.text((x + 6, 8), title, fill="black")
    filename = f"{int(row.sample_index):04d}_{row.person_id}_to_{row.cloth_id}.jpg"
    canvas.save(GALLERY_DIR / filename, quality=92)

print(f"Saved {len(results_df)} holdout comparisons: {GALLERY_DIR}")
