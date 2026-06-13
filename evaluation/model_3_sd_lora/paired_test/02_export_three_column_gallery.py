"""Export paired-test Person | Target Cloth | Try-On Result comparisons."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm


PAIRED_TEST_DIR = OUTPUT_DIR / "paired_test_metrics"
RESULT_DIR = PAIRED_TEST_DIR / "images"
COMPARISON_DIR = PAIRED_TEST_DIR / "comparisons_3_columns"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

manifest_path = PAIRED_TEST_DIR / "paired_test_manifest.csv"
metrics_path = PAIRED_TEST_DIR / "paired_test_metrics.csv"
if not manifest_path.exists() or not metrics_path.exists():
    raise FileNotFoundError(
        "Thieu ket qua Cell 15. Can co paired_test_manifest.csv va paired_test_metrics.csv."
    )

manifest = pd.read_csv(manifest_path)
metrics = pd.read_csv(metrics_path)
ok_ids = set(
    metrics.loc[metrics["status"] == "ok", "person_id"].astype(str)
)

HEADER_H = 42
GAP = 8
FONT = ImageFont.load_default()
test_dir = VITON_ROOT / "test"


def _to_u8(image):
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_person(person_id):
    image = Image.open(test_dir / "image" / f"{person_id}.jpg").convert("RGB")
    return _crop_resize(image)


def _load_target_cloth(cloth_id):
    cloth = Image.open(test_dir / "cloth" / f"{cloth_id}.jpg").convert("RGB")
    cloth_mask = Image.open(test_dir / "cloth-mask" / f"{cloth_id}.jpg").convert("L")
    return _process_cloth(cloth, cloth_mask)


def _make_comparison(person, target_cloth, result, person_id):
    panels = [
        ("Person Input", person),
        ("Target Cloth", target_cloth),
        ("Try-On Result", result),
    ]
    canvas = Image.new(
        "RGB",
        (3 * W + 2 * GAP, H + HEADER_H),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for index, (title, panel) in enumerate(panels):
        x = index * (W + GAP)
        canvas.paste(panel.resize((W, H), Image.BILINEAR), (x, HEADER_H))
        draw.text((x + 8, 8), title, fill="black", font=FONT)

    draw.text(
        (8, HEADER_H - 15),
        f"person_id={person_id} | cloth_id={person_id}",
        fill="black",
        font=FONT,
    )
    return canvas


saved = 0
missing = []
for row in tqdm(
    manifest.itertuples(index=False),
    total=len(manifest),
    desc="Exporting 3-column comparisons",
):
    person_id = str(row.person_id)
    if person_id not in ok_ids:
        continue

    result_path = RESULT_DIR / f"{person_id}_result.jpg"
    if not result_path.exists():
        missing.append(person_id)
        continue

    person = _load_person(person_id)
    target_cloth = _load_target_cloth(person_id)
    result = Image.open(result_path).convert("RGB")
    comparison = _make_comparison(person, target_cloth, result, person_id)
    comparison.save(
        COMPARISON_DIR / f"{person_id}_comparison.jpg",
        quality=95,
    )
    saved += 1

print(f"Saved comparisons: {saved}")
print(f"Missing results   : {len(missing)}")
print(f"Output directory  : {COMPARISON_DIR}")
