"""Review a manually selected shortlist of unpaired try-on results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


DATASET_SLUG = "vto-v2-epoch12-unpaired-full-eval-20260612"
SHORTLIST_PER_PAGE = 5
PAGE = 1

SHORTLIST = """
10358 12419
05883 05115s
01008 12704
03408 10901
01229 08759
14350 12813
11556 02636
14471 10567
10410 05327
10549 01260
"""



def _normalize_id(raw_id):
    raw_id = str(raw_id).strip()
    return raw_id if raw_id.endswith("_00") else f"{raw_id}_00"


def _find_dataset_root():
    candidates = [
        Path(f"/kaggle/input/{DATASET_SLUG}"),
        Path(f"/kaggle/input/datasets/khoaanh1234/{DATASET_SLUG}"),
    ]
    for candidate in candidates:
        if (candidate / "unpaired_metrics.csv").exists():
            return candidate

    for candidate in Path("/kaggle/input").glob("*"):
        if candidate.is_dir() and (candidate / "unpaired_metrics.csv").exists():
            return candidate
    return None


dataset_root = _find_dataset_root()
if dataset_root is None:
    raise FileNotFoundError(
        f"Khong tim thay dataset {DATASET_SLUG}. Hay Add Input dataset vao notebook."
    )

metrics_df = pd.read_csv(dataset_root / "unpaired_metrics.csv")
metrics_df["person_id"] = metrics_df["person_id"].astype(str)
metrics_df["cloth_id"] = metrics_df["cloth_id"].astype(str)

pairs = []
for line in SHORTLIST.strip().splitlines():
    parts = line.split()
    if len(parts) != 2:
        continue
    pairs.append((_normalize_id(parts[0]), _normalize_id(parts[1])))


def _find_comparison(row):
    old_path = Path(str(row["comparison_path"]))
    filenames = [
        old_path.name,
        f"{int(row['sample_index']):04d}_{row['person_id']}__{row['cloth_id']}_comparison.jpg",
    ]
    for folder in [dataset_root / "comparisons", dataset_root / "images"]:
        for filename in filenames:
            candidate = folder / filename
            if candidate.exists():
                return candidate
    return None


selected = []
missing = []
for rank, (person_id, cloth_id) in enumerate(pairs, start=1):
    match = metrics_df[
        (metrics_df["person_id"] == person_id)
        & (metrics_df["cloth_id"] == cloth_id)
    ]
    if len(match) == 0:
        missing.append((person_id, cloth_id))
        continue

    row = match.iloc[0].to_dict()
    row["shortlist_rank"] = rank
    row["comparison_file"] = _find_comparison(row)
    if row["comparison_file"] is None:
        missing.append((person_id, cloth_id))
        continue
    selected.append(row)

selected_df = pd.DataFrame(selected)
selected_csv = Path("/kaggle/working/manual_unpaired_shortlist.csv")
selected_df.drop(columns=["comparison_file"], errors="ignore").to_csv(selected_csv, index=False)

total_pages = max(1, (len(selected) + SHORTLIST_PER_PAGE - 1) // SHORTLIST_PER_PAGE)
page = max(1, min(PAGE, total_pages))
start = (page - 1) * SHORTLIST_PER_PAGE
page_rows = selected[start:start + SHORTLIST_PER_PAGE]

fig, axes = plt.subplots(
    len(page_rows),
    1,
    figsize=(20, 5.6 * len(page_rows)),
    squeeze=False,
)

for axis, row in zip(axes.flat, page_rows):
    with Image.open(row["comparison_file"]) as image:
        axis.imshow(image.convert("RGB").copy())
    axis.axis("off")
    axis.set_title(
        f'Candidate {int(row["shortlist_rank"]):02d} | '
        f'person={row["person_id"]} | cloth={row["cloth_id"]} | '
        f'CLIP={row["clip_garment_similarity"]:.3f}',
        fontsize=11,
        fontweight="bold",
    )

fig.suptitle(
    f"Manual Unpaired Shortlist | page {page}/{total_pages}",
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

print(f"Found: {len(selected)}/{len(pairs)} pairs")
print(f"Saved shortlist CSV: {selected_csv}")
if missing:
    print("Missing:")
    for person_id, cloth_id in missing:
        print(f"  person={person_id}, cloth={cloth_id}")
print("Doi PAGE de xem trang tiep theo.")
