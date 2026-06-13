# ================================================================
# CELL 16B: PUBLISH PAIRED TEST 3-COLUMN COMPARISONS TO KAGGLE
# Run after Cell 16.
# ================================================================

import json
import shutil
import subprocess
from pathlib import Path


KAGGLE_DATASET_OWNER = "khoaanh1234"
KAGGLE_DATASET_SLUG = "vto-sd-lora-epoch12-paired-three-column-gallery"
KAGGLE_DATASET_ID = f"{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_SLUG}"

paired_test_dir = OUTPUT_DIR / "paired_test_metrics"
comparison_dir = paired_test_dir / "comparisons_3_columns"
publish_dir = Path(f"/kaggle/working/{KAGGLE_DATASET_SLUG}")

required_paths = [
    comparison_dir,
    paired_test_dir / "paired_test_manifest.csv",
    paired_test_dir / "paired_test_metrics.csv",
    paired_test_dir / "paired_test_summary.json",
    paired_test_dir / "ssim_psnr_lpips_distributions.png",
]
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Thieu ket qua Cell 15/16:\n  " + "\n  ".join(missing_paths)
    )

comparison_count = len(list(comparison_dir.glob("*_comparison.jpg")))
if comparison_count == 0:
    raise RuntimeError(
        f"Khong co anh 3 cot trong {comparison_dir}. Hay chay Cell 16 truoc."
    )

if publish_dir.exists():
    shutil.rmtree(publish_dir)
publish_dir.mkdir(parents=True)

shutil.copytree(comparison_dir, publish_dir / "comparisons_3_columns")
for filename in [
    "paired_test_manifest.csv",
    "paired_test_metrics.csv",
    "paired_test_summary.json",
    "ssim_psnr_lpips_distributions.png",
]:
    shutil.copy2(paired_test_dir / filename, publish_dir / filename)

metadata = {
    "title": "VTO SD LoRA Epoch 12 Paired Three Column Gallery",
    "id": KAGGLE_DATASET_ID,
    "licenses": [{"name": "CC0-1.0"}],
}
with open(publish_dir / "dataset-metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print(f"Prepared images : {comparison_count}")
print(f"Publish folder  : {publish_dir}")
print(f"Dataset ID      : {KAGGLE_DATASET_ID}")

create_command = [
    "kaggle",
    "datasets",
    "create",
    "-p",
    str(publish_dir),
    "-r",
    "zip",
]
create_result = subprocess.run(create_command, text=True, capture_output=True)
combined_output = f"{create_result.stdout}\n{create_result.stderr}"
print(combined_output)

if create_result.returncode != 0:
    already_exists = any(
        phrase in combined_output.lower()
        for phrase in ["already exists", "slug is unavailable", "409"]
    )
    if not already_exists:
        raise RuntimeError("Khong publish duoc Kaggle Dataset:\n" + combined_output)

    print("Dataset da ton tai, dang tao version moi...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(publish_dir),
            "-m",
            f"Update {CHECKPOINT_NAME} paired three-column gallery",
            "-r",
            "zip",
        ],
        check=True,
    )

print("Publish completed.")
print(f"Open: https://www.kaggle.com/datasets/{KAGGLE_DATASET_ID}")
