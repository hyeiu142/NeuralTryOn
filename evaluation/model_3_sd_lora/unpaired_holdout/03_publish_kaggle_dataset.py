"""Publish the complete unpaired holdout evaluation as a Kaggle Dataset."""

import json
import shutil
import subprocess
from pathlib import Path


KAGGLE_DATASET_OWNER = "khoaanh1234"
KAGGLE_DATASET_SLUG = "vto-v2-epoch12-unpaired-full-eval-20260612"
KAGGLE_DATASET_ID = f"{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_SLUG}"

source_dir = OUTPUT_DIR / "unpaired_evaluation"
if not source_dir.exists():
    raise FileNotFoundError(f"Chua co ket qua Cell 20: {source_dir}")

required_files = [
    source_dir / "unpaired_manifest.csv",
    source_dir / "unpaired_metrics.csv",
    source_dir / "unpaired_summary.json",
]
missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError("Thieu ket qua Cell 20:\n  " + "\n  ".join(missing_files))

publish_dir = Path("/kaggle/working/vto-v2-epoch12-unpaired-full-eval-20260612")
if publish_dir.exists():
    shutil.rmtree(publish_dir)
shutil.copytree(source_dir, publish_dir)

metadata = {
    "title": "VTO V2 Epoch 12 Unpaired Full Evaluation 20260612",
    "id": KAGGLE_DATASET_ID,
    "licenses": [{"name": "CC0-1.0"}],
}
with open(publish_dir / "dataset-metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print(f"Prepared private dataset: {publish_dir}")
print(f"Dataset id: {KAGGLE_DATASET_ID}")

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

if create_result.returncode == 0:
    print(create_result.stdout)
    print("Private dataset created successfully.")
else:
    combined_output = f"{create_result.stdout}\n{create_result.stderr}"
    already_exists = any(
        phrase in combined_output.lower()
        for phrase in ["already exists", "dataset slug is unavailable", "409"]
    )
    if not already_exists:
        raise RuntimeError(
            "Khong tao duoc Kaggle Dataset private:\n" + combined_output
        )

    print("Dataset da ton tai, dang tao version moi...")
    version_command = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(publish_dir),
        "-m",
        f"Update {CHECKPOINT_NAME} full unpaired holdout results",
        "-r",
        "zip",
    ]
    subprocess.run(version_command, check=True)
    print("Private dataset version updated successfully.")

print(f"Open: https://www.kaggle.com/datasets/{KAGGLE_DATASET_ID}")
