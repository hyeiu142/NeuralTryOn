"""Publish Model 1 paired-test metrics and galleries as a Kaggle Dataset."""

import json
import shutil
import subprocess
from pathlib import Path


KAGGLE_DATASET_OWNER = "khoaanh1234"
KAGGLE_DATASET_SLUG = "vto-model1-lightweight-unet-paired-evaluation"
KAGGLE_DATASET_ID = f"{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_SLUG}"
publish_dir = Path("/kaggle/working") / KAGGLE_DATASET_SLUG

required = [
    PAIRED_DIR / "paired_test_manifest.csv",
    PAIRED_DIR / "paired_test_metrics.csv",
    PAIRED_DIR / "paired_test_summary.json",
    PAIRED_DIR / "metric_report",
    PAIRED_DIR / "comparisons_3_columns",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Thieu paired evaluation output:\n  " + "\n  ".join(missing))

if publish_dir.exists():
    shutil.rmtree(publish_dir)
shutil.copytree(PAIRED_DIR, publish_dir)
(publish_dir / "dataset-metadata.json").write_text(
    json.dumps(
        {
            "title": "VTO Model 1 Lightweight U-Net Paired Evaluation",
            "id": KAGGLE_DATASET_ID,
            "licenses": [{"name": "CC0-1.0"}],
        },
        indent=2,
    ),
    encoding="utf-8",
)

command = ["kaggle", "datasets", "create", "-p", str(publish_dir), "-r", "zip"]
result = subprocess.run(command, text=True, capture_output=True)
output = f"{result.stdout}\n{result.stderr}"
print(output)
if result.returncode != 0:
    if not any(text in output.lower() for text in ["already exists", "unavailable", "409"]):
        raise RuntimeError("Publish Kaggle Dataset that bai:\n" + output)
    subprocess.run(
        [
            "kaggle", "datasets", "version", "-p", str(publish_dir),
            "-m", "Update Model 1 paired evaluation", "-r", "zip",
        ],
        check=True,
    )
print(f"Open: https://www.kaggle.com/datasets/{KAGGLE_DATASET_ID}")
