"""Publish Model 2 paired evaluation outputs as a Kaggle Dataset."""

import json
import shutil
import subprocess
from pathlib import Path


SLUG = "vto-model2-pix2pix-paired-evaluation"
DATASET_ID = f"khoaanh1234/{SLUG}"
publish_dir = Path("/kaggle/working") / SLUG
required = [
    PAIRED_DIR / "paired_test_manifest.csv",
    PAIRED_DIR / "paired_test_metrics.csv",
    PAIRED_DIR / "paired_test_summary.json",
    PAIRED_DIR / "metric_report",
    PAIRED_DIR / "comparisons_3_columns",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Thieu paired output:\n  " + "\n  ".join(missing))
if publish_dir.exists():
    shutil.rmtree(publish_dir)
shutil.copytree(PAIRED_DIR, publish_dir)
(publish_dir / "dataset-metadata.json").write_text(
    json.dumps({"title": "VTO Model 2 Pix2Pix Paired Evaluation", "id": DATASET_ID, "licenses": [{"name": "CC0-1.0"}]}, indent=2),
    encoding="utf-8",
)
result = subprocess.run(["kaggle", "datasets", "create", "-p", str(publish_dir), "-r", "zip"], text=True, capture_output=True)
output = f"{result.stdout}\n{result.stderr}"
print(output)
if result.returncode != 0:
    if not any(text in output.lower() for text in ["already exists", "unavailable", "409"]):
        raise RuntimeError(output)
    subprocess.run(["kaggle", "datasets", "version", "-p", str(publish_dir), "-m", "Update Model 2 paired evaluation", "-r", "zip"], check=True)
print(f"Open: https://www.kaggle.com/datasets/{DATASET_ID}")
