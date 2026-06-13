"""Publish Model 2 unpaired holdout outputs as a Kaggle Dataset."""

import json
import shutil
import subprocess
from pathlib import Path


SLUG = "vto-model2-pix2pix-unpaired-holdout"
DATASET_ID = f"khoaanh1234/{SLUG}"
publish_dir = Path("/kaggle/working") / SLUG
required = [
    UNPAIRED_DIR / "unpaired_manifest.csv",
    UNPAIRED_DIR / "unpaired_results.csv",
    UNPAIRED_DIR / "unpaired_summary.json",
    UNPAIRED_DIR / "qualitative_report",
    UNPAIRED_DIR / "comparisons_3_columns",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Thieu holdout output:\n  " + "\n  ".join(missing))
if publish_dir.exists():
    shutil.rmtree(publish_dir)
shutil.copytree(UNPAIRED_DIR, publish_dir)
(publish_dir / "dataset-metadata.json").write_text(
    json.dumps({"title": "VTO Model 2 Pix2Pix Unpaired Holdout", "id": DATASET_ID, "licenses": [{"name": "CC0-1.0"}]}, indent=2),
    encoding="utf-8",
)
result = subprocess.run(["kaggle", "datasets", "create", "-p", str(publish_dir), "-r", "zip"], text=True, capture_output=True)
output = f"{result.stdout}\n{result.stderr}"
print(output)
if result.returncode != 0:
    if not any(text in output.lower() for text in ["already exists", "unavailable", "409"]):
        raise RuntimeError(output)
    subprocess.run(["kaggle", "datasets", "version", "-p", str(publish_dir), "-m", "Update Model 2 holdout", "-r", "zip"], check=True)
print(f"Open: https://www.kaggle.com/datasets/{DATASET_ID}")
