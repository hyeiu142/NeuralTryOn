"""Run resumable unpaired holdout inference using Model 1's shifted-cloth protocol."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


UNPAIRED_IMAGE_DIR = UNPAIRED_DIR / "images"
UNPAIRED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

source_df = pd.read_csv(CSV_ROOT / "holdout_test.csv")
persons = source_df["person_id"].astype(str).tolist()
cloths = source_df["cloth_id"].astype(str).tolist()
shifted_cloths = cloths[1:] + cloths[:1]
manifest_df = pd.DataFrame(
    {
        "sample_index": range(len(persons)),
        "person_id": persons,
        "source_cloth_id": cloths,
        "cloth_id": shifted_cloths,
    }
)
manifest_path = UNPAIRED_DIR / "unpaired_manifest.csv"
manifest_df.to_csv(manifest_path, index=False)

metrics_path = UNPAIRED_DIR / "unpaired_results.csv"
try:
    results_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
except pd.errors.EmptyDataError:
    results_df = pd.DataFrame()
records = results_df.to_dict("records")
finished = set()
if {"person_id", "cloth_id", "status"}.issubset(results_df.columns):
    finished = {
        (str(row.person_id), str(row.cloth_id))
        for row in results_df.itertuples(index=False)
        if row.status == "ok"
    }

print(f"Holdout rows       : {len(manifest_df)}")
print(f"Already finished   : {len(finished)}")
print("Protocol           : cloth list shifted by one position")

for row in tqdm(manifest_df.itertuples(index=False), total=len(manifest_df), desc="Unpaired holdout"):
    key = (str(row.person_id), str(row.cloth_id))
    if key in finished:
        continue
    started = time.time()
    stem = f"{int(row.sample_index):04d}_{row.person_id}_to_{row.cloth_id}"
    try:
        output = run_inference(row.person_id, row.cloth_id, mode="unpaired", return_debug=True)
        result_path = UNPAIRED_IMAGE_DIR / f"{stem}_result.jpg"
        Image.fromarray((np.clip(output["image"], 0, 1) * 255).round().astype(np.uint8)).save(
            result_path, quality=95
        )
        record = {
            "sample_index": int(row.sample_index),
            "person_id": str(row.person_id),
            "source_cloth_id": str(row.source_cloth_id),
            "cloth_id": str(row.cloth_id),
            "seconds": float(time.time() - started),
            "result_path": str(result_path),
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        record = {
            "sample_index": int(row.sample_index),
            "person_id": str(row.person_id),
            "source_cloth_id": str(row.source_cloth_id),
            "cloth_id": str(row.cloth_id),
            "seconds": float(time.time() - started),
            "status": "error",
            "error": str(exc),
        }
    records.append(record)
    pd.DataFrame(records).to_csv(metrics_path, index=False)

results_df = pd.DataFrame(records)
ok_df = results_df[results_df["status"] == "ok"]
summary = {
    "model": MODEL_NAME,
    "evaluation_source": str(CSV_ROOT / "holdout_test.csv"),
    "evaluation_mode": "unpaired holdout: cloth list shifted by one position",
    "samples_requested": int(len(manifest_df)),
    "samples_ok": int(len(ok_df)),
    "samples_error": int(len(results_df) - len(ok_df)),
    "mean_seconds": float(ok_df["seconds"].mean()) if len(ok_df) else None,
}
(UNPAIRED_DIR / "unpaired_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
