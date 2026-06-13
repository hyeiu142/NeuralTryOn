"""Run resumable Model 2 inference on the original holdout person-cloth pairs."""

import json
import time

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


UNPAIRED_IMAGE_DIR = UNPAIRED_DIR / "images"
UNPAIRED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
manifest_df = pd.read_csv(CSV_ROOT / "holdout_test.csv")
manifest_df["person_id"] = manifest_df["person_id"].astype(str)
manifest_df["cloth_id"] = manifest_df["cloth_id"].astype(str)
manifest_df = manifest_df.reset_index(drop=True)
manifest_df["sample_index"] = manifest_df.index
manifest_df.to_csv(UNPAIRED_DIR / "unpaired_manifest.csv", index=False)

results_path = UNPAIRED_DIR / "unpaired_results.csv"
try:
    results_df = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
except pd.errors.EmptyDataError:
    results_df = pd.DataFrame()
records = results_df.to_dict("records")
finished = {
    (str(row.person_id), str(row.cloth_id))
    for row in results_df.itertuples(index=False)
    if getattr(row, "status", "") == "ok"
}

print(f"Holdout rows     : {len(manifest_df)}")
print(f"Already finished : {len(finished)}")
print("Protocol         : original person_id and cloth_id from holdout_test.csv")

for row in tqdm(manifest_df.itertuples(index=False), total=len(manifest_df), desc="Model 2 holdout"):
    key = (str(row.person_id), str(row.cloth_id))
    if key in finished:
        continue
    started = time.time()
    stem = f"{int(row.sample_index):04d}_{row.person_id}_to_{row.cloth_id}"
    try:
        output = run_inference(row.person_id, row.cloth_id, return_debug=True)
        result_path = UNPAIRED_IMAGE_DIR / f"{stem}_result.jpg"
        Image.fromarray((np.clip(output["image"], 0, 1) * 255).round().astype(np.uint8)).save(
            result_path, quality=95
        )
        record = {
            "sample_index": int(row.sample_index),
            "person_id": str(row.person_id),
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
            "cloth_id": str(row.cloth_id),
            "seconds": float(time.time() - started),
            "status": "error",
            "error": str(exc),
        }
    records.append(record)
    pd.DataFrame(records).to_csv(results_path, index=False)

results_df = pd.DataFrame(records)
ok_df = results_df[results_df["status"] == "ok"]
summary = {
    "model": MODEL_NAME,
    "evaluation_source": str(CSV_ROOT / "holdout_test.csv"),
    "evaluation_mode": "unpaired holdout: original person_id and cloth_id",
    "samples_requested": int(len(manifest_df)),
    "samples_ok": int(len(ok_df)),
    "samples_error": int(len(results_df) - len(ok_df)),
    "mean_seconds": float(ok_df["seconds"].mean()) if len(ok_df) else None,
}
(UNPAIRED_DIR / "unpaired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
