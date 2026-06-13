"""Run resumable paired-test inference and compute SSIM, PSNR, and LPIPS."""

import json
import time
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm.auto import tqdm


PAIRED_IMAGE_DIR = PAIRED_DIR / "images"
PAIRED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Preserve the protocol used by the completed Model 2 notebook evaluation.
paired_ids = sorted(
    path.stem for path in (TEST_DIR / "image").iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)
manifest_df = pd.DataFrame(
    {
        "sample_index": range(len(paired_ids)),
        "person_id": paired_ids,
        "cloth_id": paired_ids,
    }
)
manifest_path = PAIRED_DIR / "paired_test_manifest.csv"
manifest_df.to_csv(manifest_path, index=False)

metrics_path = PAIRED_DIR / "paired_test_metrics.csv"
try:
    existing_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
except pd.errors.EmptyDataError:
    existing_df = pd.DataFrame()

finished = set()
records = []
if {"person_id", "status"}.issubset(existing_df.columns):
    existing_df = existing_df.drop_duplicates("person_id", keep="last")
    records = existing_df.to_dict("records")
    finished = set(existing_df.loc[existing_df["status"] == "ok", "person_id"].astype(str))

lpips_model = lpips.LPIPS(net="vgg").to(DEVICE).eval()


def _lpips_value(target, prediction):
    target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    pred_tensor = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return float(lpips_model(target_tensor * 2 - 1, pred_tensor * 2 - 1).item())


def _save_rgb(image, path):
    Image.fromarray((np.clip(image, 0, 1) * 255).round().astype(np.uint8)).save(
        path, quality=95
    )


print(f"Paired test samples : {len(manifest_df)}")
print(f"Already finished    : {len(finished)}")
print("Protocol            : cloth_id = person_id, all files in test/image")

for row in tqdm(manifest_df.itertuples(index=False), total=len(manifest_df), desc="Paired test"):
    person_id = str(row.person_id)
    if person_id in finished:
        continue

    started = time.time()
    try:
        output = run_inference(person_id, person_id, return_debug=True)
        target = output["person"]
        prediction = output["image"]
        result_path = PAIRED_IMAGE_DIR / f"{person_id}_result.jpg"
        _save_rgb(prediction, result_path)

        record = {
            "checkpoint": CHECKPOINT_NAME,
            "sample_index": int(row.sample_index),
            "person_id": person_id,
            "cloth_id": person_id,
            "ssim": float(
                structural_similarity(target, prediction, data_range=1.0, channel_axis=2)
            ),
            "psnr": float(peak_signal_noise_ratio(target, prediction, data_range=1.0)),
            "lpips": _lpips_value(target, prediction),
            "seconds": float(time.time() - started),
            "result_path": str(result_path),
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        record = {
            "checkpoint": CHECKPOINT_NAME,
            "sample_index": int(row.sample_index),
            "person_id": person_id,
            "cloth_id": person_id,
            "seconds": float(time.time() - started),
            "status": "error",
            "error": str(exc),
        }

    records.append(record)
    pd.DataFrame(records).to_csv(metrics_path, index=False)

metrics_df = pd.DataFrame(records)
ok_df = metrics_df[metrics_df["status"] == "ok"]
metric_names = ["ssim", "psnr", "lpips", "seconds"]
summary = {
    "model": MODEL_NAME,
    "checkpoint": CHECKPOINT_NAME,
    "evaluation_source": str(TEST_DIR / "image"),
    "evaluation_mode": "paired reconstruction: all test/image files, cloth_id = person_id",
    "samples_requested": int(len(manifest_df)),
    "samples_ok": int(len(ok_df)),
    "samples_error": int(len(metrics_df) - len(ok_df)),
    "metrics_mean": {name: float(ok_df[name].mean()) for name in metric_names if len(ok_df)},
    "metrics_std": {name: float(ok_df[name].std()) for name in metric_names if len(ok_df) > 1},
}
summary_path = PAIRED_DIR / "paired_test_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"Metrics : {metrics_path}")
print(f"Summary : {summary_path}")
