# ================================================================
# CELL 15: FULL PAIRED TEST EVALUATION - SSIM / PSNR / LPIPS
# Run after Cell 12, Cell 13, and Cell 14.
# ================================================================

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm.auto import tqdm

try:
    import lpips
except ImportError as exc:
    raise ImportError("Chay Cell 1 de cai lpips truoc khi danh gia.") from exc


PAIRED_TEST_SEED = 2026
PAIRED_TEST_LIMIT = None  # None = chay full clean_vto_dataset_test.csv
SAVE_RESULT_IMAGES = True

PAIRED_TEST_DIR = OUTPUT_DIR / "paired_test_metrics"
PAIRED_TEST_IMAGE_DIR = PAIRED_TEST_DIR / "images"
PAIRED_TEST_DIR.mkdir(parents=True, exist_ok=True)
if SAVE_RESULT_IMAGES:
    PAIRED_TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

print("Cell 15 started.", flush=True)
print(f"Output directory: {PAIRED_TEST_DIR}", flush=True)


required_runtime_objects = [
    "CSV_ROOT",
    "VITON_ROOT",
    "OUTPUT_DIR",
    "CHECKPOINT_NAME",
    "CHECKPOINT_SOURCE",
    "DEVICE",
    "run_inference",
    "_crop_resize",
]
missing_runtime_objects = [
    name for name in required_runtime_objects if name not in globals()
]
if missing_runtime_objects:
    raise RuntimeError(
        "Cell 15 thieu bien tu Cell 12-14: "
        + ", ".join(missing_runtime_objects)
    )


test_csv = CSV_ROOT / "clean_vto_dataset_test.csv"
print(f"Reading paired test IDs from: {test_csv}", flush=True)
test_df = pd.read_csv(test_csv)
if "person_id" not in test_df.columns:
    raise ValueError(f"{test_csv} thieu cot person_id")

# The original test CSV is unpaired. For reconstruction metrics, force
# cloth_id = person_id so that test/image/{person_id}.jpg is the ground truth.
paired_ids = test_df["person_id"].astype(str).drop_duplicates().tolist()
test_split_dir = VITON_ROOT / "test"
paired_ids = [
    sample_id
    for sample_id in paired_ids
    if (test_split_dir / "image" / f"{sample_id}.jpg").exists()
    and (test_split_dir / "cloth" / f"{sample_id}.jpg").exists()
    and (test_split_dir / "cloth-mask" / f"{sample_id}.jpg").exists()
    and (test_split_dir / "agnostic-v3.2" / f"{sample_id}.jpg").exists()
]
if PAIRED_TEST_LIMIT is not None:
    paired_ids = paired_ids[:PAIRED_TEST_LIMIT]

manifest = pd.DataFrame(
    {
        "sample_index": range(len(paired_ids)),
        "person_id": paired_ids,
        "cloth_id": paired_ids,
        "seed": [PAIRED_TEST_SEED + i for i in range(len(paired_ids))],
    }
)
manifest_path = PAIRED_TEST_DIR / "paired_test_manifest.csv"
manifest.to_csv(manifest_path, index=False)
print(f"Manifest ready: {len(manifest)} samples", flush=True)


metrics_path = PAIRED_TEST_DIR / "paired_test_metrics.csv"
if metrics_path.exists():
    try:
        metrics_df = pd.read_csv(metrics_path)
    except pd.errors.EmptyDataError:
        metrics_df = pd.DataFrame()
else:
    metrics_df = pd.DataFrame()

selected_ids = set(paired_ids)
if {"person_id", "status"}.issubset(metrics_df.columns):
    metrics_df["person_id"] = metrics_df["person_id"].astype(str)
    metrics_df = metrics_df[metrics_df["person_id"].isin(selected_ids)].copy()
    metrics_df = metrics_df.drop_duplicates(subset=["person_id"], keep="last")
else:
    metrics_df = pd.DataFrame()

finished_ids = set()
if len(metrics_df) > 0:
    finished_ids = set(
        metrics_df.loc[metrics_df["status"] == "ok", "person_id"].astype(str)
    )

print("Loading LPIPS AlexNet model. This step may take a few minutes...", flush=True)
lpips_model = lpips.LPIPS(net="alex").to(DEVICE).eval()
lpips_model.requires_grad_(False)
print("LPIPS model ready.", flush=True)


def _load_ground_truth(person_id):
    image = Image.open(test_split_dir / "image" / f"{person_id}.jpg").convert("RGB")
    image = _crop_resize(image)
    return np.asarray(image).astype(np.float32) / 255.0


def _lpips_score(gt_np, result_np):
    gt = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    result = torch.from_numpy(result_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        return float(lpips_model(gt * 2.0 - 1.0, result * 2.0 - 1.0).item())


def _save_result_image(image_np, path):
    image_u8 = (np.clip(image_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(image_u8).save(path, quality=95)


records = metrics_df.to_dict("records")


def _save_records(current_records):
    current_df = pd.DataFrame(current_records)
    current_df = current_df.drop_duplicates(subset=["person_id"], keep="last")
    current_df = current_df.sort_values("sample_index")
    current_df.to_csv(metrics_path, index=False)
    return current_df


print(f"Checkpoint       : {CHECKPOINT_NAME}", flush=True)
print(f"Checkpoint source: {CHECKPOINT_SOURCE}", flush=True)
print(f"Evaluation CSV   : {test_csv}", flush=True)
print(f"Paired samples   : {len(manifest)}", flush=True)
print(f"Already finished : {len(finished_ids)}", flush=True)
print("Rule             : cloth_id = person_id", flush=True)

for row in tqdm(
    manifest.itertuples(index=False),
    total=len(manifest),
    desc="Paired test SSIM/PSNR/LPIPS",
):
    person_id = str(row.person_id)
    if person_id in finished_ids:
        continue

    started = time.time()
    print(
        f"[{int(row.sample_index) + 1}/{len(manifest)}] "
        f"Inferencing {person_id} with cloth_id={person_id}...",
        flush=True,
    )
    try:
        result = run_inference(
            person_id=person_id,
            cloth_id=person_id,
            split="test",
            seed=int(row.seed),
            return_debug=False,
            show_progress=False,
        )
        gt = _load_ground_truth(person_id)

        record = {
            "checkpoint": CHECKPOINT_NAME,
            "sample_index": int(row.sample_index),
            "person_id": person_id,
            "cloth_id": person_id,
            "seed": int(row.seed),
            "ssim": float(
                structural_similarity(gt, result, data_range=1.0, channel_axis=2)
            ),
            "psnr": float(peak_signal_noise_ratio(gt, result, data_range=1.0)),
            "lpips": _lpips_score(gt, result),
            "seconds": float(time.time() - started),
            "status": "ok",
            "error": "",
        }
        if SAVE_RESULT_IMAGES:
            _save_result_image(
                result,
                PAIRED_TEST_IMAGE_DIR / f"{person_id}_result.jpg",
            )
        print(
            f"  done in {record['seconds']:.1f}s | "
            f"SSIM={record['ssim']:.4f} | PSNR={record['psnr']:.2f} | "
            f"LPIPS={record['lpips']:.4f}",
            flush=True,
        )
    except Exception as exc:
        record = {
            "checkpoint": CHECKPOINT_NAME,
            "sample_index": int(row.sample_index),
            "person_id": person_id,
            "cloth_id": person_id,
            "seed": int(row.seed),
            "seconds": float(time.time() - started),
            "status": "error",
            "error": str(exc),
        }
        print(f"  ERROR: {exc}", flush=True)

    records.append(record)
    _save_records(records)


metrics_df = _save_records(records)
ok_df = metrics_df[metrics_df["status"] == "ok"].copy()
error_df = metrics_df[metrics_df["status"] != "ok"].copy()

summary = {
    "checkpoint": CHECKPOINT_NAME,
    "checkpoint_source": str(CHECKPOINT_SOURCE),
    "evaluation_source": str(test_csv),
    "evaluation_mode": "paired test reconstruction: cloth_id = person_id",
    "samples_requested": int(len(manifest)),
    "samples_ok": int(len(ok_df)),
    "samples_error": int(len(error_df)),
    "metrics_mean": {
        metric: float(ok_df[metric].mean())
        for metric in ["ssim", "psnr", "lpips"]
        if len(ok_df) > 0
    },
    "metrics_std": {
        metric: float(ok_df[metric].std())
        for metric in ["ssim", "psnr", "lpips"]
        if len(ok_df) > 1
    },
}

summary_path = PAIRED_TEST_DIR / "paired_test_summary.json"
with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

if len(ok_df) > 0:
    metric_specs = [
        ("ssim", "SSIM", "higher is better", "#4C78A8"),
        ("psnr", "PSNR (dB)", "higher is better", "#59A14F"),
        ("lpips", "LPIPS", "lower is better", "#E15759"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, (column, label, direction, color) in zip(axes, metric_specs):
        values = ok_df[column].dropna()
        axis.hist(values, bins=30, color=color, edgecolor="white", alpha=0.9)
        axis.axvline(
            values.mean(),
            color="black",
            linestyle="--",
            label=f"Mean={values.mean():.4f}",
        )
        axis.set_title(f"{label} ({direction})")
        axis.set_xlabel(label)
        axis.set_ylabel("Samples")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    fig.suptitle(f"{CHECKPOINT_NAME} - Paired Test Metrics", fontweight="bold")
    fig.tight_layout()
    chart_path = PAIRED_TEST_DIR / "ssim_psnr_lpips_distributions.png"
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

print(json.dumps(summary, indent=2))
print(f"Metrics : {metrics_path}")
print(f"Summary : {summary_path}")
print(f"Chart   : {PAIRED_TEST_DIR / 'ssim_psnr_lpips_distributions.png'}")
